#include "hf/fock_builder.h"
#include "hf/hfdefaults.h"
#include "ints/aofactory.h"

#include <utility>
#include <iomanip>

namespace hf {
	
fockbuilder::fockbuilder(desc::molecule& mol, desc::options& opt, MPI_Comm comm) :
	m_mol(mol), 
	m_opt(opt), 
	m_comm(comm),
	m_restricted(opt.get<bool>("restricted", true)),
	m_use_df(opt.get<bool>("use_df", HF_USE_DF)) {

	// build the integrals 
	ints::aofactory ao(m_mol, m_comm);
	
	std::cout << std::scientific;
	std::cout << std::setprecision(12);
	
	if (!m_use_df) {
	
		std::cout << "COMPUTING 2EINTS" << std::endl;
		m_2e_ints = std::make_optional<dbcsr::tensor<4>>(
			ao.compute<4>({.op = "coulomb",  .bas = "bbbb", .name="i_bbbb", .map1 = {0,1}, .map2 = {2,3}}));
		
		std::cout << "INTS:" << std::endl;
		dbcsr::print(*m_2e_ints);
		
	}
	
	// set up tensors
	dbcsr::pgrid<2> grid({.comm = m_comm});
	
	auto b = m_mol.dims().b();
	
	dbcsr::tensor<2> j_bb({.name = "j_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = {b,b}});
	m_j_bb = std::move(j_bb);
	
	dbcsr::tensor<2> k_bb_A({.name = "k_bb_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = {b,b}});
	m_k_bb_A = std::move(k_bb_A);
	
	dbcsr::tensor<2> f_bb_A({.name = "f_bb_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = {b,b}});
	m_f_bb_A = std::move(f_bb_A);
	
	if (!m_restricted && m_mol.nocc_beta() != 0) {
		dbcsr::tensor<2> k_bb_B({.name = "k_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = {b,b}});
		m_k_bb_B = std::move(k_bb_B);
		dbcsr::tensor<2> f_bb_B({.name = "f_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = {b,b}});
		m_f_bb_B = std::move(f_bb_B);
	}

	grid.destroy();

}

void fockbuilder::build_j(compute_param&& pms) {
	
	// build PT = PA + PB
	auto pt = (pms.p_B) ? *pms.p_A + *pms.p_B : 2.0 * (*pms.p_A);
	
	std::cout << "PT" << std::endl;
	dbcsr::print(pt);
	
	if (!m_use_df) {
		
		auto pt_bbY = dbcsr::add_dummy(pt);
		auto j_bbY = dbcsr::add_dummy(m_j_bb);
	
		dbcsr::einsum<3,4,3>({.x = "LS_, MNSL -> MN_", .t1 = pt_bbY, .t2 = *m_2e_ints, .t3 = j_bbY});
		
		std::cout << "DONE CONTRACTING..." << std::endl;
		
		dbcsr::print(j_bbY);
		
		m_j_bb = dbcsr::remove_dummy(j_bbY, vec<int>{0}, vec<int>{1});
		
		std::cout << "J" << std::endl;
		dbcsr::print(m_j_bb);
		
		
		pt_bbY.destroy();
		j_bbY.destroy();
		
	}
	
	pt.destroy();
	
}

void fockbuilder::build_k(compute_param&& pms) {
	
	if (!m_use_df) {
		
		auto make_k = [&](dbcsr::tensor<2,double>& p_bb, dbcsr::tensor<2,double>& k_bb, std::string x) {
		
			auto p_bbY = dbcsr::add_dummy(p_bb);
			auto k_bbY = dbcsr::add_dummy(k_bb);
			
			dbcsr::einsum<3,4,3>({.x = "LS_, MNSL -> MN_", .t1 = p_bbY, .t2 = *m_2e_ints, .t3 = k_bbY, /*.unit_nr = 6, .log = true*/});
		
			std::cout << "DONE CONTRACTING..." << std::endl;
		
			k_bb = dbcsr::remove_dummy(k_bbY, vec<int>{0}, vec<int>{1});
			
			dbcsr::print(k_bb);
			
			k_bb.scale(-1);
			
			p_bbY.destroy();
			k_bbY.destroy();
		
		};
		
		make_k(*pms.p_A, m_k_bb_A, "A");
		if (!m_restricted && m_mol.nocc_beta() != 0) 
			make_k(*pms.p_B, m_k_bb_B, "B");
		
	}
	
	
}

void fockbuilder::compute(compute_param&& pms) {
	
	build_j(std::forward<compute_param>(pms));
	build_k(std::forward<compute_param>(pms));
	
	std::cout << "core" << std::endl;
	
	dbcsr::copy<2>({.t_in = *pms.core, .t_out = m_f_bb_A, .sum = false});
	dbcsr::copy<2>({.t_in = m_j_bb, .t_out = m_f_bb_A, .sum = true, .move_data = true});
	dbcsr::copy<2>({.t_in = m_k_bb_A, .t_out = m_f_bb_A, .sum = true, .move_data = true});
	
	dbcsr::print(m_f_bb_A);
	
	std::cout << "DONE." << std::endl;
	
	if (!m_restricted && m_mol.nocc_beta() != 0)
		m_f_bb_B = *pms.core + m_j_bb + m_k_bb_B;
	
	m_j_bb.clear();
	m_k_bb_A.clear();
	
}



		
} // end namespace
