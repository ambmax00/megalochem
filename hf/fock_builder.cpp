#include "hf/fock_builder.h"
#include "hf/hfdefaults.h"
#include "ints/aofactory.h"
#include "math/linalg/symmetrize.h"
#include "math/linalg/inverse.h"

#include <utility>
#include <iomanip>

namespace hf {
	
fockbuilder::fockbuilder(desc::smolecule mol, desc::options opt, MPI_Comm comm, int print,
	util::mpi_time& TIME_IN) :
	m_mol(mol), 
	m_opt(opt), 
	m_comm(comm),
	LOG(comm,print),
	TIME(TIME_IN),
	m_use_df(opt.get<bool>("use_df", HF_USE_DF)) {

	// build the integrals 
	ints::aofactory ao(*m_mol, m_comm);
	
	m_restricted = (m_mol->nele_alpha() == m_mol->nele_beta()) ? true : false;
	
	std::cout << std::scientific;
	std::cout << std::setprecision(12);
	
	if (!m_use_df) {
		
		auto& t_int2ele = TIME.sub("2e electron integrals");
		
		t_int2ele.start();
	
		LOG.os<>("Computing 2e integrals...\n");
		m_2e_ints = ao.compute<4>({.op = "coulomb",  .bas = "bbbb", .name="i_bbbb", .map1 = {0,1}, .map2 = {2,3}});
		
		t_int2ele.finish();
		
		LOG.os<>("Done with 2e integrals...\n");
		
	} else {
		
		auto& t_int3c2e = TIME.sub("3-centre-2-electron integrals");
		
		t_int3c2e.start();
		
		LOG.os<>("Computing 3c2e integrals...\n");
		m_3c2e_ints = ao.compute<3>({.op = "coulomb", .bas = "xbb", .name="d_xbb", .map1 = {0}, .map2 = {1,2}});
		
		t_int3c2e.finish();
		
		LOG.os<>("Done with 3c2e integrals...\n");
		
		std::cout << "PLEV: " << LOG.global_plev() << std::endl;
		
		if (LOG.global_plev() >= 3) {
			dbcsr::print(*m_3c2e_ints);
		}
		
		auto& t_metric = TIME.sub("Metric Matrix");
		
		t_metric.start();
		
		LOG.os<>("Computing metric...\n");
		auto m_xx = ao.compute<2>({.op = "coulomb", .bas = "xx", .name="i_xx", .map1={0}, .map2={1}});
		
		t_metric.finish();
		
		LOG.os<>("Done.\n");
		
		//m_xx = math::symmetrize(m_xx, "xx");
		
		auto& t_inv = TIME.sub("Inverse of metric matrix");
		
		t_inv.start();
		
		LOG.os<>("Computing inverse...\n");
		m_inv_xx = math::eigen_inverse(m_xx, "inv_xx");
		LOG.os<>("Done.\n");
		
		t_inv.finish();
		
		m_xx->destroy();
		
	}
		
	// set up tensors
	dbcsr::pgrid<2> grid({.comm = m_comm});
	
	auto b = m_mol->dims().b();
	
	m_j_bb = dbcsr::make_stensor<2>({.name = "j_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = {b,b}});
	m_k_bb_A = dbcsr::make_stensor<2>({.name = "k_bb_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = {b,b}});
	m_f_bb_A = dbcsr::make_stensor<2>({.name = "f_bb_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = {b,b}});
	
	if (!m_restricted && m_mol->nele_beta() != 0) {
		m_k_bb_B = dbcsr::make_stensor<2>(
			{.name = "k_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = {b,b}});
		m_f_bb_B = dbcsr::make_stensor<2>(
			{.name = "f_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = {b,b}});	
	}

	grid.destroy();

}

void fockbuilder::build_j(compute_param&& pms) {
	
	// build PT = PA + PB
	dbcsr::pgrid<2> grid({.comm = m_comm});
	dbcsr::tensor<2> pt({.name = "ptot", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = pms.p_A->blk_size()});
	
	dbcsr::copy<2>({.t_in = *pms.p_A, .t_out = pt});
	
	if (pms.p_B) {
		std::cout << "Probs here..." << std::endl;
		dbcsr::copy<2>({.t_in = *pms.p_B, .t_out = pt, .sum = true});
	} else {
		pt.scale(2.0);
	}
	
	grid.destroy();
	
	auto& t_j = TIME.sub("Coulomb Term");
	
	t_j.start();
	
	LOG.os<1>("Computing coulomb term... \n");
	
	bool log = false;
	int u = 0;
	
	if (LOG.global_plev() >= 2) {
		u = 6;
		log = true;
	}
	
	if (!m_use_df) {
		
		auto pt_bbY = dbcsr::add_dummy(pt);
		auto j_bbY = dbcsr::add_dummy(*m_j_bb);
	
		dbcsr::einsum<3,4,3>({.x = "LS_, MNLS -> MN_", .t1 = pt_bbY, .t2 = *m_2e_ints, .t3 = j_bbY, .unit_nr = u, .log = log});
		
		*m_j_bb = dbcsr::remove_dummy(j_bbY, vec<int>{0}, vec<int>{1});
		
		pt_bbY.destroy();
		j_bbY.destroy();
		
	} else {
		
		dbcsr::pgrid<2> grid2({.comm = m_comm});
		dbcsr::pgrid<3> grid3({.comm = m_comm});
		
		vec<vec<int>> sizes = {m_mol->dims().x(),vec<int>{1}};
		
		dbcsr::tensor<2> c_xY({.name = "c_xY", .pgridN = grid2, .map1 = {0}, .map2 = {1}, .blk_sizes = sizes});
		dbcsr::tensor<2> d_xY({.name = "d_xY", .pgridN = grid2, .map1 = {0}, .map2 = {1}, .blk_sizes = sizes});
		
		auto pt_bbY = dbcsr::add_dummy(pt);
		auto j_bbY = dbcsr::add_dummy(*m_j_bb);
		
		//cX("M") = PT("mu,nu") * B("M,mu,nu");
		dbcsr::einsum<3,3,2>({.x = "MN_, XMN -> X_", .t1 = pt_bbY, .t2 = *m_3c2e_ints, .t3 = c_xY, .unit_nr = u, .log = log});
		
		//dX("M") = Jinv("M,N") * cX("N");
		dbcsr::einsum<2,2,2>({.x = "XY, Y_ -> X_", .t1 = *m_inv_xx, .t2 = c_xY, .t3 = d_xY, .unit_nr = u, .log = log});
		
		//j("mu,nu") = dX("M") * B("M,mu,nu");
		dbcsr::einsum<2,3,3>({.x = "X_, XMN -> MN_", .t1 = d_xY, .t2 = *m_3c2e_ints, .t3 = j_bbY, .unit_nr = u, .log = log});
		
		m_j_bb->filter();
	
		*m_j_bb = dbcsr::remove_dummy(j_bbY, vec<int>{0}, vec<int>{1});
		
		//m_j_bb = math::symmetrize(m_j_bb, "j_bb");
	
		c_xY.destroy();
		d_xY.destroy();
		j_bbY.destroy();
		pt_bbY.destroy();
		grid2.destroy();
		grid3.destroy();
	
	}
	
	t_j.finish();
	
	LOG.os<1>("Done.\n");
	
	if (LOG.global_plev() >= 2) dbcsr::print(*m_j_bb);
	
	pt.destroy();
	
}

void fockbuilder::build_k(compute_param&& pms) {
	
	bool log = false;
	int u = 0;
	
	auto& t_k = TIME.sub("Exchange Term");
	
	t_k.start();
	
	if (LOG.global_plev() >= 2) {
		u = 6;
		log = true;
	}
	
	if (!m_use_df) {
		
		auto make_k = [&](dbcsr::tensor<2,double>& p_bb, dbcsr::tensor<2,double>& k_bb, std::string x) {
		
			LOG.os<1>("Computing exchange term (", x, ") ... \n");
		
			auto p_bbY = dbcsr::add_dummy(p_bb);
			auto k_bbY = dbcsr::add_dummy(k_bb);
			
			dbcsr::einsum<3,4,3>({.x = "LS_, MLSN -> MN_", .t1 = p_bbY, .t2 = *m_2e_ints, .t3 = k_bbY, .alpha = -1.0, .unit_nr = u, .log = log});
		
			//std::cout << "DONE CONTRACTING..." << std::endl;
		
			k_bb = dbcsr::remove_dummy(k_bbY, vec<int>{0}, vec<int>{1});
			
			//dbcsr::print(k_bb);
			
			p_bbY.destroy();
			k_bbY.destroy();
		
		};
		
		make_k(*pms.p_A, *m_k_bb_A, "A");
		
		if (pms.p_B && pms.c_B) 
			make_k(*pms.p_B, *m_k_bb_B, "B");
			
		
		
	} else {
		
		auto make_k = [&](dbcsr::tensor<2,double>& p_bb, dbcsr::tensor<2,double>& c_bm, 
			dbcsr::tensor<2,double>& k_bb, std::string x, bool SAD_iter) {
				
			LOG.os<1>("Computing exchange term (", x, ") ... \n");
			
			dbcsr::pgrid<3> grid3({.comm = m_comm});
			
			vec<int> o = c_bm.blk_size()[1];
			
			vec<vec<int>> HTsizes = {m_mol->dims().x(), m_mol->dims().b(), o};
			
			dbcsr::tensor<3> HT({.name = "HT"+x, .pgridN = grid3, .map1 = {0,1}, .map2 = {2}, .blk_sizes = HTsizes});	
			dbcsr::tensor<3> D({.name = "D"+x, .pgridN = grid3, .map1 = {0}, .map2 = {1,2}, .blk_sizes = HTsizes});
			
			// HTa("M,mu,i") = Coa("nu,i") * B("M,mu,nu");
			if (!SAD_iter) {
				
				int nocc = (x == "A") ? m_mol->nocc_alpha() - 1 : m_mol->nocc_beta() - 1;	
				
				std::cout << "NOCC: " << nocc << std::endl;
							
				vec<vec<int>> occ_bounds = {{0,nocc}};
				
				dbcsr::einsum<2,3,3>({.x = "Ni, XMN -> XMi", .t1 = c_bm, .t2 = *m_3c2e_ints, .t3 = HT, .b2 = occ_bounds, .unit_nr = u, .log = log});
				
			} else {
				
				dbcsr::einsum<2,3,3>({.x = "Ni, XMN -> XMi", .t1 = c_bm, .t2 = *m_3c2e_ints, .t3 = HT, .unit_nr = u, .log = log});
				
			}
			
			HT.filter();
			
			//Da("M,mu,i") = HTa("N,mu,i") * Jinv("N,M");
			dbcsr::einsum<3,2,3>({.x = "XMi, XY -> YMi", .t1 = HT, .t2 = *m_inv_xx, .t3 = D, .unit_nr = u, .log = log});
			
			D.filter();
			//std::cout << "Stuck here: " << x << " AS" << std::endl;
			
			//ka("mu,nu") = HTa("M,mu,i") * Da("M,nu,i");
			dbcsr::einsum<3,3,2>({.x = "XMi, XNi -> MN", .t1 = HT, .t2 = D, .t3 = k_bb, .alpha = -1.0, .unit_nr = u, .log = log}); //<- REORDERING!!!
			
			k_bb.filter();
			//std::cout << "NOPE." << std::endl;
			//k_bb = math::symmetrize(k_bb, k_bb.name());
			
			HT.destroy();
			D.destroy();
			grid3.destroy();
		
		};
		
		bool SAD_iter = (pms.SAD_iter) ? *pms.SAD_iter : false;
		
		make_k(*pms.p_A, *pms.c_A, *m_k_bb_A, "A", SAD_iter);
		
		if (pms.p_B && pms.c_B) 
			make_k(*pms.p_B, *pms.c_B, *m_k_bb_B, "B", SAD_iter);
			
	}
	
	t_k.finish();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_k_bb_A);
		if (pms.p_B && pms.c_B) dbcsr::print(*m_k_bb_B);
	}
	
	
}

void fockbuilder::compute(compute_param&& pms) {
	
	build_j(std::forward<compute_param>(pms));
	build_k(std::forward<compute_param>(pms));
	
	dbcsr::copy<2>({.t_in = *pms.core, .t_out = *m_f_bb_A, .sum = false});
	dbcsr::copy<2>({.t_in = *m_j_bb, .t_out = *m_f_bb_A, .sum = true, .move_data = false});
	dbcsr::copy<2>({.t_in = *m_k_bb_A, .t_out = *m_f_bb_A, .sum = true, .move_data = true});
	
	if (m_f_bb_B) {
		dbcsr::copy<2>({.t_in = *pms.core, .t_out = *m_f_bb_B, .sum = false});
		dbcsr::copy<2>({.t_in = *m_j_bb, .t_out = *m_f_bb_B, .sum = true, .move_data = true});
		dbcsr::copy<2>({.t_in = *m_k_bb_B, .t_out = *m_f_bb_B, .sum = true, .move_data = true});
	}
	
	LOG.os<2>("Finished Fock Matrix:\n");
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_f_bb_A);
		if (m_f_bb_B) dbcsr::print(*m_f_bb_B);
	}
	
}



		
} // end namespace
