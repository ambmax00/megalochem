#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include <dbcsr_tensor_ops.hpp>

namespace fock {
	
JK_common::JK_common(dbcsr::world& w, desc::options opt) :
	m_world(w), m_opt(opt),
	LOG(m_world.comm(),m_opt.get<int>("print", FOCK_PRINT_LEVEL)),
	TIME(m_world.comm(), "Fock Builder", LOG.global_plev()) {}
	
void J::init() {
	
	// set up J
	dbcsr::pgrid<2> grid2(m_world.comm());
	auto b = m_p_A->row_blk_sizes();
	arrvec<int,2> bb = {b,b};
	m_J = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().name("J_bb")
		.ngrid(grid2).map1({0}).map2({1}).blk_sizes(bb));
	
}

void K::init() {
	
	// set up K's
	dbcsr::pgrid<2> grid2(m_world.comm());
	auto b = m_p_A->row_blk_sizes();
	arrvec<int,2> bb = {b,b};
	m_K_A = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().name("K_bb_A")
		.ngrid(grid2).map1({0}).map2({1}).blk_sizes(bb));
	if (m_p_B) m_K_B = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create_template(*m_K_A).name("K_bb_B"));
	
}

EXACT_J::EXACT_J(dbcsr::world& w, desc::options& iopt) : J(w,iopt) {} 
EXACT_K::EXACT_K(dbcsr::world& w, desc::options& iopt) : K(w,iopt) {}

void EXACT_J::compute_J() {
	
	if (!m_J_bbd) {
		// allocate J and Ptot for first time
		dbcsr::pgrid<3> grid3(m_world.comm());
		auto b = m_p_A->row_blk_sizes();
		vec<int> d = {1};
		
		arrvec<int,3> bbd = {b,b,d};
		
		m_J_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create().ngrid(grid3).name("J dummy")
			.map1({0,1}).map2({2}).blk_sizes(bbd));
		
		m_ptot_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create_template(*m_J_bbd).name("ptot dummy"));
		
	}
	
	// copy P 
	if (m_p_A && !m_p_B) {
		dbcsr::copy_matrix_to_3Dtensor<double>(*m_p_A,*m_ptot_bbd,false,true);
		m_ptot_bbd->scale(2.0);
	} else {
		dbcsr::copy_matrix_to_3Dtensor<double>(*m_p_A,*m_ptot_bbd,false,true);
		dbcsr::copy_matrix_to_3Dtensor<double>(*m_p_B,*m_ptot_bbd,true,true);
	}
	
	if (LOG.global_plev() >= 3) {
		dbcsr::print(*m_ptot_bbd);
	}
	
	// grab integrals
	auto eris = m_factory->ao_eri(vec<int>{0,1},vec<int>{2,3},true,true);
	
	dbcsr::contract(*m_ptot_bbd, *eris, *m_J_bbd).print(true).perform("LS_, MNLS -> MN_");
	
	if (LOG.global_plev() >= 3) {
		dbcsr::print(*m_J_bbd);
	}

	dbcsr::copy_3Dtensor_to_2Dtensor<double>(*m_J_bbd,*m_J,false);
	
	m_ptot_bbd->clear();
	m_J_bbd->clear();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_J);
	}

}

void EXACT_K::compute_K() {
	
	if (!m_K_bbd) {
		// allocate J and Ptot for first time
		dbcsr::pgrid<3> grid3(m_world.comm());
		auto b = m_p_A->row_blk_sizes();
		vec<int> d = {1};
		
		arrvec<int,3> bbd = {b,b,d};
		
		m_K_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create().ngrid(grid3).name("K dummy")
			.map1({0,1}).map2({2}).blk_sizes(bbd));
		
		m_p_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create_template(*m_K_bbd).name("p dummy"));
		
	}
	
	auto eris = m_factory->ao_eri(vec<int>{0,2},vec<int>{1,3},true,true);
	
	auto compute_K_single = [&](dbcsr::smat_d& p, dbcsr::stensor2_d& k, std::string x) {
		
		dbcsr::copy_matrix_to_3Dtensor<double>(*p,*m_p_bbd,false,true);
		
		LOG.os<1>("Computing exchange term (", x, ") ... \n");
			
		dbcsr::contract(*m_p_bbd, *eris, *m_K_bbd).alpha(-1.0).perform("LS_, MLSN -> MN_");

		dbcsr::copy_3Dtensor_to_2Dtensor(*m_K_bbd,*k);
		
		dbcsr::print(*m_K_bbd);
		
		m_K_bbd->clear();
		m_p_bbd->clear();
		
		if (LOG.global_plev() >= 2) {
			dbcsr::print(*k);
		}
		
	};
	
	compute_K_single(m_p_A, m_K_A, "A");
	if (m_K_B) compute_K_single(m_p_B, m_K_B, "B");
	
}
		
	
} // end namespace
