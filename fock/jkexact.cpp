#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include "math/linalg/LLT.h"
#include <dbcsr_tensor_ops.hpp>

namespace fock {

EXACT_J::EXACT_J(dbcsr::world& w, desc::options& iopt) : J(w,iopt) {} 
EXACT_K::EXACT_K(dbcsr::world& w, desc::options& iopt) : K(w,iopt) {}

void EXACT_J::init_tensors() {
	
	dbcsr::pgrid<3> grid3(m_world.comm());
	auto b = m_p_A->row_blk_sizes();
	vec<int> d = {1};
	
	arrvec<int,3> bbd = {b,b,d};
	
	m_J_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create().ngrid(grid3).name("J dummy")
		.map1({0,1}).map2({2}).blk_sizes(bbd));
	
	m_ptot_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create_template(*m_J_bbd).name("ptot dummy"));
	
}

void EXACT_K::init_tensors() {
	
	dbcsr::pgrid<3> grid3(m_world.comm());
	auto b = m_p_A->row_blk_sizes();
	vec<int> d = {1};
	
	arrvec<int,3> bbd = {b,b,d};
	
	m_K_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create().ngrid(grid3).name("K dummy")
		.map1({0,1}).map2({2}).blk_sizes(bbd));
	
	m_p_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create_template(*m_K_bbd).name("p dummy"));
	
}	

void EXACT_J::compute_J() {
	
	// copy P 
	
	dbcsr::mat_d ptot = dbcsr::mat_d::create_template(*m_p_A).name("ptot");
	
	if (m_p_A && !m_p_B) {
		ptot.copy_in(*m_p_A);
		ptot.scale(2.0);
		dbcsr::copy_matrix_to_3Dtensor_new(ptot,*m_ptot_bbd,true);
		ptot.clear();
	} else {
		ptot.copy_in(*m_p_A);
		ptot.add(1.0, 1.0, *m_p_B);
		dbcsr::copy_matrix_to_3Dtensor_new<double>(ptot,*m_ptot_bbd,true);
		ptot.clear();
	}
	
	if (LOG.global_plev() >= 3) {
		dbcsr::print(*m_ptot_bbd);
	}
	
	// grab integrals
	auto eris = m_reg.get_tensor<4,double>(m_mol->name() + "_i_bbbb_(01|23)", true, true);
	
	dbcsr::contract(*m_ptot_bbd, *eris, *m_J_bbd).print(false).perform("LS_, MNLS -> MN_");
	
	if (LOG.global_plev() >= 3) {
		dbcsr::print(*m_J_bbd);
	}

	dbcsr::copy_3Dtensor_to_matrix_new<double>(*m_J_bbd,*m_J);
	
	m_ptot_bbd->clear();
	m_J_bbd->clear();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_J);
	}

}

void EXACT_K::compute_K() {
	
	auto eris = m_reg.get_tensor<4,double>(m_mol->name() + "_i_bbbb_(02|13)", true, true);
	
	auto compute_K_single = [&](dbcsr::smat_d& p, dbcsr::smat_d& k, std::string x) {
		
		dbcsr::copy_matrix_to_3Dtensor_new<double>(*p,*m_p_bbd,true);
		
		LOG.os<1>("Computing exchange term (", x, ") ... \n");
			
		dbcsr::contract(*m_p_bbd, *eris, *m_K_bbd).alpha(-1.0).perform("LS_, MLSN -> MN_");

		dbcsr::copy_3Dtensor_to_matrix_new(*m_K_bbd,*k);
		
		//dbcsr::print(*m_K_bbd);
		
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
