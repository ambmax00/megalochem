#include "fock/jkbuilder.hpp"
#include "fock/fock_defaults.hpp"
#include "math/linalg/LLT.hpp"
#include <dbcsr_tensor_ops.hpp>

namespace megalochem {

namespace fock {

void EXACT_J::init() {
	
	init_base();

	auto b = m_mol->dims().b();
	vec<int> d = {1};
	
	arrvec<int,3> bbd = {b,b,d};
	int nbf = std::accumulate(b.begin(),b.end(),0);
	std::array<int,3> tsizes = {nbf,nbf,1};
	
	m_spgrid_bbd = dbcsr::pgrid<3>::create(m_cart.comm()).tensor_dims(tsizes).build();
	
	m_J_bbd = dbcsr::tensor<3>::create().name("J_bbd").set_pgrid(*m_spgrid_bbd)
		.map1({0,1}).map2({2}).blk_sizes(bbd).build();
	
	m_ptot_bbd = dbcsr::tensor<3>::create_template(*m_J_bbd)
		.name("p dummy").build();
		
}

void EXACT_K::init() {
	
	init_base();
	
	auto b = m_mol->dims().b();
	vec<int> d = {1};
	
	arrvec<int,3> bbd = {b,b,d};
	int nbf = std::accumulate(b.begin(),b.end(),0);
	std::array<int,3> tsizes = {nbf,nbf,1};
	
	m_spgrid_bbd = dbcsr::pgrid<3>::create(m_cart.comm()).tensor_dims(tsizes).build();
	
	m_K_bbd = dbcsr::tensor<3>::create().name("K dummy").set_pgrid(*m_spgrid_bbd)
		.map1({0,1}).map2({2}).blk_sizes(bbd).build();
	
	m_p_bbd = dbcsr::tensor<3>::create_template(*m_K_bbd)
		.name("p dummy").build();
	
	
}	

void EXACT_J::compute_J() {
	
	// copy P 
	
	auto ptot = dbcsr::matrix<>::create_template(*m_p_A)
		.name("ptot").build();
	
	if (m_p_A && !m_p_B) {
		ptot->copy_in(*m_p_A);
		ptot->scale(2.0);
		dbcsr::copy_matrix_to_3Dtensor_new(*ptot,*m_ptot_bbd,m_sym);
		ptot->clear();
	} else {
		ptot->copy_in(*m_p_A);
		ptot->add(1.0, 1.0, *m_p_B);
		dbcsr::copy_matrix_to_3Dtensor_new<double>(*ptot,*m_ptot_bbd,m_sym);
		ptot->clear();
	}
	
	if (LOG.global_plev() >= 3) {
		dbcsr::print(*m_ptot_bbd);
	}
		
	m_ptot_bbd->filter(dbcsr::global::filter_eps);
	
	//m_J_bbd->batched_contract_init();
	m_eri4c2e_batched->decompress_init({2,3},vec<int>{0,1},vec<int>{2,3});
	
	for (int imu = 0; imu != m_eri4c2e_batched->nbatches(2); ++imu) {
		for (int inu = 0; inu != m_eri4c2e_batched->nbatches(3); ++inu) {
			
			m_eri4c2e_batched->decompress({imu,inu});
			auto eri_01_23 = m_eri4c2e_batched->get_work_tensor();
			
			vec<vec<int>> ls_bounds = {
				m_eri4c2e_batched->bounds(2, imu),
				m_eri4c2e_batched->bounds(3, inu)
			};
			
			dbcsr::contract(1.0, *m_ptot_bbd, *eri_01_23, 1.0, *m_J_bbd)
				.bounds1(ls_bounds).perform("LS_, MNLS -> MN_");
				
		}
	}
			
	m_eri4c2e_batched->decompress_finalize();
	//m_J_bbd->batched_contract_finalize();
	
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
	
	auto compute_K_single = [&](dbcsr::shared_matrix<double>& p, 
		dbcsr::shared_matrix<double>& k, std::string x) {
		
		dbcsr::copy_matrix_to_3Dtensor_new<double>(*p,*m_p_bbd,true);
		m_p_bbd->filter(dbcsr::global::filter_eps);		
	
		LOG.os<1>("Computing exchange term (", x, ") ... \n");
		
		//m_K_bbd->batched_contract_init();
		m_eri4c2e_batched->decompress_init({1,3}, vec<int>{0,2}, vec<int>{1,3});
	
		for (int imu = 0; imu != m_eri4c2e_batched->nbatches(1); ++imu) {
			for (int inu = 0; inu != m_eri4c2e_batched->nbatches(3); ++inu) {
			
				m_eri4c2e_batched->decompress({imu,inu});
				
				auto eri_02_13 = m_eri4c2e_batched->get_work_tensor();
				
				vec<vec<int>> ls_bounds = {
					m_eri4c2e_batched->bounds(1, imu),
					m_eri4c2e_batched->bounds(3, inu)
				};
				
				dbcsr::contract(-1.0, *m_p_bbd, *eri_02_13, 1.0, *m_K_bbd)
					.bounds1(ls_bounds).perform("LS_, MLNS -> MN_");
					
			}
		}
			
		m_eri4c2e_batched->decompress_finalize();
		//m_K_bbd->batched_contract_finalize();
	
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

} // end mega
