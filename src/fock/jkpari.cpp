#include "fock/jkbuilder.h"
#include "ints/aofactory.h"
#include "ints/screening.h"
#include "math/linalg/SVD.h"
#include "extern/lapack.h"
#include <Eigen/Core>
#include <Eigen/SVD>

namespace fock {

BATCHED_PARI_K::BATCHED_PARI_K(dbcsr::world& w, desc::options& opt) 
	: K(w,opt,"PARI-K") {}

void BATCHED_PARI_K::init() {
	
	init_base();
	
	auto b = m_mol->dims().b();
	arrvec<int,2> bb = {b,b};
	
	m_eri_batched = m_reg.get_btensor<3,double>("i_xbb_batched");
	
	m_cfit_xbb = m_reg.get_tensor<3,double>("c_xbb_pari");
	
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	m_K_01 = dbcsr::tensor_create<2,double>().pgrid(m_spgrid2).name("K_01")
		.map1({0}).map2({1}).blk_sizes(bb).get();
		
	m_p_bb = dbcsr::tensor_create_template<2,double>(m_K_01)
			.name("p_bb_0_1").map1({0}).map2({1}).get();
	
	m_s_xx = m_reg.get_tensor<2,double>("s_xx");
	
}

void BATCHED_PARI_K::compute_K() {
	
	TIME.start();
	
	auto& time_reo_int1 = TIME.sub("Reordering integrals (1)");
	auto& time_fetch_ints = TIME.sub("Fetching ints");
	auto& time_reo_cbar = TIME.sub("Reordering c_bar");
	auto& time_reo_ctil = TIME.sub("Reordering c_tilde");
	auto& time_form_cbar = TIME.sub("Forming c_bar");
	auto& time_form_ctil = TIME.sub("Forming c_til");
	auto& time_copy_cfit = TIME.sub("Copying c_fit");
	auto& time_form_K1 = TIME.sub("Forming K1");
	auto& time_form_K2 = TIME.sub("Forming K2");
	
	// allocate tensors
	
	auto cfit_xbb_01_2 = m_cfit_xbb;
	
	auto cfit_xbb_0_12 = dbcsr::tensor_create_template(m_cfit_xbb)
		.name("cfit_xbb_0_12")
		.map1({0}).map2({1,2})
		.get();
	
	auto cbar_xbb_01_2 = dbcsr::tensor_create_template(m_cfit_xbb)
		.name("cbar_xbb_01_2")
		.map1({0,1}).map2({2})
		.get();
		
	auto cbar_xbb_02_1 = dbcsr::tensor_create_template(m_cfit_xbb)
		.name("cbar_xbb_02_1")
		.map1({0,2}).map2({1})
		.get();
	
	auto ctil_xbb_0_12 = dbcsr::tensor_create_template(m_cfit_xbb)
		.name("ctil_xbb_0_12")
		.map1({0}).map2({1,2})
		.get();
		
	auto ctil_xbb_02_1 = dbcsr::tensor_create_template(m_cfit_xbb)
		.name("ctil_xbb_02_1")
		.map1({0,2}).map2({1})
		.get();
	
	auto K1 = dbcsr::tensor_create_template(m_K_01)
		.name("K1").get();
		
	auto K2 = dbcsr::tensor_create_template(m_K_01)
		.name("K2").get();
		
	time_reo_int1.start();
	m_eri_batched->decompress_init({0}, vec<int>{0,2},vec<int>{1});
	time_reo_int1.finish();	
	
	dbcsr::copy_matrix_to_tensor(*m_p_A, *m_p_bb);
	
	// Loop ix
	for (int ix = 0; ix != m_eri_batched->nbatches(0); ++ix) {
		
		LOG.os<1>("Fetching integrals.\n");
		time_fetch_ints.start();
		m_eri_batched->decompress({ix});
		auto eri_02_1 = m_eri_batched->get_work_tensor();
		time_fetch_ints.finish();
		
		// Loop inu
		for (int isig = 0; isig != m_eri_batched->nbatches(2); ++isig) {
			
			LOG.os<1>("Loop PARI K, batch nr ", ix, " ", isig, '\n');
			
			vec<vec<int>> xm_bounds = {
				m_eri_batched->bounds(0,ix),
				m_eri_batched->full_bounds(1)
			};
			
			vec<vec<int>> s_bounds = {
				m_eri_batched->bounds(2,isig)
			};
			
			LOG.os<1>("Forming cbar.\n");
			
			// form cbar
			
			time_form_cbar.start();
			dbcsr::contract(*cfit_xbb_01_2, *m_p_bb, *cbar_xbb_01_2)
				.bounds2(xm_bounds)
				.bounds3(s_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Qml, ls -> Qms");
			time_form_cbar.finish();	
				
			vec<vec<int>> rns_bounds = {
				m_eri_batched->full_bounds(0),
				m_eri_batched->full_bounds(1),
				m_eri_batched->bounds(2,isig)
			};
			 
			LOG.os<1>("Copying...\n");
			time_copy_cfit.start();
			dbcsr::copy(*cfit_xbb_01_2,*cfit_xbb_0_12)
				.bounds(rns_bounds)
				.perform();
			time_copy_cfit.finish();
		
			vec<vec<int>> ns_bounds = {
				m_eri_batched->full_bounds(1),
				m_eri_batched->bounds(2,isig)
			};
			
			vec<vec<int>> x_bounds = {
				m_eri_batched->bounds(0,ix)
			};
			
			LOG.os<1>("Forming ctil.\n");
			time_form_ctil.start();
			dbcsr::contract(*cfit_xbb_0_12, *m_s_xx, *ctil_xbb_0_12)
				.bounds2(ns_bounds)
				.bounds3(x_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Rns, RQ -> Qns");	
			time_form_ctil.finish();
			
			cfit_xbb_0_12->clear();
			
			LOG.os<1>("Reordering ctil.\n");
			time_reo_ctil.start();
			dbcsr::copy(*ctil_xbb_0_12, *ctil_xbb_02_1).move_data(true).perform();
			time_reo_ctil.finish();
			
			LOG.os<1>("Reordering cbar.\n");
			time_reo_cbar.start();
			dbcsr::copy(*cbar_xbb_01_2, *cbar_xbb_02_1).move_data(true).perform();
			time_reo_cbar.finish();
			
			vec<vec<int>> xs_bounds = {
				m_eri_batched->bounds(0,ix),
				m_eri_batched->bounds(2,isig)
			};
			
			LOG.os<1>("Forming K (1).\n");
			
			time_form_K1.start();
			dbcsr::contract(*cbar_xbb_02_1, *eri_02_1, *K1)
				.bounds1(xs_bounds)
				.filter(dbcsr::global::filter_eps/m_eri_batched->nbatches(0))
				.beta(1.0)
				.perform("Qms, Qns -> mn");
			time_form_K1.finish();
							
			LOG.os<1>("Forming K (2).\n");
			
			time_form_K2.start();
			dbcsr::contract(*ctil_xbb_02_1, *cbar_xbb_02_1, *K2)
				.bounds1(xs_bounds)
				.filter(dbcsr::global::filter_eps/m_eri_batched->nbatches(0))
				.beta(1.0)
				.perform("Qns, Qms -> mn");	
			time_form_K2.finish();
			
			ctil_xbb_02_1->clear();
			cbar_xbb_02_1->clear();
		
			LOG.os<1>("Done.\n");
				
		} // end loop sig
	} // end loop x
	
	m_eri_batched->decompress_finalize();
		
	K2->scale(-0.5);	
	
	//dbcsr::print(*K1);
	//dbcsr::print(*K2);
		
	dbcsr::copy(*K1, *m_K_01).move_data(true).sum(true).perform();
	dbcsr::copy(*K2, *m_K_01).move_data(true).sum(true).perform();		
			
	//dbcsr::print(*m_K_01);
	
	LOG.os<1>("Forming final K.\n");
	
	auto K_copy = dbcsr::tensor_create_template<2,double>(m_K_01)
		.name("K copy").get();
		
	dbcsr::copy(*m_K_01, *K_copy).order({1,0}).perform();
	dbcsr::copy(*K_copy, *m_K_01).move_data(true).sum(true).perform();
		
	dbcsr::copy_tensor_to_matrix(*m_K_01, *m_K_A);
	
	m_K_01->clear();
	
	m_K_A->scale(-1.0);
	//dbcsr::print(*m_K_A);
	
	//time_reo_int2.start();
	//m_eri_batched->reorder(vec<int>{0}, vec<int>{1,2});
	//time_reo_int2.finish();	
		
	TIME.finish();
			
}

} // end namespace
