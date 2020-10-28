#include "fock/jkbuilder.h"

namespace fock {
	
BATCHED_QRDF_K::BATCHED_QRDF_K(dbcsr::world& w, desc::options& opt) 
	: K(w,opt,"BATCHED_QRDF_K") {}
void BATCHED_QRDF_K::init() {
	
	init_base();
		
	m_c_xbb_batched = m_reg.get<dbcsr::sbtensor<3,double>>(Kkey::dfit_qr_xbb);
	//m_c_xbb_batched_02 = c_xbb_batched->get_access_duplicate();
	//m_c_xbb_batched_02 = c_xbb_batched->get_access_duplicate();
	
	m_v_xx = m_reg.get<dbcsr::shared_matrix<double>>(Kkey::v_xx);
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,3> xbb = {x,b,b};
	arrvec<int,2> bb = {b,b};
	arrvec<int,2> xx = {x,x};
	
	m_spgrid3_xbb = m_c_xbb_batched->spgrid();
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	m_cbar_xbb_01_2 = dbcsr::tensor_create<3,double>()
		.name("Cbar_xbb_01_2")
		.pgrid(m_spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0,1}).map2({2})
		.get();
	
	m_cbar_xbb_1_02 = 
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("Cbar_xbb_1_02")
		.map1({1}).map2({0,2})
		.get();
	
	m_cpq_xbb_0_12 =
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("cpq_xbb_0_12")
		.map1({0}).map2({1,2})
		.get();
		
	m_cpq_xbb_01_2 = 
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("cpq_xbb_01_2")
		.get();
	
	m_K_01 = dbcsr::tensor_create<2,double>()
		.pgrid(m_spgrid2)
		.name("K_01")
		.map1({0}).map2({1})
		.blk_sizes(bb)
		.get();
		
	m_p_bb = dbcsr::tensor_create_template<2,double>(m_K_01)
			.name("p_bb_0_1")
			.map1({0})
			.map2({1})
			.get();
			
	m_v_xx_01 = dbcsr::tensor_create<2,double>()
		.name("v_xx_01")
		.pgrid(m_spgrid2)
		.blk_sizes(xx)
		.map1({0}).map2({1})
		.get();

}

void BATCHED_QRDF_K::compute_K() {
	
	TIME.start();
	
	dbcsr::copy_matrix_to_tensor(*m_v_xx, *m_v_xx_01);
	
	auto compute_K_single = 
	[&] (dbcsr::smat_d& p_bb, dbcsr::smat_d& k_bb, std::string x) {
		
		LOG.os<1>("Computing exchange part (", x, ")\n");
		
		dbcsr::copy_matrix_to_tensor(*p_bb, *m_p_bb);
				
		//dbcsr::print(*c_bm);
		//dbcsr::print(*m_c_bm);	
		
		// LOOP OVER X
		
		auto& reo_int = TIME.sub("Reordering ints " + x);
		auto& reo_1_batch = TIME.sub("Reordering (1)/batch " + x);
		auto& con_1_batch = TIME.sub("Contraction (1)/batch " + x);
		auto& con_2_batch = TIME.sub("Contraction (2)/batch " + x);
		auto& fetch = TIME.sub("Fetching integrals/batch " + x);
		auto& fetch2 = TIME.sub("Fetching fitting coeffs/batch " + x);
		//auto& retint = TIME.sub("Returning integrals/batch " + x);
	
		//m_c_xbb_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
		
		int nxbatches = m_c_xbb_batched->nbatches(0);
		int nnbatches = m_c_xbb_batched->nbatches(2);
		
		auto c_xbb = m_c_xbb_batched->get_work_tensor();
		
		dbcsr::contract(*m_v_xx_01, *c_xbb, *m_cpq_xbb_0_12)
			.perform("XY, Ymr -> Xmr");
			
		dbcsr::contract(*m_cpq_xbb_0_12, *m_p_bb, *m_cbar_xbb_01_2)
			.perform("Xmr, sr -> Xms");
			
		dbcsr::contract(*c_xbb, *m_cbar_xbb_01_2, *m_K_01)
			.perform("Xns, Xms -> mn");
		
		m_K_01->batched_contract_finalize();
		
		dbcsr::copy_tensor_to_matrix(*m_K_01,*k_bb);
		m_K_01->clear();
		m_p_bb->clear();
		k_bb->scale(-1.0);
		
		LOG.os<1>("Done with exchange.\n");
		
	}; // end lambda function
	
	compute_K_single(m_p_A, m_K_A, "A");
	
	if (m_K_B) compute_K_single(m_p_B, m_K_B, "B");
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_K_A);
		if (m_K_B) dbcsr::print(*m_K_B);
	}
	
	m_v_xx_01->clear();
	
	TIME.finish();
			
}
	
	
	
} // end namespace
