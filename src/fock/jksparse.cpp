#include "fock/jkbuilder.h"

namespace fock {
/*
SPARSE_K::SPARSE_K(dbcsr::world& w, desc::options& opt) 
	: K(w,opt,"SPARSE_K") {}
void SPARSE_K::init() {
	
	init_base();
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,3> xbb = {x,b,b};
	arrvec<int,2> bb = {b,b};
	arrvec<int,2> xx = {x,x};
	
	m_spgrid3_xbb = m_eri3c2e_batched->spgrid();
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
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
		
	// 
	
	
		
}

void BATCHED_DFSPARSE_K::compute_K() {
	
	TIME.start();
	
	dbcsr::copy_matrix_to_tensor(*m_v_xx, *m_v_xx_01);
	
	auto compute_K_single = 
	[&] (dbcsr::smat_d& p_bb, dbcsr::smat_d& k_bb, std::string x) {
		
		LOG.os<1>("Computing exchange part (", x, ")\n");
		
		m_eri3c2e_batched->decompress_init({0}, vec<int>{0}, vec<int>{1,2});
		
		for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
			
			auto xbds = m_eri3c2e_batched->bounds(0, ix);
			auto eri_xbb_A = m_eri3c2e_batched->get_work_tensor();
			
			
		
		
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
			
}*/
	
} // end namespace
