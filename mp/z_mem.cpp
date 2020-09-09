#include "mp/z_builder.h"

namespace mp {

void LLMP_MEM_Z::init_tensors() {
	
	LOG.os<>("Setting up tensors in LLMP_MEM.\n");
	
	m_eri_batched = m_reg.get_btensor<3,double>("i_xbb_batched");
	auto eri = m_eri_batched->get_stensor();
	
	auto xbb = eri->blk_sizes();
	
	auto x = xbb[0];
	auto b = xbb[1];
	
	arrvec<int,2> xx = {x,x};
	
	m_zmat = dbcsr::create<double>()
		.set_world(m_world)
		.name("zmat")
		.row_blk_sizes(x)
		.col_blk_sizes(x)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	m_zmat_01 = dbcsr::tensor_create<2,double>()
		.pgrid(m_spgrid2)
		.name("zmat_0_1")
		.map1({0}).map2({1})
		.blk_sizes(xx)
		.get();
		
}

void LLMP_MEM_Z::compute() {
	
	TIME.start();
	
	auto& time_reo_int1 = TIME.sub("Reordering integrals 0|12 -> 1|02");
	auto& time_reo_int2 = TIME.sub("Reordering integrals 1|02 -> 0|12");
	auto& time_reo1 = TIME.sub("Reordering intermed tensor (1)");
	auto& time_reo2 = TIME.sub("Reordering intermed tensor (2)");
	auto& time_reo3 = TIME.sub("Reordering intermed tensor (3)");
	auto& time_tran1 = TIME.sub("First transformation");
	auto& time_tran2 = TIME.sub("Second transformation");
	auto& time_tran3 = TIME.sub("Third transformation");
	auto& time_write = TIME.sub("Writing tensor");
	auto& time_read = TIME.sub("Reading tensor");
	auto& time_formz = TIME.sub("Forming Z");
	auto& time_setview = TIME.sub("Setting view");
	auto& time_fetchints1 = TIME.sub("Fetching ints (1)");
	auto& time_fetchints2 = TIME.sub("Fetching ints (2)");
	
	auto b = m_locc->row_blk_sizes();
	auto o = m_locc->col_blk_sizes();
	auto x = m_zmat->row_blk_sizes();
	
	arrvec<int,2> bo = {b,o};
	arrvec<int,2> bb = {b,b};
	arrvec<int,3> xob = {x,o,b};
	
	int nbtot = std::accumulate(b.begin(),b.end(),0);
	int notot = std::accumulate(o.begin(),o.end(),0);
	int nxtot = std::accumulate(x.begin(),x.end(),0);
	
	std::array<int,3> xobsizes = {nxtot,notot,nbtot};
	
	// ======= TAKE CARE OF MATRIX STUFF ============
	
	m_locc_01 = dbcsr::tensor_create<2,double>()
		.name("locc_01")
		.pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bo).get();
		
	m_pvir_01 = dbcsr::tensor_create<2,double>()
		.name("pvir_01")
		.pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bb).get();
		
	//copy over
	dbcsr::copy_matrix_to_tensor(*m_locc, *m_locc_01);
	
	// ============= TAKE CARE OF TENSOR STUFF =============
	
	auto eri = m_eri_batched->get_stensor();
	
	auto spgrid3_xob = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(xobsizes).get();
	
	auto b_xob_1_02 = dbcsr::tensor_create<3,double>()
		.pgrid(spgrid3_xob)
		.name("b_xob_1_02")
		.map1({1}).map2({0,2})
		.blk_sizes(xob).get();
		
	auto b_xob_2_01 = 
		dbcsr::tensor_create_template<3,double>(b_xob_1_02)
		.name("b_xob_2_01").map1({2}).map2({0,1}).get();
		
	auto b2_xob_2_01 = 
		dbcsr::tensor_create_template<3,double>(b_xob_1_02)
		.name("b2_xob_2_01").map1({2}).map2({0,1}).get(); 
		
	auto b2_xob_1_02 = 
		dbcsr::tensor_create_template<3,double>(b_xob_1_02)
		.name("b2_xob_1_02").map1({1}).map2({0,2}).get();
	
	auto eri_1_02 = 
		dbcsr::tensor_create_template<3,double>(eri)
		.name("eri_1_02").map1({1}).map2({0,2}).get();
	
	auto b2_xbb_1_02 = 
		dbcsr::tensor_create_template<3,double>(eri)
		.name("b2_xbb_1_02").map1({1}).map2({0,2}).get();
	
	auto b2_xbb_0_12 =  
		dbcsr::tensor_create_template<3,double>(eri)
		.name("b2_xbb_0_12").map1({0}).map2({1,2}).get();
	
	//time_reo_int1.start();
	//m_eri_batched->reorder(vec<int>{1},vec<int>{0,2});
	//time_reo_int1.finish();
	
	m_eri_batched->decompress_init({0});
	
	dbcsr::sbtensor<3,double> eri2_batched
		= std::make_shared<dbcsr::btensor<3,double>>(*m_eri_batched);
		
	eri2_batched->decompress_init({0,2});
	
	LOG.os<>("Starting batching over auxiliary functions.\n");
	
	int nxbatches = m_eri_batched->nbatches_dim(0);
	int nnbatches = m_eri_batched->nbatches_dim(2);
		
	// ===== LOOP OVER BATCHES OF AUXILIARY FUNCTIONS ==================
	for (int ix = 0; ix != nxbatches; ++ix) {
		
		LOG.os<1>("-- (X) Batch ", ix, "\n");
		
		LOG.os<1>("-- Fetching ao ints...\n");
		time_fetchints1.start();
		m_eri_batched->decompress({ix});
		time_fetchints1.finish();
		
		auto eri_0_12 = m_eri_batched->get_stensor();
		
		vec<vec<int>> xbb_bounds = {
			m_eri_batched->bounds(0)[ix],
			m_eri_batched->full_bounds(1),
			m_eri_batched->full_bounds(2)
		};
		
		dbcsr::copy(*eri_0_12, *eri_1_02).bounds(xbb_bounds).perform(); 
		
		// first transform 
	    LOG.os<1>("-- First transform.\n");
	    
	    vec<vec<int>> x_nu_bounds = { 
			m_eri_batched->bounds(0)[ix], 
			m_eri_batched->full_bounds(2)
		};
	    
	    time_tran1.start();
	    dbcsr::contract(*m_locc_01, *eri_1_02, *b_xob_1_02)
			.bounds3(x_nu_bounds).perform("mi, Xmn -> Xin");
	    time_tran1.finish();
	    
	    eri_1_02->clear();
	    
	    // reorder
		LOG.os<1>("-- Reordering B_xob.\n");
		time_reo1.start();
		dbcsr::copy(*b_xob_1_02, *b_xob_2_01).move_data(true).perform();
		time_reo1.finish();
		
		//dbcsr::print(*B_xob_2_01);
		
		//copy over vir density
		dbcsr::copy_matrix_to_tensor(*m_pvir, *m_pvir_01);
	
		// new loop over nu
		for (int inu = 0; inu != nnbatches; ++inu) {
			
			LOG.os<1>("---- (NU) Batch ", inu, "\n");
			
			// second transform
			LOG.os<1>("---- Second transform.\n");
			
			vec<vec<int>> nu_bounds = { m_eri_batched->bounds(2)[inu] };
			vec<vec<int>> x_u_bounds = { 
				m_eri_batched->bounds(0)[ix],
				vec<int>{0, notot - 1}
			};
			
			time_tran2.start();
			dbcsr::contract(*m_pvir_01, *b_xob_2_01, *b2_xob_2_01)
				.bounds2(nu_bounds).bounds3(x_u_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Nn, Xin -> XiN");
			time_tran2.finish();
		
			// reorder
			LOG.os<1>("---- Reordering B_xoB.\n");
			time_reo2.start();
			dbcsr::copy(*b2_xob_2_01, *b2_xob_1_02).move_data(true).perform();
			time_reo2.finish();
			
			//dbcsr::print(*B_xob_1_02);
	
			// final contraction
			//B_xBB_1_02->reserve_template(*B_xbb_1_02);
		
			bool force_sparsity = false;
			if (m_shellpair_info) {
				
				force_sparsity = true;
				arrvec<int,3> res;
				auto& shellmat = *m_shellpair_info;
				
				auto xblkbounds = m_eri_batched->blk_bounds(0)[ix];
				auto bblkbounds = m_eri_batched->blk_bounds(1)[inu];

				for (int mublk = 0; mublk != b.size(); ++mublk) {
					for (int nublk = bblkbounds[0]; nublk != bblkbounds[1]+1; ++nublk) {
						
						if (!shellmat(mublk,nublk)) continue;
						
						for (int xblk = xblkbounds[0]; xblk != xblkbounds[1]+1; ++xblk) {
							
							std::array<int,3> idx = {xblk,mublk,nublk};
							if (m_world.rank() != b2_xbb_1_02->proc(idx)) continue;

							res[0].push_back(xblk);
							res[1].push_back(mublk);
							res[2].push_back(nublk);
						}
					}
				}
				
				b2_xbb_1_02->reserve(res);

			}
		
			LOG.os<1>("-- Final transform.\n");
			
			vec<vec<int>> x_nu_bounds = {
				m_eri_batched->bounds(0)[ix],
				m_eri_batched->bounds(2)[inu]
			};
											
			time_tran3.start();
			dbcsr::contract(*m_locc_01, *b2_xob_1_02, *b2_xbb_1_02)
				.bounds3(x_nu_bounds)
				.retain_sparsity(force_sparsity)
				.perform("Mi, XiN -> XMN");
			time_tran3.finish();
	
			// reorder
			LOG.os<1>("---- B_xBB.\n");
			time_reo3.start();
			dbcsr::copy(*b2_xbb_1_02, *b2_xbb_0_12).move_data(true).perform();
			time_reo3.finish(); 
			
			for (int iy = 0; iy != m_eri_batched->nbatches_dim(0); ++iy) {
				
				eri2_batched->decompress({iy,inu});
				auto eri2_0_12 = eri2_batched->get_stensor();
				
				vec<vec<int>> mn_bounds = {
					m_eri_batched->full_bounds(1),
					m_eri_batched->bounds(1)[inu]
				};
				
				vec<vec<int>> x_bounds = {
					m_eri_batched->bounds(0)[ix]
				};
				
				vec<vec<int>> y_bounds = {
					m_eri_batched->bounds(0)[iy]
				};
				
				time_formz.start();
				dbcsr::contract(*b2_xbb_0_12, *eri2_0_12, *m_zmat_01)
					.beta(1.0)
					.bounds1(mn_bounds)
					.bounds2(x_bounds)
					.bounds3(y_bounds)
					.filter(dbcsr::global::filter_eps / nxbatches)
					.perform("Mmn, Nmn -> MN");
				time_formz.finish();
				
			}
			
			b2_xbb_0_12->clear();
						
		}
		
	}
	
	m_eri_batched->decompress_finalize();
	eri2_batched->decompress_finalize();
	
	LOG.os<>("Finished batching.\n");
	
	// copy
	dbcsr::copy_tensor_to_matrix(*m_zmat_01, *m_zmat);
	
	m_zmat_01->clear();
	
	TIME.finish();
	
}

void LLMP_ASYM_Z::init_tensors() {
	
	LOG.os<>("Setting up tensors in LLMP_ASYM.\n");
	
	m_eri_batched = m_reg.get_btensor<3,double>("i_xbb_batched");
	auto eri = m_eri_batched->get_stensor();
	
	m_t_batched = m_reg.get_btensor<3,double>("t_xbb_batched");
	
	auto xbb = eri->blk_sizes();
	
	auto x = xbb[0];
	auto b = xbb[1];
	
	arrvec<int,2> xx = {x,x};
	
	m_zmat = dbcsr::create<double>()
		.set_world(m_world)
		.name("zmat")
		.row_blk_sizes(x)
		.col_blk_sizes(x)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	m_zmat_01 = dbcsr::tensor_create<2,double>()
		.pgrid(m_spgrid2)
		.name("zmat_0_1")
		.map1({0}).map2({1})
		.blk_sizes(xx)
		.get();
		
}

void LLMP_ASYM_Z::compute() {
	
	TIME.start();
	
	auto& time_reo_tensor = TIME.sub("Reordering tensor 0|12 -> 1|02");
	auto& time_reo1 = TIME.sub("Reordering intermed tensor (1)");
	auto& time_reo2 = TIME.sub("Reordering intermed tensor (2)");
	auto& time_reo3 = TIME.sub("Reordering intermed tensor (3)");
	auto& time_tran1 = TIME.sub("First transformation");
	auto& time_tran2 = TIME.sub("Second transformation");
	auto& time_tran3 = TIME.sub("Third transformation");
	auto& time_formz = TIME.sub("Forming Z");
	auto& time_fetchtensor = TIME.sub("Fetching tensor");
	auto& time_fetchints = TIME.sub("Fetching ints");
	
	auto b = m_locc->row_blk_sizes();
	auto o = m_locc->col_blk_sizes();
	auto x = m_zmat->row_blk_sizes();
	
	arrvec<int,2> bo = {b,o};
	arrvec<int,2> bb = {b,b};
	arrvec<int,3> xob = {x,o,b};
	
	int nbtot = std::accumulate(b.begin(),b.end(),0);
	int notot = std::accumulate(o.begin(),o.end(),0);
	int nxtot = std::accumulate(x.begin(),x.end(),0);
	
	std::array<int,3> xobsizes = {nxtot,notot,nbtot};
	
	// ======= TAKE CARE OF MATRIX STUFF ============
	
	m_locc_01 = dbcsr::tensor_create<2,double>()
		.name("locc_01")
		.pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bo).get();
		
	m_pvir_01 = dbcsr::tensor_create<2,double>()
		.name("pvir_01")
		.pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bb).get();
		
	//copy over
	dbcsr::copy_matrix_to_tensor(*m_locc, *m_locc_01);
	
	// ============= TAKE CARE OF TENSOR STUFF =============
	
	auto eri = m_eri_batched->get_stensor();
	
	auto spgrid3_xob = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(xobsizes).get();
	
	auto b_xob_1_02 = dbcsr::tensor_create<3,double>()
		.pgrid(spgrid3_xob)
		.name("b_xob_1_02")
		.map1({1}).map2({0,2})
		.blk_sizes(xob).get();
		
	auto b_xob_2_01 = 
		dbcsr::tensor_create_template<3,double>(b_xob_1_02)
		.name("b_xob_2_01").map1({2}).map2({0,1}).get();
		
	auto b2_xob_2_01 = 
		dbcsr::tensor_create_template<3,double>(b_xob_1_02)
		.name("b2_xob_2_01").map1({2}).map2({0,1}).get(); 
		
	auto b2_xob_1_02 = 
		dbcsr::tensor_create_template<3,double>(b_xob_1_02)
		.name("b2_xob_1_02").map1({1}).map2({0,2}).get();
	
	auto b2_xbb_1_02 = 
		dbcsr::tensor_create_template<3,double>(eri)
		.name("b2_xbb_1_02").map1({1}).map2({0,2}).get();
	
	auto b2_xbb_0_12 =  
		dbcsr::tensor_create_template<3,double>(eri)
		.name("b2_xbb_0_12").map1({0}).map2({1,2}).get();
	
	time_reo_tensor.start();
	m_t_batched->reorder(vec<int>{1},vec<int>{0,2});
	time_reo_tensor.finish();
	
	m_eri_batched->decompress_init({0,2});
	m_t_batched->decompress_init({0,2});
	
	LOG.os<>("Starting batching over auxiliary functions.\n");
	
	int nxbatches = m_eri_batched->nbatches_dim(0);
	int nnbatches = m_eri_batched->nbatches_dim(2);
	
	auto xbounds = m_eri_batched->bounds(0);
	auto bbounds = m_eri_batched->bounds(2);
	auto fullbbounds = m_eri_batched->full_bounds(2);
		
	// ===== LOOP OVER BATCHES OF AUXILIARY FUNCTIONS ==================
	for (int ix = 0; ix != nxbatches; ++ix) {
		
		for (int inuP = 0; inuP != nnbatches; ++inuP) {
			
			LOG.os<1>("Batch x/n ", ix, " ", inuP, '\n');
			
			for (int inu = 0; inu != nnbatches; ++inu) {
				
				time_fetchtensor.start();
				m_t_batched->decompress({ix,inu}); 
				time_fetchtensor.finish();
				
				auto t_1_02 = m_t_batched->get_stensor();
				
				LOG.os<1>("First transform");
				
				vec<vec<int>> x_nu_bounds = { 
					xbounds[ix], 
					bbounds[inu]
				};
				
				time_tran1.start();
				dbcsr::contract(*m_locc_01, *t_1_02, *b_xob_1_02)
					.bounds3(x_nu_bounds)
					.filter(dbcsr::global::filter_eps)
					.perform("mi, Xmn -> Xin");
				time_tran1.finish();
				
				// reorder
				LOG.os<1>("-- Reordering B_xob.\n");
				time_reo1.start();
				dbcsr::copy(*b_xob_1_02, *b_xob_2_01).move_data(true).perform();
				time_reo1.finish();
		
				//copy over vir density
				dbcsr::copy_matrix_to_tensor(*m_pvir, *m_pvir_01);
			
				// second transform
				LOG.os<1>("---- Second transform.\n");
				
				vec<vec<int>> nu_bounds = { bbounds[inu] };
				vec<vec<int>> nuP_bounds = { bbounds[inuP] };
				vec<vec<int>> x_u_bounds = { 
					xbounds[ix],
					vec<int>{0, notot - 1}
				};
			
				time_tran2.start();
				dbcsr::contract(*m_pvir_01, *b_xob_2_01, *b2_xob_2_01)
					.bounds1(nu_bounds)
					.bounds2(nuP_bounds)
					.bounds3(x_u_bounds)
					.filter(dbcsr::global::filter_eps)
					.beta(1.0)
					.perform("Nn, Xin -> XiN");
				time_tran2.finish();
				
			} // end loop inu
		
			// reorder
			LOG.os<1>("---- Reordering B_xoB.\n");
			time_reo2.start();
			dbcsr::copy(*b2_xob_2_01, *b2_xob_1_02).move_data(true).perform();
			time_reo2.finish();
		
			LOG.os<1>("-- Final transform.\n");
			
			vec<vec<int>> x_nuP_bounds = {
				xbounds[ix],
				bbounds[inuP]
			};
											
			time_tran3.start();
			dbcsr::contract(*m_locc_01, *b2_xob_1_02, *b2_xbb_1_02)
				.bounds3(x_nuP_bounds)
				.perform("Mi, XiN -> XMN");
			time_tran3.finish();
			
			b2_xob_1_02->clear();
	
			// reorder
			LOG.os<1>("---- B_xBB.\n");
			time_reo3.start();
			dbcsr::copy(*b2_xbb_1_02, *b2_xbb_0_12).move_data(true).perform();
			time_reo3.finish(); 
			
			for (int iy = 0; iy != m_eri_batched->nbatches_dim(0); ++iy) {
				
				m_eri_batched->decompress({iy,inuP});
				auto eri_0_12 = m_eri_batched->get_stensor();
				
				vec<vec<int>> mn_bounds = {
					fullbbounds,
					bbounds[inuP]
				};
				
				vec<vec<int>> x_bounds = {
					xbounds[ix]
				};
				
				vec<vec<int>> y_bounds = {
					xbounds[iy]
				};
				
				time_formz.start();
				dbcsr::contract(*b2_xbb_0_12, *eri_0_12, *m_zmat_01)
					.beta(1.0)
					.bounds1(mn_bounds)
					.bounds2(x_bounds)
					.bounds3(y_bounds)
					.filter(dbcsr::global::filter_eps / nxbatches)
					.perform("Mmn, Nmn -> MN");
				time_formz.finish();
				
			} // end loop iy
			
			b2_xbb_0_12->clear();
						
		} // end loop inuP
		
	} // end loop X
	
	m_eri_batched->decompress_finalize();
	m_t_batched->decompress_finalize();
	
	LOG.os<>("Finished batching.\n");
	
	// copy
	dbcsr::copy_tensor_to_matrix(*m_zmat_01, *m_zmat);
	
	m_zmat_01->clear();
	
	TIME.finish();
	
}
	
} // end namespace
