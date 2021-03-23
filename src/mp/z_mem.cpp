#include "mp/z_builder.hpp"

namespace mp {

void LLMP_MEM_Z::init() {
	
	LOG.os<1>("Setting up tensors in LLMP_MEM.\n");
	
	auto x = m_mol->dims().x();
	
	arrvec<int,2> xx = {x,x};
	
	m_zmat = dbcsr::matrix<>::create()
		.set_world(m_world)
		.name("zmat")
		.row_blk_sizes(x)
		.col_blk_sizes(x)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	m_spgrid2 = dbcsr::pgrid<2>::create(m_world.comm()).build();
	
	m_zmat_01 = dbcsr::tensor<2>::create()
		.set_pgrid(*m_spgrid2)
		.name("zmat_0_1")
		.map1({0}).map2({1})
		.blk_sizes(xx)
		.build();
	
	LOG.os<1>("Computing shellpir info...\n");
	m_shellpair_info = get_shellpairs(m_eri3c2e_batched);
		
}

void LLMP_MEM_Z::compute() {
	
	TIME.start();
	
	auto& time_reo_int1 = TIME.sub("Reordering integrals 0|12 -> 1|02");
	auto& time_reo1 = TIME.sub("Reordering intermed tensor (1)");
	auto& time_reo2 = TIME.sub("Reordering intermed tensor (2)");
	auto& time_reo3 = TIME.sub("Reordering intermed tensor (3)");
	auto& time_tran1 = TIME.sub("First transformation");
	auto& time_tran2 = TIME.sub("Second transformation");
	auto& time_tran3 = TIME.sub("Third transformation");
	auto& time_formz = TIME.sub("Forming Z");
	auto& time_fetchints1 = TIME.sub("Fetching ints (1)");
	auto& time_fetchints2 = TIME.sub("Fetching ints (2)");
	
	auto b = m_locc->row_blk_sizes();
	auto o = m_locc->col_blk_sizes();
	auto x = m_zmat->row_blk_sizes();
	
	arrvec<int,2> bo = {b,o};
	arrvec<int,2> bb = {b,b};
	arrvec<int,3> xob = {x,o,b};
	arrvec<int,3> xbb = {x,b,b};
	
	int nbtot = std::accumulate(b.begin(),b.end(),0);
	int notot = std::accumulate(o.begin(),o.end(),0);
	int nxtot = std::accumulate(x.begin(),x.end(),0);
	
	std::array<int,3> xobsizes = {nxtot,notot,nbtot};
	
	// ======= TAKE CARE OF MATRIX STUFF ============
	
	m_locc_01 = dbcsr::tensor<2>::create()
		.name("locc_01")
		.set_pgrid(*m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bo).build();
		
	m_pvir_01 = dbcsr::tensor<2>::create()
		.name("pvir_01")
		.set_pgrid(*m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bb).build();
		
	//copy over
	dbcsr::copy_matrix_to_tensor(*m_locc, *m_locc_01);
	
	// ============= TAKE CARE OF TENSOR STUFF =============
	
	auto spgrid3_xbb = m_eri3c2e_batched->spgrid();
		
	auto spgrid3_xob = dbcsr::pgrid<3>::create(m_world.comm())
		.tensor_dims(xobsizes).build();
	
	auto b_xob_1_02 = dbcsr::tensor<3>::create()
		.set_pgrid(*spgrid3_xob)
		.name("b_xob_1_02")
		.map1({1}).map2({0,2})
		.blk_sizes(xob).build();
		
	auto b_xob_2_01 = 
		dbcsr::tensor<3>::create_template(*b_xob_1_02)
		.name("b_xob_2_01").map1({2}).map2({0,1}).build();
		
	auto b2_xob_2_01 = 
		dbcsr::tensor<3>::create_template(*b_xob_1_02)
		.name("b2_xob_2_01").map1({2}).map2({0,1}).build(); 
		
	auto b2_xob_1_02 = 
		dbcsr::tensor<3>::create_template(*b_xob_1_02)
		.name("b2_xob_1_02").map1({1}).map2({0,2}).build();
	
	auto eri_1_02 = dbcsr::tensor<3>::create()
		.name("eri_1_02")
		.set_pgrid(*spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({1}).map2({0,2})
		.build();
	
	auto b2_xbb_1_02 = 
		dbcsr::tensor<3>::create_template(*eri_1_02)
		.name("b2_xbb_1_02").map1({1}).map2({0,2}).build();
	
	auto b2_xbb_0_12 =  
		dbcsr::tensor<3>::create_template(*eri_1_02)
		.name("b2_xbb_0_12").map1({0}).map2({1,2}).build();
	
	time_reo_int1.start();
	m_eri3c2e_batched->decompress_init({0}, vec<int>{0}, vec<int>{1,2});
	time_reo_int1.finish();
	
	LOG.os<1>("Starting batching over auxiliary functions.\n");
	
	int nxbatches = m_eri3c2e_batched->nbatches(0);
	int nnbatches = m_eri3c2e_batched->nbatches(2);
		
	// ===== LOOP OVER BATCHES OF AUXILIARY FUNCTIONS ==================
	for (int ix = 0; ix != nxbatches; ++ix) {
		
		LOG.os<1>("-- (X) Batch ", ix, "\n");
		
		LOG.os<1>("-- Fetching ao ints...\n");
		time_fetchints1.start();
		m_eri3c2e_batched->decompress({ix});
		auto eri_0_12 = m_eri3c2e_batched->get_work_tensor();
		time_fetchints1.finish();
		
		// batch inits
		for (int inu = 0; inu != nnbatches; ++inu) {
		
			vec<vec<int>> xbb_bounds = {
				m_eri3c2e_batched->bounds(0,ix),
				m_eri3c2e_batched->full_bounds(1),
				m_eri3c2e_batched->bounds(2,inu)
			};
			
			dbcsr::copy(*eri_0_12, *eri_1_02)
				.bounds(xbb_bounds)
				.perform(); 
					
			// first transform 
			LOG.os<1>("-- First transform.\n");
			
			vec<vec<int>> x_nu_bounds = { 
				m_eri3c2e_batched->bounds(0,ix), 
				m_eri3c2e_batched->bounds(2,inu)
			};
			
			time_tran1.start();
			dbcsr::contract(1.0, *m_locc_01, *eri_1_02, 0.0, *b_xob_1_02)
				.bounds3(x_nu_bounds)
				.perform("mi, Xmn -> Xin");
			time_tran1.finish();
			
			eri_1_02->clear();
			
			 // reorder
			LOG.os<1>("-- Reordering B_xob.\n");
			time_reo1.start();
			dbcsr::copy(*b_xob_1_02, *b_xob_2_01)
				.move_data(true)
				.sum(true)
				.perform();
			time_reo1.finish();
			
		}
		
		//dbcsr::print(*B_xob_2_01);
		
		//copy over vir density
		dbcsr::copy_matrix_to_tensor(*m_pvir, *m_pvir_01);
	
		// new loop over nu
		for (int inu = 0; inu != nnbatches; ++inu) {
			
			LOG.os<1>("---- (NU) Batch ", inu, "\n");
			
			// second transform
			LOG.os<1>("---- Second transform.\n");
			
			vec<vec<int>> nu_bounds = { 
				m_eri3c2e_batched->bounds(2,inu)
			};
			
			vec<vec<int>> x_u_bounds = { 
				m_eri3c2e_batched->bounds(0,ix),
				vec<int>{0, notot - 1}
			};
			
			time_tran2.start();
			dbcsr::contract(1.0, *m_pvir_01, *b_xob_2_01, 0.0, *b2_xob_2_01)
				.bounds2(nu_bounds).bounds3(x_u_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Nn, Xin -> XiN");
			time_tran2.finish();
		
			// reorder
			LOG.os<1>("---- Reordering B_xoB.\n");
			time_reo2.start();
			dbcsr::copy(*b2_xob_2_01, *b2_xob_1_02).move_data(true).perform();
			time_reo2.finish();
			
			// final contraction
			//B_xBB_1_02->reserve_template(*B_xbb_1_02);
		
			bool force_sparsity = false;
			if (true) {
				
				force_sparsity = true;
				arrvec<int,3> res;
				auto& shellmat = *m_shellpair_info;
				
				auto xblkbounds = m_eri3c2e_batched->blk_bounds(0,ix);
				auto bblkbounds = m_eri3c2e_batched->blk_bounds(1,inu);

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
				m_eri3c2e_batched->bounds(0,ix),
				m_eri3c2e_batched->bounds(2,inu)
			};
											
			time_tran3.start();
			dbcsr::contract(1.0, *m_locc_01, *b2_xob_1_02, 0.0, *b2_xbb_1_02)
				.bounds3(x_nu_bounds)
				.retain_sparsity(force_sparsity)
				.perform("Mi, XiN -> XMN");
			time_tran3.finish();
			
			b2_xob_1_02->clear();
	
			// reorder
			LOG.os<1>("---- B_xBB.\n");
			time_reo3.start();
			dbcsr::copy(*b2_xbb_1_02, *b2_xbb_0_12)
				.move_data(true)
				.sum(true)
				.perform();
			time_reo3.finish(); 
			
		}
			
		for (int iy = 0; iy != m_eri3c2e_batched->nbatches(0); ++iy) {
			
			m_eri3c2e_batched->decompress({iy});
			auto eri_0_12 = m_eri3c2e_batched->get_work_tensor();
			
			vec<vec<int>> x_bounds = {
				m_eri3c2e_batched->bounds(0,ix)
			};
			
			vec<vec<int>> y_bounds = {
				m_eri3c2e_batched->bounds(0,iy)
			};
			
			time_formz.start();
			dbcsr::contract(1.0, *b2_xbb_0_12, *eri_0_12, 1.0, *m_zmat_01)
				.bounds2(x_bounds)
				.bounds3(y_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Mmn, Nmn -> MN");
			time_formz.finish();
			
		}
		
		b2_xbb_0_12->clear();
					
	}
	
	m_eri3c2e_batched->decompress_finalize();
	
	LOG.os<1>("Finished batching.\n");
	
	// copy
	dbcsr::copy_tensor_to_matrix(*m_zmat_01, *m_zmat);
	
	m_zmat_01->clear();
	
	TIME.finish();
	
}

/*
void LLMP_ASYM_Z::init() {
	
	LOG.os<1>("Setting up tensors in LLMP_ASYM.\n");
	
	auto x = m_mol->dims().x();
	
	arrvec<int,2> xx = {x,x};
	
	m_zmat = dbcsr::matrix<>::create()
		.set_world(m_world)
		.name("zmat")
		.row_blk_sizes(x)
		.col_blk_sizes(x)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	m_spgrid2 = dbcsr::pgrid<2>::create(m_world.comm()).build();
	
	m_zmat_01 = dbcsr::tensor<2>::create()
		.set_pgrid(*m_spgrid2)
		.name("zmat_0_1")
		.map1({0}).map2({1})
		.blk_sizes(xx)
		.build();
		
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
	
	m_locc_01 = dbcsr::tensor<2>::create()
		.name("locc_01")
		.set_pgrid(*m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bo).build();
		
	m_pvir_01 = dbcsr::tensor<2>::create()
		.name("pvir_01")
		.set_pgrid(*m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bb).build();
		
	//copy over
	dbcsr::copy_matrix_to_tensor(*m_locc, *m_locc_01);
	
	// ============= TAKE CARE OF TENSOR STUFF =============
	
	auto spgrid3_xob = dbcsr::pgrid<3>::create(m_world.comm())
		.tensor_dims(xobsizes).build();
	
	auto b_xob_1_02 = dbcsr::tensor<3>::create()
		.set_pgrid(*spgrid3_xob)
		.name("b_xob_1_02")
		.map1({1}).map2({0,2})
		.blk_sizes(xob).build();
		
	auto b_xob_2_01 = 
		dbcsr::tensor<3>::create_template(*b_xob_1_02)
		.name("b_xob_2_01").map1({2}).map2({0,1}).build();
		
	auto b2_xob_2_01 = 
		dbcsr::tensor<3>::create_template(*b_xob_1_02)
		.name("b2_xob_2_01").map1({2}).map2({0,1}).build(); 
		
	auto b2_xob_1_02 = 
		dbcsr::tensor<3>::create_template(*b_xob_1_02)
		.name("b2_xob_1_02").map1({1}).map2({0,2}).build();
	
	auto b2_xbb_1_02 = m_t3c2e_right_batched->get_template(
		"b2_xbb_1_02", vec<int>{1}, vec<int>{0,2});
		
	auto b2_xbb_0_12 = m_t3c2e_right_batched->get_template(
		"b2_xbb_0_12", vec<int>{0}, vec<int>{1,2});
	
	time_reo_tensor.start();
	m_t3c2e_left_batched->decompress_init({0}, vec<int>{0}, vec<int>{1,2});
	m_t3c2e_right_batched->decompress_init({0}, vec<int>{1}, vec<int>{0,2});
	time_reo_tensor.finish();
	
	LOG.os<1>("Starting batching over auxiliary functions.\n");
	
	int nxbatches = m_t3c2e_left_batched->nbatches(0);
	int nnbatches = m_t3c2e_left_batched->nbatches(2);
	
	// ===== LOOP OVER BATCHES OF AUXILIARY FUNCTIONS ==================
	for (int ix = 0; ix != nxbatches; ++ix) {
		
		LOG.os<1>("-- (X) Batch ", ix, "\n");
		
		LOG.os<1>("-- Fetching tensor...\n");
		time_fetchtensor.start();
		m_t3c2e_right_batched->decompress({ix});
		auto t_1_02 = m_t3c2e_right_batched->get_work_tensor();
		time_fetchtensor.finish();
		
		// batch inits
		for (int inu = 0; inu != nnbatches; ++inu) {
		
			vec<vec<int>> xbb_bounds = {
				m_t3c2e_left_batched->bounds(0,ix),
				m_t3c2e_left_batched->full_bounds(1),
				m_t3c2e_left_batched->bounds(2,inu)
			};
			
			// first transform 
			LOG.os<1>("-- First transform.\n");
			
			vec<vec<int>> x_nu_bounds = { 
				m_t3c2e_left_batched->bounds(0,ix), 
				m_t3c2e_left_batched->bounds(2,inu)
			};
			
			time_tran1.start();
			dbcsr::contract(1.0, *m_locc_01, *t_1_02, 0.0, *b_xob_1_02)
				.bounds3(x_nu_bounds)
				.perform("mi, Xmn -> Xin");
			time_tran1.finish();
			
			 // reorder
			LOG.os<1>("-- Reordering B_xob.\n");
			time_reo1.start();
			dbcsr::copy(*b_xob_1_02, *b_xob_2_01)
				.move_data(true)
				.sum(true)
				.perform();
			time_reo1.finish();
			
		}
		
		//dbcsr::print(*B_xob_2_01);
		
		//copy over vir density
		dbcsr::copy_matrix_to_tensor(*m_pvir, *m_pvir_01);
	
		// new loop over nu
		for (int inu = 0; inu != nnbatches; ++inu) {
			
			LOG.os<1>("---- (NU) Batch ", inu, "\n");
			
			// second transform
			LOG.os<1>("---- Second transform.\n");
			
			vec<vec<int>> nu_bounds = { 
				m_t3c2e_left_batched->bounds(2,inu)
			};
			
			vec<vec<int>> x_u_bounds = { 
				m_t3c2e_left_batched->bounds(0,ix),
				vec<int>{0, notot - 1}
			};
			
			time_tran2.start();
			dbcsr::contract(1.0, *m_pvir_01, *b_xob_2_01, 0.0, *b2_xob_2_01)
				.bounds2(nu_bounds).bounds3(x_u_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Nn, Xin -> XiN");
			time_tran2.finish();
		
			// reorder
			LOG.os<1>("---- Reordering B_xoB.\n");
			time_reo2.start();
			dbcsr::copy(*b2_xob_2_01, *b2_xob_1_02).move_data(true).perform();
			time_reo2.finish();
			
			// final contraction
			//B_xBB_1_02->reserve_template(*B_xbb_1_02);
		
			bool force_sparsity = false;
			if (true) {
				
				force_sparsity = true;
				arrvec<int,3> res;
				auto& shellmat = *m_shellpair_info;
				
				auto xblkbounds = m_t3c2e_left_batched->blk_bounds(0,ix);
				auto bblkbounds = m_t3c2e_left_batched->blk_bounds(1,inu);

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
				m_t3c2e_left_batched->bounds(0,ix),
				m_t3c2e_left_batched->bounds(2,inu)
			};
											
			time_tran3.start();
			dbcsr::contract(1.0, *m_locc_01, *b2_xob_1_02, 0.0, *b2_xbb_1_02)
				.bounds3(x_nu_bounds)
				.retain_sparsity(force_sparsity)
				.perform("Mi, XiN -> XMN");
			time_tran3.finish();
			
			b2_xob_1_02->clear();
	
			// reorder
			LOG.os<1>("---- B_xBB.\n");
			time_reo3.start();
			dbcsr::copy(*b2_xbb_1_02, *b2_xbb_0_12)
				.move_data(true)
				.sum(true)
				.perform();
			time_reo3.finish(); 
			
		}
			
		for (int iy = 0; iy != m_t3c2e_right_batched->nbatches(0); ++iy) {
			
			m_t3c2e_left_batched->decompress({iy});
			auto eri_0_12 = m_t3c2e_left_batched->get_work_tensor();
			
			vec<vec<int>> x_bounds = {
				m_t3c2e_left_batched->bounds(0,ix)
			};
			
			vec<vec<int>> y_bounds = {
				m_t3c2e_left_batched->bounds(0,iy)
			};
			
			time_formz.start();
			dbcsr::contract(1.0, *b2_xbb_0_12, *eri_0_12, 1.0, *m_zmat_01)
				.bounds2(x_bounds)
				.bounds3(y_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Mmn, Nmn -> MN");
			time_formz.finish();
			
		}
		
		b2_xbb_0_12->clear();
					
	}
	
	m_t3c2e_left_batched->decompress_finalize();
	m_t3c2e_right_batched->decompress_finalize();
	
	LOG.os<1>("Finished batching.\n");
	
	// copy
	dbcsr::copy_tensor_to_matrix(*m_zmat_01, *m_zmat);
	
	m_zmat_01->clear();
	
	TIME.finish();
	
}*/
	
} // end namespace
