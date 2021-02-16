#include "mp/z_builder.hpp"

namespace mp {

void LLMP_FULL_Z::init() {
	
	LOG.os<1>("Setting up tensors in LLMP_FULL.\n");
	
	auto x = m_mol->dims().x();
	auto b = m_mol->dims().b();
	
	arrvec<int,2> xx = {x,x};
	arrvec<int,3> xbb = {x,b,b};
	
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
		
	auto bdims = m_eri3c2e_batched->batch_dims();
	
	auto blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
	auto blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
	
	arrvec<int,3> blkmap_xbb = { blkmap_x, blkmap_b, blkmap_b };	
		
	auto spgrid3_xbb = m_eri3c2e_batched->spgrid();	
	
	m_FT3c2e_batched = dbcsr::btensor_create<3>()
		.set_pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.name(m_mol->name() + "_z_xbb_batched")
		.blk_map(blkmap_xbb)
		.batch_dims(bdims)
		.btensor_type(m_intermeds)
		.print(LOG.global_plev())
		.build();
		
}

void LLMP_FULL_Z::compute() {
	
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
		
	auto spgrid3_xob = dbcsr::pgrid<3>::create(m_world.comm())
		.tensor_dims(xobsizes).build();
		
	auto spgrid3_xbb = m_eri3c2e_batched->spgrid();
	
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
		
	auto b2_xbb_1_02 = dbcsr::tensor<3>::create()
		.name("b2_xbb_1_02")
		.set_pgrid(*spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({1}).map2({0,2})
		.build();
	
	auto b2_xbb_0_12 =  
		dbcsr::tensor<3>::create_template(*b2_xbb_1_02)
		.name("b2_xbb_0_12").map1({0}).map2({1,2}).build();
	
	time_reo_int1.start();
	LOG.os<1>("Reordering ints to 1|02.\n");
	m_eri3c2e_batched->decompress_init({0},vec<int>{1},vec<int>{0,2});
	time_reo_int1.finish();
	
	m_FT3c2e_batched->compress_init({0,2}, vec<int>{0}, vec<int>{1,2});
	
	LOG.os<1>("Starting batching over auxiliary functions.\n");
		
	// ===== LOOP OVER BATCHES OF AUXILIARY FUNCTIONS ==================
	for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
		
		LOG.os<1>("-- (X) Batch ", ix, "\n");
		
		LOG.os<1>("-- Fetching ao ints...\n");
		time_fetchints1.start();
		m_eri3c2e_batched->decompress({ix});
		time_fetchints1.finish();
		
		auto eri_1_02 = m_eri3c2e_batched->get_work_tensor();
					
		// first transform 
	    LOG.os<1>("-- First transform.\n");
	    
	    vec<vec<int>> x_nu_bounds = { 
			m_eri3c2e_batched->bounds(0,ix),
			m_eri3c2e_batched->full_bounds(2)
		};
		
		//std::cout << "ERI" << std::endl;
		//dbcsr::print(*eri_1_02);
	    
	    time_tran1.start();
	    dbcsr::contract(*m_locc_01, *eri_1_02, *b_xob_1_02)
			.bounds3(x_nu_bounds).perform("mi, Xmn -> Xin");
	    time_tran1.finish();
	    
	    // reorder
		LOG.os<1>("-- Reordering B_xob.\n");
		time_reo1.start();
		dbcsr::copy(*b_xob_1_02, *b_xob_2_01).move_data(true).perform();
		time_reo1.finish();
		
		//std::cout << "THERE" << std::endl;
		//dbcsr::print(*b_xob_2_01);
		
		//copy over vir density
		dbcsr::copy_matrix_to_tensor(*m_pvir, *m_pvir_01);
	
		// new loop over nu
		for (int inu = 0; inu != m_eri3c2e_batched->nbatches(2); ++inu) {
			
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
			
			//dbcsr::print(*b_xob_2_01);
			
			time_tran2.start();
			dbcsr::contract(*m_pvir_01, *b_xob_2_01, *b2_xob_2_01)
				.bounds2(nu_bounds).bounds3(x_u_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Nn, Xin -> XiN");
			time_tran2.finish();
		
			//dbcsr::print(*b2_xob_2_01);
			
			// reorder
			LOG.os<1>("---- Reordering B_xoB.\n");
			time_reo2.start();
			dbcsr::copy(*b2_xob_2_01, *b2_xob_1_02).move_data(true).perform();
			time_reo2.finish();
		
			LOG.os<1>("-- Final transform.\n");
			
			vec<vec<int>> x_nu_bounds = {
				m_eri3c2e_batched->bounds(0,ix),
				m_eri3c2e_batched->bounds(2,inu)
			};
			
			bool force_sparsity = false;
			if (true) {
				
				force_sparsity = true;
				auto& shellmat = *m_shellpair_info;
				
				arrvec<int,3> res;
				
				auto xblkbounds = m_eri3c2e_batched->blk_bounds(0,ix);
				auto bblkbounds = m_eri3c2e_batched->blk_bounds(2,inu);

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
			
			//dbcsr::print(*b2_xob_1_02);
											
			time_tran3.start();
			dbcsr::contract(*m_locc_01, *b2_xob_1_02, *b2_xbb_1_02)
				.bounds3(x_nu_bounds)
				.retain_sparsity(force_sparsity)
				.perform("Mi, XiN -> XMN");
			time_tran3.finish();
	
			//dbcsr::print(*b2_xbb_1_02);
	
			// reorder
			LOG.os<1>("---- B_xBB.\n");
			time_reo3.start();
			dbcsr::copy(*b2_xbb_1_02, *b2_xbb_0_12)
				.move_data(true)
				//.sum(true)
				.perform();
			time_reo3.finish();
			
			//dbcsr::print(*b2_xbb_0_12);
			
			//dbcsr::print(*B_xBB_0_12_wr);
			
			//dbcsr::print(*B_xBB_0_12);
			
			LOG.os<1>("---- Writing B_xBB to memory.\n");
			time_write.start();
			m_FT3c2e_batched->compress({ix,inu},b2_xbb_0_12);
			time_write.finish();
						
		}
		
		//dbcsr::print(*b2_xbb_0_12);
		
	}
	
	m_eri3c2e_batched->decompress_finalize();
	m_FT3c2e_batched->compress_finalize();
	
	LOG.os<1>("Occupation of FT3c2e: ", m_FT3c2e_batched->occupation()*100, "%\n"); 
	
	LOG.os<1>("Finished batching.\n");
	
	LOG.os<1>("Reordering ints 1|02 -> 0|12 \n");
	
	time_reo_int2.start();
	m_eri3c2e_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	time_reo_int2.finish();
	
	LOG.os<1>("Setting up decompression.\n");
	
	time_setview.start();
	m_FT3c2e_batched->decompress_init({0,2}, vec<int>{0}, vec<int>{1,2});
	time_setview.finish();
	
	LOG.os<1>("Computing Z_XY.\n");
	
	//m_zmat_01->batched_contract_init();
	
	for (int inu = 0; inu != m_eri3c2e_batched->nbatches(2); ++inu) {
		
		LOG.os<1>("Batch: ", inu, '\n'); 	
		LOG.os<1>("Fetching integrals...\n");
		
		time_fetchints2.start();
		m_eri3c2e_batched->decompress({inu});
		time_fetchints2.finish();
		
		auto eri_0_12 = m_eri3c2e_batched->get_work_tensor();
	
		for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
			
			LOG.os<1>("-- Batch: ", ix, '\n');
			
			LOG.os<1>("-- Fetching intermediate...\n");
			
			time_read.start();
			m_FT3c2e_batched->decompress({ix,inu});
			time_read.finish();
			
			auto z_xbb_0_12 = m_FT3c2e_batched->get_work_tensor();
			
			vec<vec<int>> x_bounds = {
				m_eri3c2e_batched->bounds(0,ix)
			};
			
			vec<vec<int>> mn_bounds = {
				m_eri3c2e_batched->full_bounds(1),
				m_eri3c2e_batched->bounds(2,inu)
			};
			
			// form Z
			LOG.os<1>("-- Forming Z.\n");
			
			time_formz.start();
			dbcsr::contract(*z_xbb_0_12, *eri_0_12, *m_zmat_01)
				.beta(1.0)
				.bounds1(mn_bounds)
				.bounds2(x_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Mmn, Nmn -> MN");
			time_formz.finish();
			
		}
	}
	
	m_eri3c2e_batched->decompress_finalize();
	m_FT3c2e_batched->decompress_finalize();
	
	m_FT3c2e_batched->reset();

	//m_zmat_01->batched_contract_finalize();	
	
	LOG.os<1>("Finished batching.\n");

	// copy
	dbcsr::copy_tensor_to_matrix(*m_zmat_01, *m_zmat);
	
	m_zmat_01->clear();
	
	TIME.finish();
	
}

#if 0

void LL_Z::init() {
	
	LOG.os<1>("Setting up tensors in LLMP_FULL.\n");
	
	auto x = m_mol->dims().x();
	auto b = m_mol->dims().b();
	
	arrvec<int,2> xx = {x,x};
	arrvec<int,3> xbb = {x,b,b};
	
	m_zmat = dbcsr::create<double>()
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
		
	auto bdims = m_eri3c2e_batched->batch_dims();
	
	auto blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
	auto blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
	
	arrvec<int,3> blkmap_xbb = { blkmap_x, blkmap_b, blkmap_b };	
		
	auto spgrid3_xbb = m_eri3c2e_batched->spgrid();	
	
	m_FT3c2e_batched = dbcsr::btensor_create<3>()
		.set_pgrid(*spgrid3_xbb)
		.blk_sizes(xbb)
		.name(m_mol->name() + "_z_xbb_batched")
		.blk_map(blkmap_xbb)
		.batch_dims(bdims)
		.btensor_type(m_intermeds)
		.print(LOG.global_plev())
		.build();
		
}

void LL_Z::compute() {
	
	TIME.start();
	
	auto b = m_locc->row_blk_sizes();
	auto o = m_locc->col_blk_sizes();
	auto v = m_lvir->col_blk_sizes();
	auto x = m_zmat->row_blk_sizes();
	
	arrvec<int,2> bo = {b,o};
	arrvec<int,2> bv = {b,v};
	arrvec<int,2> bb = {b,b};
	arrvec<int,3> xov = {x,o,v};
	arrvec<int,3> xob = {x,o,b};
	arrvec<int,3> xbb = {x,b,b};
	
	int nbtot = std::accumulate(b.begin(),b.end(),0);
	int notot = std::accumulate(o.begin(),o.end(),0);
	int nvtot = std::accumulate(v.begin(),v.end(),0);
	int nxtot = std::accumulate(x.begin(),x.end(),0);
	
	std::array<int,3> xobsizes = {nxtot,notot,nbtot};
	std::array<int,3> xovsizes = {nxtot,nvtot,nbtot};
	
	// ======= TAKE CARE OF MATRIX STUFF ============
	
	m_locc_01 = dbcsr::tensor<2>::create()
		.name("locc_01")
		.set_pgrid(*m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bo).build();
		
	m_lvir_01 = dbcsr::tensor<2>::create()
		.name("pvir_01")
		.set_pgrid(*m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bv).build();
		
	//copy over
	dbcsr::copy_matrix_to_tensor(*m_locc, *m_locc_01);
	dbcsr::copy_matrix_to_tensor(*m_lvir, *m_lvir_01);
	
	// ============= TAKE CARE OF TENSOR STUFF =============
		
	auto spgrid3_xob = dbcsr::pgrid<3>::create(m_world.comm())
		.tensor_dims(xobsizes).build();
		
	auto spgrid3_xov = dbcsr::pgrid<3>::create(m_world.comm())
		.tensor_dims(xovsizes).build();
		
	auto spgrid3_xbb = m_eri3c2e_batched->spgrid();
	
	auto b_xob_1_02 = dbcsr::tensor<3>::create()
		.set_pgrid(*spgrid3_xob)
		.name("b_xob_1_02")
		.map1({1}).map2({0,2})
		.blk_sizes(xob).build();
		
	auto b_xob_2_01 = 
		dbcsr::tensor<3>::create_template(b_xob_1_02)
		.name("b_xob_2_01").map1({2}).map2({0,1}).build();
		
	auto b_xov_2_01 = 
		dbcsr::tensor<3>::create_template(b_xob_1_02)
		.name("b2_xob_2_01").map1({2}).map2({0,1}).build(); 
		
	auto b_xov_0_12 = 
		dbcsr::tensor<3>::create_template(b_xob_1_02)
		.name("b2_xob_1_02").map1({1}).map2({0,2}).build();
	
	auto blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
	auto blkmap_o = vec<int>(o.size());
	auto blkmap_v = vec<int>(v.size());
	
	std::iota(blkmap_o.begin(), blkmap_o.end(), 0);
	std::iota(blkmap_v.begin(), blkmap_v.end(), 0);
	
	arrvec<int,3> blkmaps_xov = {blkmap_x, blkmap_o, blkmap_v};
	
	int nxbatches = m_eri3c2e_batched->nbatches(0);
	int nbbatches = m_eri3c2e_batched->nabtches(1);
		
	std::array<int,3> bdims_xov = {nxbatches,nbbatches,nbbatches};
		
	auto b_xov_batched = dbcsr::btensor_create<3>()
		.name("b_xov_batched")
		.set_pgrid(*spgrid3_xov)
		.blk_sizes(xov)
		.blk_map(blkmaps_xov)
		.batch_dims(bdims_xov)
		.btensor_type(m_intermeds)
		.print(LOG.global_plev())
		.build();
	
	time_reo_int1.start();
	LOG.os<1>("Reordering ints to 1|02.\n");
	m_eri3c2e_batched->decompress_init({0},vec<int>{1},vec<int>{0,2});
	time_reo_int1.finish();
	
	b_xov_batched->compress_init({0}, {0}, {1,2});
	
	LOG.os<1>("Starting batching over auxiliary functions.\n");
		
	// ===== LOOP OVER BATCHES OF AUXILIARY FUNCTIONS ==================
	for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
		
		LOG.os<1>("-- (X) Batch ", ix, "\n");
		
		LOG.os<1>("-- Fetching ao ints...\n");
		time_fetchints1.start();
		m_eri3c2e_batched->decompress({ix});
		time_fetchints1.finish();
		
		auto eri_1_02 = m_eri3c2e_batched->get_work_tensor();
					
		// first transform 
	    LOG.os<1>("-- First transform.\n");
	    
	    vec<vec<int>> x_nu_bounds = { 
			m_eri3c2e_batched->bounds(0,ix),
			m_eri3c2e_batched->full_bounds(2)
		};
		
		//std::cout << "ERI" << std::endl;
		//dbcsr::print(*eri_1_02);
	    
	    time_tran1.start();
	    dbcsr::contract(*m_locc_01, *eri_1_02, *b_xob_1_02)
			.bounds3(x_nu_bounds).perform("mi, Xmn -> Xin");
	    time_tran1.finish();
	    
	    // reorder
		LOG.os<1>("-- Reordering B_xob.\n");
		time_reo1.start();
		dbcsr::copy(*b_xob_1_02, *b_xob_2_01).move_data(true).perform();
		time_reo1.finish();
		
		//std::cout << "THERE" << std::endl;
		//dbcsr::print(*b_xob_2_01);
		
		//copy over vir density
		dbcsr::copy_matrix_to_tensor(*m_pvir, *m_pvir_01);
	
		// new loop over nu
		for (int inu = 0; inu != m_eri3c2e_batched->nbatches(2); ++inu) {
			
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
			
			//dbcsr::print(*b_xob_2_01);
			
			time_tran2.start();
			dbcsr::contract(*m_pvir_01, *b_xob_2_01, *b2_xob_2_01)
				.bounds2(nu_bounds).bounds3(x_u_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Nn, Xin -> XiN");
			time_tran2.finish();
		
			//dbcsr::print(*b2_xob_2_01);
			
			// reorder
			LOG.os<1>("---- Reordering B_xoB.\n");
			time_reo2.start();
			dbcsr::copy(*b2_xob_2_01, *b2_xob_1_02).move_data(true).perform();
			time_reo2.finish();
		
			LOG.os<1>("-- Final transform.\n");
			
			vec<vec<int>> x_nu_bounds = {
				m_eri3c2e_batched->bounds(0,ix),
				m_eri3c2e_batched->bounds(2,inu)
			};
			
			bool force_sparsity = false;
			if (true) {
				
				force_sparsity = true;
				auto& shellmat = *m_shellpair_info;
				
				arrvec<int,3> res;
				
				auto xblkbounds = m_eri3c2e_batched->blk_bounds(0,ix);
				auto bblkbounds = m_eri3c2e_batched->blk_bounds(2,inu);

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
			
			//dbcsr::print(*b2_xob_1_02);
											
			time_tran3.start();
			dbcsr::contract(*m_locc_01, *b2_xob_1_02, *b2_xbb_1_02)
				.bounds3(x_nu_bounds)
				.retain_sparsity(force_sparsity)
				.perform("Mi, XiN -> XMN");
			time_tran3.finish();
	
			//dbcsr::print(*b2_xbb_1_02);
	
			// reorder
			LOG.os<1>("---- B_xBB.\n");
			time_reo3.start();
			dbcsr::copy(*b2_xbb_1_02, *b2_xbb_0_12)
				.move_data(true)
				//.sum(true)
				.perform();
			time_reo3.finish();
			
			//dbcsr::print(*b2_xbb_0_12);
			
			//dbcsr::print(*B_xBB_0_12_wr);
			
			//dbcsr::print(*B_xBB_0_12);
			
			LOG.os<1>("---- Writing B_xBB to memory.\n");
			time_write.start();
			m_FT3c2e_batched->compress({ix,inu},b2_xbb_0_12);
			time_write.finish();
						
		}
		
		//dbcsr::print(*b2_xbb_0_12);
		
	}
	
	m_eri3c2e_batched->decompress_finalize();
	m_FT3c2e_batched->compress_finalize();
	
	LOG.os<1>("Occupation of FT3c2e: ", m_FT3c2e_batched->occupation()*100, "%\n"); 
	
	LOG.os<1>("Finished batching.\n");
	
	LOG.os<1>("Reordering ints 1|02 -> 0|12 \n");
	
	time_reo_int2.start();
	m_eri3c2e_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	time_reo_int2.finish();
	
	LOG.os<1>("Setting up decompression.\n");
	
	time_setview.start();
	m_FT3c2e_batched->decompress_init({0,2}, vec<int>{0}, vec<int>{1,2});
	time_setview.finish();
	
	LOG.os<1>("Computing Z_XY.\n");
	
	//m_zmat_01->batched_contract_init();
	
	for (int inu = 0; inu != m_eri3c2e_batched->nbatches(2); ++inu) {
		
		LOG.os<1>("Batch: ", inu, '\n'); 	
		LOG.os<1>("Fetching integrals...\n");
		
		time_fetchints2.start();
		m_eri3c2e_batched->decompress({inu});
		time_fetchints2.finish();
		
		auto eri_0_12 = m_eri3c2e_batched->get_work_tensor();
	
		for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
			
			LOG.os<1>("-- Batch: ", ix, '\n');
			
			LOG.os<1>("-- Fetching intermediate...\n");
			
			time_read.start();
			m_FT3c2e_batched->decompress({ix,inu});
			time_read.finish();
			
			auto z_xbb_0_12 = m_FT3c2e_batched->get_work_tensor();
			
			vec<vec<int>> x_bounds = {
				m_eri3c2e_batched->bounds(0,ix)
			};
			
			vec<vec<int>> mn_bounds = {
				m_eri3c2e_batched->full_bounds(1),
				m_eri3c2e_batched->bounds(2,inu)
			};
			
			// form Z
			LOG.os<1>("-- Forming Z.\n");
			
			time_formz.start();
			dbcsr::contract(*z_xbb_0_12, *eri_0_12, *m_zmat_01)
				.beta(1.0)
				.bounds1(mn_bounds)
				.bounds2(x_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("Mmn, Nmn -> MN");
			time_formz.finish();
			
		}
	}
	
	m_eri3c2e_batched->decompress_finalize();
	m_FT3c2e_batched->decompress_finalize();
	
	m_FT3c2e_batched->reset();

	//m_zmat_01->batched_contract_finalize();	
	
	LOG.os<1>("Finished batching.\n");

	// copy
	dbcsr::copy_tensor_to_matrix(*m_zmat_01, *m_zmat);
	
	m_zmat_01->clear();
	
	TIME.finish();
	
}

#endif
	
} // end namespace
