#include "ints/gentran.h"

namespace ints {

dbcsr::sbtensor<3,double> transform3(dbcsr::sbtensor<3,double>& d_xbb_batched, 
	dbcsr::shared_tensor<2,double>& cm1, dbcsr::shared_tensor<2,double>& cm2, 
	dbcsr::shared_pgrid<3> pgrid3, int nbatches, dbcsr::btype mtype, std::string name) {

	/*
	auto comm = cm1->comm();

	auto d_xbb = d_xbb_batched->get_stensor();
	
	auto xblksizes = d_xbb->blk_sizes()[0];
	auto bblksizes = d_xbb->blk_sizes()[1];

	auto cm1blksizes = cm1->blk_sizes()[1];
	auto cm2blksizes = cm2->blk_sizes()[1];
	
	int nx = std::accumulate(xblksizes.begin(),xblksizes.end(),0);
	int nb = std::accumulate(bblksizes.begin(),bblksizes.end(),0);
	int nm1 = std::accumulate(cm1blksizes.begin(),cm1blksizes.end(),0);
	int nm2 = std::accumulate(cm2blksizes.begin(),cm2blksizes.end(),0);
	
	std::cout << nx << " " << nb << " " << nm1 << " " << nm2 << std::endl;
	
	arrvec<int,3> blksizes_hti = {xblksizes,cm1blksizes,bblksizes};
	arrvec<int,3> blksizes_full = {xblksizes,cm1blksizes,cm2blksizes};
	
	std::array<int,3> HTI_sizes = {nx,nm1,nb};
	std::array<int,3> FTI_sizes = {nx,nm1,nm2};
	
	auto spgrid3_xmb = dbcsr::create_pgrid<3>(comm)
		.tensor_dims(HTI_sizes).get();
		
	auto spgrid3_xmm = dbcsr::create_pgrid<3>(comm)
		.tensor_dims(FTI_sizes).get();
		
	pgrid3 = spgrid3_xmm;
	
	auto HTI_02_1 = dbcsr::tensor_create<3,double>().name("HTI")
		.pgrid(spgrid3_xmb).map1({0,2}).map2({1})
		.blk_sizes(blksizes_hti)
		.get();
		
	auto HTI_01_2 = dbcsr::tensor_create<3,double>().name("HTI")
		.pgrid(spgrid3_xmb).map1({0,1}).map2({2})
		.blk_sizes(blksizes_hti)
		.get();
	
	auto FTI_0_12 = dbcsr::tensor_create<3,double>()
		.name(name).pgrid(spgrid3_xmm)
		.map1({0}).map2({1,2})
		.blk_sizes(blksizes_full)
		.get();
	
	auto FTI_01_2 = dbcsr::tensor_create<3,double>()
		.name(name).pgrid(spgrid3_xmm)
		.map1({0,1}).map2({2})
		.blk_sizes(blksizes_full)
		.get();
		
	std::array 
		
	auto FTI_batched = std::make_shared<dbcsr::btensor<3,double>>(
		FTI_0_12, nbatches, mtype, 1);
		
	int xbatches = FTI_batched->nbatches_dim(0);
	int m1batches = FTI_batched->nbatches_dim(1);
	
	auto xblks = FTI_batched->bounds(0);
	auto m1blks = FTI_batched->bounds(1);
	auto bblksfull = d_xbb_batched->full_bounds(2);
	
	d_xbb_batched->reorder(vec<int>{0,2},vec<int>{1});
	
	d_xbb_batched->decompress_init({0});
	FTI_batched->compress_init({0,1});
	
	dbcsr::print(*cm1);
	dbcsr::print(*cm2);
	
	auto d_xbb_02_1 = d_xbb_batched->get_stensor();
	
	for (int ix = 0; ix != xbatches; ++ix) {
		
		d_xbb_batched->decompress({ix});
		
		for (int im1 = 0; im1 != m1batches; ++im1) {

			vec<vec<int>> xbbounds = {
				xblks[ix],
				bblksfull
			};
			
			vec<vec<int>> m1bounds = {
				m1blks[im1]
			};
			
			// contract 1
			dbcsr::contract(*d_xbb_02_1, *cm1, *HTI_02_1)
				.bounds2(xbbounds)
				.bounds3(m1bounds)
				.perform("XMN, Mi -> XiN");
			
			dbcsr::copy(*HTI_02_1, *HTI_01_2).move_data(true).perform();
			
			dbcsr::print(*HTI_01_2);
			
			vec<vec<int>> xm1_bounds = {
				xblks[ix],
				m1blks[im1]
			};
			
			// contract 2
			dbcsr::contract(*HTI_01_2, *cm2, *FTI_01_2)
				.bounds2(xm1_bounds)
				.perform("XiN, Nj -> Xij");
				
			HTI_02_1->clear();
				
			dbcsr::copy(*FTI_01_2, *FTI_0_12).move_data(true).perform();
			
			FTI_batched->compress({ix,im1}, FTI_0_12);
			
		}
		
	}
	
	d_xbb_batched->decompress_finalize();
	FTI_batched->compress_finalize();
	
	d_xbb_batched->reorder(vec<int>{0},vec<int>{1,2});
	
	return FTI_batched;*/
	
}
		
}
