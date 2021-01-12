#include "mp/z_builder.h"

namespace mp {
	
SMatrixXi get_shellpairs(dbcsr::sbtensor<3,double> eri_batched) {
	
	auto blksizes = eri_batched->blk_sizes();
		
	int nblkb = blksizes[1].size();
	
	Eigen::MatrixXi idx_loc = Eigen::MatrixXi::Zero(nblkb,nblkb);
	Eigen::MatrixXi idx_tot = Eigen::MatrixXi::Zero(nblkb,nblkb);
	
	auto add_idx = [&idx_loc](dbcsr::shared_tensor<3,double> eri) {
		
		dbcsr::iterator_t<3,double> iter(*eri);
		iter.start();
		while (iter.blocks_left()) {
			iter.next_block();
			auto& tidx = iter.idx();
			
			int mu = tidx[1];
			int nu = tidx[2];
			
			idx_loc(mu,nu) = 1;
			
		}
		iter.stop();
	
	};

#ifndef _CORE_VECTOR
	if (eri_batched->get_type() == dbcsr::btype::core) {
		
		add_idx(eri_batched->get_work_tensor());
	
	} else {
#endif
		 
		eri_batched->decompress_init({0}, vec<int>{0}, vec<int>{1,2});
		
		for (int ix = 0; ix != eri_batched->nbatches(0); ++ix) {
			
			eri_batched->decompress({ix});
			add_idx(eri_batched->get_work_tensor());
			
		}
		
		eri_batched->decompress_finalize();
#ifndef _CORE_VECTOR	
	} 
#endif
	
	MPI_Allreduce(idx_loc.data(), idx_tot.data(), nblkb*nblkb, MPI_INT,
		MPI_LOR, eri_batched->comm());
	
	//std::cout << "IDXLOC" << std::endl;
	//std::cout << idx_loc << std::endl;
	
	SMatrixXi out = std::make_shared<Eigen::MatrixXi>(std::move(idx_tot));
	
	return out;
	
}
		
} // end namespace
