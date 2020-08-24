#include "mp/z_builder.h"

namespace mp {
	
Eigen::MatrixXi Z::get_shellpairs(dbcsr::sbtensor<3,double> eri_batched) {
	
	auto eri = eri_batched->get_stensor();
	auto ndims = eri->nblks_total();
	
	int nblkb = ndims[1];
	
	Eigen::MatrixXi idx_loc = Eigen::MatrixXi::Zero(nblkb,nblkb);
	Eigen::MatrixXi idx_tot = Eigen::MatrixXi::Zero(nblkb,nblkb);
	
	auto add_idx = [&eri,&idx_loc]() {
		
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
	
	if (eri_batched->get_type() == dbcsr::core) {
		
		add_idx();
	
	} else {
		 
		eri_batched->decompress_init({0});
		
		for (int ix = 0; ix != eri_batched->nbatches_dim(0); ++ix) {
			
			eri_batched->decompress({ix});
			
			add_idx();
			
		}
		
		eri_batched->decompress_finalize();
		
	} 
	
	MPI_Allreduce(idx_loc.data(), idx_tot.data(), nblkb*nblkb, MPI_INT,
		MPI_LOR, eri->comm());
	
	//std::cout << "IDXLOC" << std::endl;
	//std::cout << idx_loc << std::endl;
	
	return idx_tot;
	
}
		
} // end namespace
