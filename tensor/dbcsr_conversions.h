#ifndef DBCSR_CONVERSIONS_HPP
#define DBCSR_CONVERSIONS_HPP

#include <Eigen/Core>
#include "tensor/dbcsr.hpp"
#include "utils/mpi_log.h"

#include <limits>

template <class T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

namespace dbcsr {

template <typename T = double>
MatrixX<T> tensor_to_eigen(dbcsr::tensor<2,T>& array, int l = 0) {
	
	int myrank, mpi_size;
	
	MPI_Comm comm = array.comm();
	
	MPI_Comm_rank(comm, &myrank); 
	MPI_Comm_size(comm, &mpi_size);
	
	arr<int,2> tsize = array.nfull_total();
	
	MatrixX<T> m_out(tsize[0],tsize[1]);
	
	/* we loop over each process, from which we broadcast
	 * each local block to all the other processes 
	 */
	 
	iterator_t<2,T> iter(array);
	iter.start();
	
	for (int p = 0; p != mpi_size; ++p) {
		
		int numblocks = -1;
		
		if (p == myrank) numblocks = array.num_blocks(); 
		
		MPI_Bcast(&numblocks,1,MPI_INT,p,comm);
	
		for (int iblk = 0; iblk != numblocks; ++iblk) {
			
			// needed: blocksize, blockoffset, block
			index<2> idx;
			index<2> blkoff;
			index<2> blksize;
			
			if (p == myrank) {
				iter.next();
				idx = iter.idx();
				blkoff = iter.offset();
				blksize = iter.size();
				
			}	
			
			MPI_Bcast(blkoff.data(),2,MPI_INT,p,comm);
			MPI_Bcast(blksize.data(),2,MPI_INT,p,comm);
			
			block<2,T> blk(blksize);
			
			if (p == myrank) {
				bool found;
				blk = array.get_block(idx,blksize,found);
			}
			
			MPI_Bcast(blk.data(),blk.ntot(),MPI_DOUBLE,p,comm);	
			
			m_out.block(blkoff[0], blkoff[1], blksize[0], blksize[1]) 
				= Eigen::Map<MatrixX<T>>(blk.data(), blksize[0], blksize[1]);
				
		}
		
	}
	
	iter.stop();
	
	return m_out;
	
}


template <typename T = double>
dbcsr::tensor<2,T> eigen_to_tensor(MatrixX<T>& M, std::string name, 
	dbcsr::pgrid<2>& grid, vec<int> map1, vec<int> map2, arrvec<int,2> blk_sizes, double eps = block_threshold) {
	
	dbcsr::tensor<2,T> out = typename dbcsr::tensor<2,T>::create().name(name).ngrid(grid)
		.map1(map1).map2(map2).blk_sizes(blk_sizes);
		
	out.reserve_all();
	
	#pragma omp parallel 
	{
		dbcsr::iterator_t<2> iter(out);
		iter.start();
	
		while (iter.blocks_left()) {
			
			iter.next();
			
			auto& idx = iter.idx();
			auto& size = iter.size();
			auto& off = iter.offset();
			
			MatrixX<T> eigenblk = M.block(off[0],off[1],size[0],size[1]);
				
			dbcsr::block<2> blk(size,eigenblk.data());
			
			out.put_block(idx,blk);
			
		}
		
		out.finalize();
		iter.stop();
	}	
	
	return out;

}
	
}
	
	
#endif
