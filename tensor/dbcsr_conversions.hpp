#ifndef DBCSR_CONVERSIONS_HPP
#define DBCSR_CONVERSIONS_HPP

#include <Eigen/Core>
#include "extern/scalapack.h"
#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>
#include <limits>

template <class T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

namespace dbcsr {

template <typename T = double>
MatrixX<T> matrix_to_eigen(dbcsr::matrix<T>& mat) {
	int row = mat.nfullrows_total();
	int col = mat.nfullcols_total();
	
	mat.replicate_all();
	
	MatrixX<T> eigenmat(row,col);
	
	iterator<T> iter(mat);
	
	iter.start();
	
	while (iter.blocks_left()) {
		
		iter.next_block();
		for (int j = 0; j != iter.col_size(); ++j) {
			for (int i = 0; i != iter.row_size(); ++i) {
				eigenmat(i + iter.row_offset(), j + iter.col_offset())
					= iter(i,j);
			}
		}
	
	}
	
	iter.stop();
	
	mat.distribute();
	
	return eigenmat;
	
}

template <typename T = double>
matrix<T> eigen_to_matrix(MatrixX<T>& mat, world& w, std::string name, vec<int>& row_blk_sizes, 
	vec<int>& col_blk_sizes, char type) {
	
	matrix<T> out = typename matrix<T>::create().name(name).set_world(w).row_blk_sizes(row_blk_sizes)
		.col_blk_sizes(col_blk_sizes).type(type);
	
	out.reserve_all();
	
	iterator<T> iter(out);
	
	iter.start();
	
	while (iter.blocks_left()) {
		
		iter.next_block();
		
		for (int j = 0; j != iter.col_size(); ++j) {
			for (int i = 0; i != iter.row_size(); ++i) {
				iter(i,j) = mat(i + iter.row_offset(), j + iter.col_offset());
			}
		}
	}
	
	iter.stop();
	out.finalize();
	return out;
	
}

#ifdef USE_SCALAPACK

template <typename T = double>
scalapack::distmat<T> matrix_to_scalapack(matrix<T>& mat_in, std::string nameint, 
										scalapack::grid& igrid, int nsplitrow, int nsplitcol) {
	
	world mworld = mat_in.get_world();	
	
	int nrows = mat_in.nfullrows_total();
	int ncols = mat_in.nfullcols_total();		
	
	auto split_range = [](int n, int split) {
	
				// number of intervals
				int nblock = n%split == 0 ? n/split : n/split + 1;
				bool even = n%split == 0 ? true : false;
				
				if (even) {
					std::vector<int> out(nblock,split);
					return out;
				} else {
					std::vector<int> out(nblock,split);
					out[nblock-1] = n%split;
					return out;
				}
	};
	
	// make distvecs
	vec<int> rowsizes = split_range(nrows, nsplitrow);
	vec<int> colsizes = split_range(ncols, nsplitcol);
	
	auto cyclic_dist = [](int dist_size, int nbins) {
  
		std::vector<int> distv(dist_size);
		for(int i=0; i < dist_size; i++)
			distv[i] = i % nbins;

		return distv;
	};
	
	vec<int> rowdist = cyclic_dist(rowsizes.size(),mworld.dims()[0]);
	vec<int> coldist = cyclic_dist(colsizes.size(),mworld.dims()[1]);
	
	dist scaldist = dist::create().set_world(mworld).row_dist(rowdist).col_dist(coldist);
	
	matrix<T> mat_out = typename matrix<T>::create().name(mat_in.name() + " redist.").set_dist(scaldist)
		.row_blk_sizes(rowsizes).col_blk_sizes(colsizes).type(dbcsr_type_no_symmetry);

	if (mat_in.has_symmetry()) {
		matrix<T> mat_in_nosym = mat_in.desymmetrize();
		mat_out.redistribute(mat_in_nosym, false);
		mat_in_nosym.release();
	} else {
		mat_out.redistribute(mat_in,false);
	}
	
	dbcsr::print(mat_out);
	
	scalapack::distmat<double> scamat(igrid,nrows,ncols,nsplitrow,nsplitcol);
	
	dbcsr::iterator iter(mat_out);

#pragma omp parallel
{
	iter.start();
	
	while (iter.blocks_left()) {
		
		iter.next_block();
		
		int ioff = iter.row_offset();
		int joff = iter.col_offset();
		
		for (int j = 0; j != iter.col_size(); ++j) {
			for (int i = 0; i != iter.row_size(); ++i) {
				// use global indices for scamat
				scamat.global_access(i + ioff, j + joff) = iter(i,j);
			}
		}
		
	}
	
	iter.stop();
	mat_out.finalize();
}	
	mat_out.release();
	
	scamat.print();
	
	return scamat;
	
}

#endif

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
