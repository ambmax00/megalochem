#ifndef DBCSR_CONVERSIONS_HPP
#define DBCSR_CONVERSIONS_HPP

#include <Eigen/Core>
#include "math/tensor/dbcsr.hpp"
#include "utils/mpi_log.h"

template <class T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

namespace dbcsr {

template <class T>
MatrixX<T> tensor_to_eigen(dbcsr::tensor<2,T>& array, int l = 0) {
	
	int myrank, mpi_size;
	
	MPI_Comm comm = MPI_COMM_WORLD;
	
	MPI_Comm_rank(comm, &myrank); 
	MPI_Comm_size(comm, &mpi_size);
	
	std::cout << myrank << std::endl;
	
	auto total_elem = array.nfull_tot();
	auto nblocks = array.nblks_tot();
	auto blk_size = array.blk_size();
	auto blk_offset = array.blk_offset();
	
	auto dim1 = total_elem[0];
	auto dim2 = total_elem[1];
	auto blk1 = nblocks[0];
	auto blk2 = nblocks[1];
	
	util::mpi_log LOG(l);
	
	LOG.os<1>("Dimensions: ", dim1, " ", dim2, '\n');
	
	MatrixX<T> m_out = MatrixX<T>::Zero(dim1,dim2);
	
	int proc = -1;
	
	dbcsr::iterator<2,T> iter(array);
    
    for (int p = 0; p != mpi_size; ++p) {
		
		bool blks_left = true;
		dbcsr::index<2> idx;
		dbcsr::block<2,T> blk;
		
		if (myrank == p) {
			blks_left = iter.blocks_left();
		}
		
		MPI_Bcast(&blks_left, 1, MPI_C_BOOL, p, comm);
		
		MPI_Barrier(comm);
		
		vec<int> sizes(2);
		
		//bool end = false;
		
		while (blks_left) {
			
			int size = 0;
			T* data = nullptr;
				 
			if (p == myrank) {
				
				iter.next();
				blks_left = iter.blocks_left();
				bool found = false;
				idx = iter.idx();
				
				std::cout << idx[0] << " " << idx[1] << std::endl;
				
				blk = array.get_block({.idx = idx, .found = found});
				sizes = blk.sizes();
				
				std::cout << "blk0 " << blk(0) << std::endl;
				
				data = blk.data();
				
			}
			
			MPI_Bcast(&idx[0], 2, MPI_INT, p, comm);
			int off1 = blk_offset[0][idx[0]];
			int off2 = blk_offset[1][idx[1]];
			
			LOG.os<>("Offsets: ", off1, " ", off2, '\n');
			
			MPI_Bcast(&sizes[0], 2, MPI_INT, p, comm);
			std::cout << "Sizes: " << sizes[0] << " " << sizes[1] << std::endl;
			
			std::vector<T> vec(sizes[0]*sizes[1],T());
			if (p != myrank) {
				data =  vec.data();
			}
			
			LOG.os<1>("Broadcasting from: ", p, '\n');
			MPI_Bcast(data, sizes[0]*sizes[1], MPI_DOUBLE, p, comm);
			
			LOG.os<1>(data[0], "HERE\n");
			
			m_out.block(off1, off2, sizes[0], sizes[1]) 
				= Eigen::Map<MatrixX<T>>(data, sizes[0], sizes[1]);
			
			//LOG.os<1>(myblock, '\n');
			
			//m_out.block(off1, off2, sizes[0], sizes[1]) = myblock;
			
			//if (end) break;
			
			MPI_Bcast(&blks_left, 1, MPI_C_BOOL, p, comm);
			
			//if (!blks_left) end = true;
			
		}
		
	}
	
	LOG.os<1,0>(m_out, '\n');
	MPI_Barrier(comm);
	LOG.os<1,1>(m_out, '\n');
	
	return m_out;
	
	
}


template <typename T>
dbcsr::tensor<2,T> eigen_to_tensor(MatrixX<T>& M, std::string name, 
	dbcsr::pgrid<2>& grid, vec<int> map1, vec<int> map2, vec<vec<int>> blk_sizes, double eps = 1e-6) {
	
	dbcsr::tensor<2,T> out({.name = name, .pgridN = grid, .map1 = map1,
		.map2 = map2, .blk_sizes = blk_sizes});
		
	auto blkloc = out.blks_local();
	auto blkoff = out.blk_offset();
	
	std::cout << M << std::endl;
	
	for (int i = 0; i != blkloc[0].size(); ++i) {
		for (int j = 0; j != blkloc[1].size(); ++j) {
			
			int ix = blkloc[0][i];
			int jx = blkloc[1][j];
	
			dbcsr::index<2> idx = {ix,jx};
			vec<int> off = {blkoff[0][ix],blkoff[1][jx]};
			vec<int> sizes = {blk_sizes[0][ix],blk_sizes[1][jx]};
			
			auto eigen_blk = M.block(off[0],off[1],sizes[0],sizes[1]);
			
			std::cout << eigen_blk << std::endl;
			
			if (eigen_blk.norm() >= 1e-6) {
			
				std::cout << "RESERVED" << std::endl;
				out.reserve({{ix},{jx}});
				
				dbcsr::block<2,T> blk(sizes);
				
				for (int n = 0; n != sizes[0]; ++n) {
					for (int m = 0; m != sizes[1]; ++m) {
						blk(n,m) = eigen_blk(n,m);
				}}
				
				out.put_block({.idx = idx, .blk = blk});
			
			}
	}}
			
			
	
	return out;

}
	
}
	
	
#endif
