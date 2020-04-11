#ifndef DBCSR_CONVERSIONS_HPP
#define DBCSR_CONVERSIONS_HPP

#include <Eigen/Core>
#include "math/tensor/dbcsr.hpp"
#include "utils/mpi_log.h"

#include <limits>

template <class T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

namespace dbcsr {

template <class T>
MatrixX<T> tensor_to_eigen(dbcsr::tensor<2,T>& array, int l = 0) {
	
	int myrank, mpi_size;
	
	MPI_Comm comm = array.comm();
	
	MPI_Comm_rank(comm, &myrank); 
	MPI_Comm_size(comm, &mpi_size);
	
	//std::cout << myrank << std::endl;
	
	auto total_elem = array.nfull_tot();
	auto nblocks = array.nblks_tot();
	auto blk_size = array.blk_size();
	auto blk_offset = array.blk_offset();
	
	auto dim1 = total_elem[0];
	auto dim2 = total_elem[1];
	auto blk1 = nblocks[0];
	auto blk2 = nblocks[1];
	
	util::mpi_log LOG(array.comm(), l);
	
	//LOG.os<1>("Dimensions: ", dim1, " ", dim2, '\n');
	
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
				
				//std::cout << idx[0] << " " << idx[1] << std::endl;
				
				blk = array.get_block({.idx = iter.idx(), .blk_size = {blk_size[0][idx[0]],blk_size[1][idx[1]]}, 
					.found = found});
				sizes = blk.sizes();
				
				//std::cout << "blk0 " << blk(0) << std::endl;
				
				data = blk.data();
				
			}
			
			MPI_Bcast(&idx[0], 2, MPI_INT, p, comm);
			int off1 = blk_offset[0][idx[0]];
			int off2 = blk_offset[1][idx[1]];
			
			//LOG.os<>("Offsets: ", off1, " ", off2, '\n');
			
			MPI_Bcast(&sizes[0], 2, MPI_INT, p, comm);
			//std::cout << "Sizes: " << sizes[0] << " " << sizes[1] << std::endl;
			
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
	
	//LOG.os<1,0>(m_out, '\n');
	MPI_Barrier(comm);
	//LOG.os<1,1>(m_out, '\n');
	
	return std::move(m_out);
	
	
}


template <typename T>
dbcsr::tensor<2,T> eigen_to_tensor(MatrixX<T>& M, std::string name, 
	dbcsr::pgrid<2>& grid, vec<int> map1, vec<int> map2, vec<vec<int>> blk_sizes, double eps = eps_filter) {
	
	dbcsr::tensor<2,T> out({.name = name, .pgridN = grid, .map1 = map1,
		.map2 = map2, .blk_sizes = blk_sizes});
		
	auto blkloc = out.blks_local();
	auto blkoff = out.blk_offset();
	
	std::vector<std::vector<int>> rsv(2);
	
	//std::cout << "CONVERTING: " << name << std::endl;
	// reserving blocks:
	for (int i = 0; i != blkloc[0].size(); ++i) {
		for (int j = 0; j != blkloc[1].size(); ++j) {
			rsv[0].push_back(blkloc[0][i]);
			rsv[1].push_back(blkloc[1][j]);
		}
	}
	
	out.reserve(rsv);
	
	dbcsr::iterator<2> iter(out);
	
	while (iter.blocks_left()) {
		
		iter.next();
		
		auto idx = iter.idx();
		auto blksize = iter.sizes();
		
		dbcsr::block<2> blk(blksize);
		auto eigenblk = M.block(blkoff[0][idx[0]],
			blkoff[1][idx[1]],blksize[0],blksize[1]);
			
		//std::cout << "IDX: " << idx[0] << " " << idx[1] << std::endl;
		//std::cout << eigenblk << std::endl;
		
		for (int n = 0; n != blksize[0]; ++n) {
					for (int m = 0; m != blksize[1]; ++m) {
						//double val = eigen_blk(n,m);
						//blk(n,m) = val; //(fabs(val) < std::numeric_limits<double>::epsilon())
							//? 0.0 : val;
						blk(n,m) = eigenblk(n,m);
		}}
		
		out.put_block({.idx = idx, .blk = blk});
		
	}
	
	//out.filter();	
	//std::cout << "Done!" << std::endl;
	//print(out);		
	
	return out;

}
	
}
	
	
#endif
