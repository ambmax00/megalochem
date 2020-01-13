#include "ints/integrals.h"
#include <stdexcept>
#include <map>
#include <mpi.h>
#include <omp.h>

#include <iostream>

namespace ints {

// return just block?
void calc_ints(dbcsr::tensor<2,double>& t, util::ShrPool<libint2::Engine>& engine, 
	std::vector<desc::cluster_basis>& basvec) {

	// refactor into previous function
	// make sure to use move for decreasing number of local variables
	// make template, put shell loop in kernels which are specialized
	
	int myrank = 0;
	int mpi_size = 0;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	auto v = t.blks_local();
	auto nblks = t.nblks_tot();
	auto blk_size = t.blk_size();
	
	std::map<int, dbcsr::block<2,double>*> blocks;
	vec<vec<int>> reserve(2);
	
	desc::cluster_basis& cbas1 = basvec[0];
	desc::cluster_basis& cbas2 = basvec[1];

#pragma omp parallel 
{	
	
	auto& loc_eng = engine->local();
	const auto &results = loc_eng.results();
	
	#pragma omp for collapse(2)	
	for (int i1 = 0; i1 != v[0].size(); ++i1) {
		for (int i2 = 0; i2 != v[1].size(); ++i2) {
			
			int idx1 = v[0][i1];
			int idx2 = v[1][i2];
			
			int blkidx = idx1 * nblks[1] + idx2;
			
			/*
			#pragma omp critical 
			{
				for (int r = 0; r != mpi_size; ++r) {
					if (r == myrank) {
						std::cout << "Hello from process " << r << " out of "
						<< mpi_size << " and from thread " << omp_get_thread_num() << " out of " 
						<< omp_get_num_threads() << " " << idx1 << " " << idx2 << " : " << blkidx << std::endl;
					}
					MPI_Barrier(MPI_COMM_WORLD);
				}
			} 
			*/
			
			std::vector<libint2::Shell>& c1 = cbas1[idx1];
			std::vector<libint2::Shell>& c2 = cbas2[idx2];
			
			//std::cout << "Block: " << i1 << " " << i2 << std::endl;
			
			int off1 = 0;
			int off2 = 0;
			
			vec<int> sizes = {blk_size[0][idx1],blk_size[1][idx2]};
			auto blk_ptr = new dbcsr::block<2,double>(sizes);
			
			for (int s1 = 0; s1!= c1.size(); ++s1) {
				
				auto& sh1 = c1[s1];
				
				for (int s2 = 0; s2 != c2.size(); ++s2) {
					
					auto& sh2 = c2[s2];
					
					//std::cout << "Shells: " << s1 << " " << s2 << std::endl;
					//std::cout << "Sizes: " << sh1.size() << " " << sh2.size() << std::endl;
					//std::cout << "Offset: " << off1 << " " << off2 << std::endl;
					
					loc_eng.compute(sh1,sh2);
					
					auto ints_shellsets = results[0];
					
					if (ints_shellsets != nullptr) {
						int idx = 0;
						for (int i = 0; i != sh1.size(); ++i) {
							for (int j = 0; j != sh2.size(); ++j) {
								blk_ptr->operator()(i + off1, j + off2) = ints_shellsets[idx++];
								
								std::cout << blk_ptr->operator()(i + off1, j + off2) << std::endl;
						}}
					} //endif
					
					
					off2 += c2[s2].size();
				}//endfor s2
				
				off2 = 0;
				off1 += c1[s1].size();
			}//endofor s1
			
			#pragma omp critical 
			{
				blocks[blkidx] = blk_ptr;
				reserve[0].push_back(i1);
				reserve[1].push_back(i2);
			}
			
		}//endfor i2
	}//endfor i1
}//end parallel omp	
	
	//std::cout << "Done." << std::endl;
	
	t.reserve(reserve);
	
	for (auto v : reserve) {
		for (auto x : v) {
			std::cout << x << " ";
		} std::cout << std::endl;
	}
	
	std::cout << "Reserved." << std::endl;
	
	dbcsr::iterator<2,double> iter(t);
	
	while (iter.blocks_left()) {
		iter.next();
		auto idx = iter.idx();
		
		
		for (int r = 0; r != mpi_size; ++r) {
			if (r == myrank) {
				std::cout << r << ": " << idx[0] << " " << idx[1] << std::endl;
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		
		
		t.put_block({.idx = idx, .blk = *blocks[idx[0] * nblks[1] + idx[1]]});
		
	}
		
	std::cout << "Done reading." << std::endl;

}
	
	
template <int N>
dbcsr::tensor<N,double> integrals(MPI_Comm comm, util::ShrPool<libint2::Engine>& engine, 
	std::vector<desc::cluster_basis>& basvec, std::string name, 
	std::vector<int> map1, std::vector<int> map2) {
		
	if (basvec.size() != N) throw std::runtime_error("Basis vector incompatible with tensor dimensions.");
	
	//create pgrid
	dbcsr::pgrid<N> pgridN({.comm = comm});
	
	vec<vec<int>> blk_sizes;
	
	for (int i = 0; i != N; ++i) {
		blk_sizes.push_back(basvec[i].cluster_sizes());
	}
	
	//create tensor 
	dbcsr::tensor<N,double> out({.name = "test", .pgridN = pgridN, .map1 = map1, 
		.map2 = map2, .blk_sizes = blk_sizes});
	
	calc_ints(out, engine, basvec);
	
	return out;
		
}

template dbcsr::tensor<2,double> integrals(MPI_Comm comm, util::ShrPool<libint2::Engine>& engine, 
	std::vector<desc::cluster_basis>& basvec, std::string name, 
	std::vector<int> map1, std::vector<int> map2);


}
