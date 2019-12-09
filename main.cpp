#include <mpi.h>
#include <random>
#include "math/laplace/laplace.h"
#include "math/tensor/dbcsr.hpp"

int main(int argv, char** argc) {

	MPI_Init(&argv, &argc);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	dbcsr::init();
	
	dbcsr::pgrid<4> pgrid4d(_comm = MPI_COMM_WORLD);
							 
	/* Alternatively:
	 * pgrid<4> pgrid4d(MPI_COMM_WORLD, 1, 1);
	 */
							 
	std::cout << "Created!" << std::endl;
	
	vec<int> blk1 = {3, 9, 12, 1};
    vec<int> blk2 = {4, 2, 3, 1, 9, 2, 32, 10, 5, 8, 7};
    vec<int> blk3 = {7, 3, 8, 7, 9, 5, 10, 23, 2};
    vec<int> blk4 = {8, 1, 4, 13, 6};
    vec<int> blk5 = {4, 2, 22};
	
	vec<int> d1 = dbcsr::random_dist(blk1.size(), pgrid4d.dims()[0]);
	vec<int> d2 = dbcsr::random_dist(blk2.size(), pgrid4d.dims()[1]);
	vec<int> d3 = dbcsr::random_dist(blk4.size(), pgrid4d.dims()[2]);
	vec<int> d4 = dbcsr::random_dist(blk5.size(), pgrid4d.dims()[3]);
	
	vec<int> map12 = {3,2};
	vec<int> map22 = {1,0};
	vec<vec<int>> dists = {d1,d2,d3,d4};
	
	dbcsr::dist<4> dist4d(_pgrid = pgrid4d, 
						   _map12 = map12,
						   _map22 = map22,
						   _nd_dists = dists);
						   
	vec<vec<int>> blk_sizes = {blk1,blk2,blk4,blk5};
						   
    dbcsr::tensor<4,double> tensor4d(_name = "HELLO", _dist = dist4d, _map12 = map12,
    _map22 = map22, _blk_sizes = blk_sizes);
	
	std::cout << "Tensor created." << std::endl;
	
	vec<int> nz21 = { 0, 0, 0, 0, 0, 1, 1, 1,  1,  1, 
             1, 1, 1, 1, 1, 1, 1, 1,  1,  1, 
             2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 
             3, 3, 3, 3, 3, 3 };
    vec<int> nz22 = { 0, 2, 3, 5, 9,  1, 1, 3,  4,  4, 
             5, 5, 5, 6,  6,  8, 8, 8, 9, 10, 
             0, 2, 2, 3,  4,  5, 7, 8, 10, 10, 
             0, 2, 3, 5, 9, 10 };
	vec<int> nz24 = { 2, 4, 1, 2,  1,  2, 4, 0,  0,  3, 
             1, 2, 3, 0,  3,  2, 3, 3,  1,  0, 
             2, 0, 0, 2,  3,  2, 3, 1,  1,  2, 
             0, 0, 2, 1,  4,  4 };
    vec<int> nz25 = { 0, 2, 1, 0,  0,  1, 2,  0,  2, 0, 
             1, 2, 1, 0,  2,  1, 2,  1,  0, 1, 
             2, 0, 1, 2,  1,  1, 1,  2,  0, 1, 
             0, 2, 1, 0,  2,  1 };		
    
    vec<vec<int>> nz = {nz21,nz22,nz24,nz25};
    
    tensor4d.reserve(nz);
    
    dbcsr::iterator<4,double> iter(tensor4d);
    
    std::cout << iter.blocks_left() << std::endl;
    
    while (iter.blocks_left()) {
		  
		iter.next();
		
		dbcsr::block<4,double> blk(iter.sizes());
		
		/*
		for (auto i : iter.sizes()) {
			std::cout << i << " ";
		}
		std::cout << std::endl;
		*/
		
		blk.fill_rand();
		
		/*
		for (auto i : blk.sizes()) {
			std::cout << i << " ";
		}
		std::cout << std::endl;
		
		
		for (int i = 0; i != blk.ntot(); ++i) {
			std::cout << blk(i) << " ";
		}
		
		std::cout << std::endl;
		*/
		auto idx = iter.idx();
		
		std::cout << "PRE: " << std::endl;
		std::cout << blk(0) << " " << idx[0] << std::endl;
	
		tensor4d.put_block(idx, blk);
		
		std::cout << "POST: " << std::endl;
		std::cout << blk(0) << " " << idx[0] << std::endl;	
		
	}
    
    std::cout << iter.sizes()[0] << std::endl;
	
	tensor4d.destroy();	   
	pgrid4d.destroy();
	dist4d.destroy();

	dbcsr::finalize();

	MPI_Finalize();

	return 0;

}
