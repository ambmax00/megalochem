#include <mpi.h>
#include "math/laplace/laplace.h"
#include "math/tensor/dbcsr.hpp"

int main(int argv, char** argc) {

	MPI_Init(&argv, &argc);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	dbcsr::init();
	
	std::optional<int> nsplit = 1;
	std::optional<int> ndim = 1;
	
	
	dbcsr::pgrid<4> p("comm"_arg = MPI_COMM_WORLD, "dimsplit"_arg = ndim, "nsplit"_arg = nsplit);
	
	auto dims = p.dims();
	
	auto dist1 = dbcsr::random_dist(dims.size(), dims[0]);
	auto dist2 = dbcsr::random_dist(dims.size(), dims[1]);
	auto dist3 = dbcsr::random_dist(dims.size(), dims[2]);
	auto dist4 = dbcsr::random_dist(dims.size(), dims[3]);
	
	if (rank == 0) {
		for (auto i : dist1) {
			std::cout << i << " ";
		}
		std::cout << std::endl;
		
		for (auto i : dist2) {
			std::cout << i << " ";
		}
		std::cout << std::endl;
		
		for (auto i : dist3) {
			std::cout << i << " ";
		}
		std::cout << std::endl;
		
		for (auto i : dist4) {
			std::cout << i << " ";
		}
		std::cout << std::endl;
	}
	
	dbcsr::dist<4> d1(p, vec<int>{0,1}, vec<int>{2,3}, vec<vec<int>>{dist1,dist2,dist3,dist4});

	//math::laplace lp(6, -1, -2, 2, 1000);

	//lp.compute();	
	
	p.destroy();

	dbcsr::finalize();

	MPI_Finalize();

	return 0;

}
