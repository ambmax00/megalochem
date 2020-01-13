#ifndef INTS_INTEGRALS_H
#define INTS_INTEGRALS_H

#include "math/tensor/dbcsr.hpp"
#include "utils/pool.h"
#include "desc/basis.h"
#include <vector>
#include <mpi.h>
#include <libint2.hpp>

namespace ints {

template <int N>
dbcsr::tensor<N,double> integrals(MPI_Comm comm, util::ShrPool<libint2::Engine>& engine, 
	std::vector<desc::cluster_basis>& basvec, std::string name, 
	std::vector<int> map1, std::vector<int> map2);
	
}

#endif
