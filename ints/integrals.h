#ifndef INTS_INTEGRALS_H
#define INTS_INTEGRALS_H

#include "math/tensor/dbcsr.hpp"
#include "ints/screening.h"
#include "utils/pool.h"
#include "desc/basis.h"
#include <vector>
#include <mpi.h>
#include <libint2.hpp>

namespace ints {
	
static int multiplicity(int i, int j) {
	return (i == j) ? 1 : 2;
}

static int multiplicity(int i, int j, int k) {
	return (j != k) ? 2 : 1;
}

static bool is_canonical(int i, int j) {
	return (i >= j);
}

static bool is_canonical(int i, int j, int k) {
	return (j >= k);
}

//static bool is_canonical(int i, int j, int k, int l) {
//    return (i > k) || (i == k && j >= l);
//}

template <int N>
struct integral_parameters {
	required<MPI_Comm,val> comm;
	required<util::ShrPool<libint2::Engine>,ref> engine;
	required<std::vector<desc::cluster_basis>,ref> basvec;
	optional<Zmat,ref> bra;
	optional<Zmat,ref> ket;
	required<std::string,val> name;
	required<std::vector<int>,val> map1;
	required<std::vector<int>,val> map2;
};

template <int N>
dbcsr::tensor<N,double> integrals(integral_parameters<N>&& p);
	
}

#endif
