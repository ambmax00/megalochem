#ifndef INTS_INTEGRALS_H
#define INTS_INTEGRALS_H

#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>
#include "utils/pool.h"
#include "utils/params.hpp"
#include "desc/basis.h"
#include "ints/screening.h"
#include "math/other/blockmap.h"
#include <vector>
#include <mpi.h>
#include <libint2.hpp>

namespace ints {

void calc_ints(dbcsr::mat_d& m_out, util::ShrPool<libint2::Engine>& engine,
	std::vector<const desc::cluster_basis*>& basvec);

void calc_ints(dbcsr::tensor<2>& t_out, util::ShrPool<libint2::Engine>& eng,
	std::vector<const desc::cluster_basis*>& basvec);
	
void calc_ints(dbcsr::tensor<3>& t_out, util::ShrPool<libint2::Engine>& eng,
	std::vector<const desc::cluster_basis*>& basvec, screener* scr = nullptr);
	
void calc_ints(dbcsr::tensor<4>& t_out, util::ShrPool<libint2::Engine>& eng,
	std::vector<const desc::cluster_basis*>& basvec);
	
void calc_ints_schwarz_mn(dbcsr::mat_d& m_out, util::ShrPool<libint2::Engine>& engine,
	std::vector<const desc::cluster_basis*>& basvec);
	
void calc_ints_schwarz_x(dbcsr::mat_d& m_out, util::ShrPool<libint2::Engine>& engine,
	std::vector<const desc::cluster_basis*>& basvec);

void calc_ints(math::blockmap<3>& b_in, arrvec<int,3>& newblks, util::ShrPool<libint2::Engine>& engine,
	std::vector<const desc::cluster_basis*>& basvec);

}

#endif
