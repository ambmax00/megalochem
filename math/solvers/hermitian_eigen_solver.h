#ifndef MATH_HERMITIAN_EIGEN_SOLVER_H
#define MATH_HERMITIAN_EIGEN_SOLVER_H

#include <dbcsr_matrix.hpp>
#include "extern/scalapack.h"
#include "utils/mpi_log.h"

namespace math {

using smatrix = dbcsr::smatrix_d;
using matrix = dbcsr::matrix_d;

class hermitian_eigen_solver {
private:

	smatrix m_mat_in;
	smatrix m_eigvec;
	std::vector<double> m_eigval;
	dbcsr::world m_world;
	util::mpi_log LOG;

	int m_blksize;
	char m_jobz;
	
	std::optional<vec<int>> m_blksizes_out 
			= std::nullopt; //block sizes for eigenvector matrix
	
public:

	inline hermitian_eigen_solver& set_blksizes(vec<int>& blksizes) {
		m_blksizes_out = std::make_optional<vec<int>>(blksizes);
		return *this;
	}

	hermitian_eigen_solver(smatrix& mat_in, char jobz, int print = 0) :
		m_mat_in(mat_in), m_world(mat_in->get_world()),
		m_blksize(4), LOG(m_world.comm(), print),
		m_jobz(jobz) {}

	void compute();
	
	vec<double>& eigvals() {
		return m_eigval;
	}
	
	
};

}

#endif
