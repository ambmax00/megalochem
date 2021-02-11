#ifndef MATH_HERMITIAN_EIGEN_SOLVER_H
#define MATH_HERMITIAN_EIGEN_SOLVER_H

#include <dbcsr_matrix.hpp>
#include "extern/scalapack.hpp"
#include "utils/mpi_log.hpp"

namespace math {

using smatrix = dbcsr::smat_d;
using matrix = dbcsr::mat_d;

class hermitian_eigen_solver {
private:

	dbcsr::shared_matrix<double> m_mat_in;
	dbcsr::shared_matrix<double> m_eigvec;
	std::vector<double> m_eigval;
	dbcsr::world m_world;
	util::mpi_log LOG;

	char m_jobz;
	
	std::optional<vec<int>> m_rowblksizes_out 
			= std::nullopt; //block sizes for eigenvector matrix
	std::optional<vec<int>> m_colblksizes_out
			= std::nullopt;
	
public:

	inline hermitian_eigen_solver& eigvec_rowblks(vec<int>& blksizes) {
		m_rowblksizes_out = std::make_optional<vec<int>>(blksizes);
		return *this;
	}
	
	inline hermitian_eigen_solver& eigvec_colblks(vec<int>& blksizes) {
		m_colblksizes_out = std::make_optional<vec<int>>(blksizes);
		return *this;
	}

	hermitian_eigen_solver(dbcsr::shared_matrix<double>& mat_in, 
		char jobz, bool print = false) :
		m_mat_in(mat_in), m_world(mat_in->get_world()),
		LOG(m_world.comm(), (print) ? 0 : -1),
		m_jobz(jobz) {}

	void compute();
	
	vec<double>& eigvals() {
		return m_eigval;
	}
	
	dbcsr::shared_matrix<double> eigvecs() {
		return m_eigvec;
	}
	
	dbcsr::shared_matrix<double> inverse();
	
	dbcsr::shared_matrix<double> inverse_sqrt();
	
};

}

#endif
