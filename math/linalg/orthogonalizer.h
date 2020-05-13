#ifndef MATH_ORTHOGONALIZER_H
#define MATH_ORTHOGONALIZER_H

#include <dbcsr_matrix.hpp>
#include <string>
#include <Eigen/Core>

namespace math {
	
class orthogonalizer {
private:

	dbcsr::smatrix<double> m_mat_in;
	dbcsr::smatrix<double> m_mat_out;
	int m_plev;

public:

	orthogonalizer(dbcsr::smatrix<double>& m) : m_mat_in(m), m_plev(0) {};
	
	void compute();

	dbcsr::smatrix<double> result() { return m_mat_out; }
	
};

} // end namespace

#endif
