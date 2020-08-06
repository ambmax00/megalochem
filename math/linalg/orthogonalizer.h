#ifndef MATH_ORTHOGONALIZER_H
#define MATH_ORTHOGONALIZER_H

#include <dbcsr_matrix.hpp>
#include <string>
#include <Eigen/Core>

namespace math {
	
class orthogonalizer {
private:

	dbcsr::shared_matrix<double> m_mat_in;
	dbcsr::shared_matrix<double> m_mat_out;
	bool m_print;

public:

	orthogonalizer(dbcsr::shared_matrix<double>& m, bool print = false) : m_mat_in(m), m_print(print) {};
	
	void compute();

	dbcsr::smatrix<double> result() { return m_mat_out; }
	
};

} // end namespace

#endif
