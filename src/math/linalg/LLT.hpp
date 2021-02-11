#ifndef MATH_LLT_H
#define MATH_LLT_H

#include <dbcsr_matrix_ops.hpp>
#include "extern/scalapack.hpp"
#include "utils/mpi_log.hpp"

namespace math {
	
class LLT {
private:

	dbcsr::shared_matrix<double> m_mat_in;
	std::shared_ptr<
	 scalapack::distmat<double>> m_L;
	std::shared_ptr<
	 scalapack::distmat<double>> m_L_inv;
	 
	util::mpi_log LOG;
	
	void compute_L_inverse();
	 
public:

	LLT(dbcsr::shared_matrix<double>& mat, int print) : 
		m_mat_in(mat), LOG(m_mat_in->get_world().comm(),print) {}
	~LLT() {}
	
	void compute();
	
	dbcsr::shared_matrix<double> L(vec<int> b);
	
	dbcsr::shared_matrix<double> L_inv(vec<int> b);
	
	dbcsr::shared_matrix<double> inverse(vec<int> b);
	
};

} // end namespace

#endif
