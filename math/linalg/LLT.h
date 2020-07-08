#ifndef MATH_LLT_H
#define MATH_LLT_H

#include <dbcsr_matrix_ops.hpp>
#include "extern/scalapack.h"
#include "utils/mpi_log.h"

namespace math {
	
class LLT {
private:

	dbcsr::smat_d m_mat_in;
	std::shared_ptr<
	 scalapack::distmat<double>> m_L;
	std::shared_ptr<
	 scalapack::distmat<double>> m_L_inv;
	 
	util::mpi_log LOG;
	
	void compute_L_inverse();
	 
public:

	LLT(dbcsr::smat_d& mat, int print) : 
		m_mat_in(mat), LOG(m_mat_in->get_world().comm(),print) {}
	~LLT() {}
	
	void compute();
	
	dbcsr::smat_d L(vec<int> b);
	
	dbcsr::smat_d L_inv(vec<int> b);
	
	dbcsr::smat_d inverse(vec<int> b);
	
};

} // end namespace

#endif
