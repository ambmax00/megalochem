#ifndef MATH_SVD_H
#define MATH_SVD_H

#include <dbcsr_matrix_ops.hpp>
#include "extern/scalapack.h"
#include "utils/mpi_log.h"

namespace math {
	
class SVD {
private:

	dbcsr::smat_d m_mat_in;
	char m_jobu, m_jobvt;
	
	std::shared_ptr<scalapack::distmat<double>> m_U;
	std::shared_ptr<scalapack::distmat<double>> m_Vt;
	std::shared_ptr<std::vector<double>> m_s;
	 
	util::mpi_log LOG;
	 
public:

	SVD(dbcsr::smat_d& mat, char jobu, char jobvt, int print) : 
		m_mat_in(mat), LOG(m_mat_in->get_world().comm(),print),
		m_jobu(jobu), m_jobvt(jobvt) {}
	~SVD() {}
	
	void compute();
	
};

} // end namespace

#endif
