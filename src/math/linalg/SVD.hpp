#ifndef MATH_SVD_H
#define MATH_SVD_H

#include <dbcsr_matrix_ops.hpp>
#include "extern/scalapack.hpp"
#include "utils/mpi_log.hpp"

namespace math {
	
class SVD {
private:

	dbcsr::shared_matrix<double> m_mat_in;
	char m_jobu, m_jobvt;
	
	std::shared_ptr<scalapack::distmat<double>> m_U;
	std::shared_ptr<scalapack::distmat<double>> m_Vt;
	std::shared_ptr<std::vector<double>> m_s;
	 
	util::mpi_log LOG;
	
	int m_rank;
	 
public:

	SVD(dbcsr::shared_matrix<double>& mat, char jobu, char jobvt, 
		int print) : 
		m_mat_in(mat), LOG(m_mat_in->get_world().comm(),print),
		m_jobu(jobu), m_jobvt(jobvt) {}
	~SVD() {}
	
	void compute(double eps = 1e-10);
	
	dbcsr::shared_matrix<double> inverse();
	
	dbcsr::shared_matrix<double> U(std::vector<int> rowblksizes, std::vector<int> colblksizes);
	dbcsr::shared_matrix<double> Vt(std::vector<int> rowblksizes, std::vector<int> colblksizes);
	std::vector<double> s();
	
	int rank() { return m_rank; }
	
};

} // end namespace

#endif
