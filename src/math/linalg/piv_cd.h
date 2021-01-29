#ifndef PIV_CD_H
#define PIV_CD_H

#include <dbcsr_conversions.hpp>

#include "utils/mpi_log.h"

#include <utility>
#include <string>

#define _USE_SPARSE_COMPUTE

namespace math {
	
class pivinc_cd {
private:

	dbcsr::shared_matrix<double> m_mat_in;
	
#ifdef _USE_SPARSE_COMPUTE
	dbcsr::shared_matrix<double> m_L;
#else
	std::shared_ptr<scalapack::distmat<double>> m_L;
#endif
	
	util::mpi_log LOG;
	
	int m_rank = -1;
	double m_thresh;
	
	void reorder_and_reduce(scalapack::distmat<double>& L);

public:

	pivinc_cd(dbcsr::shared_matrix<double> mat_in, int print) : 
		m_mat_in(mat_in), LOG(m_mat_in->get_world().comm(), print) {}
	~pivinc_cd() {}
	
	void compute(std::optional<int> force_rank = std::nullopt);
	
	void compute_sparse();
	
	int rank() { return m_rank; }
	
	dbcsr::smat_d L(std::vector<int> rowblksizes, std::vector<int> colblksizes);
	
};
	
} // end namespace

#endif
