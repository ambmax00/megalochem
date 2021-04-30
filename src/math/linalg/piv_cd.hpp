#ifndef PIV_CD_H
#define PIV_CD_H

#include <dbcsr_conversions.hpp>

#include "utils/mpi_log.hpp"
#include "megalochem.hpp"
#include <utility>
#include <string>

#define _USE_SPARSE_COMPUTE

namespace megalochem {

namespace math {
	
class pivinc_cd {
private:

	world m_world;

	dbcsr::shared_matrix<double> m_mat_in;
	
#ifdef _USE_SPARSE_COMPUTE
	dbcsr::shared_matrix<double> m_L;
#else
	std::shared_ptr<scalapack::distmat<double>> m_L;
#endif
	
	util::mpi_log LOG;
	
	int m_rank = -1;
	
	std::vector<int> m_perm;
	
	void reorder_and_reduce(scalapack::distmat<double>& L);

public:

	pivinc_cd(world w, dbcsr::shared_matrix<double> mat_in, int print) : 
		m_world(w), m_mat_in(mat_in), LOG(m_mat_in->get_cart().comm(), print) {}
	~pivinc_cd() {}
	
	void compute(std::optional<int> force_rank = std::nullopt,
		std::optional<double> eps = std::nullopt);
	
	void compute_sparse();
	
	int rank() { return m_rank; }
	
	std::vector<int> perm() {
		return m_perm;
	}
	
	dbcsr::shared_matrix<double> L(std::vector<int> rowblksizes, std::vector<int> colblksizes);
	
};
	
} // end namespace

} // end mega

#endif
