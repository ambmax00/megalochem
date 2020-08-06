#ifndef PIV_CD_H
#define PIV_CD_H

#include <dbcsr_conversions.hpp>

#include "utils/mpi_log.h"

#include <utility>
#include <string>

namespace math {
	
class pivinc_cd {
private:

	dbcsr::shared_matrix<double> m_mat_in;
	std::shared_ptr<scalapack::distmat<double>> m_L;
	
	util::mpi_log LOG;
	
	int m_rank = -1;
	double m_thresh;
	
	std::optional<std::string> m_reorder_method = std::nullopt;
	std::optional<bool> m_reduce = std::nullopt;
	
	void reorder_and_reduce(scalapack::distmat<double>& L);

public:

	inline pivinc_cd& reorder(std::string method) {
		m_reorder_method = method;
		return *this;
	}
	
	inline pivinc_cd& reduce(bool red) {
		m_reduce = red;
		return *this;
	}
		

	pivinc_cd(dbcsr::shared_matrix<double> mat_in, int print) : 
		m_mat_in(mat_in), LOG(m_mat_in->get_world().comm(), print) {}
	~pivinc_cd() {}
	
	void compute();
	
	int rank() { return m_rank; }
	
	dbcsr::smat_d L(std::vector<int> rowblksizes, std::vector<int> colblksizes);
	
};
	
} // end namespace

#endif
