#ifndef MATH_LAPLACE_H
#define MATH_LAPLACE_H

#include <vector>
#include <limits>
#include "utils/mpi_log.h"
#include <laplace_minimax_c.h>

namespace math {
	
class laplace {
	
private:

	util::mpi_log LOG;
	int m_k;

public:

	laplace() : 
		m_k(t_k), m_omega(t_k), m_alpha(t_k), 
		LOG(MPI_COMM_WORLD, print), m_limit(std::numeric_limits<double>::epsilon()) {};
	
	std::vector<double> omega() {
		return m_omega_db;
	}
	
	std::vector<double> alpha() {
		return m_alpha_db;
	}
	
	~laplace() {}
	
	void compute() {}
	
	
};
		
} // end namespace

#endif
