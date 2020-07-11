#ifndef MATH_LAPLACE_H
#define MATH_LAPLACE_H

#include <vector>
#include <limits>
#include "utils/mpi_log.h"
#include <laplace_minimax_c.h>

namespace math {
	
class laplace {
	
private:
	
	MPI_Comm m_comm;
	util::mpi_log LOG;
	
	int m_nlap;
	double m_err;
	std::vector<double> m_omega;
	std::vector<double> m_alpha;
	
	

public:

	laplace(MPI_Comm comm, int print) :  
		LOG(comm, print), m_nlap(0), m_err(0.0) {}
		
	void compute(int nlap, double ymin, double ymax) {
		
		m_omega.clear();
		m_alpha.clear();
		
		m_omega.resize(nlap);
		m_alpha.resize(nlap);
		
		e_minimax(errmax,xpnts,wghts,nlap,ymin,ymax,&
                           mxiter,iprint,stepmx,tolrng,tolpar,tolerr,&
                           delta,afact,do_rmsd,do_init,do_nlap)
		
		c_laplace_minimax(&m_err, m_omega.data(), m_alpha.data(), 
			&m_nlap, ymin, ymax, 
			(m_mxiter) ? &*m_mxiter : nullptr , 
			&LOG.global_plev(), 
			(m_stepmx) ? &*m_stepmx : nullptr, 
			(m_tolrng) ? &*m_tolrng : nullptr, 
			(m_tolpar) ? &*m_tolpar : nullptr, 
			(m_tolerr) ? &*m_tolerr : nullptr, 
			(m_delta) ? &*m_delta : nullptr,
			(m_afact) ? &*m_afact : nullptr,
			(m_do_rmsd) ? &*m_do_rmsd : nullptr,
			(m_do_init) ? &*m_do_init : nullptr,
			nullptr);
			
	
	std::vector<double> omega() {
		return m_omega;
	}
	
	std::vector<double> alpha() {
		return m_alpha;
	}
	
	~laplace() {}
	
	void compute() {}
	
	
};
		
} // end namespace

#endif
