#ifndef MATH_LAPLACE_H
#define MATH_LAPLACE_H

#include <vector>
#include <limits>
#include "utils/mpi_log.hpp"
#include "math/laplace/laplace_minimax_c.hpp"
#include <optional>

namespace megalochem {

namespace math {
	
class laplace {
	
private:
	
	MPI_Comm m_comm;
	util::mpi_log LOG;
	
	int m_nlap;
	double m_err;
	std::vector<double> m_omega;
	std::vector<double> m_alpha;
	
/*	mxiter               (optional) maximum number of iterations. Used for each
                           of the iterative prodecures (Remez + Newton(-Maehly))

   stepmx               (optional) maximum step length used for each of the 
                           Newton type procedures

   tolrng               (optional) tolerance threshold for the Newton-Maehly
                           procedure that determines the extremum points

   tolpar               (optional) tolerance threshold for the Newton procedure
                           that computes the Laplace parameters at each extremum
                           point

   tolerr               (optional) tolerance threshold for the maximum quadrature
                           error obtained by the minimax algorithm

   delta                (optional) shift parameter for initializating the next
                           extremum point to be determined by Newton-Maehly

   afact                (optional) factor for the line search algorithm used
                           in combination with the Newton algorithm to determine
                           the Laplace parameters

   do_rmsd              (optional) compute an RMS error after the optimization
                           of the Laplace parameters
*/	
	
#define set_var(type, name) \
	std::optional<type> m_##name = std::nullopt; \
	laplace& set_##name(type name) { \
		m_##name = name; \
		return *this; \
	} 
	
	set_var(int, mxiter)
	set_var(double,stepmx)
	set_var(double,tolrng)
	set_var(double,tolpar)
	set_var(double,tolerr)
	set_var(double,delta)
	set_var(double,afact)
	set_var(bool,do_rmsd)

public:

	laplace(MPI_Comm comm, int print) :  
		m_comm(comm), LOG(comm, print), m_nlap(0), m_err(0.0) {}
		
	void compute(int nlap, double ymin, double ymax) {
	
		int rank = -1;
		
		MPI_Comm_rank(m_comm, &rank);
		
		m_omega.clear();
		m_alpha.clear();
		
		m_omega.resize(nlap);
		m_alpha.resize(nlap);
		
		if (rank == 0) {
		
			int nprint = LOG.global_plev();
			
			c_laplace_minimax(&m_err, m_alpha.data(), m_omega.data(), 
				&nlap, ymin, ymax, 
				(m_mxiter) ? &*m_mxiter : nullptr , 
				&nprint, 
				(m_stepmx) ? &*m_stepmx : nullptr, 
				(m_tolrng) ? &*m_tolrng : nullptr, 
				(m_tolpar) ? &*m_tolpar : nullptr, 
				(m_tolerr) ? &*m_tolerr : nullptr, 
				(m_delta) ? &*m_delta : nullptr,
				(m_afact) ? &*m_afact : nullptr,
				(m_do_rmsd) ? &*m_do_rmsd : nullptr,
				nullptr, nullptr);
				
		}
		
		MPI_Bcast(m_omega.data(), nlap, MPI_DOUBLE, 0, m_comm);
		MPI_Bcast(m_alpha.data(), nlap, MPI_DOUBLE, 0, m_comm);
			
	}
			
	
	std::vector<double> omega() const {
		return m_omega;
	}
	
	std::vector<double> alpha() const {
		return m_alpha;
	}
	
	~laplace() {}
	
};
		
} // end namespace

} // end megalochem

#endif
