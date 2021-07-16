#ifndef MATH_MINIMAX_HPP
#define MATH_MINIMAX_HPP

#include <vector>
#include <optional>

extern "C" {
  void c_laplace_minimax(double* errmax, double* xpnts, double* wghts, int* nlap,
    double ymin, double ymax, int* mxiter, int* iprint, double* stepmx,
    double* tolrng, double* tolpar, double* tolerr, double* delta, double* afact,
    bool* do_rmsd, bool* do_init, bool* do_nlap);
}

namespace megalochem {

namespace math {
	
class minimax {
private:
		
	int m_nlap, m_print;
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
	minimax& set_##name(type name) { \
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

	minimax(int print = 0) :  
		m_nlap(0), m_print(print), m_err(0.0) {}
		
	void compute(int nlap, double ymin, double ymax) {
		
		m_omega.clear();
		m_alpha.clear();
		
		m_omega.resize(nlap);
		m_alpha.resize(nlap);
    
    c_laplace_minimax(&m_err, m_alpha.data(), m_omega.data(), 
      &nlap, ymin, ymax, 
      (m_mxiter) ? &*m_mxiter : nullptr , 
      &m_print, 
      (m_stepmx) ? &*m_stepmx : nullptr, 
      (m_tolrng) ? &*m_tolrng : nullptr, 
      (m_tolpar) ? &*m_tolpar : nullptr, 
      (m_tolerr) ? &*m_tolerr : nullptr, 
      (m_delta) ? &*m_delta : nullptr,
      (m_afact) ? &*m_afact : nullptr,
      (m_do_rmsd) ? &*m_do_rmsd : nullptr,
      nullptr, nullptr);
			
	}
			
	
	std::vector<double> weights() const {
		return m_omega;
	}
	
	std::vector<double> exponents() const {
		return m_alpha;
	}
	
	~minimax() {}
	
};

}  // namespace math

}  // namespace megalochem

#endif
