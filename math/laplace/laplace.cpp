#include "math/laplace/laplace.h"
#include "math/laplace/lap_data.h"

#include <algorithm>

namespace math {
	
void laplace::minmax_read() {
	

	double minval, maxval;
	double R_target = -1;
	
	laplace_data LOCAL_DATA;
	
	// GET MINIMUM AND MAXIMUM
	for (auto lp : LAPLACE_DATA) {
		if ((lp.k == m_k) && (m_R > lp.R) && (R_target < lp.R)) {
			R_target = lp.R, R_target;
			LOCAL_DATA = lp;
		}	
	}
	
	if (R_target < 0) throw std::runtime_error("R is too small.");
	
	LOG.os<>("Target Value: ", R_target, '\n');
	
	for (int i = 0; i != m_k; ++i) {
		m_omega(i) = LOCAL_DATA.omega[i];
		m_alpha(i) = LOCAL_DATA.alpha[i];
	}
	
}
	
void laplace::compute() {
	
	double E_min = 2*(m_elumo - m_ehomo);
	double E_max = 2*(m_emax - m_emin);
	
	LOG.os<>("E_min, E_max ", (double)E_min, " ", (double)E_max, '\n');
	
	m_R = E_max/E_min;
	LOG.os<>("R ", m_R, '\n');

	// READ IN THE INITIAL SET FOR ALPHAS AND OMEGAS
	minmax_read();
	
	VectorX128 omega_old, alpha_old;
	
	// ENTER THE LOOP 
	int loop_i = 0;
	float128 conv = 1;
	
	while ((++loop_i < 100) && (conv > m_limit) ) {
	
		omega_old = m_omega;
		alpha_old = m_omega;
		
		// FIND EXTREMUM POINTS
		
		minmax_root();
		
		//OPTIMIZE PARAMETERS
		
		minmax_paraopt();
		
		conv = (m_omega - omega_old).norm();
		
		LOG.os<>("Iteration: ", loop_i, " with RMS of omega ", conv, '\n');
		
		
	}
	
	if (conv > m_limit) {
		LOG.error<>("Laplace quadrature FAILED!\n");
	}
	
	// scale
	
	m_omega /= (float128)E_min;
	m_alpha /= (float128)E_min;
	
	LOG.os<>("Laplace quadrature succeeded!\n");
	LOG.os<>("Final omega coeficients:\n", m_omega, '\n', "Final alpha coefficients:\n", m_alpha, '\n');
	
	// Test accuracy
	int nquad = 1000;
	double dx = (E_max - E_min)/nquad;
	Eigen::VectorXd xfrac(nquad);
	Eigen::VectorXd xlap(nquad);
	
	for (int i = 0; i != nquad; ++i) {
		float128 x = E_min + i*dx;
		
		xfrac(i) = 1/(double)x;
		
		float128 lquad = 0;
		for (int k = 0; k != m_k; ++k) {
			lquad += m_omega(k) * exp(-m_alpha(k) * x); 
		}
		
		xlap(i) = (double)lquad;
		
		//std::cout << x << " " << xfrac(i) << " " << xlap(i) << std::endl;
		
	}
	
	LOG.os<>("Error: ", (xfrac - xlap).norm() / nquad, '\n');
	
}

} // end namespace






