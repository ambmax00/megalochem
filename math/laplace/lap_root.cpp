#include "math/laplace/laplace.h"

#include <cmath>
#include <iostream>	
#include <iomanip>
#include <string>

namespace math {
	

void laplace::NewtonKahan(int i, int method) {
	
	// Newton procedure with Kahan Summation 
	// For f == 0, the standard Newton Raphson approach is used
	// Else, Newton-Maehly is used (modified Newton Raphson with deflation)
	
	float128& xi = m_xi(i);
	
	auto NRaphson = [&](int& n) { return dEta(m_xi(n))/ddEta(m_xi(n)); };
	auto NMaehly = [&](int& n) { 
		
		float128 fd = dEta(m_xi(n));
		float128 sd = ddEta(m_xi(n));
		
		float128 sum = 0;
		
		for (int j = 0; j != n; ++j) {
			sum += 1.0/(m_xi(n) - m_xi(j));
		}
		
		return fd/(sd - fd*sum);
	
	};
		
	float128 sum = 0;
	int iter = 0;
	float128 xold = 0;
	
	while ((fabs(xold - xi) > m_limit) && (iter < 50)) {
		
		xold = xi;
		
		if (method == 0) {
			xi -= NRaphson(i);
		} else {
			xi -= NMaehly(i);
		}
		
		++iter;
	}
		
		
}
	
void laplace::minmax_root() {	
	
	// ===================================================================	
	// ========= STEP 1: FIND THE EXTREMUM POINTS OF DELTA  ==============
	// ===================================================================
	
	// Find the extrema of the error distribution function Eta
	// by Newton-Raphson and Newton-Maehly procedure
	// Eta(x, omega, alpha) = sum^k_nu=1 omega_nu exp(-alpha_nu * x) - 1/x
	
	
	// STEP 1.1: INITIALIZING
	
	// xi : extrema of the error curve
	
	LOG.os<1>("\t Searching extremum points...\n");
	
	m_xi(0) = 1.0;
	m_xi(1) = 1.0;
	
	NewtonKahan(1, 0);
	
	// STEP 1.2 LOOP FOR i > 2
	
	double delta = 1e-4;
	
	for (int i = 2; i != 2*m_k; ++i){
		m_xi(i) = m_xi(i-1)*(1 + delta);
		NewtonKahan(i, 1);
	}
	
	m_xi(2*m_k) = m_R;
	
	
	//for (myfloat x = 1.0; x < 10; x += 0.01) {
	//	std::cout << x << " " << Eta(omega, alpha, k, x) << " " << dEta(omega, alpha, k, x) << std::endl;
	//}

}

}


