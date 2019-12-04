#include "math/laplace/laplace.h"
#include <cmath>
#include <iostream>
#include <Eigen/LU> 

namespace math {


MatrixX128 laplace::Jacobi() {
	
		// INITIALIZE (2k+1)² MATRIX
		
		/*STRCUTURE:
		|	d/d w_j f(x_1) 	   ... d/d a_j f(x_1)     ... d/d deltak f(x_1)		 |
		|		...                 ...					....					 |
		|	d/d w_j f(x_2*k+1) ... d/d a_j f(x_2*k+1) ... d/d deltak f(x_2*k+1)  |
		*/
	
		// COLUMN-MAJOR !!!!
		MatrixX128 Jac(2*m_k+1, 2*m_k+1);
		
		
		for (int i = 0; i != 2*m_k+1; ++i) {
		
			// OMEGA GRADIENTS
			for (int j = 0; j != m_k; ++j) {
				Jac(i,j) = exp(- m_alpha(j)*m_xi(i));
			}
			
			//ALPHA GRADIENTS
			for (int j = m_k; j != 2*m_k; ++j) {
				Jac(i,j) = -m_xi(i)*m_omega(j-m_k)*exp(-m_alpha(j-m_k)*m_xi(i));
			}
			
			//delta max Gradient
			Jac(i, 2*m_k) = - pow(-1.0Q, (float128)i);
		}
	
		return Jac;
	
}




void laplace::minmax_paraopt() {
	
	// =================================================================================	
	// ========= STEP 2: OPTIMIZE OMEGA AND ALPHA BY MULTIVARIATE NEWTON-RAPHSON =======
	// =================================================================================
	
	// Calculate Chebyshev norm delta_k[1,R](omega, alpha) = max | eta_k(x, omega, alpha) |
	
	float128 maxnorm = deltak();
	
	int iter = 0;
	float128 conv = 10;
	
	VectorX128 old_para(2*m_k+1);
	VectorX128 new_para(2*m_k+1);
	
	for (int i = 0; i != m_k; ++i) {
			old_para(i) = m_omega(i);
			old_para(i+m_k) = m_alpha(i);
	}
	
	old_para(2*m_k) = maxnorm;
	
	// THE SYSTEM OF EQUATION TO SOLVE IS:
	// Eta(xi, omega, alpha) - (-1)^i delta_k = 0
	// with the 2*k+1 paramters {omega}, {alpha} and delta_k
	
	LOG.os<1>("\t Multivariate Newton-Raphson...\n");
		
	while ((++iter < 10) && (conv > m_limit) ) {
		
		// FORM THE JACOBI MATRIX
		auto J = Jacobi();
		
		//std::cout << J << std::endl;
		
		// FORM THE FUNCTIONAL -f(x)
		VectorX128 bx(2*m_k+1);
		
		for (int i = 0; i != 2*m_k+1; ++i) {
			bx(i) = - Eta(m_xi(i)) + pow(-1.0Q, (float128)i)*old_para(2*m_k);
		}
		
		//std::cout << bx << std::endl;
		
		Eigen::PartialPivLU<MatrixX128> LU(J); 
		
		auto X = LU.solve(bx);
		
		float128 LUerr = (J*X-bx).norm() / bx.norm();
		
		// ADD dx to x
		
		new_para = old_para + X;
		
		// COMPUTE RMS
		
		conv = (new_para - old_para).norm();
		
		old_para = new_para;
		
		LOG.os<1>("\t Iteration: ", iter, " LU err: ", LUerr, " conv: ", conv, '\n');
		
		
		for (int i = 0; i != m_k; ++i) {
			m_omega(i) = new_para(i);
			m_alpha(i) = new_para(i+m_k);
		}
		
		
	}
	
}

}

