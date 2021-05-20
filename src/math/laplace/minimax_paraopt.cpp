#include "math/laplace/minimax.hpp"
#include <stdexcept>

namespace megalochem {
	
namespace math {

float128 minimax::eta(float128 x, Eigen::VectorXq& omegas, Eigen::VectorXq& alphas) 
{
	int k = omegas.size();
	Eigen::VectorXq vals(k);
	
	for (int ii = 0; ii != k; ++ii) {
		vals(ii) = omegas(ii) * exp(-alphas(ii) * x);
	}
	
	return kahan_summation(vals) - pow(x,-1.0);
	
} 

std::tuple<Eigen::VectorXq,Eigen::VectorXq> 
	minimax::para_opt(Eigen::VectorXq& expts,
	Eigen::VectorXq omegas, Eigen::VectorXq alphas)
{
	os<1>("Performing paramter optimization.\n");
	
	int k = omegas.size();
	
	os<1>("Initializing trial vector f\n");
	
	Eigen::VectorXq f(2*k);
	
	
	for (int ii = 0; ii != 2*k; ++ii) {
		f(ii) = eta(expts(ii),omegas,alphas) + eta(expts(ii+1),omegas,alphas);
	}
		
	os<1>("Starting multidimensional Newton-Raphson\n");
	
	int niter = 0;
	
	while (niter < _max_iter_newton) {
		
		Eigen::VectorXq old_omegas = omegas;
		Eigen::VectorXq old_alphas = alphas;
		
		os<1>("---- Iteration ", niter, '\n');
		os<1>("---- Forming Jacobi Matrix\n");
		
		Eigen::MatrixXq jacobi(2*k,2*k);
		
		for (int icol = 0; icol != k; ++icol) {
			for (int irow = 0; irow != 2*k; ++irow) {
				
				float128 a = alphas(icol);
				float128 w = omegas(icol);
				
				float128 x0 = expts(irow);
				float128 x1 = expts(irow+1);
				
				float128 exp0 = exp(-a * x0);
				float128 exp1 = exp(-a * x1);
				
				// df/dw_i
				jacobi(irow,icol) = exp0 + exp1;
				
				// df_da_i
				jacobi(irow,icol+k) = - x0 * w * exp0 - x1 * w * exp1;
			}
		}
		
		os<2>("---- Jacobi Matrix:\n");
		os<2>(jacobi, '\n');
		
		os<1>("---- Solving set of equations\n");
		Eigen::VectorXq delta_y = jacobi.householderQr().solve(f);
		
		os<1>("---- Correction vector:\n", delta_y, '\n');
		
		os<1>("---- Done!\n");
		for (int ii = 0; ii != k; ++ii) {
			omegas(ii) -= delta_y(ii);
			alphas(ii) -= delta_y(ii + k);
		}
		
		os<1>("---- Computing residuals\n");
		Eigen::VectorXq r_omega = omegas - old_omegas;
		Eigen::VectorXq r_alpha = alphas - old_alphas;
		
		float128 norm_alpha = r_alpha.cwiseAbs().norm();
		float128 norm_omega = r_omega.cwiseAbs().norm();
		
		float128 eps = max(norm_alpha, norm_omega);
		
		os<1>("---- Norm: ", eps, '\n');
		os<1>("---- New omegas/alphas:\n");
		os<2>(omegas, '\n', alphas, '\n');
	
		if (eps < _itol_newton) break;
	
		for (int ii = 0; ii != 2*k; ++ii) {
			f(ii) = eta(expts(ii),omegas,alphas) + eta(expts(ii+1),omegas,alphas);
		}
		
		++niter;
				
	}
	
	if (niter == _max_iter_newton) {
		throw std::runtime_error("MINIMAX: Multi-dimensional Newton did not converge");
	}
	
	return std::make_tuple(omegas,alphas);
	
}
		
} // namespace math

} // namespace megalochem
	
