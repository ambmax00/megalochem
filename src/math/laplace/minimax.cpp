#include "math/laplace/minimax.hpp"
#include <iomanip>
#include <stdexcept>

namespace megalochem {

namespace math {

void minimax::compute(int k, double ymin, double ymax)
{
  os<1>("Starting Minimax...\n");

  double R = ymax / ymin;

  os<1>("R: ", R, '\n');

  // load initial alpha and omega
  auto [omega_guess, alpha_guess] = read_guess(R, k);

  int niter = 0;

  while (niter < _max_iter_remez) {
    os<1>("Remez Iteration ", niter, '\n');

    // perform Newton-Maehly
    auto expts = newton_maehly(R, k, omega_guess, alpha_guess);

    // perform para opt
    auto [new_omegas, new_alphas] = para_opt(expts, omega_guess, alpha_guess);

    Eigen::VectorXq r_omega = new_omegas - omega_guess;
    Eigen::VectorXq r_alpha = new_alphas - alpha_guess;

    float128 norm_alpha = r_alpha.cwiseAbs().norm();
    float128 norm_omega = r_omega.cwiseAbs().norm();

    float128 eps = max(norm_alpha, norm_omega);

    os<1>("Remez Iteration ", niter, " Error ", eps, '\n');

    omega_guess = new_omegas;
    alpha_guess = new_alphas;

    if (eps < _itol_remez)
      break;

    ++niter;
  }

  if (niter == _max_iter_remez) {
    throw std::runtime_error("MINIMAX: Remez did not converge");
  }

  omega_guess /= (float128)ymin;
  alpha_guess /= (float128)ymin;

  os<1>("Final scaled weights:\n", omega_guess, '\n');
  os<1>("Final scaled exponents:\n", alpha_guess, '\n');

  _weights.resize(k);
  _exponents.resize(k);

  for (int ii = 0; ii != k; ++ii) {
    _weights[ii] = (double)omega_guess(ii);
    _exponents[ii] = (double)alpha_guess(ii);
  }
}

}  // namespace math

}  // namespace megalochem
