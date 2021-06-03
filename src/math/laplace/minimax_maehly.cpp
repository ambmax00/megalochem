#include <stdexcept>
#include "math/laplace/minimax.hpp"

namespace megalochem {

namespace math {

float128 minimax::kahan_summation(Eigen::VectorXq& array)
{
  float128 sum = 0.0;
  float128 c = 0.0;

  for (int ii = 0; ii != array.size(); ++ii) {
    float128 y = array(ii) - c;
    float128 t = sum + y;

    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

float128 minimax::dEta(
    float128 x, Eigen::VectorXq& omegas, Eigen::VectorXq& alphas)
{
  int k = omegas.size();

  Eigen::VectorXq vals(k);

  for (int ilap = 0; ilap != k; ++ilap) {
    vals(ilap) = alphas(ilap) * omegas(ilap) * exp(-alphas(ilap) * x);
  }

  return -kahan_summation(vals) + pow(x, -2.0);
}

float128 minimax::ddEta(
    float128 x, Eigen::VectorXq& omegas, Eigen::VectorXq& alphas)
{
  int k = omegas.size();

  Eigen::VectorXq vals(k);

  for (int ilap = 0; ilap != k; ++ilap) {
    vals(ilap) =
        alphas(ilap) * alphas(ilap) * omegas(ilap) * exp(-alphas(ilap) * x);
  }

  return kahan_summation(vals) - float128(2.0) * pow(x, -3.0);
}

float128 minimax::newton(float128 x0, newton_function& update)
{
  float128 xi = x0;
  os<1>("---- Performing Newton iterations.\n");

  os<1>("---- Guess: ", x0, '\n');

  int niter = 0;

  while (niter < _max_iter_newton) {
    float128 delta_x = update(xi);
    xi -= delta_x;
    os<1>("---- Iteration ", niter, "\t", xi, "\t", delta_x, '\n');
    if (fabs(delta_x) < _itol_newton)
      break;

    ++niter;
  }

  if (niter == _max_iter_newton) {
    throw std::runtime_error("MINIMAX: Newton did not converge");
  }

  return xi;
}

Eigen::VectorXq minimax::newton_maehly(
    double R, int k, Eigen::VectorXq& omegas, Eigen::VectorXq& alphas)
{
  Eigen::VectorXq expts(2 * k + 1);

  expts(0) = 1.0;

  float128 xval0 = expts(0);

  os<1>("Determine extremum points...\n");

  for (int ixpt = 1; ixpt < 2 * k; ++ixpt) {
    float128 xval = xval0;

    os<1>("--- Finding root ", ixpt, '\n');

    newton_function nm_func = [this, &expts, &ixpt, &omegas,
                               &alphas](float128 x) {
      auto de = dEta(x, omegas, alphas);
      auto dde = ddEta(x, omegas, alphas);

      float128 sum = 0.0;

      for (int jj = 0; jj < ixpt; ++jj) { sum += pow(x - expts(jj), -1.0); }

      return de / (dde - de * sum);
    };

    newton_function nr_func = [this, &omegas, &alphas](float128 x) {
      auto de = dEta(x, omegas, alphas);
      auto dde = ddEta(x, omegas, alphas);

      return de / dde;
    };

    os<1>("--- Performing Newton Maehly\n");

    xval = newton(xval, nm_func);

    os<1>("--- Performing Newton Raphson\n");

    xval = newton(xval, nr_func);

    expts(ixpt) = xval;
    xval0 = xval * (float128(1.0) + _delta);
  }

  expts(2 * k) = (float128)R;

  os<1>("New EPs:\n", expts, '\n');

  return expts;
}

}  // namespace math

}  // namespace megalochem
