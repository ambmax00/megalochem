#ifndef MATH_MINIMAX_HPP
#define MATH_MINIMAX_HPP

#include <functional>
#include <tuple>
#include <vector>
#include "math/laplace/quadmath.hpp"

namespace megalochem {

namespace math {

class minimax {
 private:
  // ==== member variables ====
  int _print_level;

  float128 _delta = 0.0001;
  float128 _itol_newton = 1e-16;
  float128 _itol_remez = 1e-16;

  int _max_iter_newton = 50;
  int _max_iter_remez = 50;

  std::vector<double> _weights;
  std::vector<double> _exponents;

  // ==== other ====

  using newton_function = std::function<float128(float128)>;

  // ==== main functions ======

  std::tuple<Eigen::VectorXq, Eigen::VectorXq> read_guess(double R, int k);

  float128 eta(float128 x, Eigen::VectorXq& omegas, Eigen::VectorXq& alphas);

  float128 dEta(float128 x, Eigen::VectorXq& omegas, Eigen::VectorXq& alphas);

  float128 ddEta(float128 x, Eigen::VectorXq& omegas, Eigen::VectorXq& alphas);

  Eigen::VectorXq newton_maehly(
      double R, int k, Eigen::VectorXq& omega, Eigen::VectorXq& alpha);

  float128 newton(float128 x0, newton_function& update);

  float128 kahan_summation(Eigen::VectorXq& array);

  std::tuple<Eigen::VectorXq, Eigen::VectorXq> para_opt(
      Eigen::VectorXq& expts, Eigen::VectorXq omegas, Eigen::VectorXq alphas);

  /*float128 newton_maehly_step(Eigen::VectorXq expts,
          Eigen::VectorXq omegas, Eigen::VectorXq alphas, int i);

  float128 eta(float128 x, Eigen::VectorXq omegas,
          Eigen::VectorXq alphas);

  float128 deta(float128 x, Eigen::VectorXq omegas,
          Eigen::VectorXq alphas);

  float128 d2eta(float128 x, Eigen::VectorXq omegas,
          Eigen::VectorXq alphas);*/

  // ==== utility functions ======

  void print_()
  {
    std::cout << std::flush;
  }

  template <typename T, typename... Args>
  void print_(T in, Args... args)
  {
    std::cout << in;
    print_(args...);
  }

  template <int nprint = 0, typename T, typename... Args>
  void os(T in, Args... args)
  {
    if (nprint <= _print_level) {
      print_(in, args...);
    }
  }

 public:
  minimax(int print_level = 0) : _print_level(print_level)
  {
  }

  void compute(int k, double ymin, double ymax);

  std::vector<double> weights()
  {
    return _weights;
  }

  std::vector<double> exponents()
  {
    return _exponents;
  }
};

}  // namespace math

}  // namespace megalochem

#endif
