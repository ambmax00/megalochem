#ifndef MATH_HERMITIAN_EIGEN_SOLVER_H
#define MATH_HERMITIAN_EIGEN_SOLVER_H

#include <dbcsr_matrix.hpp>
#include "extern/scalapack.hpp"
#include "megalochem.hpp"
#include "utils/mpi_log.hpp"

namespace megalochem {

namespace math {

using smatrix = dbcsr::shared_matrix<double>;
using matrix = dbcsr::matrix<double>;

class hermitian_eigen_solver {
 private:
  world m_world;

  dbcsr::shared_matrix<double> m_mat_in;
  dbcsr::shared_matrix<double> m_eigvec;
  std::vector<double> m_eigval;
  dbcsr::cart m_cart;
  util::mpi_log LOG;

  char m_jobz;

  std::optional<vec<int>> m_rowblksizes_out =
      std::nullopt;  // block sizes for eigenvector matrix
  std::optional<vec<int>> m_colblksizes_out = std::nullopt;

 public:
  inline hermitian_eigen_solver& eigvec_rowblks(vec<int>& blksizes)
  {
    m_rowblksizes_out = std::make_optional<vec<int>>(blksizes);
    return *this;
  }

  inline hermitian_eigen_solver& eigvec_colblks(vec<int>& blksizes)
  {
    m_colblksizes_out = std::make_optional<vec<int>>(blksizes);
    return *this;
  }

  hermitian_eigen_solver(
      world w,
      dbcsr::shared_matrix<double>& mat_in,
      char jobz,
      bool print = false) :
      m_world(w),
      m_mat_in(mat_in), m_cart(mat_in->get_cart()),
      LOG(m_cart.comm(), (print) ? 0 : -1), m_jobz(jobz)
  {
  }

  void compute();

  vec<double>& eigvals()
  {
    return m_eigval;
  }

  dbcsr::shared_matrix<double> eigvecs()
  {
    return m_eigvec;
  }

  dbcsr::shared_matrix<double> inverse();

  dbcsr::shared_matrix<double> inverse_sqrt();
};

}  // namespace math

}  // namespace megalochem

#endif
