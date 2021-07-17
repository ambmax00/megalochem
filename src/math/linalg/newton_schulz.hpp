#ifndef MEGALOCHEM_MATH_NEWTON_SCHULZ_HPP
#define MEGALOCHEM_MATH_NEWTON_SCHULZ_HPP

#include <stdexcept>
#include <tuple>
#include "megalochem.hpp"
#include "utils/mpi_log.hpp"
#include <dbcsr_matrix_ops.hpp>

namespace megalochem {
namespace math {
  
class newton_schulz {
 private:
  
  world m_world;
  util::mpi_log LOG;
  dbcsr::shared_matrix<double> m_mat, m_mat_invsqrt, m_mat_sqrt;
  
  double m_conv = 1e-12;
  int m_max_iter = 20;
  int m_blksize = 8;
  double m_eps_div = 100.0;
  double m_filter_eps = dbcsr::global::filter_eps;
  double m_filter_eps_iter = m_filter_eps/m_eps_div;
  
  std::tuple<double,double> get_extremum_eigenvalues();
  
  void fill_identity(dbcsr::matrix<double>& mat);
  
  double mapping(double val);
  
  dbcsr::shared_matrix<double> taylor(dbcsr::matrix<double>& Xk);
  
 public:
 
  newton_schulz(world w, dbcsr::shared_matrix<double> mat, int print) :
    m_world(w), m_mat(mat), LOG(w.comm(), print) 
  {
    
    if (mat->matrix_type() != dbcsr::type::symmetric) {
      throw std::runtime_error("Newton Schulz will not work for non-symmetric matrices!");
    }
  }
  
  void compute();
    
  dbcsr::shared_matrix<double> sqrt() {
    return m_mat_sqrt;
  }
  
  dbcsr::shared_matrix<double> inverse_sqrt() {
    return m_mat_invsqrt;
  }
  
  dbcsr::shared_matrix<double> compute_inverse();
  
};
  
  
} // namespace math
} // namespace megalochem

#endif
