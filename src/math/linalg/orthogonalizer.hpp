#ifndef MATH_ORTHOGONALIZER_H
#define MATH_ORTHOGONALIZER_H

#include <Eigen/Core>
#include <dbcsr_matrix.hpp>
#include <string>
#include "megalochem.hpp"

namespace megalochem {

namespace math {

class orthogonalizer {
 private:
  world m_world;

  dbcsr::shared_matrix<double> m_mat_in;
  dbcsr::shared_matrix<double> m_mat_out;
  bool m_print;

 public:
  orthogonalizer(world w, dbcsr::shared_matrix<double>& m, bool print = false) :
      m_world(w), m_mat_in(m), m_print(print){};

  void compute();

  dbcsr::smatrix<double> result()
  {
    return m_mat_out;
  }
};

}  // namespace math

}  // namespace megalochem

#endif
