#ifndef MATH_SCALE_H
#define MATH_SCALE_H

#include "math/tensor/dbcsr.hpp"
#include <vector>

namespace math {

// do t[ij] = t[ij] * v[j]
void scale(dbcsr::tensor<2>& t, std::vector<double>& v);

}

#endif
