#ifndef MATH_SCALE_H
#define MATH_SCALE_H

#include "tensor/dbcsr.hpp"
#include <vector>

namespace math {

// do t[ij] = t[ij] * v[j]
void scale(dbcsr::tensor<2>& t_in, std::vector<double>& v_in, 
	std::optional<std::vector<int>> bounds = std::nullopt);

}

#endif
