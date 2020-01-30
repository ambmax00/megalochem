#ifndef MATH_LINALG_SYMMETRIZE_H
#define MATH_LINALG_SYMMETRIZE_H

#include <string>

#include "math/tensor/dbcsr.hpp"

namespace math {
	
dbcsr::tensor<2> symmetrize(dbcsr::tensor<2>& unsym_tensor, std::string name);


}

#endif
