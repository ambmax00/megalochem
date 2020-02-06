#ifndef MATH_LINALG_INVERSE_H
#define MATH_LINALG_INVERSE_H
#include "math/tensor/dbcsr.hpp"
#include <string>


namespace math {
	
dbcsr::stensor<2> eigen_inverse(dbcsr::stensor<2>& t_in, std::string name);
dbcsr::stensor<2> eigen_sqrt_inverse(dbcsr::stensor<2>& t_in, std::string name);

} // end namespace

#endif
