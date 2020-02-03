#ifndef MATH_LINALG_INVERSE_H
#define MATH_LINALG_INVERSE_H
#include "math/tensor/dbcsr.hpp"
#include <string>


namespace math {
	
dbcsr::tensor<2> eigen_inverse(dbcsr::tensor<2>& t_in, std::string name);
dbcsr::tensor<2> eigen_sqrt_inverse(dbcsr::tensor<2>& t_in, std::string name);

} // end namespace

#endif
