#ifndef INTS_GENTRAN_H
#define INTS_GENTRAN_H

#include "math/tensor/dbcsr.hpp"
#include <string>

namespace ints {

struct tranp3 {
	required<dbcsr::tensor<3>,ref>	t_in;
	required<dbcsr::tensor<2>,ref>	c_1;
	required<dbcsr::tensor<2>,ref>	c_2;
	required<std::string,val>		name;
};
dbcsr::stensor<3> transform3(tranp3&& p);

}//namespace ints

#endif
