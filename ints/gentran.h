#ifndef INTS_GENTRAN_H
#define INTS_GENTRAN_H

#include "tensor/dbcsr.hpp"
#include <string>

namespace ints {

dbcsr::stensor<3> transform3(dbcsr::stensor<3>& d_xab, dbcsr::stensor<2>& ca, dbcsr::stensor<2>& cb, std::string name);

}//namespace ints

#endif
