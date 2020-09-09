#ifndef INTS_GENTRAN_H
#define INTS_GENTRAN_H

#include <dbcsr_tensor.hpp>
#include <dbcsr_btensor.hpp>
#include <string>

namespace ints {

dbcsr::sbtensor<3,double> transform3(dbcsr::sbtensor<3,double>& d_xab, 
	dbcsr::shared_tensor<2,double>& ca, dbcsr::shared_tensor<2,double>& cb, 
	dbcsr::shared_pgrid<3> pgrid3, int nbatches,
	dbcsr::btype mytype, std::string name);

}//namespace ints

#endif
