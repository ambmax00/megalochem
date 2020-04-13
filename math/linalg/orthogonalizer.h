#ifndef MATH_ORTHOGONALIZER_H
#define MATH_ORTHOGONALIZER_H

#include "tensor/dbcsr.hpp"
#include <string>
#include <Eigen/Core>

namespace math {
	
class orthgon {
private:

	dbcsr::stensor<2,double>& m_tensor;
	Eigen::MatrixXd m_out;
	int m_plev;

public:

	orthgon(dbcsr::stensor<2,double>& t) : m_tensor(t), m_plev(0) {};
	
	void compute();

	dbcsr::stensor<2,double> result(std::string name);
	
};

} // end namespace

#endif
