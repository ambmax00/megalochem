#ifndef MATH_ORTHOGONALIZER_H
#define MATH_ORTHOGONALIZER_H

#include "math/tensor/dbcsr.hpp"
#include <Eigen/Core>

namespace math {
	
class orthgon {
private:

	dbcsr::tensor<2,double>& m_tensor;
	Eigen::MatrixXd m_out;

public:

	orthgon(dbcsr::tensor<2,double>& t) : m_tensor(t) {};
	
	void compute();

	Eigen::MatrixXd result() {
		return m_out;
	};
	
};

} // end namespace

#endif
