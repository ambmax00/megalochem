#include "math/linalg/orthogonalizer.h"
#include "math/tensor/dbcsr_conversions.hpp"
#include "math/linalg/symmetrize.h"

#include <Eigen/Eigenvalues>

#include <stdexcept>

namespace math {
	
void orthgon::compute() {
	
	//auto t = symmetrize(m_tensor, "SYM");
	
	auto mat = dbcsr::tensor_to_eigen(m_tensor);
	
	std::cout << "MATRIX S" << std::endl;
	std::cout << mat << std::endl;
	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	
	es.compute(mat);
	
	auto eigval = es.eigenvalues();
	m_out = es.eigenvectors();
	
	std::cout << "EIGENVALUES: " << std::endl;
	std::cout << eigval << std::endl;
	
	if (es.info() != Eigen::Success) throw std::runtime_error("Eigen hermitian eigensolver failed.");
	
	// for now, we dont throw any away.
	
	for (int i = 0; i != m_out.rows(); ++i) {
		for (int j = 0; j != m_out.cols(); ++j) {
			m_out(i,j) /= sqrt(eigval(j));
		}
	}
	
	
}
	
}
