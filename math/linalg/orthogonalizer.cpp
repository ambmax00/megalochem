#include "math/linalg/orthogonalizer.h"
#include "math/tensor/dbcsr_conversions.hpp"
#include "math/linalg/symmetrize.h"
#include "utils/mpi_log.h"

#include <Eigen/Eigenvalues>

#include <stdexcept>

namespace math {
	
void orthgon::compute() {
	
	//auto t = symmetrize(m_tensor, "SYM");
	util::mpi_log LOG(m_tensor->comm(), m_plev);
	
	//dbcsr::print(*m_tensor);
	
	auto mat = dbcsr::tensor_to_eigen(*m_tensor);
	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	
	es.compute(mat);
	
	//std::cout << "MAT" << std::endl;
	//std::cout << mat << std::endl;
	
	auto eigval = es.eigenvalues();
	/*
	m_out = es.eigenvectors();
	
	std::cout << m_out << std::endl;
	*/
	LOG.os<2>("Eigenvalues: ");
	LOG.os<2>(eigval);
	
	if (es.info() != Eigen::Success) throw std::runtime_error("Eigen hermitian eigensolver failed.");
	
	// for now, we dont throw any away.
	
	//for (int i = 0; i != m_out.rows(); ++i) {
	//	for (int j = 0; j != m_out.cols(); ++j) {
	//		m_out(i,j) /= sqrt(eigval(j));
	//	}
	//}
	
	/*		auto X = m_out;
	auto S = mat;
	
	auto XSX = X.transpose() * S * X;
	
	std::cout << XSX << std::endl;
	exit(0);*/
	
	auto eigvec = es.eigenvectors();
	
	for (int i = 0; i != eigval.size(); ++i) {
		eigval(i) = 1/sqrt(eigval(i));
	}
	
	m_out = eigvec * eigval.asDiagonal() * eigvec.transpose();

}


dbcsr::stensor<2,double> orthgon::result(std::string name) {
	
	dbcsr::pgrid<2> grid({.comm = m_tensor->comm()});
		
	auto t_out = 
		dbcsr::eigen_to_tensor(m_out, name, grid, vec<int>{0}, vec<int>{1}, m_tensor->blk_size());
		
	m_out.resize(0,0);
		
	return t_out.get_stensor();
}
	
}
