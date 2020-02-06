#include "math/linalg/orthogonalizer.h"
#include "math/tensor/dbcsr_conversions.hpp"
#include "math/linalg/symmetrize.h"
#include "utils/mpi_log.h"

#include <Eigen/Eigenvalues>

#include <stdexcept>

namespace math {
	
void orthgon::compute() {
	
	//auto t = symmetrize(m_tensor, "SYM");
	util::mpi_log LOG(m_plev, m_tensor->comm());
	
	auto mat = dbcsr::tensor_to_eigen(*m_tensor);
	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	
	es.compute(mat);
	
	auto eigval = es.eigenvalues();
	m_out = es.eigenvectors();
	
	LOG.os<2>("Eigenvalues: ");
	LOG.os<2>(eigval);
	
	if (es.info() != Eigen::Success) throw std::runtime_error("Eigen hermitian eigensolver failed.");
	
	// for now, we dont throw any away.
	
	for (int i = 0; i != m_out.rows(); ++i) {
		for (int j = 0; j != m_out.cols(); ++j) {
			m_out(i,j) /= sqrt(eigval(j));
		}
	}
}

dbcsr::stensor<2,double> orthgon::result(std::string name) {
	
	dbcsr::pgrid<2> grid({.comm = m_tensor->comm()});
		
	auto t_out = 
		dbcsr::eigen_to_tensor(m_out, name, grid, vec<int>{0}, vec<int>{1}, m_tensor->blk_size());
		
	m_out.resize(0,0);
		
	return t_out.get_stensor();
}
	
}
