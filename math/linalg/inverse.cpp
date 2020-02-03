#include "math/linalg/inverse.h"
#include "math/tensor/dbcsr_conversions.hpp"
#include <Eigen/Eigenvalues>

namespace math {
	
dbcsr::tensor<2> eigen_inverse(dbcsr::tensor<2>& t_in, std::string name) {
	
	auto M = dbcsr::tensor_to_eigen(t_in); 
		
	dbcsr::pgrid<2> grid({.comm = t_in.comm()});
	
	Eigen::MatrixXd inv = M.inverse();
	
	//auto I = M * M.inverse();
	//std::cout << "IDENTITY" << std::endl;
	//std::cout << I << std::endl;
	
	auto t_out = dbcsr::eigen_to_tensor(inv, name, grid, vec<int>{0}, vec<int>{1}, t_in.blk_size());  
	
	grid.destroy();
	
	return t_out;
	
}

dbcsr::tensor<2> eigen_sqrt_inverse(dbcsr::tensor<2>& t_in, std::string name) {
	
	auto M = dbcsr::tensor_to_eigen(t_in); 
		
	dbcsr::pgrid<2> grid({.comm = t_in.comm()});
	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> SAES(M);
	
	Eigen::MatrixXd inv = SAES.operatorInverseSqrt();
	
	auto t_out = dbcsr::eigen_to_tensor(inv, name, grid, vec<int>{0}, vec<int>{1}, t_in.blk_size());
	
	grid.destroy();
	
	return t_out;
	
}

} // end namespace
