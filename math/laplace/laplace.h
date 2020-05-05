#ifndef MATH_LAPLACE_H
#define MATH_LAPLACE_H

#include <vector>
#include <limits>
#include <Eigen/Core>
#include "utils/mpi_log.h"
#include <boost/multiprecision/float128.hpp> 

namespace boost{ 
namespace multiprecision{

class float128_backend;

typedef number<float128_backend, et_off> float128;

}}

using namespace boost::multiprecision;

typedef Eigen::Matrix<float128,Eigen::Dynamic,Eigen::Dynamic> MatrixX128;
typedef Eigen::Matrix<float128,Eigen::Dynamic, 1> VectorX128;

namespace math {
	
class laplace {
	
private:

	util::mpi_log LOG;

	int m_k;
	double m_emin, m_ehomo, m_elumo, m_emax, m_R;
	double m_limit;
	VectorX128 m_omega, m_alpha, m_xi;
	
	std::vector<double> m_omega_db;
	std::vector<double> m_alpha_db;
	
	void minmax_read();
	void minmax_root();
	void minmax_newton();
	void minmax_paraopt();
	
	void NewtonKahan(int i, int method);
	MatrixX128 Jacobi();
	
	float128 Eta(float128 x);
	float128 dEta(float128 x);
	float128 ddEta(float128 x);
	float128 deltak();
	

public:

	laplace(int t_k, double t_emin, double t_ehomo, 
		double t_elumo, double t_emax) :  
		m_k(t_k), m_emin(t_emin), m_ehomo(t_ehomo),
		m_elumo(t_elumo), m_emax(t_emax),
		m_omega(t_k), m_alpha(t_k), m_xi(2*t_k + 1),
		m_omega_db(t_k), m_alpha_db(t_k),
		LOG(MPI_COMM_WORLD, 1), m_limit(std::numeric_limits<double>::epsilon()) {};
	
	std::vector<double> omega() {
		return m_omega_db;
	}
	
	std::vector<double> alpha() {
		return m_alpha_db;
	}
	
	~laplace() {}
	
	void compute();
	
	
};
		
} // end namespace

#endif
