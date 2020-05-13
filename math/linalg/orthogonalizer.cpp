#include "math/linalg/orthogonalizer.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include <dbcsr_matrix_ops.hpp>
#include "utils/mpi_log.h"

#include <stdexcept>

namespace math {
	
int threshold = 1e-6;
	
void orthogonalizer::compute() {
	
	util::mpi_log LOG(m_mat_in->get_world().comm(), m_plev);
	
	hermitian_eigen_solver solver(m_mat_in, 'V');
	
	solver.compute();
	
	auto eigvecs = solver.eigvecs();
	auto eigvals = solver.eigvals();
	
	std::for_each(eigvals.begin(),eigvals.end(),[](double& d) { d = (d < threshold) ? 0 : 1/sqrt(d); });
	
	dbcsr::mat_d eigvec_copy = dbcsr::mat_d::copy<double>(*eigvecs);
	
	eigvec_copy.scale(eigvals, "right");
	
	dbcsr::mat_d out = dbcsr::mat_d::create_template(*m_mat_in).name(m_mat_in->name() + " orthogonalized")
		.type(dbcsr_type_symmetric);
		
	dbcsr::multiply('N', 'T', eigvec_copy, *eigvecs, out).perform(); 
		
	m_mat_out = out.get_smatrix();
	
}
	
}
