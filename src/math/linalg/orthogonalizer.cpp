#include "math/linalg/orthogonalizer.hpp"
#include "math/solvers/hermitian_eigen_solver.hpp"
#include <dbcsr_matrix_ops.hpp>
#include "utils/mpi_log.hpp"

#include <stdexcept>

namespace math {
	
int threshold = 1e-6;
	
void orthogonalizer::compute() {
	
	util::mpi_log LOG(m_mat_in->get_world().comm(), (m_print) ? 0 : -1);
	
	hermitian_eigen_solver solver(m_mat_in, 'V', m_print);
	
	solver.compute();
	
	auto eigvecs = solver.eigvecs();
	auto eigvals = solver.eigvals();
	
	/*std::cout << "EIGVALS: " << std::endl;
	for (auto d : eigvals) {
		std::cout << d << std::endl;
	}*/
	
	std::for_each(eigvals.begin(),eigvals.end(),[](double& d) { d = (d < threshold) ? 0 : 1/sqrt(d); });
	
	auto eigvec_copy = dbcsr::matrix<>::copy(*eigvecs)
		.build();
	
	eigvec_copy->scale(eigvals, "right");
	
	m_mat_out = dbcsr::matrix<>::create_template(*m_mat_in)
		.name(m_mat_in->name() + " orthogonalized")
		.matrix_type(dbcsr::type::symmetric)
		.build();
		
	dbcsr::multiply('N', 'T', *eigvec_copy, *eigvecs, *m_mat_out).perform(); 
			
}
	
}
