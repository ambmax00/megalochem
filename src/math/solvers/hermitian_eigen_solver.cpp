#include "math/solvers/hermitian_eigen_solver.hpp"
#include <dbcsr_matrix_ops.hpp>
#include "extern/scalapack.hpp"

#include <dbcsr_conversions.hpp>
#include <cmath>

namespace megalochem {

namespace math {
	
void hermitian_eigen_solver::compute() {
	
	int lwork;
	
	auto sgrid = m_world.scalapack_grid();
	
	int n = m_mat_in->nfullrows_total();
	int nb = scalapack::global::block_size;
	int nprow = m_cart.dims()[0];
	int npcol = m_cart.dims()[1];
	
	LOG.os<>("Running SCALAPACK pdsyev calculation\n");
	
	if (m_jobz == 'N') {
		lwork = 5*n + nb * ((n - 1)/nprow + nb) + 1;
	} else {
		int A = (n/(nb*nprow)) + (n/(nb*npcol));
		lwork = 5*n + std::max(2*n,(3+A)*nb*nb) + nb*((n-1)/(nb*nprow*npcol)+1)*n + 1;
	}
	
	LOG.os<>("-- Allocating work space of size ", lwork, '\n');
	double* work = new double[lwork];
	
	// convert array
	
	LOG.os<>("-- Converting input matrix to SCALAPACK format.\n");
		
	int ori_proc = m_mat_in->proc(0,0);
	int ori_coord[2];
	
	if (m_world.rank() == ori_proc) {
		ori_coord[0] = sgrid.myprow();
		ori_coord[1] = sgrid.mypcol();
	}
	
	MPI_Bcast(&ori_coord[0],2,MPI_INT,ori_proc,m_world.comm());
		
	scalapack::distmat<double> sca_mat_in = dbcsr::matrix_to_scalapack(m_mat_in, 
		sgrid, nb, nb, ori_coord[0], ori_coord[1]);
	
	std::optional<scalapack::distmat<double>> sca_eigvec_opt;
	
	if (m_jobz == 'V') {
		sca_eigvec_opt.emplace(sgrid, n, n, nb, nb, ori_coord[0], ori_coord[1]);
	} else {
		sca_eigvec_opt = std::nullopt;
	}
	
	m_eigval.resize(n);
	
	double* a_ptr = sca_mat_in.data();
	double* z_ptr = (sca_eigvec_opt) ? sca_eigvec_opt->data() : nullptr;
	auto sca_desc = sca_mat_in.desc();
	int info;
	
	LOG.os<>("-- Starting pdsyev...\n");
	c_pdsyev(m_jobz,'U',n,a_ptr,0,0,sca_desc.data(),m_eigval.data(),z_ptr,
		0,0,sca_desc.data(),work,lwork,&info);
	LOG.os<>("-- Subroutine pdsyev finished with exit code ", info, '\n');
	
	delete [] work; // if only it was this easy to get rid of your work
	sca_mat_in.release();
	
	if (sca_eigvec_opt) {
		
		// convert to dbcsr matrix
		std::vector<int> rowblksizes = (m_rowblksizes_out) 
			? *m_rowblksizes_out : m_mat_in->row_blk_sizes();
			
		std::vector<int> colblksizes = (m_colblksizes_out) 
			? *m_colblksizes_out : m_mat_in->col_blk_sizes();
			
		//sca_eigvec_opt->print();
		
		m_eigvec = dbcsr::scalapack_to_matrix(*sca_eigvec_opt, 
			m_cart, "eigenvectors", rowblksizes, colblksizes); 
				
		sca_eigvec_opt->release();
		
	}
		
	return;
}

smatrix hermitian_eigen_solver::inverse() {
	
	auto eigvec_copy = dbcsr::matrix<>::copy(*m_eigvec).build();
	
	//dbcsr::print(eigvec_copy);
	
	std::vector<double> eigval_copy = m_eigval;
	
	std::for_each(eigval_copy.begin(),eigval_copy.end(),
		[](double& d) { 
			d = (fabs(d) < 1e-12) ? 0 : 1.0/d; 
		});
	
	eigvec_copy->scale(eigval_copy, "right");
	
	//dbcsr::print(eigvec_copy);
	
	auto inv = dbcsr::matrix<>::create_template(*m_mat_in)
		.name(m_mat_in->name() + "^-1")
		.matrix_type(dbcsr::type::symmetric)
		.build();
	
	dbcsr::multiply('N', 'T', 1.0, *eigvec_copy, *m_eigvec, 0.0, *inv)
		.perform(); 
	
	//eigvec_copy.release();
	
	return inv;
	
}

smatrix hermitian_eigen_solver::inverse_sqrt() {
	
	auto eigvec_copy = dbcsr::matrix<>::copy(*m_eigvec)
		.build();
	
	//dbcsr::print(eigvec_copy);
	
	std::vector<double> eigval_copy = m_eigval;
	
	std::for_each(eigval_copy.begin(),eigval_copy.end(),[](double& d) { d = 1.0/sqrt(d); });
	
	eigvec_copy->scale(eigval_copy, "right");
	
	//dbcsr::print(eigvec_copy);
	
	auto inv = dbcsr::matrix<>::create_template(*m_mat_in)
		.name(m_mat_in->name() + "^-1/2")
		.matrix_type(dbcsr::type::symmetric)
		.build();
	
	dbcsr::multiply('N', 'T', 1.0, *eigvec_copy, *m_eigvec, 0.0, *inv)
		.perform(); 
	
	//eigvec_copy.release();
	
	return inv;
	
}
	
} //end namepace

} // end megalochem
	
	
