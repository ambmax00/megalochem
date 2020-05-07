#include "math/solvers/hermitian_eigen_solver.h"
#include "extern/scalapack.h"
#include <dbcsr_conversions.hpp>
#include <cmath>

namespace math {

void hermitian_eigen_solver::compute() {
	
	int lwork;
	
	int n = m_mat_in->nfullrows_total();
	int nb = m_blksize;
	int nprow = m_world.dims()[0];
	int npcol = m_world.dims()[1];
	
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
	
	scalapack::grid mygrid(nprow,npcol,'R');
	
	scalapack::distmat<double> sca_mat_in = dbcsr::matrix_to_scalapack(*m_mat_in, 
		m_mat_in->name() + "_scalapack", mygrid, m_blksize, m_blksize);
	
	std::optional<scalapack::distmat<double>> sca_eigvec_opt = std::nullopt;
	
	if (m_jobz == 'V') {
		sca_eigvec_opt = std::make_optional<scalapack::distmat<double>>(
			scalapack::distmat<double>(mygrid, n, n, m_blksize, m_blksize));
	}
	
	m_eigval.resize(n);
	
	double* a_ptr = sca_mat_in.data();
	double* z_ptr = (sca_eigvec_opt) ? sca_eigvec_opt->data() : nullptr;
	int* desc_ptr = sca_mat_in.desc();
	int info;
	
	LOG.os<>("-- Starting pdsyev...\n");
	c_pdsyev(m_jobz,'U',n,a_ptr,0,0,desc_ptr,m_eigval.data(),z_ptr,
		0,0,desc_ptr,work,lwork,&info);
	LOG.os<>("-- Subroutine pdsyev finished with exit code ", info, '\n');
	
	if (sca_eigvec_opt) {
		
		sca_eigvec_opt->print();
		
	}
		
	return;
}

} //end namepace
	
	
