#include "math/linalg/SVD.h"
#include <dbcsr_conversions.hpp>

namespace math {

void SVD::compute() {
	
	auto wrd = m_mat_in->get_world();
	int n = m_mat_in->nfullrows_total();
	int m = m_mat_in->nfullcols_total();
	int nb = scalapack::global::block_size;
	int nprow = wrd.dims()[0];
	int npcol = wrd.dims()[1];
	int size = std::min(m,n);
	auto& _grid = scalapack::global_grid;
	
	LOG.os<>("Running SCALAPACK pdgesvd calculation\n");
	
	// convert array
	
	LOG.os<>("-- Converting input matrix to SCALAPACK format.\n");
		
	int ori_proc = m_mat_in->proc(0,0);
	int ori_coord[2];
	
	if (wrd.rank() == ori_proc) {
		ori_coord[0] = _grid.myprow();
		ori_coord[1] = _grid.mypcol();
	}
	
	MPI_Bcast(&ori_coord[0],2,MPI_INT,ori_proc,wrd.comm());
		
	auto sca_mat_in = std::make_shared<scalapack::distmat<double>>(
		dbcsr::matrix_to_scalapack(m_mat_in, m_mat_in->name() + "_scalapack", 
		nb, nb, ori_coord[0], ori_coord[1])
	);
	
	LOG.os<>("-- Setting up other arrays.\n");
	
	m_U = std::make_shared<scalapack::distmat<double>>(
		m,size,nb,nb,ori_coord[0],ori_coord[1]);
		
	m_Vt = std::make_shared<scalapack::distmat<double>>(
		size,n,nb,nb,ori_coord[0],ori_coord[1]);
		
	m_s = std::make_shared<std::vector<double>>(size,0.0);
	
	LOG.os<>("-- Computing size of work space.");
	
	double wsize = 0;
	int info = 0;
	
	auto a = sca_mat_in->data();
	auto u = m_U->data();
	auto vt = m_Vt->data();
	auto s = m_s->data();
	
	auto desca = sca_mat_in->desc().data();
	auto descu = m_U->desc().data();
	auto descvt = m_Vt->desc().data();
	
	c_pdgesvd(m_jobu, m_jobvt, m, n, a, 0, 0, desca, s, u, 0, 0,
		descu, vt, 0, 0, descvt, &wsize, -1, nullptr, &info);
	
	int lwork = (int)wsize;
	double* work = new double[lwork];
	
	LOG.os<>("-- Computing SVD.\n");
	
	c_pdgesvd(m_jobu, m_jobvt, m, n, a, 0, 0, desca, s, u, 0, 0,
		descu, vt, 0, 0, descvt, work, lwork, nullptr, &info);
	
	LOG.os<>("-- Exited with info = ", info, '\n');

	delete [] work;
	
}
	
} // end namespace
