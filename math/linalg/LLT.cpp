#include "math/linalg/LLT.h"
#include <dbcsr_conversions.hpp>

namespace math {

void LLT::compute() {
	
	auto wrd = m_mat_in->get_world();
	int n = m_mat_in->nfullrows_total();
	int nb = scalapack::global::block_size;
	int nprow = wrd.dims()[0];
	int npcol = wrd.dims()[1];
	auto& _grid = scalapack::global_grid;
	
	LOG.os<>("Running SCALAPACK pdpotrf calculation\n");
	
	// convert array
	
	LOG.os<>("-- Converting input matrix to SCALAPACK format.\n");
		
	int ori_proc = m_mat_in->proc(0,0);
	int ori_coord[2];
	
	if (wrd.rank() == ori_proc) {
		ori_coord[0] = _grid.myprow();
		ori_coord[1] = _grid.mypcol();
	}
	
	MPI_Bcast(&ori_coord[0],2,MPI_INT,ori_proc,m_mat_in->get_world().comm());
		
	m_L = std::make_shared<scalapack::distmat<double>>(
		dbcsr::matrix_to_scalapack(*m_mat_in, m_mat_in->name() + "_scalapack", 
		nb, nb, ori_coord[0], ori_coord[1])
	);
		
	LOG.os<>("-- Computing cholesky decomposition.\n");
	
	//m_L->print();
	
	int info;
	c_pdpotrf('L', n, m_L->data(), 0, 0, m_L->desc().data(), &info); 
	
	LOG.os<>("-- Exited with info = ", info, '\n');

	//m_L->print();
	
}

dbcsr::smat_d LLT::L(vec<int> blksizes) {
	
	auto w = m_mat_in->get_world();
	dbcsr::smat_d out = std::make_shared<dbcsr::mat_d>(
		dbcsr::scalapack_to_matrix(*m_L, "Cholesky decomposition of "+m_mat_in->name(),
			w,blksizes,blksizes,"lowtriang")
	);
	
	out->filter();
	return out;
	
}

void LLT::compute_L_inverse() {
	
	if (!m_L_inv) {
		
		LOG.os<>("Computing sqrt inverse of matrix.\n");
		
		// copy L
		int n = m_L->nrowstot();
		int nb = scalapack::global::block_size;
		
		m_L_inv = std::make_shared<scalapack::distmat<double>>(
			n, n, nb, nb, m_L->rsrc(), m_L->csrc());
		
		LOG.os<>("Copying L.\n");
		// sub( C ) := beta*sub( C ) + alpha*op( sub( A ) )
		c_pdgeadd('N', n, n, 1.0, m_L->data(), 0, 0, m_L->desc().data(), 0.0, 
			m_L_inv->data(), 0, 0, m_L_inv->desc().data());
			
		// triangular inverse
		int info = 0;
		LOG.os<>("Starting pdtrtri.\n");
		c_pdtrtri('L', 'N', n, m_L_inv->data(), 0, 0, m_L_inv->desc().data(), &info);
		LOG.os<>("Routine pdtrtri exited with ", info, '\n');
		
	}
	
	return;
	
}

dbcsr::smat_d LLT::L_inv(vec<int> blksizes) {
	
	compute_L_inverse();
	
	auto w = m_mat_in->get_world();
	dbcsr::smat_d out = std::make_shared<dbcsr::mat_d>(
		dbcsr::scalapack_to_matrix(*m_L_inv, "Inverted cholesky decomposition of "+m_mat_in->name(),
			w,blksizes,blksizes,"lowtriang")
	);
	
	out->filter();
	return out;
	
}
	
} // end namespace
