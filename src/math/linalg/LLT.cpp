#include "math/linalg/LLT.hpp"
#include <dbcsr_conversions.hpp>

namespace megalochem {

namespace math {

void LLT::compute() {

	auto dgrid = m_world.dbcsr_grid();
	auto sgrid = m_world.scalapack_grid();
	
	int n = m_mat_in->nfullrows_total();
	int nb = scalapack::global::block_size;
	int nprow = dgrid.dims()[0];
	int npcol = dgrid.dims()[1];
	
	LOG.os<>("Running SCALAPACK pdpotrf calculation\n");
	
	// convert array
	
	LOG.os<>("-- Converting input matrix to SCALAPACK format.\n");
		
	int ori_proc = m_mat_in->proc(0,0);
	int ori_coord[2];
	
	if (m_world.rank() == ori_proc) {
		ori_coord[0] = sgrid.myprow();
		ori_coord[1] = sgrid.mypcol();
	}
	
	MPI_Bcast(&ori_coord[0],2,MPI_INT,ori_proc,m_world.comm());
		
	m_L = std::make_shared<scalapack::distmat<double>>(
		dbcsr::matrix_to_scalapack(m_mat_in, sgrid, 
		m_mat_in->name() + "_scalapack", 
		nb, nb, ori_coord[0], ori_coord[1])
	);
		
	LOG.os<>("-- Computing cholesky decomposition.\n");
	
	//m_L->print();
	
	int info;
	c_pdpotrf('L', n, m_L->data(), 0, 0, m_L->desc().data(), &info); 
	
	LOG.os<>("-- Exited with info = ", info, '\n');

	//m_L->print();
	
}

dbcsr::shared_matrix<double> LLT::L(vec<int> blksizes) {
	
	auto out = dbcsr::scalapack_to_matrix(*m_L, m_world.dbcsr_grid(), 
			"Cholesky decomposition of "+m_mat_in->name(),
			blksizes,blksizes,"lowtriang");
	
	return out;
	
}

void LLT::compute_L_inverse() {
	
	if (!m_L_inv) {
		
		LOG.os<>("Computing sqrt inverse of matrix.\n");
		
		// copy L
		int n = m_L->nrowstot();
		int nb = scalapack::global::block_size;
		
		m_L_inv = std::make_shared<scalapack::distmat<double>>(
			m_world.scalapack_grid(), n, n, nb, nb, m_L->rsrc(), m_L->csrc());
		
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

dbcsr::shared_matrix<double> LLT::L_inv(vec<int> blksizes) {
	
	compute_L_inverse();
	
	auto w = m_world.dbcsr_grid();
	auto out =
		dbcsr::scalapack_to_matrix(*m_L_inv, w,
			"Inverted cholesky decomposition of "+m_mat_in->name(),
			blksizes,blksizes,"lowtriang");
	
	return out;
	
}

dbcsr::shared_matrix<double> LLT::inverse(vec<int> blksizes) {
	
	auto Linv = this->L_inv(blksizes);
	auto out = dbcsr::matrix<>::create_template(*Linv)
		.name("Cholesky inverse of " + m_mat_in->name())
		.matrix_type(dbcsr::type::symmetric)
		.build();
	
	dbcsr::multiply('T', 'N', 1.0, *Linv, *Linv, 0.0, *out)
		.perform();
	
	return out;
	
}
	
} // end namespace

} // end namespace megalochem
