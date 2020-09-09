#include "math/linalg/SVD.h"
#include <dbcsr_conversions.hpp>

namespace math {

void SVD::compute() {
	
	auto wrd = m_mat_in->get_world();
	int m = m_mat_in->nfullrows_total();
	int n = m_mat_in->nfullcols_total();
	int nb = scalapack::global::block_size;
	int nprow = wrd.dims()[0];
	int npcol = wrd.dims()[1];
	int size = std::min(m,n);
	auto& _grid = scalapack::global_grid;
	
	LOG.os<>("Running SCALAPACK pdgesvd calculation\n");
	
	// convert array
	
	LOG.os<>("-- Converting input matrix to SCALAPACK format.\n");
	LOG.os<>("Problem size: ", m, " ", n, '\n');
		
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
	
	MPI_Barrier(wrd.comm());
	sca_mat_in->print();
	MPI_Barrier(wrd.comm());
	
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
	
	sca_mat_in->print();
	
	auto desca = sca_mat_in->desc();
	auto descu = m_U->desc();
	auto descvt = m_Vt->desc();
	
	c_pdgesvd(m_jobu, m_jobvt, m, n, a, 0, 0, desca.data(), s, u, 0, 0,
		descu.data(), vt, 0, 0, descvt.data(), &wsize, 1, &info);
	
	int lwork = (int)wsize;
	
	LOG.os<>("-- LWORK: ", lwork, '\n');
	
	double* work = new double[lwork];
	
	LOG.os<>("-- Computing SVD.\n");
	
	c_pdgesvd(m_jobu, m_jobvt, m, n, a, 0, 0, desca.data(), s, u, 0, 0,
		descu.data(), vt, 0, 0, descvt.data(), work, lwork, &info);
	
	LOG.os<>("Eigenvalues: \n");
	for (int i = 0; i != size; ++i) {
		LOG.os<>(s[i], " ");
	} LOG.os<>('\n');
	
	LOG.os<>("-- Exited with info = ", info, '\n');

	delete [] work;
	
}

dbcsr::shared_matrix<double> SVD::inverse() {
	
	int m = m_mat_in->nfullrows_total();
	int n = m_mat_in->nfullcols_total();
	
	auto rowsizes = m_mat_in->row_blk_sizes();
	auto colsizes = m_mat_in->col_blk_sizes();
	
	vec<int> sizes = (m < n) ? rowsizes : colsizes;
	
	auto w = m_mat_in->get_world();
	
	dbcsr::shared_matrix<double> U =
		dbcsr::scalapack_to_matrix(*m_U, "U", w, 
			rowsizes, sizes);
	
	dbcsr::shared_matrix<double> Vt =
		dbcsr::scalapack_to_matrix(*m_Vt, "Vt", w, 
			sizes, colsizes);
			
	//dbcsr::print(*U);
	//dbcsr::print(*Vt);
			
	auto out = dbcsr::create<double>().set_world(w)
		.name("SVD inverse of " + m_mat_in->name())
		.row_blk_sizes(colsizes)
		.col_blk_sizes(rowsizes)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	auto sigma = *m_s;
	
	std::for_each(sigma.begin(), sigma.end(), 
		[](double& e) { 
			e = (fabs(e) < 1e-12) ? 0 : 1/e; 
			std::cout << e << std::endl; });	
	
	U->scale(sigma, "right");
	
	//dbcsr::print(*U);
	
	dbcsr::multiply('T', 'T', *Vt, *U, *out).perform();
	
	//dbcsr::print(*out);
	
	auto ide = dbcsr::create<double>().set_world(w)
		.name("IDE1")
		.row_blk_sizes(rowsizes)
		.col_blk_sizes(rowsizes)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	dbcsr::multiply('N', 'N', *m_mat_in, *out, *ide).perform();
	
	//dbcsr::print(*ide);
	
	auto ide1 = dbcsr::create<double>().set_world(w)
		.name("IDE2")
		.row_blk_sizes(rowsizes)
		.col_blk_sizes(colsizes)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	dbcsr::multiply('N', 'N', *ide, *m_mat_in, *ide1).perform();
	
	std::cout << "THIS" << std::endl;
	//dbcsr::print(*m_mat_in);
	//dbcsr::print(*ide1);
	
	ide1->add(1.0, -1.0, *m_mat_in);
	
	LOG.os<>("Error: ", ide1->norm(dbcsr_norm_frobenius), '\n');
	
		
	return out;
	
}
	
} // end namespace
