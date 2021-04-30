#include "math/linalg/SVD.hpp"
#include <dbcsr_conversions.hpp>

#include "utils/matrix_plot.hpp"

namespace megalochem {

namespace math {

void SVD::compute(double eps) {
	
	auto dcart = m_world.dbcsr_grid();
	auto sgrid = m_world.scalapack_grid();

	int m = m_mat_in->nfullrows_total();
	int n = m_mat_in->nfullcols_total();
	int nb = scalapack::global::block_size;
	int nprow = dcart.dims()[0];
	int npcol = dcart.dims()[1];
	int size = std::min(m,n);
	
	LOG.os<1>("Running SCALAPACK pdgesvd calculation\n");
	
	// convert array
	
	LOG.os<1>("-- Converting input matrix to SCALAPACK format.\n");
	LOG.os<1>("Problem size: ", m, " ", n, '\n');
		
	int ori_proc = m_mat_in->proc(0,0);
	int ori_coord[2];
	
	if (m_world.rank() == ori_proc) {
		ori_coord[0] = sgrid.myprow();
		ori_coord[1] = sgrid.mypcol();
	}
	
	MPI_Bcast(&ori_coord[0],2,MPI_INT,ori_proc,m_world.comm());
		
	auto sca_mat_in = std::make_shared<scalapack::distmat<double>>(
		dbcsr::matrix_to_scalapack(m_mat_in, sgrid, 
		m_mat_in->name() + "_scalapack", 
		nb, nb, ori_coord[0], ori_coord[1])
	);
	
	//MPI_Barrier(wrd.comm());
	//sca_mat_in->print();
	//MPI_Barrier(wrd.comm());
	
	LOG.os<1>("-- Setting up other arrays.\n");
	
	m_U = std::make_shared<scalapack::distmat<double>>(
		sgrid,m,size,nb,nb,ori_coord[0],ori_coord[1]);
		
	m_Vt = std::make_shared<scalapack::distmat<double>>(
		sgrid,size,n,nb,nb,ori_coord[0],ori_coord[1]);
	
	m_s = std::make_shared<std::vector<double>>(size,0.0);
	
	LOG.os<1>("-- Computing size of work space.");
	
	double wsize = 0;
	int info = 0;
	
	auto a = sca_mat_in->data();
	auto u = m_U->data();
	auto vt = m_Vt->data();
	auto s = m_s->data();
	
	//sca_mat_in->print();
	
	auto desca = sca_mat_in->desc();
	auto descu = m_U->desc();
	auto descvt = m_Vt->desc();
	
	c_pdgesvd(m_jobu, m_jobvt, m, n, a, 0, 0, desca.data(), s, u, 0, 0,
		descu.data(), vt, 0, 0, descvt.data(), &wsize, -1, &info);
	
	int lwork = (int)wsize;
	
	LOG.os<1>("-- LWORK: ", lwork, '\n');
	
	double* work = new double[lwork];
	
	LOG.os<1>("-- Computing SVD.\n");
	
	c_pdgesvd(m_jobu, m_jobvt, m, n, a, 0, 0, desca.data(), s, u, 0, 0,
		descu.data(), vt, 0, 0, descvt.data(), work, lwork, &info);
	
	m_rank = 0;
	
	LOG.os<2>("Eigenvalues: \n");
	for (int i = 0; i != size; ++i) {
		LOG.os<2>(s[i], " ");
		if (fabs(s[i]) > eps) m_rank++;
	} LOG.os<2>('\n');
	
	LOG.os<1>("RANK: ", m_rank, '\n');	
	
	LOG.os<1>("-- Exited with info = ", info, '\n');
	
	if (m_jobu == 'V') {
		
		LOG.os<1>("Truncating U\n");
		
		auto newU = std::make_shared<scalapack::distmat<double>>(
			sgrid,m,m_rank,nb,nb,ori_coord[0],ori_coord[1]);
			
		c_pdgeadd('N', m, m_rank, 1.0, m_U->data(), 0, 0,
			m_U->desc().data(), 0.0, newU->data(), 0, 0, newU->desc().data());
			
		m_U = newU;
		
	}
	
	if (m_jobvt == 'V') {
		
		LOG.os<1>("Truncating Vt\n");
		
		auto newVt = std::make_shared<scalapack::distmat<double>>(
			sgrid, m_rank,n,nb,nb,ori_coord[0],ori_coord[1]);
			
		c_pdgeadd('N', m_rank, n, 1.0, m_Vt->data(), 0, 0,
			m_Vt->desc().data(), 0.0, newVt->data(), 0, 0, newVt->desc().data());
			
		m_Vt = newVt;
		
	}


	delete [] work;
	
}

dbcsr::shared_matrix<double> SVD::inverse() {
	
	int m = m_mat_in->nfullrows_total();
	int n = m_mat_in->nfullcols_total();
	
	auto rowsizes = m_mat_in->row_blk_sizes();
	auto colsizes = m_mat_in->col_blk_sizes();
	
	vec<int> sizes = (m < n) ? rowsizes : colsizes;
	
	auto dcart = m_mat_in->get_cart();
	
	dbcsr::shared_matrix<double> U =
		dbcsr::scalapack_to_matrix(*m_U, dcart, "U", 
			rowsizes, sizes);
	
	dbcsr::shared_matrix<double> Vt =
		dbcsr::scalapack_to_matrix(*m_Vt, dcart, "Vt", 
			sizes, colsizes);
			
	//dbcsr::print(*U);
	//dbcsr::print(*Vt);
			
	auto out = dbcsr::matrix<>::create()
		.set_cart(dcart)
		.name("SVD inverse of " + m_mat_in->name())
		.row_blk_sizes(colsizes)
		.col_blk_sizes(rowsizes)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	auto sigma = *m_s;
	
	std::for_each(sigma.begin(), sigma.end(), 
		[](double& e) { 
			e = (fabs(e) < 1e-12) ? 0 : 1/e; 
			std::cout << e << std::endl; });	
	
	U->scale(sigma, "right");
	
	//dbcsr::print(*U);
	
	dbcsr::multiply('T', 'T', 1.0, *Vt, *U, 0.0, *out).perform();
	
	//dbcsr::print(*out);
	
	auto ide = dbcsr::matrix<>::create()
		.set_cart(dcart)
		.name("IDE1")
		.row_blk_sizes(rowsizes)
		.col_blk_sizes(rowsizes)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	dbcsr::multiply('N', 'N', 1.0, *m_mat_in, *out, 0.0, *ide).perform();
	
	//dbcsr::print(*ide);
	
	auto ide1 = dbcsr::matrix<>::create()
		.name("IDE2")
		.row_blk_sizes(rowsizes)
		.col_blk_sizes(colsizes)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	dbcsr::multiply('N', 'N', 1.0, *ide, *m_mat_in, 0.0, *ide1).perform();
	
	//dbcsr::print(*m_mat_in);
	//dbcsr::print(*ide1);
	
	ide1->add(1.0, -1.0, *m_mat_in);
	
	LOG.os<>("Error: ", ide1->norm(dbcsr_norm_frobenius), '\n');
	
		
	return out;
	
}

dbcsr::shared_matrix<double> SVD::U(std::vector<int> rowblksizes, 
	std::vector<int> colblksizes) {
		
	//m_U->print();
	auto dcart = m_world.dbcsr_grid();
	
	auto out = dbcsr::scalapack_to_matrix(*m_U, dcart, 
		"SVD U matrix of " + m_mat_in->name(), rowblksizes, colblksizes);
			
	//dbcsr::print(*out);
	
	//util::plot(out, 1e-4);
	
	return out;
	
}

dbcsr::shared_matrix<double> SVD::Vt(std::vector<int> rowblksizes, 
	std::vector<int> colblksizes) {
	
	auto dcart = m_mat_in->get_cart();
	
	auto out = dbcsr::scalapack_to_matrix(*m_Vt, dcart, 
		"SVD Vt matrix of " + m_mat_in->name(), rowblksizes, colblksizes);
			
	//util::plot(out, 1e-4);
	
	return out;
	
}

std::vector<double> SVD::s() {
	
	std::vector<double> out(m_rank);
	std::copy(m_s->begin(), m_s->begin() + m_rank, out.begin());
	return out;
	
}
	
} // end namespace

} // end megalochem
