#include "locorb/locorb.h"

namespace locorb {

using smat = dbcsr::shared_matrix<double>;

smat transform(smat c, smat dip) {
		
	auto b = c->row_blk_sizes();
	auto m = c->col_blk_sizes();
	auto w = c->get_world();
	
	auto temp = dbcsr::create<double>()
		.name("temp")
		.set_world(w)
		.row_blk_sizes(b)
		.col_blk_sizes(m)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto dip_mm = dbcsr::create<double>()
		.name("dip_mm")
		.set_world(w)
		.row_blk_sizes(m)
		.col_blk_sizes(m)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	dbcsr::multiply('N', 'N', *dip, *c, *temp)
		.filter_eps(dbcsr::global::filter_eps)
		.perform();
		
	dbcsr::multiply('T', 'N', *c, *temp, *dip_mm)
		.filter_eps(dbcsr::global::filter_eps)
		.perform();
		
	return dip_mm;
		
}

smat compute_D(smat dip_x, smat dip_y, smat dip_z) {
		
	auto diag_x = dip_x->get_diag();
	auto diag_y = dip_y->get_diag();
	auto diag_z = dip_z->get_diag();
	
	auto copy_x = dbcsr::copy(dip_x).get();
	auto copy_y = dbcsr::copy(dip_y).get();
	auto copy_z = dbcsr::copy(dip_z).get();
	
	copy_x->scale(diag_x, "right");
	copy_y->scale(diag_y, "right");
	copy_z->scale(diag_z, "right");
	
	copy_x->add(1.0, 1.0, *copy_y);
	copy_x->add(1.0, 1.0, *copy_z);
	
	copy_x->setname("D");
	return copy_x;
	
}

double jacobi_sweep(smat& c_dist, smat& x_dist, smat& y_dist, smat& z_dist) {
	
	auto c = dbcsr::matrix_to_eigen(c_dist);
	auto x = dbcsr::matrix_to_eigen(x_dist);
	auto y = dbcsr::matrix_to_eigen(y_dist);
	auto z = dbcsr::matrix_to_eigen(z_dist);
	auto w = c_dist->get_world();
	
	int nbas = c_dist->nfullrows_total();
	int norb = c_dist->nfullcols_total();
	
	auto b = c_dist->row_blk_sizes();
	auto m = c_dist->col_blk_sizes();
	
	double t12 = 1e-12;
	double t8 = 1e-8;
	
	double max_diff = 0.0;
	
	auto mat_update = [&norb](Eigen::MatrixXd& mat, double ca, 
		double sa, int i, int j) 
	{
			
		double mii = mat(i,i);
		double mjj = mat(j,j);
		double mij = mat(i,j);
		
		for (int k = 0; k != norb; ++k) {
			
			double mik = mat(i,k);
			double mjk = mat(j,k);
			
			mat(i,k) = ca * mik + sa * mjk;
			mat(k,i) = mat(i,k);
			mat(j,k) = ca * mjk - sa * mik;
			mat(k,j) = mat(j,k);
			
		}
		
		mat(i,i) = (pow(ca,2.0) - pow(sa,2.0)) * mij + ca*sa*(mjj - mii);
		mat(j,i) = mat(i,j);
		mat(i,i) = pow(ca,2.0)*mii + pow(sa,2.0)*mjj + 2*ca*sa*mij;
		mat(j,j) = pow(sa,2.0)*mii + pow(ca,2.0)*mjj - 2*ca*sa*mij;
		
	};
	
	
	if (w.rank() == 0) {
		std::cout << "COLD" << std::endl;
		std::cout << c << std::endl;
		std::cout << "X:" << std::endl;
		std::cout << x << std::endl;
		std::cout << "Y:" << std::endl;
		std::cout << y << std::endl;
		std::cout << "Z:" << std::endl;
		std::cout << z << std::endl;
	}
	
	
	
	if (w.rank() == 0) {
	
		for (int i = 0; i < norb; ++i) {
			for (int j = i+1; j < norb; ++j) {
				
				if (w.rank() == 0) {
					std::cout << "I,J: " << i << " " << j << std::endl;
				}
				
				double A = pow(x(i,j),2.0) + pow(y(i,j),2.0) + pow(z(i,j),2.0)
					- 0.25 * (pow(x(i,i),2.0) + pow(y(i,i),2.0) + pow(z(i,i),2.0))
					- 0.25 * (pow(x(j,j),2.0) + pow(y(j,j),2.0) + pow(z(j,j),2.0))
					+ 0.5 * (x(i,i) * x(j,j) + y(i,i) * y(j,j) + z(i,i) * z(j,j));
					
				double B = x(i,j)*(x(i,i) - x(j,j))
					+ y(i,j) * (y(i,i) - y(j,j))
					+ z(i,j) * (z(i,i) - z(j,j));
					
				if (fabs(B) < t12) continue;
				if (fabs(A) < t8) throw std::runtime_error("PANIC!");
				
				// update
				
				double alpha4 = atan(-B/A);
				double pi = 2 * acos(0.0);
				
				if (alpha4 < 0.0 && B > 0.0) alpha4 += pi;
				if (alpha4 > 0.0 && B < 0.0) alpha4 -= pi;
				
				double ca = cos(alpha4/4.0);
				double sa = sin(alpha4/4.0);
				
				if (0 == w.rank()) {
					std::cout << "ca & sa: " <<  ca << " " << sa << std::endl;
				}
				
				max_diff = std::max(max_diff, A + sqrt(pow(A,2.0) + pow(B,2.0)));
				
				// update c
				for (int k = 0; k != nbas; ++k) {
					double cki = c(k,i);
					double ckj = c(k,j);
					
					std::cout << k << " " << c(k,i) << std::endl;
					std::cout << k << " " << c(k,j) << std::endl;
					
					c(k,i) = ca*cki + sa*ckj;
					c(k,j) = -sa*cki + ca*ckj;
					
					std::cout << k << " " << c(k,i) << std::endl;
					std::cout << k << " " << c(k,j) << std::endl;
					
				}
				
				std::cout << "C: " << std::endl;
				std::cout << c << std::endl;
				
				exit(0);
				
				// update x,y,z
				mat_update(x, ca, sa, i, j);
				mat_update(y, ca, sa, i, j);
				mat_update(z, ca, sa, i, j);
				
			}
		}
		
	}
	
	MPI_Bcast(c.data(), nbas*norb, MPI_DOUBLE, 0, w.comm());
	MPI_Bcast(x.data(), norb*norb, MPI_DOUBLE, 0, w.comm());
	MPI_Bcast(y.data(), norb*norb, MPI_DOUBLE, 0, w.comm());
	MPI_Bcast(z.data(), norb*norb, MPI_DOUBLE, 0, w.comm());
	MPI_Bcast(&max_diff, 1, MPI_DOUBLE, 0, w.comm());
		
	c_dist = dbcsr::eigen_to_matrix(c, w, "c_new", b, m, dbcsr::type::no_symmetry);
	x_dist = dbcsr::eigen_to_matrix(x, w, "x_new", m, m, dbcsr::type::no_symmetry);
	y_dist = dbcsr::eigen_to_matrix(y, w, "y_new", m, m, dbcsr::type::no_symmetry);
	z_dist = dbcsr::eigen_to_matrix(z, w, "z_new", m, m, dbcsr::type::no_symmetry);
	
	return max_diff;
	
}

double jacobi_sweep_mpi(smat& c_dist, smat& x_dist, smat& y_dist, smat& z_dist) {
	
	auto w = c_dist->get_world();
	int my_rank = w.rank();
	
	if (w.size() < 2) {
		throw std::runtime_error(
			"Jacobi sweep (MPI) needs nproc >= 3");
	}
	
	int nbas = c_dist->nfullrows_total();
	int norb = c_dist->nfullcols_total();
	
	auto b = c_dist->row_blk_sizes();
	auto m = c_dist->col_blk_sizes();
	
	auto c = dbcsr::matrix_to_eigen(c_dist);
	auto x = dbcsr::matrix_to_eigen(x_dist);
	auto y = dbcsr::matrix_to_eigen(y_dist);
	auto z = dbcsr::matrix_to_eigen(z_dist);
	
	double t12 = 1e-12;
	double t8 = 1e-8;
	double max_diff = 0.0;
	
	MPI_Comm comm_i = MPI_COMM_NULL;
	MPI_Comm comm_j = MPI_COMM_NULL;
	int color = my_rank % 2;
	
	int color_i = (my_rank == 0 || color == 0) ? 0 : MPI_UNDEFINED;
	int color_j = (my_rank == 0 || color == 1) ? 1 : MPI_UNDEFINED;
	
	int mpirank_i = -1;
	int mpisize_i = 0;
	int mpirank_j = -1;
	int mpisize_j = 0;
	
	MPI_Comm_split(w.comm(), color_i, my_rank, &comm_i);
	MPI_Comm_split(w.comm(), color_j, my_rank, &comm_j);
	
	if (comm_i != MPI_COMM_NULL) {
		MPI_Comm_rank(comm_i, &mpirank_i);
		MPI_Comm_size(comm_i, &mpisize_i);
	}
	
	if (comm_j != MPI_COMM_NULL) {
		MPI_Comm_rank(comm_j, &mpirank_j);
		MPI_Comm_size(comm_j, &mpisize_j);
	}
	
	
	for (int i = 0; i != w.size(); ++i) {
		if (i == w.rank()) {
			std::cout << "RANK: " << w.rank() << std::endl;
			std::cout << "COLOR 0: " << color_i << std::endl;
			std::cout << "COLOR 1: " << color_j << std::endl;
			std::cout << "COLPROC 0: " << mpirank_i << std::endl;
			std::cout << "COLPROC 1: " << mpirank_j << std::endl;
		}
		MPI_Barrier(w.comm());
	}
	
	std::vector<int> chunk_sizes_norb_i(mpisize_i), 
		chunk_sizes_nbas_i(mpisize_i),
		chunk_sizes_norb_j(mpisize_j),
		chunk_sizes_nbas_j(mpisize_j),
		chunk_off_norb_i(mpisize_i,0),
		chunk_off_nbas_i(mpisize_i,0),
		chunk_off_norb_j(mpisize_j,0),
		chunk_off_nbas_j(mpisize_j,0);
		
	auto get_chunk_size = [](int n, int mpi_size, int mpi_rank) {
		int size = mpi_size - 1;
		switch (mpi_rank) {
			case 0:
				return 0;
			case 1:
				return n / size + n % size;
			default:
				return n / size;
		}
	};
	
	int off_norb = 0;
	int off_nbas = 0;
	
	for (int n = 0; n != mpisize_i; ++n) {
		
		int norb_size = get_chunk_size(norb, mpisize_i, n);
		int nbas_size = get_chunk_size(nbas, mpisize_i, n);
		
		chunk_sizes_norb_i[n] = norb_size;
		chunk_sizes_nbas_i[n] = nbas_size;
		chunk_off_norb_i[n] = off_norb;
		chunk_off_nbas_i[n] = off_nbas;
		
		off_norb += norb_size;
		off_nbas += nbas_size;
		
	}
	
	off_norb = 0;
	off_nbas = 0;
	
	for (int n = 0; n != mpisize_j; ++n) {
		
		int norb_size = get_chunk_size(norb, mpisize_j, n);
		int nbas_size = get_chunk_size(nbas, mpisize_j, n);
		
		chunk_sizes_norb_j[n] = norb_size;
		chunk_sizes_nbas_j[n] = nbas_size;
		chunk_off_norb_j[n] = off_norb;
		chunk_off_nbas_j[n] = off_nbas;
		
		off_norb += norb_size;
		off_nbas += nbas_size;
		
	} 
	
	auto printvec = [](auto vec) {
		for (auto a : vec) {
			std::cout << a << " ";
		} std::cout << std::endl;
	};
	
	if (w.rank() == 0) {
		std::cout << "NORB: " << norb << std::endl;
		std::cout << "NBAS: " << nbas << std::endl;
		std::cout << "X:" << std::endl;
		std::cout << x << std::endl;
		std::cout << "Y:" << std::endl;
		std::cout << y << std::endl;
		std::cout << "Z:" << std::endl;
		std::cout << z << std::endl;
	}
	
	for (int i = 0; i != w.size(); ++i) {
		if (i == w.rank()) {
			std::cout << "RANK: " << w.rank() << std::endl;
			std::cout << "COLOR 0: " << color_i << std::endl;
			std::cout << "COLOR 1: " << color_j << std::endl;
			std::cout << "COLPROC 0: " << mpirank_i << std::endl;
			std::cout << "COLPROC 1: " << mpirank_j << std::endl;
			printvec(chunk_sizes_norb_i);
			printvec(chunk_sizes_norb_j);
			printvec(chunk_sizes_nbas_i);
			printvec(chunk_sizes_nbas_j);
			printvec(chunk_off_norb_i);
			printvec(chunk_off_norb_j);
			printvec(chunk_off_nbas_i);
			printvec(chunk_off_nbas_j);
		}
		MPI_Barrier(w.comm());
	}
	
	if (w.rank() == 0) {
		std::cout << "COLD" << std::endl;
		std::cout << c << std::endl;
	}
	
	// now begin sweep
	for (int i = 0; i < norb; ++i) {
		for (int j = i+1; j < norb; ++j) {
			
			if (w.rank() == 0) {
				std::cout << "I,J: " << i << " " << j << std::endl;
			}
			
			double A = pow(x(i,j),2.0) + pow(y(i,j),2.0) + pow(z(i,j),2.0)
					- 0.25 * (pow(x(i,i),2.0) + pow(y(i,i),2.0) + pow(z(i,i),2.0))
					- 0.25 * (pow(x(j,j),2.0) + pow(y(j,j),2.0) + pow(z(j,j),2.0))
					+ 0.5 * (x(i,i) * x(j,j) + y(i,i) * y(j,j) + z(i,i) * z(j,j));
					
			double B = x(i,j)*(x(i,i) - x(j,j))
				+ y(i,j) * (y(i,i) - y(j,j))
				+ z(i,j) * (z(i,i) - z(j,j));
			
			if (fabs(B) < t12) continue;
			if (fabs(A) < t8) throw std::runtime_error("PANIC!");
				
			// update
			
			double alpha4 = atan(-B/A);
			double pi = 2 * acos(0.0);
			
			if (alpha4 < 0.0 && B > 0.0) alpha4 += pi;
			if (alpha4 > 0.0 && B < 0.0) alpha4 -= pi;
			
			double ca = cos(alpha4/4.0);
			double sa = sin(alpha4/4.0);
		
			max_diff = std::max(max_diff, A + sqrt(pow(A,2.0) + pow(B,2.0)));
			
			for (int ip = 0; ip != w.size(); ++ip) {
				if (ip == w.rank()) {
					std::cout << "ca & sa: " <<  ca << " " << sa << std::endl;
				}
				MPI_Barrier(w.comm());
			}
			
			// update c
			if (mpirank_i > 0) {
				
				int chunk_off = chunk_off_nbas_i[mpirank_i];
				int chunk_size = chunk_sizes_nbas_i[mpirank_i];
				
				for (int k = chunk_off; (k < chunk_off + chunk_size) && (k < nbas); ++k) {
					
					double cki = c(k,i);
					double ckj = c(k,j);
				
					c(k,i) = ca*cki + sa*ckj; 
								
				}
			}
			
			MPI_Barrier(w.comm());
			
			if (mpirank_j > 0) {
				
				int chunk_off = chunk_off_nbas_j[mpirank_j];
				int chunk_size = chunk_sizes_nbas_j[mpirank_j];
				
				for (int k = chunk_off; (k < chunk_off + chunk_size) && (k < nbas); ++k) {
					double cki = c(k,i);
					double ckj = c(k,j);
					
					c(k,j) = - sa*cki + ca*ckj; 
				}
			}
			
			MPI_Barrier(w.comm());
			
			for (int p = 0; p != w.size(); ++p) {
				if (p == w.rank()) {
					std::cout << "C " << p << std::endl;
					std::cout << c << std::endl;
				}
				MPI_Barrier(w.comm());
			}
			
			// Gather col i
			
			double* c_ptr_recv_i = c.data() + nbas * i;
			double* c_ptr_recv_j = c.data() + nbas * j;
			
			if (mpirank_i >= 0) {
				
				std::cout << "PROCESS 0 /" << w.rank() << std::endl;
			
				double* c_ptr_send = c.data() + nbas * i + chunk_off_nbas_i[mpirank_i];
				
				MPI_Gatherv(c_ptr_send, chunk_sizes_nbas_i[mpirank_i],
					MPI_DOUBLE, c_ptr_recv_i, chunk_sizes_nbas_i.data(),
					chunk_off_nbas_i.data(), MPI_DOUBLE, 0, comm_i);
					
			}
			
			if (mpirank_j >= 0) {
				
				std::cout << "PROCESS 1 /" << w.rank() << std::endl;
				
				double* c_ptr_send = c.data() + nbas * j + chunk_off_nbas_j[mpirank_j];
				
				MPI_Gatherv(c_ptr_send, chunk_sizes_nbas_j[mpirank_j],
					MPI_DOUBLE, c_ptr_recv_j, chunk_sizes_nbas_j.data(), 
					chunk_off_nbas_j.data(), MPI_DOUBLE, 0, comm_j);
					
			}
	
			// Send to all processes
			
			//MPI_Bcast(c_ptr_recv_i, nbas, MPI_DOUBLE, 0, w.comm());
			//MPI_Bcast(c_ptr_recv_j, nbas, MPI_DOUBLE, 0, w.comm());
	
			if (w.rank() == 0) {
				std::cout << "C:" << std::endl;
				std::cout << c << std::endl;
			}
	
			exit(0);
			
		}
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/*
	
	double max_diff = 0.0;
	
	auto mat_update = [&norb](Eigen::MatrixXd& mat, double ca, 
		double sa, int i, int j) 
	{
			
		double mii = mat(i,i);
		double mjj = mat(j,j);
		double mij = mat(i,j);
		
		for (int k = 0; k != norb; ++k) {
			
			double mik = mat(i,k);
			double mjk = mat(j,k);
			
			mat(i,k) = ca * mik + sa * mjk;
			mat(k,i) = mat(i,k);
			mat(j,k) = ca * mjk - sa * mik;
			mat(k,j) = mat(j,k);
			
		}
		
		mat(i,i) = (pow(ca,2.0) - pow(sa,2.0)) * mij + ca*sa*(mjj - mii);
		mat(j,i) = mat(i,j);
		mat(i,i) = pow(ca,2.0)*mii + pow(sa,2.0)*mjj + 2*ca*sa*mij;
		mat(j,j) = pow(sa,2.0)*mii + pow(ca,2.0)*mjj - 2*ca*sa*mij;
		
	};
	
	if (w.rank() == 0) {
	
		for (int i = 0; i != norb; ++i) {
			for (int j = i+1; j != norb; ++j) {
				
				double A = pow(x(i,j),2.0) + pow(y(i,j),2.0) + pow(z(i,j),2.0)
					- 0.25 * (pow(x(i,i),2.0) + pow(y(i,i),2.0) + pow(z(i,i),2.0))
					- 0.25 * (pow(x(j,j),2.0) + pow(y(j,j),2.0) + pow(z(j,j),2.0))
					+ 0.5 * (x(i,i) * x(j,j) + y(i,i) * y(j,j) + z(i,i) * z(j,j));
					
				double B = x(i,j)*(x(i,i) - x(j,j))
					+ y(i,j) * (y(i,i) - y(j,j))
					+ z(i,j) * (z(i,i) - z(j,j));
					
				if (fabs(B) < t12) continue;
				if (fabs(A) < t8) throw std::runtime_error("PANIC!");
				
				// update
				
				double alpha4 = atan(-B/A);
				double pi = 2 * acos(0.0);
				
				if (alpha4 < 0.0 && B > 0.0) alpha4 += pi;
				if (alpha4 > 0.0 && B < 0.0) alpha4 -= pi;
				
				double ca = cos(alpha4/4.0);
				double sa = sin(alpha4/4.0);
				
				rot = E;
				
				rot(i,i) = ca;
				rot(i,j) = -sa;
				rot(j,i) = sa;
				rot(j,j) = ca;
				
				auto rot_dist = dbcsr::eigen_to_matrix(rot, w, "rot", 
					m, m, dbcsr::type::no_symmetry);
				
				max_diff = std::max(max_diff, A + sqrt(pow(A,2.0) + pow(B,2.0)));
				
				// update c
				for (int k = 0; k != nbas; ++k) {
					double cki = c(k,i);
					double ckj = c(k,j);
					c(k,i) = ca*cki + sa*ckj;
					c(k,j) = -sa*cki + ca*ckj;
				}
				
				// update x,y,z
				mat_update(x, ca, sa, i, j);
				mat_update(y, ca, sa, i, j);
				mat_update(z, ca, sa, i, j);
				
			}
		}
		
	}
	
	MPI_Bcast(c.data(), nbas*norb, MPI_DOUBLE, 0, w.comm());
	MPI_Bcast(x.data(), norb*norb, MPI_DOUBLE, 0, w.comm());
	MPI_Bcast(y.data(), norb*norb, MPI_DOUBLE, 0, w.comm());
	MPI_Bcast(z.data(), norb*norb, MPI_DOUBLE, 0, w.comm());
	MPI_Bcast(&max_diff, 1, MPI_DOUBLE, 0, w.comm());
		
	c_dist = dbcsr::eigen_to_matrix(c, w, "c_new", b, m, dbcsr::type::no_symmetry);
	x_dist = dbcsr::eigen_to_matrix(x, w, "x_new", m, m, dbcsr::type::no_symmetry);
	y_dist = dbcsr::eigen_to_matrix(y, w, "y_new", m, m, dbcsr::type::no_symmetry);
	z_dist = dbcsr::eigen_to_matrix(z, w, "z_new", m, m, dbcsr::type::no_symmetry);
	
	return max_diff;
	
}*/
}

smat mo_localizer::compute_boys(smat c_bm) {

	// compute <chi_i|r|chi_j>
	
	auto moms = m_aofac->ao_emultipole();
	
	auto dip_bb_x = moms[0];
	auto dip_bb_y = moms[1];
	auto dip_bb_z = moms[2];
	
	dbcsr::print(*dip_bb_x);
	dbcsr::print(*dip_bb_y);
	dbcsr::print(*dip_bb_z);
	
	// guess orbitals
	auto g_bm = compute_cholesky(c_bm);
	
	dbcsr::print(*g_bm);
	
	auto dip_mm_x = transform(g_bm, dip_bb_x);
	auto dip_mm_y = transform(g_bm, dip_bb_y);
	auto dip_mm_z = transform(g_bm, dip_bb_z);
	
	// compute D
	auto D = compute_D(dip_mm_x, dip_mm_y, dip_mm_z);
	
	int max_iter = 30;
	double thresh = 1e-8;
	
	for (int iter = 0; iter != max_iter; ++iter) {
		
		double max_diff = jacobi_sweep_mpi(g_bm, dip_mm_x, dip_mm_y, dip_mm_z);
		
		D = compute_D(dip_mm_x, dip_mm_y, dip_mm_z);  
		double boys_val = D->trace();
		
		LOG.os<>("ITERATION: ", iter, '\n');
		LOG.os<>("BOYSVAL: ", boys_val, '\n');
		LOG.os<>("MAXDIFF: ", max_diff, '\n');
		
		if (fabs(max_diff) < thresh) {
			LOG.os<>("BOYS FINISHED\n");
			break;
		}
		
	}
	
	return g_bm;
			
}	
	
} // end namespace
