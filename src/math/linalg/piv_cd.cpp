 #include "math/linalg/piv_cd.h"
 #include <cmath>
 #include <limits>
 #include <numeric>
 #include <algorithm>
#include <dbcsr_matrix_ops.hpp>
 
#include "extern/scalapack.h"

#include "utils/matrix_plot.h"

namespace math {
	 
void pivinc_cd::reorder_and_reduce(scalapack::distmat<double>& L) {
	
	// REORDER COLUMNS ACCORDING TO SORT METHOD
	
	int N = L.nrowstot();
	int nb = L.rowblk_size();
	
	std::vector<int> new_col_perms(m_rank);
	std::iota(new_col_perms.begin(),new_col_perms.end(),0);
			
	LOG.os<1>("-- Reordering cholesky orbitals according to method: old", "\n");
					
	double reorder_thresh = 1e-4;
	
	// reorder it according to matrix values
	std::vector<double> lmo_pos(m_rank,0);
	
	for (int j = 0; j != m_rank; ++j) {
		
		int first_mu = -1;
		int last_mu = N-1;
		
		for (int i = 0; i != N; ++i) {
			
			double L_ij = L.get('A', ' ', i, j);
			
			if (fabs(L_ij) > reorder_thresh && first_mu == -1) {
				first_mu = i;
				last_mu = i;
			} else if (fabs(L_ij) > reorder_thresh && first_mu != -1) {
				last_mu = i;
			}
			
		}
		
		if (first_mu == -1) throw std::runtime_error("Piv. Cholesky decomposition: Reorder failed.");
		
		lmo_pos[j] = (double)(first_mu + last_mu) / 2;
		
		/*ORTHOGONALIZATION DO IT BETTER
		1. MAKE REORDERING INDEPENDENT
		2. MAKE MODULES USE LOC FUNCTIONS
		3. USE ORTHOGONALIZATION (CHOLESKY)*/
		
	}
	
	if (LOG.global_plev() >= 2) {
		LOG.os<2>("-- Orbital weights: \n");	
		for (auto x : lmo_pos) { 
			LOG.os<2>(x, " "); 
		} LOG.os<2>('\n');
	}		
	
	std::stable_sort(new_col_perms.begin(), new_col_perms.end(), 
		[&lmo_pos](int i1, int i2) { return lmo_pos[i1] < lmo_pos[i2]; });
	
	if (LOG.global_plev() >= 2) {
		LOG.os<2>("-- Reordered indices: \n");	
		for (auto x : new_col_perms) { 
			LOG.os<2>(x, " "); 
		} LOG.os<2>("\n");
	}	
	
	LOG.os<1>("-- Finished reordering.\n");
					
	m_L = std::make_shared<scalapack::distmat<double>>(N,N,nb,nb,0,0);
	
	LOG.os<1>("-- Reducing and reordering L.\n");
	
	for (int i = 0; i != m_rank; ++i) {
		
		c_pdgeadd('N', N, 1, 1.0, L.data(), 0, new_col_perms[i], L.desc().data(), 0.0,
			m_L->data(), 0, i, m_L->desc().data());
			
	}
	
}

void pivinc_cd::compute() {
	
	// convert input mat to scalapack format
	
	LOG.os<1>("Starting pivoted incomplete cholesky decomposition.\n");
	
	LOG.os<1>("-- Setting up scalapack environment and matrices.\n"); 
	
	int N = m_mat_in->nfullrows_total();
	int iter = 0;
	int* iwork = new int[N];
	char scopeR = 'R';
	char scopeC = 'C';
	char top = ' ';
	int nb = scalapack::global::block_size;
	
	auto& _grid = scalapack::global_grid;
	
	MPI_Comm comm = m_mat_in->get_world().comm();
	int myrank = m_mat_in->get_world().rank();
	int myprow = _grid.myprow();
	int mypcol = _grid.mypcol();
	
	int ori_proc = m_mat_in->proc(0,0);
	int ori_coord[2];
	
	if (myrank == ori_proc) {
		ori_coord[0] = _grid.myprow();
		ori_coord[1] = _grid.mypcol();
	}
	
	MPI_Bcast(&ori_coord[0],2,MPI_INT,ori_proc,comm);
	
	//util::plot(m_mat_in, 1e-4);
		
	scalapack::distmat<double> U = dbcsr::matrix_to_scalapack(m_mat_in, 
		m_mat_in->name() + "_scalapack", nb, nb, ori_coord[0], ori_coord[1]);

	scalapack::distmat<double> Ucopy(N,N,nb,nb,0,0);
	
	c_pdgeadd('N', N, N, 1.0, U.data(), 0, 0, U.desc().data(), 
		0.0, Ucopy.data(), 0, 0, Ucopy.desc().data());
	
	//if (LOG.global_plev() >= 3) {
	//	LOG.os<3>("-- Input matrix: \n");
	//	U.print();
	//}
	
	// permutation vector
	
	int LOCr = c_numroc(N, nb, _grid.myprow(), 0, _grid.nprow());
	int LOCc = c_numroc(N, nb, _grid.mypcol(), 0, _grid.npcol());
	
	int lipiv_r = LOCr + nb;
	int lipiv_c = LOCc + nb;
	
	int* ipiv_r = new int[lipiv_r]; // column vector distributed over rows
	int* ipiv_c = new int[lipiv_c]; // row vector distributed over cols
	
	int desc_r[9];
	int desc_c[9];
	
	int info = 0;
	c_descinit(&desc_r[0],N + nb*_grid.nprow(),1,nb,nb,0,0,_grid.ctx(),lipiv_r,&info);
	c_descinit(&desc_c[0],1,N + nb*_grid.npcol(),nb,nb,0,0,_grid.ctx(),1,&info);
	
	// vector to keep track of permutations
	std::vector<int> perms(N);
	std::iota(perms.begin(),perms.end(),1);
	
	// chol mat
	scalapack::distmat<double> L(N,N,nb,nb,0,0);
	
	auto printp = [&_grid](int* p, int n) {
		for (int ir = 0; ir != _grid.nprow(); ++ir) {
			for (int ic = 0; ic != _grid.npcol(); ++ic) {
				if (ir == _grid.myprow() && ic == _grid.mypcol()) {
					std::cout << ir << " " << ic << std::endl;
					for (int i = 0; i != n; ++i) {
						std::cout << p[i] << " ";
					} std::cout << std::endl;
				}
			}
		}
	};
	
	// get max diag element
	double max_U_diag_global = 0.0;
	for (int ix = 0; ix != N; ++ix) {
		
		max_U_diag_global = std::max(
			fabs(max_U_diag_global),
			fabs(U.get('A', ' ', ix, ix)));
			
	}
	
	LOG.os<1>("-- Problem size: ", N, '\n');
	LOG.os<1>("-- Maximum diagonal element of input matrix: ", max_U_diag_global, '\n');
	
	m_thresh = N * std::numeric_limits<double>::epsilon() * max_U_diag_global;
	double thresh = m_thresh; /*N * std::numeric_limits<double>::epsilon() * max_U_diag_global;*/
	
	LOG.os<1>("-- Threshold: ", thresh, '\n');
	
	std::function<void(int)> cd_step;
	cd_step = [&](int I) {
		
		// STEP 1: If Dimension of U is one, then set L and return 
		
		LOG.os<1>("---- Level ", I, '\n');
		
		if (I == N-1) {
			double U_II = U.get('A', ' ', I, I);
			L.set(I,I,sqrt(U_II));
			m_rank = I+1;
			return;
		}
		
		// STEP 2.0: Permutation
		
		// a) find maximum diagonal element
		
		double max_U_diag = U.get('A', ' ', I, I);
		int max_U_idx = I;
		
		for (int ix = I; ix != N; ++ix) {
			double ele = U.get('A', ' ', ix, ix);
			if (ele >= max_U_diag) {
				max_U_diag = ele;
				max_U_idx = ix;
			}
		}

		LOG.os<1>("---- MAX ", max_U_diag, " @ ", max_U_idx, '\n');
		
		// b) permute rows/cols
		// U := P * U * P^t
		
		LOG.os<1>("---- Permuting U.\n");
		/*
		for (int ix = I; ix != N; ++ix) {
			Pcol.set(ix,0,ix+1);
			Prow.set(0,ix,ix+1);
		}*/
		for (int ir = 0; ir != LOCr; ++ir) {
			ipiv_r[ir] = U.iglob(ir)+1;
		}
		for (int ic = 0; ic != LOCc; ++ic) {
			ipiv_c[ic] = U.jglob(ic)+1;
		}		
		
		//Pcol.set(I,0,max_U_idx + 1);
		//Prow.set(0,I,max_U_idx + 1);
		
		if (_grid.myprow() == U.iproc(I)) {
			ipiv_r[U.iloc(I)] = max_U_idx + 1;
		}
		if (_grid.mypcol() == U.jproc(I)) {
			ipiv_c[U.jloc(I)] = max_U_idx + 1;
		}
		
		//LOG.os<>("P\n");
		
		//c_blacs_barrier(_grid.ctx(),'A');
		
		//printp(ipiv_r,LOCr);
		
		//c_blacs_barrier(_grid.ctx(),'A');
		
		//printp(ipiv_c,LOCc);
		
		//U.print();
		
		c_pdlapiv('F', 'R', 'C', N-I, N-I, U.data(), I, I, U.desc().data(), ipiv_r, I, 0, desc_r, iwork);
		c_pdlapiv('F', 'C', 'R', N-I, N-I, U.data(), I, I, U.desc().data(), ipiv_c, 0, I, desc_c, iwork);
		
		//U.print();
		
		// STEP 3.0: Convergence criterion
		
		LOG.os<1>("---- Checking convergence.\n");
		
		double U_II = U.get('A', ' ', I, I);
		
		if (U_II < 0.0 && fabs(U_II) > thresh) {
			LOG.os<1>("fabs(U_II): ", fabs(U_II), '\n');
			throw std::runtime_error("Negative Pivot element. CD not possible.");
		}
		
		if (fabs(U_II) < thresh) {
			
			perms[I] = max_U_idx + 1;
			
			m_rank = I;
			return;
		}
		
		// STEP 3.1: Form Utilde := sub(U) - u * ut
	
		// a) get u
		// u_i
		scalapack::distmat<double> u_i(N,1,nb,nb,0,0);
		c_pdgeadd('N', N-I-1, 1, 1.0, U.data(), I+1, I, U.desc().data(), 0.0, u_i.data(), I+1, 0, u_i.desc().data());
		
		// b) form Utilde
		c_pdgemm('N', 'T', N-I-1, N-I-1, 1, -1/U_II, u_i.data(), I+1, 0, u_i.desc().data(),
			u_i.data(), I+1, 0, u_i.desc().data(), 1.0, U.data(), I+1, I+1, U.desc().data());
		
		// STEP 3.2: Solve P * Utilde * Pt = L * Lt
		
		LOG.os<1>("---- Start decomposition of submatrix of dimension ", N-I-1, '\n');
		cd_step(I+1);
		
		// STEP 3.3: Form L
		// (a) diagonal element
		
		L.set(I,I,sqrt(U_II));
		
		// (b) permute u_i
		//for (int ix = I; ix != N; ++ix) {
		//	Pcol.set(ix,0,perms[ix]);
		//}
		
		for (int ir = 0; ir != LOCr; ++ir) {
			ipiv_r[ir] = perms[U.iglob(ir)];
		}
		
		//printp(ipiv_r,LOCr);
		
		//LOG.os<>("LITTLE U\n");
		//u_i.print();
		
		c_pdlapiv('F', 'R', 'C', N-I-1, 1, u_i.data(), I+1, 0, u_i.desc().data(), ipiv_r, I+1, 0, desc_r, iwork);
		
		//u_i.print();
		
		// (c) add u_i to L
		
		//L.print();
		
		c_pdgeadd('N', N-I-1, 1, 1.0/sqrt(U_II), u_i.data(), I+1, 0, u_i.desc().data(), 0.0, L.data(), I+1, I, L.desc().data());
		
		//L.print();
		
		perms[I] = max_U_idx + 1;
		
		return;
		
	};
	
	LOG.os<1>("-- Starting recursive decomposition.\n");
	cd_step(0);
	
	LOG.os<1>("-- Rank of L: ", m_rank, '\n');
	
	for (int ir = 0; ir != LOCr; ++ir) {
		ipiv_r[ir] = perms[U.iglob(ir)];
	}
		
	//printp(ipiv_r,LOCr);
	
	LOG.os<1>("-- Permuting L.\n");
	
	//L.print();
	
	c_pdlapiv('B', 'R', 'C', N, N, L.data(), 0, 0, L.desc().data(), ipiv_r, 0, 0, desc_r, iwork);
	
	//L.print();
	
	//c_pdlapiv('B', 'C', 'C', N, N, Uc.data(), 0, 0, Uc.desc().data(), Pcol.data(), 0, 0, Pcol.desc().data(), iwork);
	//c_pdlapiv('B', 'R', 'R', N, N, Uc.data(), 0, 0, Uc.desc().data(), Prow.data(), 0, 0, Prow.desc().data(), iwork);
	
	//c_pdgeadd('N', N, N, 1.0, Ucopy.data(), 0, 0, Ucopy.desc().data(), -1.0, Uc.data(), 0, 0, Uc.desc().data());

	reorder_and_reduce(L);
	
	c_pdgemm('N', 'T', N, N, N, 1.0, m_L->data(), 0, 0, m_L->desc().data(), 
		m_L->data(), 0, 0, m_L->desc().data(), -1.0, Ucopy.data(), 0, 0, Ucopy.desc().data());
	
	double err = c_pdlange('F', N, N, Ucopy.data(), 0, 0, Ucopy.desc().data(), nullptr);
	
	LOG.os<1>("-- CD error: ", err, '\n');
	
	
	LOG.os<1>("Finished decomposition.\n");
	
	delete [] iwork;
	delete [] ipiv_r;
	delete [] ipiv_c;
		
}
	
dbcsr::smat_d pivinc_cd::L(std::vector<int> rowblksizes, std::vector<int> colblksizes) {
	
	auto w = m_mat_in->get_world();
	
	auto out = dbcsr::scalapack_to_matrix(*m_L, "Inc. Chol. Decom. of " + m_mat_in->name(), 
		w, rowblksizes, colblksizes);
		
	m_L->release();
	
	//util::plot(out, 1e-4);
	
	return out;
	
}
	
}
