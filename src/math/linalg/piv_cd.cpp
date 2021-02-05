 #include "math/linalg/piv_cd.h"
 #include <cmath>
 #include <limits>
 #include <numeric>
 #include <algorithm>
#include <dbcsr_matrix_ops.hpp>
 
#include "extern/scalapack.h"

#include "utils/matrix_plot.h"

namespace math {
	
#ifndef _USE_SPARSE_COMPUTE
	 
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
					
	m_L = std::make_shared<scalapack::distmat<double>>(N,m_rank,nb,nb,0,0);
	
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
			if (ele > max_U_diag) {
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
		
		U.print();
		
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
		
		LOG.os<>("LITTLE U\n");
		u_i.print();
		
		c_pdlapiv('F', 'R', 'C', N-I-1, 1, u_i.data(), I+1, 0, u_i.desc().data(), ipiv_r, I+1, 0, desc_r, iwork);
		
		u_i.print();
		
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
	
	L.print();
	
	exit(0);
	
	c_pdlapiv('B', 'R', 'C', N, N, L.data(), 0, 0, L.desc().data(), ipiv_r, 0, 0, desc_r, iwork);
	
	//L.print();
	
	//c_pdlapiv('B', 'C', 'C', N, N, Uc.data(), 0, 0, Uc.desc().data(), Pcol.data(), 0, 0, Pcol.desc().data(), iwork);
	//c_pdlapiv('B', 'R', 'R', N, N, Uc.data(), 0, 0, Uc.desc().data(), Prow.data(), 0, 0, Prow.desc().data(), iwork);
	
	//c_pdgeadd('N', N, N, 1.0, Ucopy.data(), 0, 0, Ucopy.desc().data(), -1.0, Uc.data(), 0, 0, Uc.desc().data());

	reorder_and_reduce(L);
	
	c_pdgemm('N', 'T', N, N, m_rank, 1.0, m_L->data(), 0, 0, m_L->desc().data(), 
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

#endif

template <typename T>
class xmatrix : public dbcsr::matrix<T> {
private:

	int _nb, _mb, _nblk_row, _nblk_col;
	int _N, _M;
	int _nproc_row, _nproc_col;
	int _rank, _prow, _pcol;
	
	std::vector<int> _rblksizes, _cblksizes;
	
public:
	
	xmatrix(dbcsr::matrix<T>&& mat_in) :
		dbcsr::matrix<T>(std::forward<dbcsr::matrix<T>>(mat_in))
	{
		_N = this->nfullrows_total();
		_M = this->nfullcols_total();
		
		_rblksizes = this->row_blk_sizes();
		_cblksizes = this->col_blk_sizes();
		
		_nb = _rblksizes[0];
		_mb = _cblksizes[0];
		
		auto w = this->get_world();
		
		_rank =w.rank();
		_nproc_row = w.dims()[0];
		_nproc_col = w.dims()[1];
		_prow = w.myprow();
		_pcol = w.mypcol();
		
		_nblk_row = _rblksizes.size();
		_nblk_col = _cblksizes.size();
		
	}

	static std::shared_ptr<xmatrix> create(dbcsr::world wrd, int N, int M, int nb, int mb) 
	{
		
		auto distrvec = dbcsr::split_range(N, nb);
		auto distcvec = dbcsr::split_range(M, mb);
		
		auto rdist = dbcsr::cyclic_dist(distrvec.size(),wrd.dims()[0]);
		auto cdist = dbcsr::cyclic_dist(distcvec.size(),wrd.dims()[1]);

		dbcsr::dist mdist = dbcsr::dist::create()
			.set_world(wrd)
			.row_dist(rdist)
			.col_dist(cdist);
				
		auto mat = dbcsr::create<T>()
			.name("submatrix")
			.set_dist(mdist)
			.row_blk_sizes(distrvec)
			.col_blk_sizes(distcvec)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
				
		auto out = std::make_shared<xmatrix<T>>(std::move(*mat));
		return out;
		
	}
	
	inline int blkidx_row(int i) {
		return i / _nb;
	}
	
	inline int blkidx_col(int j) {
		return j / _mb;
	}
	
	inline int proc_row(int iblk) {
		return iblk % _nproc_row;
	}
	
	inline int proc_col(int iblk) {
		return iblk % _nproc_col;
	}
	
	inline int blksize_row(int iblk) {
		return _rblksizes[iblk];
	} 
	
	inline int blksize_col(int iblk) {
		return _cblksizes[iblk];
	}
	
	void get_row(xmatrix& vec, int i) {
				
		vec.clear();
		vec.reserve_all();
		vec.replicate_all();
		
		int irblk = blkidx_row(i);
				
		for (int icblk = 0; icblk != _nblk_col; ++icblk) {		
			if (_prow == proc_row(irblk) && _pcol == proc_col(icblk)) {
			
				bool found = false;
				auto blk_mat = this->get_block_p(irblk,icblk,found);
				
				if (!found) continue;
				
				auto blk_vec = vec.get_block_p(0,icblk,found);
				
				int roff = irblk * _nb;
				
				for (int ic = 0; ic != blksize_col(icblk); ++ic) {
					blk_vec(0, ic) = blk_mat(i - roff, ic);
				}
			}
		}
		
		vec.sum_replicated();
		vec.distribute();
		
		//dbcsr::print(vec);
		
	}
	
	void get_col(xmatrix& vec, int i) {
				
		vec.clear();
		vec.reserve_all();
		vec.replicate_all();
		
		int icblk = blkidx_col(i);
				
		for (int irblk = 0; irblk != _nblk_row; ++irblk) {		
			if (_prow == proc_row(irblk) && _pcol == proc_col(icblk)) {
			
				bool found = false;
				auto blk_mat = this->get_block_p(irblk,icblk,found);
				
				if (!found) continue;
				
				auto blk_vec = vec.get_block_p(irblk,0,found);
				
				int coff = icblk * _mb;
				
				for (int ir = 0; ir != blksize_row(irblk); ++ir) {
					blk_vec(ir,0) = blk_mat(ir,i-coff);
				}
			}
		}
		
		vec.sum_replicated();
		vec.distribute();
		
		//dbcsr::print(vec);
		
	}
	
	void overwrite_rowvec(xmatrix& vec, int i) {
			
		this->clear();
		vec.replicate_all();
		
		int irblk = blkidx_row(i);
		
		std::vector<int> resrow, rescol;
		
		for (int icblk = 0; icblk != _nblk_col; ++icblk) {
			if (_prow == proc_row(irblk) && _pcol == proc_col(icblk)) {
				
				bool found = false;
				auto blkvec = vec.get_block_p(0,icblk,found);
				
				if (!found) continue;
				
				resrow.push_back(irblk);
				rescol.push_back(icblk);
				
			}
		}
				
		this->reserve_blocks(resrow, rescol);
		int roff = irblk * _nb;
				
		for (auto icblk : rescol) {
			
			bool found = true;
			auto blkmat = this->get_block_p(irblk,icblk,found);
			auto blkvec = vec.get_block_p(0,icblk,found);
				
			for (int ic = 0; ic != blksize_col(icblk); ++ic) {
				blkmat(i - roff, ic) = blkvec(0,ic);
			}
		}
		
		vec.distribute();
		
		//dbcsr::print(*this);
		
	}
	
	void overwrite_colvec(xmatrix& vec, int i) {
			
		this->clear();
		vec.replicate_all();
		
		int icblk = blkidx_col(i);
		
		std::vector<int> resrow, rescol;
				
		for (int irblk = 0; irblk != _nblk_row; ++irblk) {
			if (_prow == proc_row(irblk) && _pcol == proc_col(icblk)) {
				
				bool found = false;
				auto blkvec = vec.get_block_p(irblk,0,found);
				
				if (!found) continue;
				
				resrow.push_back(irblk);
				rescol.push_back(icblk);
				
			}
		}
		
		this->reserve_blocks(resrow, rescol);
		int coff = icblk * _mb;
				
		for (auto irblk : resrow) {
			
			bool found = true;
			auto blkmat = this->get_block_p(irblk,icblk,found);
			auto blkvec = vec.get_block_p(irblk,0,found);
				
			for (int ir = 0; ir != blksize_row(irblk); ++ir) {
				blkmat(ir, i - coff) = blkvec(ir,0);
			}
		}
		
		vec.distribute();
		
	}
	
	void set(int i, int j, T val) {
	
		int irblk = blkidx_row(i);
		int icblk = blkidx_col(j);
		
		if (_prow == proc_row(irblk) && _pcol == proc_col(icblk)) {
			
			bool found = false;
			
			auto blk = this->get_block_p(irblk,icblk,found);
			
			if (!found) {
				std::vector<int> rres = {irblk};
				std::vector<int> cres = {icblk};
				this->reserve_blocks(rres,cres);
				blk = this->get_block_p(irblk,icblk,found);
			}
			
			int roff = irblk * _nb;
			int coff = icblk * _mb;
			
			blk(i-roff,j-coff) = val;
			
		}
		
	}
		
};

template <typename T> 
using shared_xmatrix = std::shared_ptr<xmatrix<T>>;

void permute_rows(xmatrix<double>& mat, 
	xmatrix<double>& rowvec0, xmatrix<double>& rowvec1,
	xmatrix<double>& temp0, xmatrix<double>& temp1,
	int i, int j)
{
	
	rowvec0.clear();
	rowvec1.clear();
	temp0.clear();
	temp1.clear();

	mat.get_row(rowvec0, i);
	mat.get_row(rowvec1, j);
		
	temp0.overwrite_rowvec(rowvec0, i);
	temp1.add(1.0, -1.0, temp0);
	temp0.overwrite_rowvec(rowvec0, j);
	temp1.add(1.0, 1.0, temp0);

	temp0.overwrite_rowvec(rowvec1, j);
	temp1.add(1.0, -1.0, temp0);
	temp0.overwrite_rowvec(rowvec1, i);
	temp1.add(1.0, 1.0, temp0);
	
	mat.add(1.0, 1.0, temp1);
		
	rowvec0.clear();
	rowvec1.clear();
	temp0.clear();
	temp1.clear();
}

void permute_cols(xmatrix<double>& mat, 
	xmatrix<double>& colvec0, xmatrix<double>& colvec1,
	xmatrix<double>& temp0, xmatrix<double>& temp1,
	int i, int j)
{
		
	colvec0.clear();
	colvec1.clear();
	temp0.clear();
	temp1.clear();

	mat.get_col(colvec0, i);
	
	mat.get_col(colvec1, j);
		
	temp0.overwrite_colvec(colvec0, i);
	temp1.add(1.0, -1.0, temp0);
	temp0.overwrite_colvec(colvec0, j);
	temp1.add(1.0, 1.0, temp0);

	temp0.overwrite_colvec(colvec1, j);
	temp1.add(1.0, -1.0, temp0);
	temp0.overwrite_colvec(colvec1, i);
	temp1.add(1.0, 1.0, temp0);
	
	mat.add(1.0, 1.0, temp1);
	
	colvec0.clear();
	colvec1.clear();
	temp0.clear();
	temp1.clear();
	
}

struct alignas(alignof(double)) double_int {
	double d;
	int i;
};

struct alignas(alignof(int)) int_int {
	int i0;
	int i1;
};
		
std::tuple<double,int> get_max_diag(dbcsr::matrix<double>& mat, int istart = 0) {
	auto wrd = mat.get_world();
	 
	auto diag = mat.get_diag();
	auto max = std::max_element(diag.begin() + istart,diag.end());
	int pos = (int)(max - diag.begin());
	
	double_int buf = {*max, wrd.rank()};
	
	MPI_Allreduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE_INT, 
		MPI_MAXLOC, wrd.comm());
				
	MPI_Bcast(&pos, 1, MPI_INT, buf.i, wrd.comm());
		
	return std::make_tuple(buf.d, pos);
	
}

#ifdef _USE_SPARSE_COMPUTE
void pivinc_cd::compute(std::optional<int> force_rank) {
	
	// convert input mat to scalapack format
	
	LOG.os<1>("Starting pivoted incomplete cholesky decomposition.\n");
	
	LOG.os<1>("-- Setting up dbcsr environment and matrices.\n"); 
	
	double filter_eps = dbcsr::global::filter_eps/10;
	
	auto wrd = m_mat_in->get_world();
	
	int nrows = m_mat_in->nfullrows_total();
	int ncols = m_mat_in->nfullcols_total();
	int N = nrows;
	int nb = 4;
	
	auto splitrange = dbcsr::split_range(nrows, 4);
	vec<int> single = {1};
		
	auto U = xmatrix<double>::create(wrd, N, N, nb, nb);
	auto L = xmatrix<double>::create(wrd, N, N, nb, nb);
	
	auto tmp0 = xmatrix<double>::create(wrd, N, N, nb, nb);
	auto tmp1 = xmatrix<double>::create(wrd, N, N, nb, nb);
	
	auto rvec0 = xmatrix<double>::create(wrd, 1, N, 1, nb);
	auto rvec1 = xmatrix<double>::create(wrd, 1, N, 1, nb);
	
	auto cvec0 = xmatrix<double>::create(wrd, N, 1, nb, 1);
	auto cvec1 = xmatrix<double>::create(wrd, N, 1, nb, 1);
		
	L->reserve_all();
	
	auto mat_sym = m_mat_in->desymmetrize();
	U->complete_redistribute(*mat_sym);
	mat_sym->release();
	
	U->filter(filter_eps);
	
	LOG.os<1>("-- Occupation of U: ", U->occupation(), '\n');
	
	//dbcsr::print(*U);
	
	int nblks = U->nblkrows_total();
	
	double thresh = filter_eps;
	LOG.os<1>("-- Threshold: ", thresh, '\n');
			
	std::vector<int> rowperm(N), backperm(N);
	std::iota(rowperm.begin(), rowperm.end(), 0);
	std::iota(backperm.begin(), backperm.end(), 0);
	
	std::function<void(int)> cholesky_step;
	cholesky_step = [&](int I) {
		
		U->filter(filter_eps);
		
		// STEP 1: If Dimension of U is one, then set L and return 
		
		LOG.os<1>("---- Level ", I, '\n');
		
		int i_max;
		double diag_max;
		
		if (force_rank && *force_rank == I) {
			std::tie(diag_max, i_max) = get_max_diag(*U,I);
			
			LOG.os<1>("---- Max rank reached. Max element: ", diag_max, '\n');
			
			L->set(I,I,sqrt(diag_max));
			m_rank = *force_rank;
			return;
		}
		
		if (I == N-1) {
			std::tie(diag_max, i_max) = get_max_diag(*U,I);
			L->set(I,I,sqrt(diag_max));
			m_rank = I+1;
			return;
		}
				
		// STEP 2.0: Permutation
		
		// a) find maximum diagonal element
		
		std::tie(diag_max, i_max) = get_max_diag(*U,I);
				
		LOG.os<1>("---- Maximum element: ", diag_max, " on pos ", i_max, '\n');
				
		
		// b) permute rows/cols
		// U := P * U * P^t
		
		//std::cout << dbcsr::matrix_to_eigen(*U) << std::endl;
		
		LOG.os<1>("---- Permuting ", I , " with ", i_max, '\n');
		
		if (I != i_max) {
			permute_rows(*U, *rvec0, *rvec1, *tmp0, *tmp1, I, i_max);
			permute_cols(*U, *cvec0, *cvec1, *tmp0, *tmp1, I, i_max);
			std::swap(rowperm[I],rowperm[i_max]);
		}
		
		//std::cout << dbcsr::matrix_to_eigen(*U) << std::endl;
		
		// STEP 3.0: Convergence criterion
		
		LOG.os<1>("---- Checking convergence.\n");
		
		double U_II = diag_max;
		
		if (U_II < 0.0 && fabs(U_II) > thresh) {
			LOG.os<1>("fabs(U_II): ", fabs(U_II), '\n');
			throw std::runtime_error("Negative Pivot element. CD not possible.");
		}
				
		if (fabs(U_II) < thresh) {
			
			m_rank = I;
			
			std::swap(backperm[I],backperm[i_max]);
			
			return;
		}
				
		// STEP 3.1: Form Utilde := sub(U) - u * ut
		
		LOG.os<1>("---- Forming Utilde ", '\n');
	
		// a) get u
		// u_i
		U->get_col(*cvec0, I);
				
		auto ui = xmatrix<double>::create(wrd, N, 1, nb, 1);
		ui->copy_in(*cvec0);
				
		// b) form Utilde
		dbcsr::multiply('N', 'T', *ui, *ui, *U)
			.alpha(-1.0/U_II)
			.beta(1.0)
			.filter_eps(filter_eps)
			.first_row(I+1)
			.first_col(I+1)
			.perform();
		
		// STEP 3.2: Solve P * Utilde * Pt = L * Lt
		
		LOG.os<1>("---- Start decomposition of submatrix of dimension ", N-I-1, '\n');
		cholesky_step(I+1);
		
		// STEP 3.3: Form L
		// (a) diagonal element
		
		L->set(I,I,sqrt(U_II));
		
		// (b) permute u_i
		
		LOG.os<1>("---- Permuting u_i", '\n');
		
		auto eigencol = dbcsr::matrix_to_eigen(*ui);
		auto eigencolperm = eigencol;
		
		for (int i = 0; i != I+1; ++i) {
			eigencolperm(i,0) = 0;
			eigencol(i,0) = 0;
		}
		
		//std::cout << eigencolperm << '\n' << std::endl;
		
		for (int i = I+1; i != N; ++i) {
			eigencolperm(backperm[i],0) = eigencol(i,0);
		}
		
		//std::cout << eigencolperm << '\n' << std::endl;
		
		auto colvecperm = dbcsr::eigen_to_matrix(eigencolperm, wrd, "colperm", 
			splitrange, single, dbcsr::type::no_symmetry);
		colvecperm->filter(dbcsr::global::filter_eps);
		
		cvec0->clear();
		cvec0->copy_in(*colvecperm);
		
		tmp0->overwrite_colvec(*cvec0, I);
		
		L->add(1.0, 1.0/sqrt(U_II), *tmp0);
		
		std::swap(backperm[I],backperm[i_max]);
		
	};
	
	cholesky_step(0);
	
	LOG.os<1>("-- Returned from recursion.\n");
	
	LOG.os<1>("-- Permuting rows of L\n");
	
	//auto Leigen = dbcsr::matrix_to_eigen(L);
	//if (wrd.rank() == 0) std::cout << Leigen << std::endl;
	
	auto L_reo0 = xmatrix<double>::create(wrd, N, N, nb, nb);
	auto L_reo1 = xmatrix<double>::create(wrd, N, N, nb, nb);
	
	// reorder rows to restore initial order
	for (int irow = 0; irow != N; ++irow) {
		
		L->get_row(*rvec0, irow);
		
		tmp0->overwrite_rowvec(*rvec0, rowperm[irow]);
		L_reo0->add(1.0, 1.0, *tmp0);
		
		rvec0->clear();
		tmp0->clear();
		
	}
	
	//L_reo0->filter(filter_eps);
	
	// reorder columns to minimize matrix bandwidth 
	
	std::vector<double_int> startpos(m_rank, 
		{std::numeric_limits<double>::max(),N-1});
	std::vector<double_int> stoppos(m_rank, {0.0,0});
	
	dbcsr::iterator<double> iter(*L_reo0);
	iter.start();
	
	double T = dbcsr::global::filter_eps;
	
	while (iter.blocks_left()) {
		iter.next_block();
		
		int rsize = iter.row_size();
		int csize = iter.col_size();
		
		int roff = iter.row_offset();
		int coff = iter.col_offset();
		
		for (int ir = 0; ir != rsize; ++ir) {
			for (int ic = 0; ic != csize; ++ic) {
				
				double val = fabs(iter(ir,ic));
				
				int irow = ir + roff;
				int icol = ic + coff;
				
				if (icol > m_rank-1) continue;
				
				double prevval = startpos[icol].d;
				int prevrow = startpos[icol].i;
				
				if (val > T && irow <= prevrow) {
					startpos[icol].d = val;
					startpos[icol].i = irow;
				}
				
				prevval = stoppos[icol].d;
				prevrow = stoppos[icol].i;
				
				if (val > T && irow >= prevrow) {
					stoppos[icol].d = val;
					stoppos[icol].i = irow;
				} 
			}
		}
		
	}
	
	iter.stop();
	
	/*for (int ip = 0; ip != wrd.size(); ++ip) {
		if (ip == wrd.rank()) {
			int pos = 0;
			for (auto p : startpos) {
				std::cout << pos++ << " " << p.d << " " << p.i << std::endl;
			} std::cout << std::endl;	
			pos = 0;			
			for (auto p : stoppos) {
				std::cout << pos++ << " " << p.d << " " << p.i << std::endl;
			} std::cout << std::endl;
		}
		MPI_Barrier(wrd.comm());
	}*/
			
	
	std::vector<int> startpos_comm(m_rank), stoppos_comm(m_rank),
		startpos_red(m_rank), stoppos_red(m_rank);
	for (int i = 0; i != m_rank; ++i) {
		startpos_comm[i] = startpos[i].i;
		stoppos_comm[i] = stoppos[i].i;
	}
	
	MPI_Reduce(startpos_comm.data(), startpos_red.data(), m_rank, MPI_INT,
		MPI_MIN, 0, wrd.comm());
		
	MPI_Reduce(stoppos_comm.data(), stoppos_red.data(), m_rank, MPI_INT,
		MPI_MAX, 0, wrd.comm());

	std::vector<int> lmo_perm(m_rank,0);

	if (wrd.rank() == 0) {
		
		std::iota(lmo_perm.begin(), lmo_perm.end(), 0);
		std::vector<double> lmo_pos(m_rank);
		
		for (int i = 0; i != m_rank; ++i) {
			
			if (startpos_red[i] == N-1 && stoppos_red[i] == 0) {
				//throw std::runtime_error("Cholesky-reoredering failed.");
				lmo_pos[i] = N-1;
			} else {
				lmo_pos[i] = (double)(startpos_red[i] + stoppos_red[i]) / 2.0;
			}
	
		}
		
		std::stable_sort(lmo_perm.begin(), lmo_perm.end(), 
			[&lmo_pos](int i1, int i2) { 
				return lmo_pos[i1] < lmo_pos[i2]; 
			});
			
		LOG.os<1>("-- Reordered LMO indices: \n");
		for (int i = 0; i != lmo_perm.size(); ++i) {
			LOG.os<1>(lmo_perm[i], " ", lmo_pos[lmo_perm[i]], "\n");
		} LOG.os<1>('\n');
		
	}
	
	MPI_Bcast(lmo_perm.data(), m_rank, MPI_INT, 0, wrd.comm());
		
	// reorder rows to restore initial order
	for (int icol = 0; icol != m_rank; ++icol) {
		
		//std::cout << "PUTTING: " << icol << " INTO " << lmo_perm[icol] << std::endl;
		
		L_reo0->get_col(*cvec0, lmo_perm[icol]);
		
		tmp0->overwrite_colvec(*cvec0, icol);
		
		L_reo1->add(1.0, 1.0, *tmp0);
		
		cvec0->clear();
		tmp0->clear();
		
	}
	
	//auto eigenreo0 = dbcsr::matrix_to_eigen(*L_reo0);
	//if (wrd.rank() == 0) std::cout << eigenreo0 << std::endl;
	
	//util::plot(L_reo0, 1e-5, "unordered");
	//util::plot(L_reo1, 1e-5, "ordered");
	
	//auto eigenreo1 = dbcsr::matrix_to_eigen(*L_reo1);
	//if (wrd.rank() == 0) std::cout << eigenreo1 << std::endl;
	
	//MPI_Barrier(wrd.comm());
	//exit(0);
	
	L_reo0->clear();
	
	//auto Leigenreo = dbcsr::matrix_to_eigen(L_reo);
	//auto eigenreo1 = dbcsr::matrix_to_eigen(*L_reo1);
	//if (wrd.rank() == 0) std::cout << eigenreo1 << std::endl;
	
#if 1
	auto matrowblksizes = m_mat_in->row_blk_sizes();
	
	auto rvec = dbcsr::split_range(nrows, 8);
	
	auto Lredist = dbcsr::create<double>()
		.set_world(wrd)
		.name("Cholesky decomposition")
		.row_blk_sizes(matrowblksizes)
		.col_blk_sizes(rvec)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	LOG.os<1>("-- Redistributing L\n");
		
	Lredist->complete_redistribute(*L_reo1);
	Lredist->filter(dbcsr::global::filter_eps);
	
	auto mat_copy = dbcsr::copy(*m_mat_in).get();
	dbcsr::multiply('N', 'T', *Lredist, *Lredist, *mat_copy)
		.alpha(-1.0)
		.beta(1.0)
		.filter_eps(dbcsr::global::filter_eps)
		.perform();
	
	Lredist->clear();
	LOG.os<1>("-- Cholesky error: ", mat_copy->norm(dbcsr_norm_frobenius), '\n');
	mat_copy->clear();	
#endif

	m_L = std::make_shared<dbcsr::matrix<double>>(std::move(*L_reo1));
	
	//exit(0);

	LOG.os<1>("Finished decomposition.\n");
	
		
}

dbcsr::smat_d pivinc_cd::L(std::vector<int> rowblksizes, std::vector<int> colblksizes) {
	
	static int i = 0;
	i++;
	
	auto wrd = m_L->get_world();
	
	auto Lredist = dbcsr::create<double>()
		.set_world(wrd)
		.name("Cholesky decomposition")
		.row_blk_sizes(rowblksizes)
		.col_blk_sizes(colblksizes)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	LOG.os<1>("-- Redistributing L\n");
		
	Lredist->complete_redistribute(*m_L);
	Lredist->filter(dbcsr::global::filter_eps);
			
	return Lredist;
	
}

void reorder_and_reduce(scalapack::distmat<double>& L) {}

#endif
	
}
