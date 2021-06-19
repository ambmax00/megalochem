#include "math/linalg/piv_cd.hpp"
#include <algorithm>
#include <cmath>
#include <dbcsr_matrix_ops.hpp>
#include <limits>
#include <numeric>

#include "extern/scalapack.hpp"

#include "utils/matrix_plot.hpp"

namespace megalochem {

namespace math {
  
struct alignas(alignof(double)) double_int {
  double d;
  int i;
};

struct alignas(alignof(int)) int_int {
  int i0;
  int i1;
};

#ifndef _USE_SPARSE_COMPUTE

void pivinc_cd::reorder_and_reduce(scalapack::distmat<double>& L)
{
  
  auto sgrid = m_world.scalapack_grid();
  int N = L.nrowstot();
  
  std::vector<double_int> startpos(
      m_rank, {std::numeric_limits<double>::max(), N - 1});
      
  std::vector<double_int> stoppos(m_rank, {0.0, 0});
  double eps = dbcsr::global::filter_eps;
  
  // loop over matrix elements to find start and end positions for my process
  for (int icol = 0; icol != m_rank; ++icol) {
	  if (sgrid.mypcol() != L.jproc(icol)) continue;
	  for (int irow = 0; irow != N; ++irow) {
		  if (sgrid.myprow() != L.iproc(irow)) continue;
		  
      double val = L.global_access(irow,icol);
		  int prevrow = startpos[icol].i;
      

		  if (val > eps && irow <= prevrow) {
			  startpos[icol].d = val;
			  startpos[icol].i = irow;
		  }

			prevrow = stoppos[icol].i;

      if (val > eps && irow >= prevrow) {
        stoppos[icol].d = val;
        stoppos[icol].i = irow;
      }
      
    } // end for irow
  } // end for icol

  // Prepare to reduce over the communicator
  std::vector<int> startpos_comm(m_rank), stoppos_comm(m_rank),
      startpos_red(m_rank), stoppos_red(m_rank);
  for (int i = 0; i != m_rank; ++i) {
    startpos_comm[i] = startpos[i].i;
    stoppos_comm[i] = stoppos[i].i;
  }

  MPI_Reduce(
      startpos_comm.data(), startpos_red.data(), m_rank, MPI_INT, MPI_MIN, 0,
      m_world.comm());

  MPI_Reduce(
      stoppos_comm.data(), stoppos_red.data(), m_rank, MPI_INT, MPI_MAX, 0,
      m_world.comm());

  // reorder on rank 0
  std::vector<int> lmo_perm(m_rank, 0);

  if (m_world.rank() == 0) {
    std::iota(lmo_perm.begin(), lmo_perm.end(), 0);
    std::vector<double> lmo_pos(m_rank);

    for (int i = 0; i != m_rank; ++i) {
      if (startpos_red[i] == N - 1 && stoppos_red[i] == 0) {
        lmo_pos[i] = N - 1;
      }
      else {
        lmo_pos[i] = (double)(startpos_red[i] + stoppos_red[i]) / 2.0;
      }
    }

    std::stable_sort(
        lmo_perm.begin(), lmo_perm.end(), [&lmo_pos](int i1, int i2) {
          return lmo_pos[i1] < lmo_pos[i2];
        });

    LOG.os<1>("-- Reordered LMO indices: \n");
    for (size_t i = 0; i != lmo_perm.size(); ++i) {
      LOG.os<1>(lmo_perm[i], " ", lmo_pos[lmo_perm[i]], "\n");
    }
    LOG.os<1>('\n');
  }

  // broadcast permutation
  MPI_Bcast(lmo_perm.data(), m_rank, MPI_INT, 0, m_world.comm());

  // create a new L matrix 
  auto L_reo = scalapack::distmat<double>(sgrid, N, m_rank, L.rowblk_size(),
    L.colblk_size(), 0, 0);
    
  for (int icol = 0; icol != m_rank; ++icol) {
    c_pdgeadd('N', N, 1, 1.0, L.data(), 0, lmo_perm[icol], L.desc().data(), 
      0.0, L_reo.data(), 0, icol, L_reo.desc().data());
  }
  
  L = std::move(L_reo);

}

void pivinc_cd::compute(std::optional<int> force_rank, std::optional<double> eps)
{
  // convert input mat to scalapack format

  LOG.os<1>("Starting pivoted incomplete cholesky decomposition.\n");

  LOG.os<1>("-- Setting up scalapack environment and matrices.\n");

  util::mpi_time TIME(m_world.comm(), "CHOLESKY");
  //auto& time_reo = TIME.sub("REO");
  //auto& time_calc = TIME.sub("CALC");
  //auto& time_reol = TIME.sub("reol");
  
  //TIME.start();

  int N = m_mat_in->nfullrows_total();
  int* iwork = new int[N];
  int nb = 16;

  auto sgrid = m_world.scalapack_grid();

  MPI_Comm comm = m_world.comm();
  int myrank = m_world.rank();

  int ori_proc = m_mat_in->proc(0, 0);
  int ori_coord[2];

  if (myrank == ori_proc) {
    ori_coord[0] = sgrid.myprow();
    ori_coord[1] = sgrid.mypcol();
  }

  MPI_Bcast(&ori_coord[0], 2, MPI_INT, ori_proc, comm);

  // util::plot(m_mat_in, 1e-4);

  scalapack::distmat<double> U = dbcsr::matrix_to_scalapack(
      m_mat_in, sgrid, nb, nb, ori_coord[0],
      ori_coord[1]);

  scalapack::distmat<double> Ucopy(sgrid, N, N, nb, nb, 0, 0);

  c_pdgeadd(
      'N', N, N, 1.0, U.data(), 0, 0, U.desc().data(), 0.0, Ucopy.data(), 0, 0,
      Ucopy.desc().data());

  int LOCr = c_numroc(N, nb, sgrid.myprow(), 0, sgrid.nprow());
  int LOCc = c_numroc(N, nb, sgrid.mypcol(), 0, sgrid.npcol());

  int lipiv_r = LOCr + nb;
  int lipiv_c = LOCc + nb;

  int* ipiv_r = new int[lipiv_r];  // column vector distributed over rows
  int* ipiv_c = new int[lipiv_c];  // row vector distributed over cols

  int desc_r[9];
  int desc_c[9];

  int info = 0;
  c_descinit(
      &desc_r[0], N + nb * sgrid.nprow(), 1, nb, nb, 0, 0, sgrid.ctx(), lipiv_r,
      &info);
  c_descinit(
      &desc_c[0], 1, N + nb * sgrid.npcol(), nb, nb, 0, 0, sgrid.ctx(), 1,
      &info);

  // vector to keep track of permutations (scalapack)
  std::vector<int> perms(N);
  std::iota(perms.begin(), perms.end(), 1);
  
  // m_perms
  m_perm.resize(N);
  std::iota(m_perm.begin(), m_perm.end(), 0);

  // chol mat
  scalapack::distmat<double> L(sgrid, N, N, nb, nb, 0, 0);

  // get max diag element
  auto get_max_diag = [&](int I) {
    
    double local_max = 0.0;
    int local_idx = 0;
    
    for (int ii = I; ii != N; ++ii) {
      if (sgrid.myprow() == U.iproc(ii) && sgrid.mypcol() == U.jproc(ii)) {
        double val = U.global_access(ii,ii);
        if (fabs(val) > fabs(local_max)) {
           local_max = val;
           local_idx = ii;
        }
      }
    }
    
    double_int di_loc;
    double_int di_glob;
    
    di_loc.i = m_world.rank();
    di_loc.d = local_max;
    
    MPI_Allreduce(&di_loc, &di_glob, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
      m_world.comm());
    
    int maxrank = di_glob.i;
    di_glob.i = local_idx;
    
    MPI_Bcast(&di_glob, 1, MPI_DOUBLE_INT, maxrank, m_world.comm());
    
    return std::tie(di_glob.d, di_glob.i);
  };
     
  /*double max_U_diag_global = 0.0;
  for (int ix = 0; ix != N; ++ix) {
    max_U_diag_global =
        std::max(fabs(max_U_diag_global), fabs(U.get('A', ' ', ix, ix)));
  }*/
  auto [max_val_global, max_idx_global] = get_max_diag(0); 

  LOG.os<1>("-- Problem size: ", N, '\n');
  LOG.os<1>(
      "-- Maximum diagonal element of input matrix: ", max_val_global, '\n');

  double thresh = (eps) ? *eps : N * std::numeric_limits<double>::epsilon() * max_val_global;

  LOG.os<1>("-- Threshold: ", thresh, '\n');

  std::function<void(int)> cd_step;
  cd_step = [&](int I) {
    // STEP 1: If Dimension of U is one, then set L and return

    LOG.os<1>("---- Level ", I, '\n');
    
    auto [max_U_diag, max_U_idx] = get_max_diag(I);

    /*for (int ix = I; ix != N; ++ix) {
      double ele = U.get('A', ' ', ix, ix);
      if (ele > max_U_diag) {
        max_U_diag = ele;
        max_U_idx = ix;
      }
    }*/

    if (I == N - 1) {
      double U_II = U.get('A', ' ', I, I);
      L.set(I, I, sqrt(U_II));
      m_rank = I + 1;
      return;
    }

    // STEP 2.0: Permutation

    LOG.os<1>("---- MAX ", max_U_diag, " @ ", max_U_idx, '\n');

    // b) permute rows/cols
    // U := P * U * P^t
    //time_reo.start();
    LOG.os<1>("---- Permuting U.\n");
    /*
    for (int ix = I; ix != N; ++ix) {
            Pcol.set(ix,0,ix+1);
            Prow.set(0,ix,ix+1);
    }*/
    for (int ir = 0; ir != LOCr; ++ir) { ipiv_r[ir] = U.iglob(ir) + 1; }
    for (int ic = 0; ic != LOCc; ++ic) { ipiv_c[ic] = U.jglob(ic) + 1; }

    // Pcol.set(I,0,max_U_idx + 1);
    // Prow.set(0,I,max_U_idx + 1);

    if (sgrid.myprow() == U.iproc(I)) {
      ipiv_r[U.iloc(I)] = max_U_idx + 1;
    }
    if (sgrid.mypcol() == U.jproc(I)) {
      ipiv_c[U.jloc(I)] = max_U_idx + 1;
    }

    c_pdlapiv(
        'F', 'R', 'C', N - I, N - I, U.data(), I, I, U.desc().data(), ipiv_r, I,
        0, desc_r, iwork);
    c_pdlapiv(
        'F', 'C', 'R', N - I, N - I, U.data(), I, I, U.desc().data(), ipiv_c, 0,
        I, desc_c, iwork);
        
    std::swap(m_perm[I], m_perm[max_U_idx]);
    //time_reo.finish();
    //U.print();

    // STEP 3.0: Convergence criterion

    LOG.os<1>("---- Checking convergence.\n");

    double U_II = U.get('A', ' ', I, I);

    if (U_II < 0.0 && fabs(U_II) > thresh) {
      LOG.os<1>("fabs(U_II): ", fabs(U_II), '\n');
      throw std::runtime_error("Negative Pivot element. CD not possible.");
    }

    if (fabs(U_II) < thresh) {
      LOG.os<1>("Pivot element below threshold.\n");
      perms[I] = max_U_idx + 1;
      m_rank = I;
      return;
    } else if (force_rank && *force_rank == I) {
      LOG.os<1>("Max rank reached.\n");
      perms[I] = max_U_idx + 1;
      m_rank = I;
      return;
    }

    // STEP 3.1: Form Utilde := sub(U) - u * ut

    // a) get u
    // u_i
    //time_calc.start();
    scalapack::distmat<double> u_i(sgrid, N, 1, nb, nb, 0, 0);
    c_pdgeadd(
        'N', N - I - 1, 1, 1.0, U.data(), I + 1, I, U.desc().data(), 0.0,
        u_i.data(), I + 1, 0, u_i.desc().data());

    // b) form Utilde
    /*c_pdgemm(
        'N', 'T', N - I - 1, N - I - 1, 1, -1 / U_II, u_i.data(), I + 1, 0,
        u_i.desc().data(), u_i.data(), I + 1, 0, u_i.desc().data(), 1.0,
        U.data(), I + 1, I + 1, U.desc().data());*/
    
    c_pdger(N - I - 1, N - I - 1, -1.0/U_II, u_i.data(), I+1, 0, u_i.desc().data(),
      1, u_i.data(), I+1, 0, u_i.desc().data(), 1, U.data(), I+1, I+1, U.desc().data());
        
    //time_calc.finish();
    // STEP 3.2: Solve P * Utilde * Pt = L * Lt

    LOG.os<1>(
        "---- Start decomposition of submatrix of dimension ", N - I - 1, '\n');
    cd_step(I + 1);

    // STEP 3.3: Form L
    // (a) diagonal element

    L.set(I, I, sqrt(U_II));

    // (b) permute u_i
    // for (int ix = I; ix != N; ++ix) {
    //	Pcol.set(ix,0,perms[ix]);
    //}

    for (int ir = 0; ir != LOCr; ++ir) { ipiv_r[ir] = perms[U.iglob(ir)]; }

    // printp(ipiv_r,LOCr);

    //LOG.os<>("LITTLE U\n");
    //u_i.print();

    c_pdlapiv(
        'F', 'R', 'C', N - I - 1, 1, u_i.data(), I + 1, 0, u_i.desc().data(),
        ipiv_r, I + 1, 0, desc_r, iwork);

    //u_i.print();

    // (c) add u_i to L

    // L.print();

    c_pdgeadd(
        'N', N - I - 1, 1, 1.0 / sqrt(U_II), u_i.data(), I + 1, 0,
        u_i.desc().data(), 0.0, L.data(), I + 1, I, L.desc().data());

    // L.print();

    perms[I] = max_U_idx + 1;

    return;
  };

  LOG.os<1>("-- Starting recursive decomposition.\n");
  cd_step(0);
  
  //time_reol.start();
  LOG.os<1>("-- Rank of L: ", m_rank, '\n');

  for (int ir = 0; ir != LOCr; ++ir) { ipiv_r[ir] = perms[U.iglob(ir)]; }

  // printp(ipiv_r,LOCr);

  LOG.os<1>("-- Permuting L.\n");

  //L.print();

  // permute rows of L back to original order
  c_pdlapiv(
      'B', 'R', 'C', N, N, L.data(), 0, 0, L.desc().data(), ipiv_r, 0, 0,
      desc_r, iwork);

  reorder_and_reduce(L);
  m_L = std::make_shared<decltype(L)>(std::move(L));
  //time_reol.finish();
  
  //TIME.finish();
  //TIME.print_info();
     
  c_pdgemm(
      'N', 'T', N, N, m_rank, 1.0, m_L->data(), 0, 0, m_L->desc().data(),
      m_L->data(), 0, 0, m_L->desc().data(), -1.0, Ucopy.data(), 0, 0,
      Ucopy.desc().data());

  double err =
      c_pdlange('F', N, N, Ucopy.data(), 0, 0, Ucopy.desc().data(), nullptr);
      
  LOG.os<1>("-- CD error: ", err, '\n');

  LOG.os<1>("Finished decomposition.\n");

  delete[] iwork;
  delete[] ipiv_r;
  delete[] ipiv_c;
    
}

dbcsr::shared_matrix<double> pivinc_cd::L(
    std::vector<int> rowblksizes, std::vector<int> colblksizes)
{
  auto out = dbcsr::scalapack_to_matrix(
      *m_L, m_world.dbcsr_grid(), "Inc. Chol. Decom. of " + m_mat_in->name(),
      rowblksizes, colblksizes);

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

    auto w = this->get_cart();

    _rank = w.rank();
    _nproc_row = w.dims()[0];
    _nproc_col = w.dims()[1];
    _prow = w.myprow();
    _pcol = w.mypcol();

    _nblk_row = _rblksizes.size();
    _nblk_col = _cblksizes.size();
  }

  static std::shared_ptr<xmatrix> create(
      dbcsr::cart wrd, int N, int M, int nb, int mb)
  {
    auto distrvec = dbcsr::split_range(N, nb);
    auto distcvec = dbcsr::split_range(M, mb);

    auto rdist = dbcsr::cyclic_dist(distrvec.size(), wrd.dims()[0]);
    auto cdist = dbcsr::cyclic_dist(distcvec.size(), wrd.dims()[1]);

    auto mdist = dbcsr::dist::create()
                     .set_cart(wrd)
                     .row_dist(rdist)
                     .col_dist(cdist)
                     .build();

    auto mat = dbcsr::matrix<T>::create()
                   .name("submatrix")
                   .set_dist(*mdist)
                   .row_blk_sizes(distrvec)
                   .col_blk_sizes(distcvec)
                   .matrix_type(dbcsr::type::no_symmetry)
                   .build();

    auto out = std::make_shared<xmatrix<T>>(std::move(*mat));
    return out;
  }

  inline int blkidx_row(int i)
  {
    return i / _nb;
  }

  inline int blkidx_col(int j)
  {
    return j / _mb;
  }

  inline int proc_row(int iblk)
  {
    return iblk % _nproc_row;
  }

  inline int proc_col(int iblk)
  {
    return iblk % _nproc_col;
  }

  inline int blksize_row(int iblk)
  {
    return _rblksizes[iblk];
  }

  inline int blksize_col(int iblk)
  {
    return _cblksizes[iblk];
  }

  void get_row(xmatrix& vec, int i)
  {
    vec.clear();
    vec.reserve_all();
    vec.replicate_all();

    int irblk = blkidx_row(i);

    for (int icblk = 0; icblk != _nblk_col; ++icblk) {
      if (_prow == proc_row(irblk) && _pcol == proc_col(icblk)) {
        bool found = false;
        auto blk_mat = this->get_block_p(irblk, icblk, found);

        if (!found)
          continue;

        auto blk_vec = vec.get_block_p(0, icblk, found);

        int roff = irblk * _nb;

        for (int ic = 0; ic != blksize_col(icblk); ++ic) {
          blk_vec(0, ic) = blk_mat(i - roff, ic);
        }
      }
    }

    vec.sum_replicated();
    vec.distribute();

    // dbcsr::print(vec);
  }

  void get_col(xmatrix& vec, int i)
  {
    vec.clear();
    vec.reserve_all();
    vec.replicate_all();

    int icblk = blkidx_col(i);

    for (int irblk = 0; irblk != _nblk_row; ++irblk) {
      if (_prow == proc_row(irblk) && _pcol == proc_col(icblk)) {
        bool found = false;
        auto blk_mat = this->get_block_p(irblk, icblk, found);

        if (!found)
          continue;

        auto blk_vec = vec.get_block_p(irblk, 0, found);

        int coff = icblk * _mb;

        for (int ir = 0; ir != blksize_row(irblk); ++ir) {
          blk_vec(ir, 0) = blk_mat(ir, i - coff);
        }
      }
    }

    vec.sum_replicated();
    vec.distribute();

    // dbcsr::print(vec);
  }

  void overwrite_rowvec(xmatrix& vec, int i)
  {
    this->clear();
    vec.replicate_all();

    int irblk = blkidx_row(i);

    std::vector<int> resrow, rescol;

    for (int icblk = 0; icblk != _nblk_col; ++icblk) {
      if (_prow == proc_row(irblk) && _pcol == proc_col(icblk)) {
        bool found = false;
        auto blkvec = vec.get_block_p(0, icblk, found);

        if (!found)
          continue;

        resrow.push_back(irblk);
        rescol.push_back(icblk);
      }
    }

    this->reserve_blocks(resrow, rescol);
    int roff = irblk * _nb;

    for (auto icblk : rescol) {
      bool found = true;
      auto blkmat = this->get_block_p(irblk, icblk, found);
      auto blkvec = vec.get_block_p(0, icblk, found);

      for (int ic = 0; ic != blksize_col(icblk); ++ic) {
        blkmat(i - roff, ic) = blkvec(0, ic);
      }
    }

    vec.distribute();

    // dbcsr::print(*this);
  }

  void overwrite_colvec(xmatrix& vec, int i)
  {
    this->clear();
    vec.replicate_all();

    int icblk = blkidx_col(i);

    std::vector<int> resrow, rescol;

    for (int irblk = 0; irblk != _nblk_row; ++irblk) {
      if (_prow == proc_row(irblk) && _pcol == proc_col(icblk)) {
        bool found = false;
        auto blkvec = vec.get_block_p(irblk, 0, found);

        if (!found)
          continue;

        resrow.push_back(irblk);
        rescol.push_back(icblk);
      }
    }

    this->reserve_blocks(resrow, rescol);
    int coff = icblk * _mb;

    for (auto irblk : resrow) {
      bool found = true;
      auto blkmat = this->get_block_p(irblk, icblk, found);
      auto blkvec = vec.get_block_p(irblk, 0, found);

      for (int ir = 0; ir != blksize_row(irblk); ++ir) {
        blkmat(ir, i - coff) = blkvec(ir, 0);
      }
    }

    vec.distribute();
  }

  void set(int i, int j, T val)
  {
    int irblk = blkidx_row(i);
    int icblk = blkidx_col(j);

    if (_prow == proc_row(irblk) && _pcol == proc_col(icblk)) {
      bool found = false;

      auto blk = this->get_block_p(irblk, icblk, found);

      if (!found) {
        std::vector<int> rres = {irblk};
        std::vector<int> cres = {icblk};
        this->reserve_blocks(rres, cres);
        blk = this->get_block_p(irblk, icblk, found);
      }

      int roff = irblk * _nb;
      int coff = icblk * _mb;

      blk(i - roff, j - coff) = val;
    }
  }
};

template <typename T>
using shared_xmatrix = std::shared_ptr<xmatrix<T>>;

void permute_rows(
    xmatrix<double>& mat,
    xmatrix<double>& rowvec0,
    xmatrix<double>& rowvec1,
    xmatrix<double>& temp0,
    xmatrix<double>& temp1,
    int i,
    int j)
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

void permute_cols(
    xmatrix<double>& mat,
    xmatrix<double>& colvec0,
    xmatrix<double>& colvec1,
    xmatrix<double>& temp0,
    xmatrix<double>& temp1,
    int i,
    int j)
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

std::tuple<double, int> get_max_diag(dbcsr::matrix<double>& mat, int istart = 0)
{
  auto wrd = mat.get_cart();

  auto diag = mat.get_diag();
  auto max = std::max_element(diag.begin() + istart, diag.end());
  int pos = (int)(max - diag.begin());

  double_int buf = {*max, wrd.rank()};

  MPI_Allreduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE_INT, MPI_MAXLOC, wrd.comm());

  MPI_Bcast(&pos, 1, MPI_INT, buf.i, wrd.comm());

  return std::make_tuple(buf.d, pos);
}

#ifdef _USE_SPARSE_COMPUTE
void pivinc_cd::compute(
    std::optional<int> force_rank, std::optional<double> eps)
{
  // convert input mat to scalapack format

  LOG.os<1>("Starting pivoted incomplete cholesky decomposition.\n");

  LOG.os<1>("-- Setting up dbcsr environment and matrices.\n");

  double filter_100 = dbcsr::global::filter_eps / 100;

  auto dcart = m_world.dbcsr_grid();

  int nrows = m_mat_in->nfullrows_total();
  int N = nrows;
  int nb = 4;

  auto splitrange = dbcsr::split_range(nrows, 4);
  vec<int> single = {1};

  auto U = xmatrix<double>::create(dcart, N, N, nb, nb);
  auto L = xmatrix<double>::create(dcart, N, N, nb, nb);

  auto tmp0 = xmatrix<double>::create(dcart, N, N, nb, nb);
  auto tmp1 = xmatrix<double>::create(dcart, N, N, nb, nb);

  auto rvec0 = xmatrix<double>::create(dcart, 1, N, 1, nb);
  auto rvec1 = xmatrix<double>::create(dcart, 1, N, 1, nb);

  auto cvec0 = xmatrix<double>::create(dcart, N, 1, nb, 1);
  auto cvec1 = xmatrix<double>::create(dcart, N, 1, nb, 1);

  L->reserve_all();

  auto mat_sym = m_mat_in->desymmetrize();
  U->complete_redistribute(*mat_sym);
  mat_sym->release();

  U->filter(filter_100);

  LOG.os<1>("-- Occupation of U: ", U->occupation(), '\n');

  // dbcsr::print(*U);

  double thresh = (eps) ? *eps : filter_100;
  LOG.os<1>("-- Threshold: ", thresh, '\n');

  std::vector<int> rowperm(N), backperm(N);
  std::iota(rowperm.begin(), rowperm.end(), 0);
  std::iota(backperm.begin(), backperm.end(), 0);

  std::vector<int> max_pos;

  std::function<void(int)> cholesky_step;
  cholesky_step = [&](int I) {
    U->filter(filter_100);

    // STEP 1: If Dimension of U is one, then set L and return

    LOG.os<1>("---- Level ", I, '\n');

    int i_max;
    double diag_max;

    if (force_rank && *force_rank == I) {
      std::tie(diag_max, i_max) = get_max_diag(*U, I);

      LOG.os<1>("---- Max rank reached. Max element: ", diag_max, '\n');

      L->set(I, I, sqrt(diag_max));
      m_rank = *force_rank;
      return;
    }

    if (I == N - 1) {
      std::tie(diag_max, i_max) = get_max_diag(*U, I);
      L->set(I, I, sqrt(diag_max));
      m_rank = I + 1;
      return;
    }

    // STEP 2.0: Permutation

    // a) find maximum diagonal element

    std::tie(diag_max, i_max) = get_max_diag(*U, I);

    LOG.os<1>("---- Maximum element: ", diag_max, " on pos ", i_max, '\n');

    // b) permute rows/cols
    // U := P * U * P^t

    // std::cout << dbcsr::matrix_to_eigen(*U) << std::endl;

    LOG.os<1>("---- Permuting ", I, " with ", i_max, '\n');

    if (I != i_max) {
      permute_rows(*U, *rvec0, *rvec1, *tmp0, *tmp1, I, i_max);
      permute_cols(*U, *cvec0, *cvec1, *tmp0, *tmp1, I, i_max);
      std::swap(rowperm[I], rowperm[i_max]);
    }

    // std::cout << dbcsr::matrix_to_eigen(*U) << std::endl;

    // STEP 3.0: Convergence criterion

    LOG.os<1>("---- Checking convergence.\n");

    double U_II = diag_max;

    if (U_II < 0.0 && fabs(U_II) > thresh) {
      LOG.os<1>("fabs(U_II): ", fabs(U_II), '\n');
      throw std::runtime_error("Negative Pivot element. CD not possible.");
    }

    if (fabs(U_II) < thresh) {
      m_rank = I;

      std::swap(backperm[I], backperm[i_max]);

      return;
    }

    // STEP 3.1: Form Utilde := sub(U) - u * ut

    LOG.os<1>("---- Forming Utilde ", '\n');

    // a) get u
    // u_i
    U->get_col(*cvec0, I);

    auto ui = xmatrix<double>::create(dcart, N, 1, nb, 1);
    ui->copy_in(*cvec0);

    // b) form Utilde
    dbcsr::multiply('N', 'T', -1.0 / U_II, *ui, *ui, 1.0, *U)
        .filter_eps(filter_100)
        .first_row(I + 1)
        .first_col(I + 1)
        .perform();

    // STEP 3.2: Solve P * Utilde * Pt = L * Lt

    LOG.os<1>(
        "---- Start decomposition of submatrix of dimension ", N - I - 1, '\n');
    cholesky_step(I + 1);

    // STEP 3.3: Form L
    // (a) diagonal element

    L->set(I, I, sqrt(U_II));

    // (b) permute u_i

    LOG.os<1>("---- Permuting u_i", '\n');

    auto eigencol = dbcsr::matrix_to_eigen(*ui);
    auto eigencolperm = eigencol;

    for (int i = 0; i != I + 1; ++i) {
      eigencolperm(i, 0) = 0;
      eigencol(i, 0) = 0;
    }

    // std::cout << eigencolperm << '\n' << std::endl;

    for (int i = I + 1; i != N; ++i) {
      eigencolperm(backperm[i], 0) = eigencol(i, 0);
    }

    // std::cout << eigencolperm << '\n' << std::endl;

    auto colvecperm = dbcsr::eigen_to_matrix(
        eigencolperm, dcart, "colperm", splitrange, single,
        dbcsr::type::no_symmetry);
    colvecperm->filter(filter_100);

    cvec0->clear();
    cvec0->copy_in(*colvecperm);

    tmp0->overwrite_colvec(*cvec0, I);

    L->add(1.0, 1.0 / sqrt(U_II), *tmp0);

    std::swap(backperm[I], backperm[i_max]);
  };

  cholesky_step(0);

  m_perm = rowperm;

  LOG.os<1>("-- Returned from recursion.\n");

  LOG.os<1>("-- Permuting rows of L\n");

  // auto Leigen = dbcsr::matrix_to_eigen(L);
  // if (wrd.rank() == 0) std::cout << Leigen << std::endl;

  auto L_reo0 = xmatrix<double>::create(dcart, N, N, nb, nb);
  auto L_reo1 = xmatrix<double>::create(dcart, N, N, nb, nb);

  // reorder rows to restore initial order
  for (int irow = 0; irow != N; ++irow) {
    L->get_row(*rvec0, irow);

    tmp0->overwrite_rowvec(*rvec0, rowperm[irow]);
    L_reo0->add(1.0, 1.0, *tmp0);

    rvec0->clear();
    tmp0->clear();
  }

  // L_reo0->filter(filter_eps);

  // reorder columns to minimize matrix bandwidth

  std::vector<double_int> startpos(
      m_rank, {std::numeric_limits<double>::max(), N - 1});
  std::vector<double_int> stoppos(m_rank, {0.0, 0});

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
        double val = fabs(iter(ir, ic));

        int irow = ir + roff;
        int icol = ic + coff;

        if (icol > m_rank - 1)
          continue;

        int prevrow = startpos[icol].i;

        if (val > T && irow <= prevrow) {
          startpos[icol].d = val;
          startpos[icol].i = irow;
        }

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
                          std::cout << pos++ << " " << p.d << " " << p.i <<
  std::endl; } std::cout << std::endl; pos = 0; for (auto p : stoppos) {
                          std::cout << pos++ << " " << p.d << " " << p.i <<
  std::endl; } std::cout << std::endl;
          }
          MPI_Barrier(wrd.comm());
  }*/

  std::vector<int> startpos_comm(m_rank), stoppos_comm(m_rank),
      startpos_red(m_rank), stoppos_red(m_rank);
  for (int i = 0; i != m_rank; ++i) {
    startpos_comm[i] = startpos[i].i;
    stoppos_comm[i] = stoppos[i].i;
  }

  MPI_Reduce(
      startpos_comm.data(), startpos_red.data(), m_rank, MPI_INT, MPI_MIN, 0,
      m_world.comm());

  MPI_Reduce(
      stoppos_comm.data(), stoppos_red.data(), m_rank, MPI_INT, MPI_MAX, 0,
      m_world.comm());

  std::vector<int> lmo_perm(m_rank, 0);

  if (m_world.rank() == 0) {
    std::iota(lmo_perm.begin(), lmo_perm.end(), 0);
    std::vector<double> lmo_pos(m_rank);

    for (int i = 0; i != m_rank; ++i) {
      if (startpos_red[i] == N - 1 && stoppos_red[i] == 0) {
        // throw std::runtime_error("Cholesky-reoredering failed.");
        lmo_pos[i] = N - 1;
      }
      else {
        lmo_pos[i] = (double)(startpos_red[i] + stoppos_red[i]) / 2.0;
      }
    }

    std::stable_sort(
        lmo_perm.begin(), lmo_perm.end(), [&lmo_pos](int i1, int i2) {
          return lmo_pos[i1] < lmo_pos[i2];
        });

    LOG.os<1>("-- Reordered LMO indices: \n");
    for (size_t i = 0; i != lmo_perm.size(); ++i) {
      LOG.os<1>(lmo_perm[i], " ", lmo_pos[lmo_perm[i]], "\n");
    }
    LOG.os<1>('\n');
  }

  MPI_Bcast(lmo_perm.data(), m_rank, MPI_INT, 0, m_world.comm());

  // reorder rows to restore initial order
  for (int icol = 0; icol != m_rank; ++icol) {
    // std::cout << "PUTTING: " << icol << " INTO " << lmo_perm[icol] <<
    // std::endl;

    L_reo0->get_col(*cvec0, lmo_perm[icol]);

    tmp0->overwrite_colvec(*cvec0, icol);

    L_reo1->add(1.0, 1.0, *tmp0);

    cvec0->clear();
    tmp0->clear();
  }

  // auto eigenreo0 = dbcsr::matrix_to_eigen(*L_reo0);
  // if (wrd.rank() == 0) std::cout << eigenreo0 << std::endl;

  // util::plot(L_reo0, 1e-5, "unordered");
  // util::plot(L_reo1, 1e-5, "ordered");

  // auto eigenreo1 = dbcsr::matrix_to_eigen(*L_reo1);
  // if (wrd.rank() == 0) std::cout << eigenreo1 << std::endl;

  // MPI_Barrier(wrd.comm());
  // exit(0);

  L_reo0->clear();

  // auto Leigenreo = dbcsr::matrix_to_eigen(L_reo);
  // auto eigenreo1 = dbcsr::matrix_to_eigen(*L_reo1);
  // if (wrd.rank() == 0) std::cout << eigenreo1 << std::endl;

  #if 1
  auto matrowblksizes = m_mat_in->row_blk_sizes();

  auto rvec = dbcsr::split_range(nrows, 8);

  auto Lredist = dbcsr::matrix<double>::create()
                     .set_cart(dcart)
                     .name("Cholesky decomposition")
                     .row_blk_sizes(matrowblksizes)
                     .col_blk_sizes(rvec)
                     .matrix_type(dbcsr::type::no_symmetry)
                     .build();

  LOG.os<1>("-- Redistributing L\n");

  Lredist->complete_redistribute(*L_reo1);
  Lredist->filter(dbcsr::global::filter_eps);

  auto mat_copy = dbcsr::matrix<>::copy(*m_mat_in).build();
  dbcsr::multiply('N', 'T', -1.0, *Lredist, *Lredist, 1.0, *mat_copy)
      .filter_eps(dbcsr::global::filter_eps)
      .perform();

  Lredist->clear();
  LOG.os<1>("-- Cholesky error: ", mat_copy->norm(dbcsr_norm_frobenius), '\n');
  mat_copy->clear();
  #endif

  m_L = std::make_shared<dbcsr::matrix<double>>(std::move(*L_reo1));

  // exit(0);

  LOG.os<1>("Finished decomposition.\n");
}

dbcsr::shared_matrix<double> pivinc_cd::L(
    std::vector<int> rowblksizes, std::vector<int> colblksizes)
{

  auto Lredist = dbcsr::matrix<>::create()
                     .set_cart(m_world.dbcsr_grid())
                     .name("Cholesky decomposition")
                     .row_blk_sizes(rowblksizes)
                     .col_blk_sizes(colblksizes)
                     .matrix_type(dbcsr::type::no_symmetry)
                     .build();

  LOG.os<1>("-- Redistributing L\n");

  Lredist->complete_redistribute(*m_L);
  Lredist->filter(dbcsr::global::filter_eps);

  return Lredist;
}

#endif

}  // namespace math

}  // namespace megalochem
