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

    /*LOG.os<2>("-- Reordered LMO indices: \n");
    for (size_t i = 0; i != lmo_perm.size(); ++i) {
      LOG.os<2>(lmo_perm[i], " ", lmo_pos[lmo_perm[i]], "\n");
    }
    LOG.os<2>('\n');*/
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

void pivinc_cd::compute_old(std::optional<int> force_rank, std::optional<double> eps)
{
  // convert input mat to scalapack format

  LOG.os<1>("Starting pivoted incomplete cholesky decomposition.\n");

  LOG.os<1>("-- Setting up scalapack environment and matrices.\n");

  util::mpi_time TIME(m_world.comm(), "CHOLESKY");
  //auto& time_reo = TIME.sub("REO");
  //auto& time_calc = TIME.sub("CALC");
  //auto& time_reol = TIME.sub("reol");
  
  TIME.start();

  int N = m_mat_in->nfullrows_total();
  int* iwork = new int[N];
  int nb = 64;

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
  
  // subvectors
  scalapack::distmat<double> u_i(sgrid, N, N, nb, nb, 0, 0);
  
  // create a shared memeory window on each node
 /* LOG.os<1>("-- Setup shared memory.\n");
  
  MPI_Comm local_comm;
  int local_size(-1), local_rank(-1);
  
  MPI_Comm_split_type(
        m_world.comm(), MPI_COMM_TYPE_SHARED, m_world.rank(), MPI_INFO_NULL,
        &local_comm);

  MPI_Comm_size(local_comm, &local_size);
  MPI_Comm_rank(local_comm, &local_rank);

  MPI_Win local_window;
  double_int* local_data;
  
  MPI_Win_allocate_shared(
      sizeof(double_int), sizeof(double_int), MPI_INFO_NULL, local_comm, 
      &local_data, &local_window);
      
  // communicator which groups all first processes on each node
  MPI_Comm master_comm = MPI_COMM_NULL;
  int color = (local_rank == 0) ? 0 : MPI_UNDEFINED;
  int key = m_world.rank();
    
  MPI_Comm_split(m_world.comm(), color, key, &master_comm);

  util::mpi_time TIME0(m_world.comm(), "COMM");
  auto& time1 = TIME0.sub("1");
  auto& time2 = TIME0.sub("2");
  auto& time3 = TIME0.sub("3");*/

  // get max diag element
  auto get_max_diag = [&](int I) {
    
    //time1.start();
    
    double_int local, global;
    local.d = 0;
    local.i = 0;
    bool is_set = false;
    
    for (int ii = I; ii != N; ++ii) {
      if (sgrid.myprow() == U.iproc(ii) && sgrid.mypcol() == U.jproc(ii)) {
        double val = U.global_access(ii,ii);
        if (!is_set || fabs(val) > fabs(local.d)) {
           local.d = val;
           local.i = ii;
           is_set = true;
        }
      }
    }
    
    if (!is_set) {
      local.d = -std::numeric_limits<double>::max();
    }
    
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, m_world.comm());  
    
    return global;
  };
     
  /*double max_U_diag_global = 0.0;
  for (int ix = 0; ix != N; ++ix) {
    max_U_diag_global =
        std::max(fabs(max_U_diag_global), fabs(U.get('A', ' ', ix, ix)));
  }*/
  
  LOG.os<1>("-- Getting maximum element.\n");
  
  auto max_diag = get_max_diag(0); 
  double max_val_global = max_diag.d;

  LOG.os<1>("-- Problem size: ", N, '\n');
  LOG.os<1>(
      "-- Maximum diagonal element of input matrix: ", max_val_global, '\n');

  double thresh = (eps) ? *eps : N * std::numeric_limits<double>::epsilon() * max_val_global;

  LOG.os<1>("-- Threshold: ", thresh, '\n');

  std::function<void(int)> cd_step;
  cd_step = [&](int I) {
    // STEP 1: If Dimension of U is one, then set L and return

    LOG.os<1>("---- Level ", I, '\n');
    
    auto max_U = get_max_diag(I);
    double max_U_diag = max_U.d;
    int max_U_idx = max_U.i;

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
    //scalapack::distmat<double> u_i(sgrid, N, 1, nb, nb, 0, 0);
    c_pdgeadd(
        'N', N - I - 1, 1, 1.0, U.data(), I + 1, I, U.desc().data(), 0.0,
        u_i.data(), I + 1, I, u_i.desc().data());

    // b) form Utilde
    /*c_pdgemm(
        'N', 'T', N - I - 1, N - I - 1, 1, -1 / U_II, u_i.data(), I + 1, 0,
        u_i.desc().data(), u_i.data(), I + 1, 0, u_i.desc().data(), 1.0,
        U.data(), I + 1, I + 1, U.desc().data());*/
    
    c_pdger(N - I - 1, N - I - 1, -1.0/U_II, u_i.data(), I+1, I, u_i.desc().data(),
      1, u_i.data(), I+1, I, u_i.desc().data(), 1, U.data(), I+1, I+1, U.desc().data());
        
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
        'F', 'R', 'C', N - I - 1, 1, u_i.data(), I + 1, I, u_i.desc().data(),
        ipiv_r, I + 1, 0, desc_r, iwork);

    //u_i.print();

    // (c) add u_i to L

    // L.print();

    c_pdgeadd(
        'N', N - I - 1, 1, 1.0 / sqrt(U_II), u_i.data(), I + 1, I,
        u_i.desc().data(), 0.0, L.data(), I + 1, I, L.desc().data());

    // L.print();

    perms[I] = max_U_idx + 1;

    return;
  };

  LOG.os<1>("-- Starting recursive decomposition.\n");
  cd_step(0);
  
  TIME.finish();
  TIME.print_info();
  
  //TIME0.print_info();
  
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

void max_abs_loc(void* inputBuffer, void* outputBuffer, int* len, MPI_Datatype* datatype)
{
    double_int* input = (double_int*)inputBuffer;
    double_int* output = (double_int*)outputBuffer;
 
    for(int i = 0; i < *len; i++)
    {
        if (std::fabs(output[i].d) == std::fabs(input[i].d)) {
          output[i].d = input[i].d;
          output[i].i = std::min(output[i].i,input[i].i);
        } 
        if (std::fabs(output[i].d) < std::fabs(input[i].d)) {
          output[i].d = input[i].d;
          output[i].i = input[i].i;
        } 
    }
   
}

void pivinc_cd::compute(std::optional<int> force_rank, std::optional<double> eps)
{
  // convert input mat to scalapack format

  LOG.os<1>("Starting pivoted incomplete cholesky decomposition.\n");

  LOG.os<1>("-- Setting up scalapack environment and matrices.\n");

  util::mpi_time TIME(m_world.comm(), "CHOLESKY");
  
  TIME.start();

  int N = m_mat_in->nfullrows_total();
  int* iwork = new int[N];
  int nb = 2;

  auto sgrid = m_world.scalapack_grid();
  auto dcart = m_world.dbcsr_grid();
  auto blksizes = m_mat_in->row_blk_sizes();
  
  LOG.os<1>("Grid size: ", sgrid.nprow(), " x ", sgrid.npcol(), '\n');
  
  auto print = [&](auto& As) {
    auto dAs = dbcsr::scalapack_to_matrix(As, dcart, "name", blksizes, blksizes);
    auto eigen = dbcsr::matrix_to_eigen(*dAs);
    if (m_world.rank() == 0) {
      std::cout << "Matrix: " << eigen << '\n' << std::endl;
    }
  };

  MPI_Comm comm = m_world.comm();
  int myrank = m_world.rank();
  int myprow = sgrid.myprow();
  int mypcol = sgrid.mypcol();
  
  // create MPI operator for finding max absolute element
  MPI_Op MPI_MAXABSLOC;
  MPI_Op_create(&max_abs_loc, 1, &MPI_MAXABSLOC);
  
  scalapack::distmat<double> Aspack = dbcsr::matrix_to_scalapack(
      m_mat_in, sgrid, nb, nb, 0, 0);
      
  //if (Aspack.iproc(2) == myprow && Aspack.jproc(2) == mypcol) Aspack.global_access(2,2) = 3.0;

  scalapack::distmat<double> Aspack_copy(sgrid, N, N, nb, nb, 0, 0);

  // form "diagonal" communicator
  MPI_Comm diag_comm;
  int has_diag = 0;
  
  for (int ii = 0; ii < N; ii += nb) {
    if (Aspack.iproc(ii) == myprow && Aspack.jproc(ii) == mypcol) {
      has_diag = 1;
    }
  }
  
  //std::cout << "NPROW: " << sgrid.nprow() << " " << sgrid.npcol() << std::endl;
  
  MPI_Comm_split(m_world.comm(), has_diag, m_world.rank(), &diag_comm);

  c_pdgeadd(
      'N', N, N, 1.0, Aspack.data(), 0, 0, Aspack.desc().data(), 0.0, Aspack_copy.data(), 0, 0,
      Aspack_copy.desc().data());

  // vector to keep track of permutations 
  std::vector<int> perms(N);
  std::iota(perms.begin(), perms.end(), 0);

  // chol mat
  scalapack::distmat<double> Lspack(sgrid, N, N, nb, nb, 0, 0);
    
  auto get_max_diag = [&](int I) {
    
    //time1.start();
    
    double_int local, global;
    local.d = 0;
    local.i = -1;
    
    if (has_diag) {
      
      for (int idx = I; idx < N; idx += 1) {
        if (Aspack.iproc(idx) == myprow && Aspack.jproc(idx) == mypcol) {
          //for (int ii = idx; ii < idx+blksize; ++ii) {
            double val = Aspack.global_access(idx,idx);
            if (std::fabs(val) > std::fabs(local.d)) {
               local.d = val;
               local.i = idx;
            }
          //}
        }
      }
      
      //std::cout << myprow << " " << mypcol << " " << local.d << " " << local.i << std::endl;
      
      MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXABSLOC,diag_comm);
      
      if (sgrid.nprow() != 1) {
        c_igebs2d(sgrid.ctx(), 'C', ' ', 1, 1, &global.i, 1);
        c_dgebs2d(sgrid.ctx(), 'C', ' ', 1, 1, &global.d, 1);
      }
      
    } else {
      
      int origin = mypcol % sgrid.nprow();
    
      c_igebr2d(sgrid.ctx(), 'C', ' ', 1, 1, &global.i, 1, origin, mypcol);
      c_dgebr2d(sgrid.ctx(), 'C', ' ', 1, 1, &global.d, 1, origin, mypcol);
    
    }
    
    if (global.i == -1) {
      throw std::runtime_error("MPI_MAXABSLOC: something went wrong...");
    }
    
    return global;
  };
  
  LOG.os<1>("-- Getting maximum element.\n");
  
  auto max_diag = get_max_diag(0); 
  double max_val_global = max_diag.d;

  LOG.os<1>("-- Problem size: ", N, '\n');
  LOG.os<1>(
      "-- Maximum diagonal element of input matrix: ", max_val_global, '\n');

  double thresh = (eps) ? *eps : N * std::numeric_limits<double>::epsilon() * max_val_global;

  LOG.os<1>("-- Threshold: ", thresh, '\n');

  std::function<void(int)> cd_step;
  cd_step = [&](int I) {
    
    LOG.os<1>("---- Cholesky Level ", I, '\n');
    
    //Aspack.print();

    // If max dim reached, return

    if (I == N - 1) {
      LOG.os<1>("Reached last column.\n");
      if (sgrid.myprow() == Aspack.iproc(I) && sgrid.mypcol() == Aspack.jproc(I)) {
        Aspack.global_access(I,I) = std::sqrt(Aspack.global_access(I,I));
      }
      m_rank = I + 1;
      return;
    }
    
    // Find max diagonal element
    
    max_diag = get_max_diag(I);
    double val_max = max_diag.d;
    int idx_max = max_diag.i;
    
    LOG.os<1>("---- Maximum diagonal element: ", max_diag.d, " @ ", max_diag.i, " ", max_diag.i, '\n');
    
    // permute cols and rows

    //print(Aspack);

    c_pdswap(N, Aspack.data(), 0, I, Aspack.desc().data(), 1, 
      Aspack.data(), 0, idx_max, Aspack.desc().data(), 1);
    c_pdswap(N, Aspack.data(), I, 0, Aspack.desc().data(), N, 
      Aspack.data(), idx_max, 0, Aspack.desc().data(), N);
      
    //print(Aspack);
            
    std::swap(perms[I], perms[idx_max]);

    LOG.os<1>("---- Checking convergence.\n");

    if (val_max < 0.0 && std::fabs(val_max) > thresh) {
      LOG.os<1>("fabs(U_II): ", std::fabs(val_max), '\n');
      throw std::runtime_error("Negative Pivot element. CD not possible.");
    }

    if ((std::fabs(val_max) < thresh) || (force_rank && *force_rank == I)) {
      if (std::fabs(val_max) < thresh)  
        LOG.os<1>("Pivot element below threshold.\n");
      if (force_rank && *force_rank == I) 
        LOG.os<1>("Max rank reached.\n");
      /*if (sgrid.myprow() == Aspack.iproc(I) && sgrid.mypcol() == Aspack.jproc(I)) {
        Aspack.global_access(I,I) = std::sqrt(Aspack.global_access(I,I));
      }*/
      m_rank = I;
      return;
    }

    // STEP 3.1: Form Utilde := sub(U) - u * ut
    
    c_pdscal(N-I-1, 1.0/std::sqrt(val_max), Aspack.data(), I+1, I, Aspack.desc().data(), 1);
    
    //print(Aspack);
    
    c_pdger(N-I-1, N-I-1, -1.0, Aspack.data(), I+1, I, Aspack.desc().data(),
      1, Aspack.data(), I+1, I, Aspack.desc().data(), 1, Aspack.data(), I+1, I+1, Aspack.desc().data());
    
    //print(Aspack);
    
    LOG.os<1>(
        "---- Start decomposition of submatrix of dimension ", N-I-1, '\n');
    cd_step(I + 1);
    
    if (sgrid.myprow() == Aspack.iproc(I) && sgrid.mypcol() == Aspack.jproc(I)) {
      Aspack.global_access(I,I) = std::sqrt(val_max);
    }

    return;
  };

  LOG.os<1>("-- Starting recursive decomposition.\n");
  cd_step(0);
  
  for (int ii = 0; ii < N; ++ii) {
    c_pdgeadd('N', 1, ii+1, 1.0, Aspack.data(), ii, 0, Aspack.desc().data(),
      0.0, Lspack.data(), perms[ii], 0, Lspack.desc().data());
  }
  
  reorder_and_reduce(Lspack);
  
  c_pdgemm(
      'N', 'T', N, N, m_rank, 1.0, Lspack.data(), 0, 0, Lspack.desc().data(),
      Lspack.data(), 0, 0, Lspack.desc().data(), -1.0, Aspack_copy.data(), 0, 0,
      Aspack_copy.desc().data());

  double err =
      c_pdlange('F', N, N, Aspack_copy.data(), 0, 0, Aspack_copy.desc().data(), nullptr);
      
  LOG.os<1>("-- CD error: ", err, '\n');
    
  MPI_Op_free(&MPI_MAXABSLOC);
  
  m_L = std::make_shared<decltype(Lspack)>(std::move(Lspack));
  m_perm = perms;
  
  TIME.finish();
  TIME.print_info();

    
}

dbcsr::shared_matrix<double> pivinc_cd::L(
    std::vector<int> rowblksizes, std::vector<int> colblksizes)
{
  auto out = dbcsr::scalapack_to_matrix(
      *m_L, m_world.dbcsr_grid(), "Inc. Chol. Decom. of " + m_mat_in->name(),
      rowblksizes, colblksizes);

  return out;
}



}  // namespace math

}  // namespace megalochem
