#include "locorb/locorb.hpp"
#include "math/solvers/diis.hpp"

namespace megalochem {

namespace locorb {

using smat = dbcsr::shared_matrix<double>;

smat transform(smat c, smat dip)
{
  auto b = c->row_blk_sizes();
  auto m = c->col_blk_sizes();
  auto w = c->get_cart();

  auto temp = dbcsr::matrix<double>::create()
                  .name("temp")
                  .set_cart(w)
                  .row_blk_sizes(b)
                  .col_blk_sizes(m)
                  .matrix_type(dbcsr::type::no_symmetry)
                  .build();

  auto dip_mm = dbcsr::matrix<double>::create()
                    .name("dip_mm")
                    .set_cart(w)
                    .row_blk_sizes(m)
                    .col_blk_sizes(m)
                    .matrix_type(dbcsr::type::no_symmetry)
                    .build();

  dbcsr::multiply('N', 'N', 1.0, *dip, *c, 0.0, *temp)
      .filter_eps(dbcsr::global::filter_eps)
      .perform();

  dbcsr::multiply('T', 'N', 1.0, *c, *temp, 0.0, *dip_mm)
      .filter_eps(dbcsr::global::filter_eps)
      .perform();

  return dip_mm;
}

smat compute_D(smat dip_x, smat dip_y, smat dip_z)
{
  auto diag_x = dip_x->get_diag();
  auto diag_y = dip_y->get_diag();
  auto diag_z = dip_z->get_diag();

  auto copy_x = dbcsr::matrix<double>::copy(*dip_x).build();
  auto copy_y = dbcsr::matrix<double>::copy(*dip_y).build();
  auto copy_z = dbcsr::matrix<double>::copy(*dip_z).build();

  copy_x->scale(diag_x, "right");
  copy_y->scale(diag_y, "right");
  copy_z->scale(diag_z, "right");

  copy_x->add(1.0, 1.0, *copy_y);
  copy_x->add(1.0, 1.0, *copy_z);

  copy_x->setname("D");
  return copy_x;
}

double jacobi_sweep_serial(
    smat& c_dist, smat& x_dist, smat& y_dist, smat& z_dist)
{
  auto c = dbcsr::matrix_to_eigen(*c_dist);
  auto x = dbcsr::matrix_to_eigen(*x_dist);
  auto y = dbcsr::matrix_to_eigen(*y_dist);
  auto z = dbcsr::matrix_to_eigen(*z_dist);
  auto w = c_dist->get_cart();

  int nbas = c_dist->nfullrows_total();
  int norb = c_dist->nfullcols_total();

  auto b = c_dist->row_blk_sizes();
  auto m = c_dist->col_blk_sizes();

  double t12 = 1e-12;
  double t8 = 1e-8;

  double max_diff = 0.0;

  auto mat_update =
      [&norb](Eigen::MatrixXd& mat, double ca, double sa, int i, int j) {
        double mii = mat(i, i);
        double mjj = mat(j, j);
        double mij = mat(i, j);

        for (int k = 0; k != norb; ++k) {
          double mik = mat(i, k);
          double mjk = mat(j, k);

          mat(i, k) = ca * mik + sa * mjk;
          mat(k, i) = mat(i, k);
          mat(j, k) = ca * mjk - sa * mik;
          mat(k, j) = mat(j, k);
        }

        mat(i, j) = (pow(ca, 2.0) - pow(sa, 2.0)) * mij + ca * sa * (mjj - mii);
        mat(j, i) = mat(i, j);
        mat(i, i) = pow(ca, 2.0) * mii + pow(sa, 2.0) * mjj + 2 * ca * sa * mij;
        mat(j, j) = pow(sa, 2.0) * mii + pow(ca, 2.0) * mjj - 2 * ca * sa * mij;

        // std::cout << "X: " << std::endl;
        // std::cout << mat << std::endl;
      };

  /*if (w.rank() == 0) {
          std::cout << "COLD" << std::endl;
          std::cout << c << std::endl;
          std::cout << "X:" << std::endl;
          std::cout << x << std::endl;
          std::cout << "Y:" << std::endl;
          std::cout << y << std::endl;
          std::cout << "Z:" << std::endl;
          std::cout << z << std::endl;
  }*/

  if (w.rank() == 0) {
    int npairs = norb / 2 + norb % 2;
    int nperms = norb + norb % 2;

    // generating pairs
    std::vector<int> idx_u(npairs);
    std::vector<int> idx_l(npairs);
    int hold = -1;

    int val = 0;

    for (auto& i : idx_u) { i = val++; }

    for (auto iter = idx_l.rbegin(); iter != idx_l.rend(); ++iter) {
      *iter = val++;
    }

    idx_l.front() = (norb % 2 == 0) ? idx_l.front() : -1;

    /*for (auto a : idx_u) {
            std::cout << a << " ";
    } std::cout << std::endl;

    for (auto a : idx_l) {
            std::cout << a << " ";
    } std::cout << std::endl;*/

    auto next_perm = [](std::vector<int>& idx_u, std::vector<int>& idx_l,
                        int& hold, int& iperm) {
      if (iperm % 2 == 0) {
        hold = idx_u.back();

        for (auto iter = idx_u.rbegin(); iter != idx_u.rend() - 1; ++iter) {
          *iter = *(iter + 1);
        }

        idx_u.front() = -1;
      }
      else {
        idx_u.front() = idx_l.front();
        std::copy(idx_l.begin() + 1, idx_l.end(), idx_l.begin());
        idx_l.back() = hold;
        hold = -1;
      }

      /*for (auto a : idx_u) {
              std::cout << a << " ";
      } std::cout << std::endl;

      for (auto a : idx_l) {
              std::cout << a << " ";
      } std::cout << std::endl;*/
    };

    for (int iperm = 0; iperm != nperms; ++iperm) {
      // std::cout << "IPERM: " << iperm << std::endl;

      for (int ipair = 0; ipair != npairs; ++ipair) {
        int i = idx_u[ipair];
        int j = idx_l[ipair];

        // std::cout << "I,J: " << i << " " << j << std::endl;

        if (i == -1 || j == -1)
          continue;

        double A = pow(x(i, j), 2.0) + pow(y(i, j), 2.0) + pow(z(i, j), 2.0) -
            0.25 * (pow(x(i, i), 2.0) + pow(y(i, i), 2.0) + pow(z(i, i), 2.0)) -
            0.25 * (pow(x(j, j), 2.0) + pow(y(j, j), 2.0) + pow(z(j, j), 2.0)) +
            0.5 * (x(i, i) * x(j, j) + y(i, i) * y(j, j) + z(i, i) * z(j, j));

        double B = x(i, j) * (x(i, i) - x(j, j)) +
            y(i, j) * (y(i, i) - y(j, j)) + z(i, j) * (z(i, i) - z(j, j));

        if (fabs(B) < t12)
          continue;
        if (fabs(A) < t8)
          throw std::runtime_error("PANIC!");

        // update

        double alpha4 = atan(-B / A);
        double pi = 2 * acos(0.0);

        if (alpha4 < 0.0 && B > 0.0)
          alpha4 += pi;
        if (alpha4 > 0.0 && B < 0.0)
          alpha4 -= pi;

        double ca = cos(alpha4 / 4.0);
        double sa = sin(alpha4 / 4.0);

        max_diff = std::max(max_diff, A + sqrt(pow(A, 2.0) + pow(B, 2.0)));

        // update c
        for (int k = 0; k != nbas; ++k) {
          double cki = c(k, i);
          double ckj = c(k, j);

          c(k, i) = ca * cki + sa * ckj;
          c(k, j) = -sa * cki + ca * ckj;
        }

        // update x,y,z
        mat_update(x, ca, sa, i, j);
        mat_update(y, ca, sa, i, j);
        mat_update(z, ca, sa, i, j);
      }

      next_perm(idx_u, idx_l, hold, iperm);
    }
  }

  MPI_Bcast(c.data(), nbas * norb, MPI_DOUBLE, 0, w.comm());
  MPI_Bcast(x.data(), norb * norb, MPI_DOUBLE, 0, w.comm());
  MPI_Bcast(y.data(), norb * norb, MPI_DOUBLE, 0, w.comm());
  MPI_Bcast(z.data(), norb * norb, MPI_DOUBLE, 0, w.comm());
  MPI_Bcast(&max_diff, 1, MPI_DOUBLE, 0, w.comm());

  c_dist =
      dbcsr::eigen_to_matrix(c, w, "c_new", b, m, dbcsr::type::no_symmetry);
  x_dist =
      dbcsr::eigen_to_matrix(x, w, "x_new", m, m, dbcsr::type::no_symmetry);
  y_dist =
      dbcsr::eigen_to_matrix(y, w, "y_new", m, m, dbcsr::type::no_symmetry);
  z_dist =
      dbcsr::eigen_to_matrix(z, w, "z_new", m, m, dbcsr::type::no_symmetry);

  return max_diff;
}

template <int DIM>
class ring {
 private:
  const int EMPTY = -1;

  MPI_Comm m_comm;
  int m_rank = -1;
  int m_mpisize = 0;

  util::mpi_log LOG;

  int m_ncols;
  int m_totnrows;

  std::array<int, DIM> m_nrows;
  std::array<int, DIM> m_row_offset;

  std::vector<int> m_npairs_proc;
  std::vector<int> m_npairs_off;
  int m_npairs_tot;
  int m_npairs;

  std::vector<int> m_idx_u, m_idx_l;
  Eigen::MatrixXd m_mat_u, m_mat_l;

  template <typename T>
  void print_vec(std::vector<T>& v)
  {
    for (auto x : v) { std::cout << x << " "; }
  }

  int m_iperm = 0;
  int m_last = 0;

  void next_perm_even()
  {
    // shuffle indices
    for (int i = 0; i != m_npairs; ++i) {
      m_idx_u[m_npairs - i] = m_idx_u[m_npairs - i - 1];
    }

    // shuffle cols
    for (int i = 0; i != m_npairs; ++i) {
      m_mat_u.col(m_npairs - i) = m_mat_u.col(m_npairs - i - 1);
    }

    int left = (m_rank == 0) ? m_last : m_rank - 1;
    int right = (m_rank == m_last) ? 0 : m_rank + 1;

    // send idx

    MPI_Sendrecv(
        &m_idx_u.back(), 1, MPI_INT, right, 0, &m_idx_u.front(), 1, MPI_INT,
        left, 0, m_comm, MPI_STATUS_IGNORE);

    if (m_rank == 0)
      m_idx_u.front() = -1;

    // send cols
    double* last_col = m_mat_u.data() + m_totnrows * (m_npairs);
    double* first_col = m_mat_u.data();

    MPI_Sendrecv(
        last_col, m_totnrows, MPI_DOUBLE, right, 2, first_col, m_totnrows,
        MPI_DOUBLE, left, 2, m_comm, MPI_STATUS_IGNORE);

    if (m_rank != m_last)
      m_idx_u.back() = -1;

    // print_idx();
    // print_mat();
  }

  void next_perm_odd()
  {
    // shuffle indices
    for (int i = 0; i != m_npairs; ++i) { m_idx_l[i] = m_idx_l[i + 1]; }

    // shuffle cols
    for (int i = 0; i != m_npairs; ++i) { m_mat_l.col(i) = m_mat_l.col(i + 1); }

    // send indices
    int left = (m_rank == 0) ? m_last : m_rank - 1;
    int right = (m_rank == m_last) ? 0 : m_rank + 1;

    MPI_Sendrecv(
        &m_idx_l.front(), 1, MPI_INT, left, 1, &m_idx_l.back(), 1, MPI_INT,
        right, 1, m_comm, MPI_STATUS_IGNORE);

    if (m_rank == 0)
      m_idx_u.front() = m_idx_l.front();
    if (m_rank == m_last)
      m_idx_l.back() = m_idx_u.back();

    // send cols
    double* first_col = m_mat_l.data();
    double* last_col = m_mat_l.data() + m_totnrows * (m_npairs);

    MPI_Sendrecv(
        first_col, m_totnrows, MPI_DOUBLE, left, 4, last_col, m_totnrows,
        MPI_DOUBLE, right, 4, m_comm, MPI_STATUS_IGNORE);

    if (m_rank == 0)
      m_mat_u.col(0) = m_mat_l.col(0);
    if (m_rank == m_last)
      m_mat_l.col(m_npairs) = m_mat_u.col(m_npairs);

    if (m_rank == m_last)
      m_idx_u.back() = -1;
    m_idx_l.front() = -1;

    // print_idx();
    // print_mat();
  }

 public:
  ring(MPI_Comm comm, std::array<Eigen::MatrixXd, DIM> mats) :
      m_comm(comm), LOG(comm, 0)
  {
    MPI_Comm_rank(comm, &m_rank);
    MPI_Comm_size(comm, &m_mpisize);

    m_last = m_mpisize - 1;

    m_ncols = mats[0].cols();
    int roff = 0;

    for (int i = 0; i != DIM; ++i) {
      m_nrows[i] = mats[i].rows();
      m_row_offset[i] = roff;
      roff += m_nrows[i];
      // if (m_rank == 0) std::cout << m_nrows[i] << " ";
    }  // std::cout << std::endl;

    // "length" of caterpillar
    int cat_size = (m_ncols + m_ncols % 2) / 2;

    // divide it up
    m_npairs_proc.resize(m_mpisize);  // how many pairs on process i?
    m_npairs_off.resize(m_mpisize);

    int val = 0;

    for (int i = 0; i != m_mpisize; ++i) {
      m_npairs_proc[i] =
          cat_size / m_mpisize + ((i == 0) ? cat_size % m_mpisize : 0);

      m_npairs_off[i] = val;

      val += m_npairs_proc[i];
    }

    int off = m_npairs_off[m_rank];

    m_npairs = m_npairs_proc[m_rank];
    m_npairs_tot =
        std::accumulate(m_npairs_proc.begin(), m_npairs_proc.end(), 0);

    m_idx_u.resize(m_npairs + 1, EMPTY);
    m_idx_l.resize(m_npairs + 1, EMPTY);

    // LOG(-1).os<>("NPAIRS: ", m_npairs, '\n');

    for (int i = 0; i != m_npairs; ++i) { m_idx_u[i] = off + i; }

    for (int i = 0; i != m_npairs; ++i) {
      m_idx_l[i + 1] = m_ncols - 1 + m_ncols % 2 - off - i;
    }

    if (m_rank == 0 && m_ncols % 2 != 0) {
      m_idx_l[1] = EMPTY;
    }

    // print_idx();

    // take care of matrices
    int nrow_tot = std::accumulate(m_nrows.begin(), m_nrows.end(), 0.0);

    // std::cout << "NROWTOT: " << nrow_tot << std::endl;

    m_mat_l.resize(nrow_tot, m_npairs + 1);
    m_mat_u.resize(nrow_tot, m_npairs + 1);

    for (int i = 0; i != m_npairs; ++i) {
      int row_off = 0;

      for (int d = 0; d != DIM; ++d) {
        int idx_u = m_idx_u[i];
        int idx_l = m_idx_l[i + 1];

        // std::cout << "IDX: " << idx_u << " " << idx_l << std::endl;

        if (idx_l != EMPTY) {
          m_mat_l.block(row_off, i + 1, m_nrows[d], 1) =
              mats[d].block(0, idx_l, m_nrows[d], 1);
        }

        if (idx_u != EMPTY) {
          m_mat_u.block(row_off, i, m_nrows[d], 1) =
              mats[d].block(0, idx_u, m_nrows[d], 1);
        }

        row_off += m_nrows[d];
      }
    }

    /*if (m_rank == 0) {
            std::cout << mats[0] << std::endl;
            std::cout << mats[1] << std::endl;
            std::cout << mats[2] << std::endl;
            std::cout << mats[3] << std::endl;
    }*/

    // print_mat();

    m_totnrows = nrow_tot;
  }

  void next_perm()
  {
    if (m_iperm % 2 == 0)
      next_perm_even();
    if (m_iperm % 2 == 1)
      next_perm_odd();

    ++m_iperm;
  }

  int npairs()
  {
    return m_npairs;
  }

  int npairs_tot()
  {
    return m_npairs_tot;
  }

  std::vector<int> npairs_proc()
  {
    return m_npairs_proc;
  }

  std::vector<int> npairs_off()
  {
    return m_npairs_off;
  }

  int upper_pair(int ipair)
  {
    return m_idx_u[ipair];
  }

  int lower_pair(int ipair)
  {
    return m_idx_l[ipair + 1];
  }

  Eigen::Block<Eigen::MatrixXd> upper_col(int idim, int ipair)
  {
    // std::cout << "NDIM: " << idim << " offset: " << m_row_offset[idim] <<
    // std::endl;
    return m_mat_u.block(m_row_offset[idim], ipair, m_nrows[idim], 1);
  }

  Eigen::Block<Eigen::MatrixXd> lower_col(int idim, int ipair)
  {
    return m_mat_l.block(m_row_offset[idim], ipair + 1, m_nrows[idim], 1);
  }

  inline double& upper(int idim, int i, int j)
  {
    return m_mat_u(i + m_row_offset[idim], j);
  }

  inline double& lower(int idim, int i, int j)
  {
    return m_mat_l(i + m_row_offset[idim], j + 1);
  }

  void print_idx()
  {
    for (int i = 0; i != m_mpisize; ++i) {
      if (i == m_rank) {
        std::cout << "PROC " << i << std::endl;
        std::cout << "x ";
        print_vec<int>(m_idx_u);
        std::cout << std::endl;
        print_vec<int>(m_idx_l);
        std::cout << "x" << std::endl;
      }
      MPI_Barrier(m_comm);
    }
  }

  void print_mat()
  {
    for (int i = 0; i != m_mpisize; ++i) {
      if (i == m_rank) {
        std::cout << "PROC " << i << std::endl;
        std::cout << m_mat_u << '\n' << std::endl;
        std::cout << m_mat_l << '\n' << std::endl;
      }
      MPI_Barrier(m_comm);
    }
  }

  std::array<Eigen::MatrixXd, DIM> get_mats()
  {
    std::array<Eigen::MatrixXd, DIM> mats;

    for (int imat = 0; imat != DIM; ++imat) {
      Eigen::MatrixXd mat(m_nrows[imat], m_ncols);

      for (int iproc = 0; iproc != m_mpisize; ++iproc) {
        for (int ipair = 0; ipair != m_npairs_proc[iproc] + 1; ++ipair) {
          // send upper col/lower col
          int idx[2];

          if (iproc == m_rank) {
            idx[0] = this->upper_pair(ipair);
            idx[1] =
                (ipair == m_npairs_proc[iproc]) ? -1 : this->lower_pair(ipair);
          }

          MPI_Bcast(idx, 2, MPI_INT, iproc, m_comm);

          // if (m_rank == 0) {
          //	std::cout << idx[0] << " " << idx[1] << std::endl;
          //}

          MPI_Barrier(m_comm);

          if (idx[0] != -1 && iproc == m_rank) {
            mat.col(idx[0]) =
                m_mat_u.block(m_row_offset[imat], ipair, m_nrows[imat], 1);
          }

          if (idx[1] != -1 && iproc == m_rank) {
            mat.col(idx[1]) =
                m_mat_l.block(m_row_offset[imat], ipair + 1, m_nrows[imat], 1);
          }

          double* offset0 = mat.data() + m_nrows[imat] * idx[0];
          double* offset1 = mat.data() + m_nrows[imat] * idx[1];

          if (idx[0] != -1) {
            MPI_Bcast(offset0, m_nrows[imat], MPI_DOUBLE, iproc, m_comm);
          }

          if (idx[1] != -1) {
            MPI_Bcast(offset1, m_nrows[imat], MPI_DOUBLE, iproc, m_comm);
          }
        }
      }

      mats[imat] = std::move(mat);
    }

    return mats;
  }
};

void update_mat(
    Eigen::Block<Eigen::MatrixXd> col_i,
    Eigen::Block<Eigen::MatrixXd> col_j,
    int i,
    int j,
    int ncol,
    double omega)
{
  double ca = cos(omega);
  double sa = sin(omega);

  double mii = col_i(i, 0);
  double mjj = col_j(j, 0);
  double mij = col_i(j, 0);

  for (int k = 0; k != ncol; ++k) {
    double mik = col_i(k, 0);
    double mjk = col_j(k, 0);

    col_i(k, 0) = ca * mik + sa * mjk;
    col_j(k, 0) = ca * mjk - sa * mik;
  }

  col_i(j, 0) = (pow(ca, 2.0) - pow(sa, 2.0)) * mij + ca * sa * (mjj - mii);
  col_j(i, 0) = col_i(j, 0);
  col_i(i, 0) = pow(ca, 2.0) * mii + pow(sa, 2.0) * mjj + 2 * ca * sa * mij;
  col_j(j, 0) = pow(sa, 2.0) * mii + pow(ca, 2.0) * mjj - 2 * ca * sa * mij;
}

struct rot_info {
  int idx[2];
  double omega;
};

static MPI_Datatype MPI_ROT_INFO;

void form_type()
{
  const int num = 2;
  int ele_blklength[num] = {2, 1};

  MPI_Datatype array_of_types[num] = {MPI_INT, MPI_DOUBLE};

  MPI_Aint array_of_offsets[num];
  MPI_Aint baseadd, add1;

  rot_info myrot;

  MPI_Get_address(&myrot.idx, &baseadd);
  MPI_Get_address(&myrot.omega, &add1);

  array_of_offsets[0] = 0;
  array_of_offsets[1] = add1 - baseadd;

  MPI_Type_create_struct(
      num, ele_blklength, array_of_offsets, array_of_types, &MPI_ROT_INFO);

  MPI_Aint lb, extent;
  MPI_Type_get_extent(MPI_ROT_INFO, &lb, &extent);
  if (extent != sizeof(myrot)) {
    MPI_Datatype old = MPI_ROT_INFO;
    MPI_Type_create_resized(old, 0, sizeof(myrot), &MPI_ROT_INFO);
    MPI_Type_free(&old);
  }
  MPI_Type_commit(&MPI_ROT_INFO);
}

double jacobi_sweep_mpi(smat& c_dist, smat& x_dist, smat& y_dist, smat& z_dist)
{
  auto w = c_dist->get_cart();

  if (w.size() < 2) {
    throw std::runtime_error("Jacobi sweep (MPI) needs nproc >= 2");
  }

  util::mpi_log LOG(w.comm(), 0);

  int nbas = c_dist->nfullrows_total();
  int norb = c_dist->nfullcols_total();

  auto b = c_dist->row_blk_sizes();
  auto m = c_dist->col_blk_sizes();

  auto c = dbcsr::matrix_to_eigen(*c_dist);
  auto x = dbcsr::matrix_to_eigen(*x_dist);
  auto y = dbcsr::matrix_to_eigen(*y_dist);
  auto z = dbcsr::matrix_to_eigen(*z_dist);

  double t12 = 1e-12;
  double max_diff = 0.0;

  // caterpillar ordering
  /* Number represents column
   * One box with pairs is one process
   *
   * "even"
   *  | 0 | -> | 1 | -> | 2 | -> | 3 | ->
   *  | 7 | -- | 6 | -- | 5 | -- | 4 |
   *
   * "odd"
   *  | - | -- | 0 | -- | 1 | -- | 2 |   (3)¬
   * ^| 7 | <- | 6 | <- | 5 | <- | 4 |
   *
   * "even"
   *  | 7 | -> | 0 | -> | 1 | -> | 2 | ->
   *  | 6 | -- | 5 | -- | 4 | -- | 3 |
   *
   * (x norb)
   *  Introduce dummy column for odd column number
   */

  ring<4> ring_data(w.comm(), std::array<decltype(x), 4>{x, y, z, c});
  double pi = 2 * acos(0.0);

  int npairs = ring_data.npairs();
  int npairs_tot = ring_data.npairs_tot();
  auto npairs_proc = ring_data.npairs_proc();
  auto npairs_off = ring_data.npairs_off();

  rot_info empty;
  empty.idx[0] = -1;
  empty.idx[1] = -1;
  empty.omega = 0.0;

  std::vector<rot_info> rinfo_vec_global(npairs_tot, empty);
  std::vector<rot_info> rinfo_vec_local(npairs, empty);

  for (int iperm = 0; iperm != norb + norb % 2; ++iperm) {
    for (int ipair = 0; ipair != npairs; ++ipair) {
      int i = ring_data.upper_pair(ipair);
      int j = ring_data.lower_pair(ipair);

      // std::cout << "PAIR: " << ipair << std::endl;
      // std::cout << "IDX: " << i << " " << j << std::endl;

      if (i == -1 || j == -1) {
        // std::cout << "SKIP" << std::endl;
        continue;
      }

      auto xcol_i = ring_data.upper_col(0, ipair);
      auto xcol_j = ring_data.lower_col(0, ipair);
      auto ycol_i = ring_data.upper_col(1, ipair);
      auto ycol_j = ring_data.lower_col(1, ipair);
      auto zcol_i = ring_data.upper_col(2, ipair);
      auto zcol_j = ring_data.lower_col(2, ipair);

      double A = pow(xcol_i(j, 0), 2.0) + pow(ycol_i(j, 0), 2.0) +
          pow(zcol_i(j, 0), 2.0) -
          0.25 *
              (pow(xcol_i(i, 0), 2.0) + pow(ycol_i(i, 0), 2.0) +
               pow(zcol_i(i, 0), 2.0)) -
          0.25 *
              (pow(xcol_j(j, 0), 2.0) + pow(ycol_j(j, 0), 2.0) +
               pow(zcol_j(j, 0), 2.0)) +
          0.5 *
              (xcol_i(i, 0) * xcol_j(j, 0) + ycol_i(i, 0) * ycol_j(j, 0) +
               zcol_i(i, 0) * zcol_j(j, 0));

      double B = xcol_i(j, 0) * (xcol_i(i, 0) - xcol_j(j, 0)) +
          ycol_i(j, 0) * (ycol_i(i, 0) - ycol_j(j, 0)) +
          zcol_i(j, 0) * (zcol_i(i, 0) - zcol_j(j, 0));

      // std::cout << "A: " << A << " B: " << B << std::endl;

      max_diff = std::max(max_diff, A + sqrt(pow(A, 2.0) + pow(B, 2.0)));

      if (fabs(B) < t12) {
        // std::cout << "SKIP" << std::endl;
        continue;
      }
      double alpha4 = atan(-B / A);

      if (alpha4 < 0.0 && B > 0.0)
        alpha4 += pi;
      if (alpha4 > 0.0 && B < 0.0)
        alpha4 -= pi;

      double omega = alpha4 / 4.0;

      update_mat(xcol_i, xcol_j, i, j, norb, omega);
      update_mat(ycol_i, ycol_j, i, j, norb, omega);
      update_mat(zcol_i, zcol_j, i, j, norb, omega);

      rinfo_vec_local[ipair].idx[0] = i;
      rinfo_vec_local[ipair].idx[1] = j;
      rinfo_vec_local[ipair].omega = omega;

      // update C
      auto ccol_i = ring_data.upper_col(3, ipair);
      auto ccol_j = ring_data.lower_col(3, ipair);

      double ca = cos(omega);
      double sa = sin(omega);

      for (int k = 0; k != nbas; ++k) {
        double cki = ccol_i(k, 0);
        double ckj = ccol_j(k, 0);

        ccol_i(k, 0) = ca * cki + sa * ckj;
        ccol_j(k, 0) = -sa * cki + ca * ckj;
      }

    }  // endfor ipair

    // communicate rot_info to all processes
    MPI_Allgatherv(
        rinfo_vec_local.data(), npairs, MPI_ROT_INFO, rinfo_vec_global.data(),
        npairs_proc.data(), npairs_off.data(), MPI_ROT_INFO, w.comm());

    /*if (w.rank() == 0) {
            std::cout << "ROTINFO: " << std::endl;
            for (auto r : rinfo_vec_global) {
                    std::cout << r.idx[0] << " " << r.idx[1]
                            << " " << r.omega << std::endl;
            }
    }*/

    for (int ipair = 0; ipair != npairs_tot; ++ipair) {
      int i = rinfo_vec_global[ipair].idx[0];
      int j = rinfo_vec_global[ipair].idx[1];
      double omega = rinfo_vec_global[ipair].omega;

      // std::cout << i << " " << j << std::endl;

      if (i == -1 || j == -1 || fabs(omega) < t12)
        continue;

      double ca = cos(omega);
      double sa = sin(omega);

      for (int cu = 0; cu != npairs + 1; ++cu) {
        int icol = ring_data.upper_pair(cu);
        if (icol == -1 || icol == i)
          continue;

        // std::cout << "UPPER: " << icol << std::endl;

        for (int idim = 0; idim != 3; ++idim) {
          double mik = ring_data.upper(idim, i, cu);
          double mjk = ring_data.upper(idim, j, cu);

          ring_data.upper(idim, i, cu) = ca * mik + sa * mjk;
          ring_data.upper(idim, j, cu) = -sa * mik + ca * mjk;
        }
      }

      for (int cu = 0; cu != npairs; ++cu) {
        int icol = ring_data.lower_pair(cu);
        if (icol == -1 || icol == j)
          continue;

        // std::cout << "LOWER: " << icol << std::endl;

        for (int idim = 0; idim != 3; ++idim) {
          double mik = ring_data.lower(idim, i, cu);
          double mjk = ring_data.lower(idim, j, cu);

          ring_data.lower(idim, i, cu) = ca * mik + sa * mjk;
          ring_data.lower(idim, j, cu) = -sa * mik + ca * mjk;
        }
      }

    }  // endfor ipair 2

    // ring_data.print_mat();

    /*auto mats = ring_data.get_mats();

    if (w.rank() == 0) {
            for (int i = 0; i != 4; ++i) {
                    std::cout << mats[i] << '\n' << '\n';
            } std::cout << std::endl;
    }*/

    for (auto& r : rinfo_vec_global) { r = empty; }

    for (auto& r : rinfo_vec_local) { r = empty; }

    MPI_Barrier(w.comm());
    ring_data.next_perm();
  }

  double max_diff_all = 0.0;

  MPI_Allreduce(&max_diff, &max_diff_all, 1, MPI_DOUBLE, MPI_MAX, w.comm());

  MPI_Barrier(w.comm());

  auto new_mats = ring_data.get_mats();

  x_dist = dbcsr::eigen_to_matrix(
      new_mats[0], w, "x_mult", m, m, dbcsr::type::symmetric);
  y_dist = dbcsr::eigen_to_matrix(
      new_mats[1], w, "y_mult", m, m, dbcsr::type::symmetric);
  z_dist = dbcsr::eigen_to_matrix(
      new_mats[2], w, "z_mult", m, m, dbcsr::type::symmetric);
  c_dist = dbcsr::eigen_to_matrix(
      new_mats[3], w, "c_bm", b, m, dbcsr::type::no_symmetry);

  return max_diff_all;
}

std::tuple<smat_d, smat_d> mo_localizer::compute_boys(smat_d c_bm, smat_d s_bb)
{
  // compute <chi_i|r|chi_j>

  form_type();

  auto moms = m_aofac->ao_emultipole();
  int nproc = c_bm->get_cart().size();

  auto dip_bb_x = moms[0];
  auto dip_bb_y = moms[1];
  auto dip_bb_z = moms[2];

  // guess orbitals
  auto [g_bm, t_mm] = compute_cholesky(c_bm, s_bb);

  // dbcsr::print(*g_bm);

  auto dip_mm_x = transform(g_bm, dip_bb_x);
  auto dip_mm_y = transform(g_bm, dip_bb_y);
  auto dip_mm_z = transform(g_bm, dip_bb_z);

  // compute D
  auto D = compute_D(dip_mm_x, dip_mm_y, dip_mm_z);

  int max_iter = 100;
  double thresh = 1e-6;

  for (int iter = 0; iter != max_iter; ++iter) {
    double max_diff = (nproc > 1) ?
        jacobi_sweep_mpi(g_bm, dip_mm_x, dip_mm_y, dip_mm_z) :
        jacobi_sweep_serial(g_bm, dip_mm_x, dip_mm_y, dip_mm_z);

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

  auto u_mm = this->compute_conversion(c_bm, s_bb, g_bm);

  return std::make_tuple(g_bm, u_mm);
}

}  // namespace locorb

}  // end namespace megalochem
