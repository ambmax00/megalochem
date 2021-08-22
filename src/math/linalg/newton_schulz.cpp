#include "math/linalg/newton_schulz.hpp"
#include <dbcsr_conversions.hpp> 
#include <vector>
#include <algorithm>
#include <cmath>

#include "math/solvers/hermitian_eigen_solver.hpp"

namespace megalochem {

namespace math {

const double eigval_eps = 1e-6;

std::tuple<double,double> newton_schulz::get_extremum_eigenvalues() {

  /*auto print = [&](auto vec) {
    for (auto e : vec) {
      LOG.os<>(e, " ");
    } LOG.os<>('\n');
  };*/

  // use gershgorin's theorem to estimate eigenvalues by summing rows/cols
  int N = m_mat->nfullrows_total();
  
  std::vector<double> row_sum(N,0), col_sum(N,0), diag_local(N,0), 
    diag_global(N,0), radii(N,0);
  
  //std::cout << dbcsr::matrix_to_eigen(*m_mat) << std::endl;
  
  dbcsr::iterator iter(*m_mat);
  iter.start();
  
  while (iter.blocks_left()) {
    iter.next_block();
    int roff = iter.row_offset();
    int coff = iter.col_offset();
    for (int jj = 0; jj != iter.col_size(); ++jj) {
      for (int ii = 0; ii != iter.row_size(); ++ii) {
        int i = ii + roff;
        int j = jj + coff;
        
        if (i == j) diag_local[i] = iter(ii,jj);
        
        if (i >= j) continue;
        
        row_sum[i] += std::fabs(iter(ii,jj));
        col_sum[j] += std::fabs(iter(ii,jj));
        
      }
    }
  }
  
  iter.stop();
  
  for (int ii = 0; ii != N; ++ii) {
    row_sum[ii] += col_sum[ii];
  }
  
  MPI_Allreduce(diag_local.data(), diag_global.data(), N, MPI_DOUBLE, MPI_SUM, m_world.comm());
  MPI_Allreduce(row_sum.data(), radii.data(), N, MPI_DOUBLE, MPI_SUM, m_world.comm());
  
  //LOG.os<>("Diag global\n");
  //print(diag_global);
  
  //LOG.os<>("Radii\n");
  //print(radii);
  
  std::vector<double> max_err(N,0), min_err(N,0);

  for (int ii = 0; ii != N; ++ii) {
    max_err[ii] = diag_global[ii] + radii[ii];
    min_err[ii] = diag_global[ii] - radii[ii];
  }
  
  //LOG.os<>("MAX:\n");
  //print(max_err);
  //LOG.os<>("MIN:\n");
  //print(min_err);

  double maxeval = *(std::max_element(max_err.begin(), max_err.end()));
  double mineval = *(std::min_element(min_err.begin(), min_err.end()));
  
  LOG.os<1>("MIN/MAX eigenvalue: ", mineval, " ", maxeval, '\n');
  
  /*math::hermitian_eigen_solver herm(m_world, m_mat, 'V', true);
  herm.compute();
  
  auto evals = herm.eigvals();
  
  std::cout << "EIGENVALUES: " << std::endl;
  for (auto e : evals) {
    std::cout << e << " ";
  } std::cout << std::endl;*/
  
  
  /* the eigenvalues for positive definite matrices are always above zero
   * so in principle, we could just use mineval > 0. However, this will lead
   * to the newton-schulz iterations not converging. We need to use a small
   * but non-zero(!) value so it can converge (here eigval_eps)
   */
  mineval = std::max(mineval,eigval_eps);
  return std::make_tuple(mineval,maxeval);
  
}

void newton_schulz::fill_identity(dbcsr::matrix<double>& mat) {
  
  mat.reserve_diag_blocks();
  
  dbcsr::iterator iter(mat);
  iter.start();
  while (iter.blocks_left()) {
    iter.next_block();
    int roff = iter.row_offset();
    int coff = iter.col_offset();
    for (int jj = 0; jj != iter.row_size(); ++jj) {
      for (int ii = 0; ii != iter.col_size(); ++ii) {
        if (ii + roff == jj + coff) iter(ii,jj) = 1.0;
      }
    }
  }
  iter.stop();
  
}

void newton_schulz::taylor(
  dbcsr::matrix<double>& Xk, dbcsr::matrix<double>& Tk) {
  
  Tk.clear();
  fill_identity(Tk);
  
  Tk.add(15, -10, Xk);
    
  dbcsr::multiply('N', 'N', 3.0, Xk, Xk, 1.0, Tk)
    //.filter_eps(m_filter_eps_iter)
    .perform();
    
  Tk.scale(0.125);
    
} 

double newton_schulz::mapping(double val) {
  
  return 1.0/64.0 * val * std::pow(( 15.0-10.0*val+3.0*val*val ),2.0);
  
}

void newton_schulz::compute() {
  
  auto print = [&](auto name, auto mat) {
    LOG.os<>(name, " :\n");
    LOG.os<>(dbcsr::matrix_to_eigen(*mat), '\n');
  };
  
  auto [mineval, maxeval] = get_extremum_eigenvalues();
    
  decltype(m_mat) Xk, Zk, Yk, Tk, Zknew, Yknew;
  double lamk, emax, emin;
  
  emax = maxeval;
  emin = mineval;
  
  lamk = 2.0/(emax + emin);
  
  auto blksizes = dbcsr::split_range(m_mat->nfullrows_total(), m_blksize);
  Yk = dbcsr::matrix<double>::create()
    .set_cart(m_world.dbcsr_grid())
    .name("Yk")
    .row_blk_sizes(blksizes)
    .col_blk_sizes(blksizes)
    .matrix_type(dbcsr::type::no_symmetry)
    .build();
    
  Yk->redistribute(*(m_mat->desymmetrize()));
  
  //Yk = m_mat->desymmetrize(); //dbcsr::matrix<double>::copy(*m_mat).build(); //
  //Yk->filter(m_filter_eps_iter);
  
  LOG.os<>("OCCUPATION: ", Yk->occupation(), '\n');
  
  Xk = dbcsr::matrix<double>::create_template(*Yk)
    .matrix_type(dbcsr::type::no_symmetry)
    .name("Xk")
    .build();
    
  Tk = dbcsr::matrix<double>::create_template(*Yk)
    .matrix_type(dbcsr::type::no_symmetry)
    .name("Tk")
    .build();
  
  Zk = dbcsr::matrix<double>::create_template(*Yk)
    .matrix_type(dbcsr::type::no_symmetry)
    .name("Zk")
    .build();
    
  Yknew = dbcsr::matrix<double>::create_template(*Yk)
    .matrix_type(dbcsr::type::no_symmetry)
    .name("Yknew")
    .build();
  
  Zknew = dbcsr::matrix<double>::create_template(*Yk)
    .matrix_type(dbcsr::type::no_symmetry)
    .name("Zknew")
    .build();
    
  fill_identity(*Zk);
  
  double conv = m_conv; //std::max(m_conv, m_filter_eps);
  int k = 0;
    
  while (k < m_max_iter) {
    
    LOG.os<>("EMAX/EMIN/LAM: ", emax, " ", emin, " ", lamk, '\n');
    
    dbcsr::multiply('N', 'N', lamk, *Yk, *Zk, 0.0, *Xk)
      //.filter_eps(m_filter_eps_iter)
      .perform();
      
    //print("Xk", Xk);
    
    taylor(*Xk, *Tk);
    
    //print("Tk", Tk);
    
    Xk->add(1.0, -1.0, *Tk);
    double normx = std::sqrt(Xk->dot(*Xk));
    LOG.os<>("Iteration ", k, " NormX: ", normx, '\n');
    
    Xk->clear();
        
    dbcsr::multiply('N', 'N', std::sqrt(lamk), *Zk, *Tk, 0.0, *Zknew)
      //.filter_eps(m_filter_eps_iter)
      .perform();
    
    dbcsr::multiply('N', 'N', std::sqrt(lamk), *Tk, *Yk, 0.0, *Yknew)
      //.filter_eps(m_filter_eps_iter)
      .perform();
    
    std::swap(Yk, Yknew);
    std::swap(Zk, Zknew);
    
    Yknew->add(1.0, -1.0, *Yk);
    double normy = std::sqrt(Yknew->dot(*Yknew));
    LOG.os<>("NORM: ", normy, '\n');
    LOG.os<>("OCCUPATION Zk: ", Zk->occupation(), '\n');
    
    //if (normy < conv) break;
    
    Yknew->clear();
    Zknew->clear();
    
    double emax_new = mapping(lamk * emax);
    double emin_new = mapping(lamk * emin);
    double lamk_new = 2.0/(emax_new + emin_new);
    
    if (std::fabs(emax_new - emax) < m_conv && std::fabs(emin_new - emin) < m_conv) {
      break;
    }
    
    emax = emax_new;
    emin = emin_new;
    lamk = lamk_new;
    
    ++k;
    
  }
  
  if (k == m_max_iter) {
    throw std::runtime_error("Newton-Schulz did not converge!");
  }
  
  m_mat_sqrt = dbcsr::matrix<double>::create_template(*m_mat)
    .name("Matrix Inverse Sqrt of " + m_mat->name())
    .matrix_type(dbcsr::type::no_symmetry)
    .build();
  
  m_mat_invsqrt = dbcsr::matrix<double>::create_template(*m_mat)
    .name("Matrix Sqrt of " + m_mat->name())
    .matrix_type(dbcsr::type::no_symmetry)
    .build();
  
  m_mat_invsqrt->redistribute(*Zk);
  m_mat_sqrt->redistribute(*Yk);
  
}

dbcsr::shared_matrix<double> newton_schulz::compute_inverse() {
  auto inv = dbcsr::matrix<double>::create_template(*m_mat)
    .name("Matrix Inverse of " + m_mat->name())
    .matrix_type(dbcsr::type::symmetric)
    .build();
  dbcsr::multiply('N', 'N', 1.0, *m_mat_invsqrt, *m_mat_invsqrt, 0.0, *inv).perform();
  return inv;
}

} // namespace math 

} // namespace megalochem
