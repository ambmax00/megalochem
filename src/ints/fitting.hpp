#ifndef INTS_FITTING_H
#define INTS_FITTING_H

#include <dbcsr_btensor.hpp>
#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>
#include "desc/molecule.hpp"
#include "ints/aofactory.hpp"
#include "ints/screening.hpp"
#include "utils/mpi_time.hpp"

namespace megalochem {

namespace ints {

class dfitting {
 private:
  world m_world;
  dbcsr::cart m_cart;
  desc::shared_molecule m_mol;
  util::mpi_log LOG;
  util::mpi_time TIME;

 public:
  dfitting(world w, desc::shared_molecule smol, int print = 0) :
      m_world(w), m_cart(w.dbcsr_grid()), m_mol(smol), LOG(w.comm(), print),
      TIME(w.comm(), "Fitting Coefficients")
  {
  }

  dbcsr::sbtensor<3, double> compute(
      dbcsr::sbtensor<3, double> eris,
      dbcsr::shared_matrix<double> s_inv,
      dbcsr::btype mytype);

  dbcsr::sbtensor<3, double> compute(
      dbcsr::sbtensor<3, double> eris,
      dbcsr::shared_tensor<2, double> s_inv,
      dbcsr::btype mytype);

  dbcsr::sbtensor<3, double> compute_pari(
      dbcsr::shared_matrix<double> s_xx,
      shared_screener scr_s,
      std::array<int, 3> bdims,
      dbcsr::btype mytype);

  dbcsr::sbtensor<3, double> compute_qr_new(
      dbcsr::shared_matrix<double> s_bb,
      dbcsr::shared_matrix<double> s_xx_inv,
      dbcsr::shared_matrix<double> m_xx,
      dbcsr::shared_pgrid<3> spgrid3_xbb,
      std::array<int, 3> bdims,
      dbcsr::btype mytype);

  std::shared_ptr<Eigen::MatrixXi> compute_idx(
      dbcsr::sbtensor<3, double> cfit_xbb);

  // void compute_qr(dbcsr::sbtensor<3,double> eris,
  // dbcsr::shared_matrix<double> s_xx);

  void print_info()
  {
    TIME.print_info();
  }
};

}  // namespace ints

}  // namespace megalochem

#endif
