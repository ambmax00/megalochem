#ifndef INTS_AOFACTORY_H
#define INTS_AOFACTORY_H

#include <mpi.h>
#include <dbcsr_conversions.hpp>
#include <dbcsr_tensor.hpp>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include "desc/molecule.hpp"
#include "megalochem.hpp"

/* Loads the AO integrals
 * Op: Operator (Coulom, kinetic, ...)
 * bis: tensor space, e.g. bb is a matrix in the AO basis, xbb are the 3c2e
 * integrals etc...
 */

using eigen_smat_f = std::shared_ptr<Eigen::MatrixXf>;

namespace megalochem {

namespace ints {

struct global {
  static inline double precision = std::numeric_limits<double>::epsilon();
  static inline double omega = 0.1;
  static inline double qr_theta = 1e-5;
  static inline double qr_rho = 40;
  static inline double basis_lindep = 1e-10;
  static inline bool use_newton_schulz = false;
};

enum class metric { coulomb, erfc_coulomb, qr_fit, pari };

inline metric str_to_metric(std::string m)
{
  if (m == "coulomb")
    return metric::coulomb;
  if (m == "erfc_coulomb")
    return metric::erfc_coulomb;
  if (m == "qr_fit")
    return metric::qr_fit;
  if (m == "pari")
    return metric::pari;
  throw std::runtime_error("Invalid metric");
}

class screener;

class aofactory {
 private:
  struct impl;
  impl* pimpl;

 public:
  aofactory(desc::shared_molecule mol, world w);

  aofactory(
      world w,
      desc::shared_cluster_basis cbas,
      desc::shared_cluster_basis cdfbas = nullptr,
      desc::shared_cluster_basis cbas2 = nullptr);

  ~aofactory();

  dbcsr::shared_matrix<double> ao_overlap();
  dbcsr::shared_matrix<double> ao_overlap2();
  dbcsr::shared_matrix<double> ao_kinetic();
  dbcsr::shared_matrix<double> ao_nuclear();

  // dbcsr::shared_matrix<double> ao_diag_mnmn();
  // dbcsr::shared_matrix<double> ao_diag_mmnn();

  dbcsr::shared_matrix<double> ao_2c2e(metric m);
  dbcsr::shared_matrix<double> ao_auxoverlap();

  dbcsr::shared_matrix<double> ao_schwarz();
  dbcsr::shared_matrix<double> ao_3cschwarz();

  dbcsr::shared_matrix<double> ao_schwarz_ovlp();
  dbcsr::shared_matrix<double> ao_3cschwarz_ovlp();

  std::array<dbcsr::shared_matrix<double>, 3> ao_emultipole(
      std::array<int, 3> O = {0, 0, 0});

  void ao_2c2e_setup(metric m);

  void ao_3c1e_ovlp_setup();

  void ao_3c2e_setup(metric m);

  void ao_eri_setup(metric m);
  
  void ao_2c_fill(dbcsr::shared_matrix<double>& m_in);

  void ao_3c_fill(dbcsr::shared_tensor<3, double>& t_in);

  void ao_3c_fill(
      dbcsr::shared_tensor<3, double>& t_in,
      vec<vec<int>>& blkbounds,
      std::shared_ptr<screener> scr);

  void ao_3c_fill_idx(
      dbcsr::shared_tensor<3, double>& t_in,
      arrvec<int, 3>& idx,
      std::shared_ptr<screener> scr);

  void ao_4c_fill(
      dbcsr::shared_tensor<4, double>& t_in,
      vec<vec<int>>& blkbounds,
      std::shared_ptr<screener> scr);

  std::function<void(dbcsr::shared_tensor<3, double>&, vec<vec<int>>&)>
  get_generator(std::shared_ptr<screener> s_scr);

  desc::shared_molecule mol();

};  // end class aofactory

desc::shared_cluster_basis remove_lindep(
    world w,
    desc::shared_cluster_basis cbas,
    double cutoff,
    std::optional<std::string> opt_split,
    std::optional<int> opt_nsplit);

}  // end namespace ints

}  // namespace megalochem

#endif
