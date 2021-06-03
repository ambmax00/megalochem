#ifndef ADC_ADCMOD_H
#define ADC_ADCMOD_H

#ifndef TEST_MACRO
  #include <mpi.h>
  #include <dbcsr_matrix.hpp>
  #include <dbcsr_tensor.hpp>
  #include <tuple>
  #include "adc/adc_mvp.hpp"
  #include "desc/wfn.hpp"
  #include "ints/fitting.hpp"
  #include "ints/registry.hpp"
  #include "megalochem.hpp"
  #include "utils/mpi_time.hpp"
#endif

#include "utils/ppdirs.hpp"

namespace megalochem {

namespace adc {

enum class adcmethod { ri_ao_adc1, sos_cd_ri_adc2 };

inline adcmethod str_to_adcmethod(std::string str)
{
  if (str == "ri_ao_adc1") {
    return adcmethod::ri_ao_adc1;
  }
  else if (str == "sos_cd_ri_adc2") {
    return adcmethod::sos_cd_ri_adc2;
  }
  else {
    throw std::runtime_error("Unknwon adc method");
  }
}

#define ADCMOD_LIST \
  (((world), set_world), ((desc::shared_wavefunction), set_wfn), \
   ((util::optional<desc::shared_cluster_basis>), df_basis))

#define ADCMOD_OPTLIST \
  (((util::optional<int>), print, 0), ((util::optional<int>), nbatches_b, 5), \
   ((util::optional<int>), nbatches_x, 5), \
   ((util::optional<int>), nbatches_occ, 5), \
   ((util::optional<int>), nroots, 1), ((util::optional<int>), nguesses, 1), \
   ((util::optional<bool>), block, false), \
   ((util::optional<bool>), balanced, false), \
   ((util::optional<std::string>), method, "ri_ao_adc1"), \
   ((util::optional<std::string>), df_metric, "coulomb"), \
   ((util::optional<double>), conv, 1e-5), \
   ((util::optional<int>), dav_max_iter, 40), \
   ((util::optional<int>), diis_max_iter, 40), \
   ((util::optional<std::string>), eris, "core"), \
   ((util::optional<std::string>), imeds, "core"), \
   ((util::optional<std::string>), build_K, "dfao"), \
   ((util::optional<std::string>), build_J, "dfao"), \
   ((util::optional<std::string>), build_Z, "llmp_full"), \
   ((util::optional<double>), c_os, 1.3), \
   ((util::optional<double>), c_os_coupling, 1.17), \
   ((util::optional<int>), nlap, 5), ((util::optional<bool>), local, false), \
   ((util::optional<double>), cutoff, 1.0), \
   ((util::optional<std::string>), local_method, "pao"), \
   ((util::optional<double>), ortho_eps, 1e-12), \
   ((util::optional<std::string>), guess, "hf"))

struct eigenpair {
  std::vector<double> eigvals;
  std::vector<dbcsr::shared_matrix<double>> eigvecs;
};

class adcmod {
 private:
  struct canon_lmo {
    dbcsr::shared_matrix<double> c_ao_lmo_bo, c_ao_lmo_bv, u_lmo_cmo_oo,
        u_lmo_cmo_vv;
    std::vector<double> eps_occ, eps_vir;
  };

  struct canon_lmo_mol {
    desc::shared_molecule mol;
    dbcsr::shared_matrix<double> c_ao_lmo_bo, c_ao_lmo_bv, u_lmo_cmo_oo,
        u_lmo_cmo_vv;
    std::vector<double> eps_occ, eps_vir;
  };

  megalochem::world m_world;
  desc::shared_wavefunction m_wfn;
  dbcsr::cart m_cart;
  desc::shared_cluster_basis m_df_basis;

  MAKE_MEMBER_VARS(ADCMOD_OPTLIST)

  adcmethod m_adcmethod;

  util::mpi_log LOG;
  util::mpi_time TIME;

  std::shared_ptr<ints::aoloader> m_aoloader;

  void init_ao_tensors();

  std::shared_ptr<MVP> create_adc1(
      std::optional<canon_lmo> lmo_info = std::nullopt);
  std::shared_ptr<MVP> create_adc2(
      std::optional<canon_lmo> clmo = std::nullopt);

  /*dbcsr::shared_matrix<double> compute_diag(
          dbcsr::shared_matrix<double> c_bo,
          dbcsr::shared_matrix<double> c_bv,
          std::vector<double> eps_o,
          std::vector<double> eps_v);*/

  eigenpair guess();

  eigenpair run_adc1(eigenpair& dav);

  eigenpair run_adc1_local(eigenpair& dav);

  eigenpair run_adc2(eigenpair& dav);

  eigenpair run_adc2_local(eigenpair& dav);

  dbcsr::shared_matrix<double> compute_diag_0(
      std::vector<int> o,
      std::vector<int> v,
      std::vector<double> eps_o,
      std::vector<double> eps_v);

  // dbcsr::shared_matrix<double> compute_diag_1();

  std::vector<bool> get_significant_blocks(
      dbcsr::shared_matrix<double> u_ia, double theta);

  std::vector<bool> get_significant_atoms(dbcsr::shared_matrix<double> u_ia);

  canon_lmo get_canon_nto(dbcsr::shared_matrix<double> u_ia);

  canon_lmo get_canon_pao(dbcsr::shared_matrix<double> u_ia);

  // canon_lmo_mol get_restricted_cmos2(dbcsr::shared_matrix<double> u_ia);

  /*canon_lmo get_canon_pao(dbcsr::shared_matrix<double> u_ia,
     dbcsr::shared_matrix<double> c_bo, dbcsr::shared_matrix<double> c_bv,
     std::vector<double> eps_o, std::vector<double> eps_v, double theta);*/

  void init();

  std::tuple<dbcsr::shared_matrix<double>, dbcsr::sbtensor<3, double>>
  test_fitting(std::vector<bool> atom_idx);

  dbcsr::sbtensor<3, double> m_fit;

 public:
  MAKE_PARAM_STRUCT(create, CONCAT(ADCMOD_LIST, ADCMOD_OPTLIST), ())

  MAKE_BUILDER_CLASS(adcmod, create, CONCAT(ADCMOD_LIST, ADCMOD_OPTLIST), ())

  adcmod(create_pack&& p) :
      m_world(p.p_set_world), m_wfn(p.p_set_wfn), m_cart(m_world.dbcsr_grid()),
      m_df_basis(p.p_df_basis ? *p.p_df_basis : nullptr),
      MAKE_INIT_LIST_OPT(ADCMOD_OPTLIST), LOG(m_world.comm(), m_print),
      TIME(m_world.comm(), "adcmod", m_print)
  {
    init();
  }

  ~adcmod()
  {
  }

  desc::shared_wavefunction compute();
};

}  // namespace adc

}  // namespace megalochem

#endif
