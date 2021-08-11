#ifndef AOLOADER_H
#define AOLOADER_H

#ifndef TEST_MACRO
  #include <dbcsr_common.hpp>
  #include "desc/molecule.hpp"
  #include "ints/aofactory.hpp"
  #include "ints/registry.hpp"
  #include "ints/screening.hpp"
  #include "utils/mpi_time.hpp"
#endif

#include "utils/ppdirs.hpp"

namespace megalochem {

namespace ints {

// clang-format off
enum class key {
  pgrid2 = 0,
  pgrid3 = 1,
  pgrid4 = 2,  // processor grids
  
  ovlp_bb = 3,  // overlap matrix
  ovlp_xx = 4,  // overlap matrix for aux basis
  pot_bb = 5,  // nuc. potential ints
  kin_bb = 6,  // kin. ints
  ovlp_bb_inv = 7,
  ovlp_xx_inv = 8,
  
  coul_bbbb = 9,
  
  coul_xbb = 10,  // 3c2e integrals
  erfc_xbb = 11,  // 3c2e integrals in coulomb metric
  
  coul_xx = 12,  // 2c2e ints in coulomb metric
  erfc_xx = 13,
  
  coul_xx_inv = 14,
  erfc_xx_inv = 15,
  coul_xx_invsqrt = 16,
  erfc_xx_invsqrt = 17,
  
  erfc_xx_prod = 18, // (erfc_xx)^-1 * coul_xx * erfc_xx^-1
  
  dfit_coul_xbb = 19,  // fitting coefficients
  dfit_erfc_xbb = 20,
  dfit_pari_xbb = 21,
  dfit_qr_xbb = 22,
  scr_xbb = 23,
  qr_xbb = 24,
  pari_xbb = 25,
  
  NUM_KEYS = 26
};

inline std::map<ints::key,std::vector<ints::key>> dependencies = {
  {ints::key::coul_xx_inv,        {ints::key::coul_xx}},
  {ints::key::coul_xx_invsqrt,    {ints::key::coul_xx}},
  {ints::key::erfc_xx_inv,        {ints::key::erfc_xx}},
  {ints::key::erfc_xx_invsqrt,    {ints::key::erfc_xx}},
  {ints::key::ovlp_bb_inv,        {ints::key::ovlp_bb}},
  {ints::key::erfc_xx_prod,       {ints::key::erfc_xx_inv, ints::key::erfc_xx, ints::key::coul_xx}},
  
  {ints::key::coul_xbb,           {ints::key::scr_xbb}},
  {ints::key::erfc_xbb,           {ints::key::scr_xbb}},
  {ints::key::qr_xbb,             {ints::key::ovlp_xx_inv, ints::key::ovlp_bb}},
  
  {ints::key::dfit_coul_xbb,      {ints::key::coul_xx_inv, ints::key::coul_xbb}},
  {ints::key::dfit_erfc_xbb,      {ints::key::erfc_xx, ints::key::erfc_xx_inv, ints::key::coul_xx, 
                                      ints::key::erfc_xx_prod}},
  {ints::key::dfit_pari_xbb,      {ints::key::pari_xbb, ints::key::coul_xx}},
  {ints::key::dfit_qr_xbb,        {ints::key::ovlp_bb, ints::key::ovlp_xx_inv, 
                                        ints::key::qr_xbb, ints::key::coul_xx}}
};
// clang-format on

class aoloader {
 private:
  world m_world;
  dbcsr::cart m_cart;
  desc::shared_molecule m_mol;
  dbcsr::btype m_btype_eris;
  dbcsr::btype m_btype_intermeds;
  int m_nbatches_b;
  int m_nbatches_x;

  util::mpi_log LOG;
  util::mpi_time TIME;

  ints::key_registry<key> m_reg;

  std::array<bool, static_cast<int>(key::NUM_KEYS)> m_to_compute;
  std::array<bool, static_cast<int>(key::NUM_KEYS)> m_to_keep;

  std::pair<dbcsr::shared_matrix<double>, dbcsr::shared_matrix<double>> invert(
      dbcsr::shared_matrix<double> mat, bool do_inv, bool do_invsqrt, 
      std::optional<double> cutoff);

  bool comp(key k)
  {
    return m_to_compute[static_cast<int>(k)];
  }

 public:
#define AOLOADER_CREATE_LIST \
  (((world), set_world), ((desc::shared_molecule), set_molecule), \
   ((util::optional<int>), print), ((util::optional<int>), nbatches_b), \
   ((util::optional<int>), nbatches_x), \
   ((util::optional<dbcsr::btype>), btype_eris), \
   ((util::optional<dbcsr::btype>), btype_intermeds))

  MAKE_PARAM_STRUCT(create, AOLOADER_CREATE_LIST, ())
  MAKE_BUILDER_CLASS(aoloader, create, AOLOADER_CREATE_LIST, ())

  aoloader(create_pack&& p) :
      m_world(p.p_set_world), m_cart(p.p_set_world.dbcsr_grid()),
      m_mol(p.p_set_molecule),
      m_btype_eris((p.p_btype_eris) ? *p.p_btype_eris : dbcsr::btype::core),
      m_btype_intermeds(
          (p.p_btype_intermeds) ? *p.p_btype_intermeds : dbcsr::btype::core),
      m_nbatches_b((p.p_nbatches_b) ? *p.p_nbatches_b : 5),
      m_nbatches_x((p.p_nbatches_x) ? *p.p_nbatches_x : 5),
      LOG(p.p_set_world.comm(), (p.p_print) ? *p.p_print : 0),
      TIME(p.p_set_world.comm(), "AO-loader")
  {
    for (auto& a : m_to_compute) a = false;
    for (auto& a : m_to_keep) a = false;
  }

  aoloader& request(key k)
  {
    
    int kint = static_cast<int>(k);
    
    m_to_compute[kint] = true;
    m_to_keep[kint] = true;

    auto dep = dependencies.find(k);
    if (dep != dependencies.end()) {
      // request dependencies, too
      for (auto dkey : dep->second) {
        int dkint = static_cast<int>(dkey);
        m_to_compute[dkint] = true;
      }
    }
    
    return *this;
  }

  void compute();

  ~aoloader()
  {
  }

  void print_info()
  {
    TIME.print_info();
  }

  const ints::key_registry<key> get_registry()
  {
    return m_reg;
  }

};  // end class

}  // namespace ints

}  // end namespace megalochem

#endif
