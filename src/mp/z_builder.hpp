#ifndef MP_Z_BUILDER_H
#define MP_Z_BUILDER_H

#ifndef TEST_MACRO
  #include <Eigen/Core>
  #include <dbcsr_btensor.hpp>
  #include <dbcsr_conversions.hpp>
  #include <dbcsr_matrix_ops.hpp>
  #include <dbcsr_tensor_ops.hpp>
  #include "desc/molecule.hpp"
  #include "ints/aoloader.hpp"
  #include "megalochem.hpp"
  #include "utils/mpi_time.hpp"
#endif

#include "utils/ppdirs.hpp"

namespace megalochem {

namespace mp {

enum class zmethod { llmp_full, llmp_mem, ll_full };

inline zmethod str_to_zmethod(std::string str)
{
  if (str == "llmp_full") {
    return zmethod::llmp_full;
  }
  else if (str == "llmp_mem") {
    return zmethod::llmp_mem;
  }
  else if (str == "ll_full") {
    return zmethod::ll_full;
  }
  else {
    throw std::runtime_error("Invalid zbuilder mathod.");
  }
}

using SMatrixXi = std::shared_ptr<Eigen::MatrixXi>;
SMatrixXi get_shellpairs(dbcsr::sbtensor<3, double> eri_batched);

class Z {
 protected:
  world m_world;
  dbcsr::cart m_cart;
  desc::shared_molecule m_mol;
  util::mpi_log LOG;
  util::mpi_time TIME;

  dbcsr::shared_pgrid<2> m_spgrid2;

  dbcsr::shared_matrix<double> m_zmat;
  dbcsr::shared_tensor<2, double> m_zmat_01;

  dbcsr::shared_matrix<double> m_pocc;
  dbcsr::shared_matrix<double> m_pvir;
  dbcsr::shared_matrix<double> m_locc;
  dbcsr::shared_matrix<double> m_lvir;

  SMatrixXi m_shellpair_info;

 public:
  Z(world w, desc::shared_molecule smol, int nprint, std::string mname) :
      m_world(w), m_cart(w.dbcsr_grid()), m_mol(smol), LOG(w.comm(), nprint),
      TIME(w.comm(), mname)
  {
  }

  Z& set_occ_density(dbcsr::shared_matrix<double>& pocc)
  {
    m_pocc = pocc;
    return *this;
  }

  Z& set_vir_density(dbcsr::shared_matrix<double>& pvir)
  {
    m_pvir = pvir;
    return *this;
  }

  Z& set_occ_coeff(dbcsr::shared_matrix<double>& locc)
  {
    m_locc = locc;
    return *this;
  }

  Z& set_vir_coeff(dbcsr::shared_matrix<double>& lvir)
  {
    m_lvir = lvir;
    return *this;
  }

  Z& set_shellpair_info(SMatrixXi& spinfo)
  {
    m_shellpair_info = spinfo;
    return *this;
  }

  virtual void init() = 0;
  virtual void compute() = 0;

  virtual ~Z()
  {
  }

  dbcsr::shared_matrix<double> zmat()
  {
    return m_zmat;
  }

  void print_info()
  {
    TIME.print_info();
  }
};

#define Z_INIT_LIST \
  (((world), set_world), ((desc::shared_molecule), set_molecule), \
   ((util::optional<int>), print))

#define Z_INIT_CON(name) \
  Z(p.p_set_world, p.p_set_molecule, (p.p_print) ? *p.p_print : 0, #name)

class LLMP_FULL_Z : public Z {
 private:
  dbcsr::sbtensor<3, double> m_eri3c2e_batched;
  dbcsr::btype m_intermeds;

  dbcsr::sbtensor<3, double> m_FT3c2e_batched;
  dbcsr::shared_tensor<2, double> m_locc_01;
  dbcsr::shared_tensor<2, double> m_pvir_01;

 public:
#define LLMP_FULL_Z_LIST \
  (((dbcsr::sbtensor<3, double>), eri3c2e_batched), ((dbcsr::btype), intermeds))

  MAKE_PARAM_STRUCT(create, CONCAT(Z_INIT_LIST, LLMP_FULL_Z_LIST), ())
  MAKE_BUILDER_CLASS(
      LLMP_FULL_Z, create, CONCAT(Z_INIT_LIST, LLMP_FULL_Z_LIST), ())

  LLMP_FULL_Z(create_pack&& p) :
      Z_INIT_CON(LLMP_FULL_Z), m_eri3c2e_batched(p.p_eri3c2e_batched),
      m_intermeds(p.p_intermeds)
  {
  }

  void init() override;
  void compute() override;

  ~LLMP_FULL_Z() override
  {
  }
};

class LL_FULL_Z : public Z {
 private:
  dbcsr::sbtensor<3, double> m_eri3c2e_batched;
  dbcsr::btype m_intermeds;

  dbcsr::shared_tensor<2, double> m_locc_01;
  dbcsr::shared_tensor<2, double> m_lvir_01;

 public:
#define LL_FULL_Z_LIST \
  (((dbcsr::sbtensor<3, double>), eri3c2e_batched), ((dbcsr::btype), intermeds))

  MAKE_PARAM_STRUCT(create, CONCAT(Z_INIT_LIST, LL_FULL_Z_LIST), ())
  MAKE_BUILDER_CLASS(LL_FULL_Z, create, CONCAT(Z_INIT_LIST, LL_FULL_Z_LIST), ())

  LL_FULL_Z(create_pack&& p) :
      Z_INIT_CON(LL_FULL_Z), m_eri3c2e_batched(p.p_eri3c2e_batched),
      m_intermeds(p.p_intermeds)
  {
  }

  void init() override;
  void compute() override;

  ~LL_FULL_Z() override
  {
  }
};

/*
class create_LL_Z_base;

class LL_Z : public Z {
private:

        dbcsr::sbtensor<3,double> m_eri3c2e_batched;
        dbcsr::btype m_intermeds;

        dbcsr::shared_tensor<2,double> m_locc_01;
        dbcsr::shared_tensor<2,double> m_pvir_01;

public:

        LL_Z(dbcsr::cart w, desc::shared_molecule smol, int nprint) :
                Z(w,smol,nprint,"LL") {}

        void init() override;
        void compute() override;

        ~LL_Z() override {}


};

MAKE_STRUCT(
        LL_Z, Z,
        (
                (cart, (dbcsr::cart)),
                (mol, (desc::shared_molecule)),
                (print, (int))
        ),
        (
                (eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
                (intermeds, (dbcsr::btype), required, val)
        )
)
*/

class LLMP_MEM_Z : public Z {
 private:
  dbcsr::sbtensor<3, double> m_eri3c2e_batched;

  dbcsr::shared_tensor<2, double> m_locc_01;
  dbcsr::shared_tensor<2, double> m_pvir_01;

 public:
#define LLMP_MEM_Z_LIST (((dbcsr::sbtensor<3, double>), eri3c2e_batched))

  MAKE_PARAM_STRUCT(create, CONCAT(Z_INIT_LIST, LLMP_MEM_Z_LIST), ())
  MAKE_BUILDER_CLASS(
      LLMP_MEM_Z, create, CONCAT(Z_INIT_LIST, LLMP_MEM_Z_LIST), ())

  LLMP_MEM_Z(create_pack&& p) :
      Z_INIT_CON(LLMP_MEM_Z), m_eri3c2e_batched(p.p_eri3c2e_batched)
  {
  }

  void init() override;
  void compute() override;

  ~LLMP_MEM_Z() override
  {
  }
};

/*
class LLMP_ASYM_Z : public Z {
private:

        dbcsr::sbtensor<3,double> m_t3c2e_right_batched;
        dbcsr::sbtensor<3,double> m_t3c2e_left_batched;

        dbcsr::shared_tensor<2,double> m_locc_01;
        dbcsr::shared_tensor<2,double> m_pvir_01;

public:

#:set list = [&
        ['cart', 'dbcsr::cart', _REQ, _VAL],&
        ['molecule', 'desc::shared_molecule', _REQ, _VAL],&
        ['print', 'int', _OPT, _VAL],&
        ['t3c2e_left_batched', 'dbcsr::sbtensor<3,double>', _REQ, _VAL],&
        ['t3c2e_right_batched', 'dbcsr::sbtensor<3,double>', _REQ, _VAL]]

${_MAKE_PARAM_STRUCT('create', list)}$
${_MAKE_BUILDER_CLASS('LLMP_ASYM_Z', 'create', list, True)}$

        LLMP_ASYM_Z(create_pack&& p) :
                m_t3c2e_left_batched(p.p_t3c2e_left_batched),
                m_t3c2e_right_batched(p.p_t3c2e_right_batched),
                ${_INIT('LLMP_ASYM')}$
        {}

        void init() override;
        void compute() override;

        ~LLMP_ASYM_Z() override {}

};*/

inline void load_zints(
    [[maybe_unused]] zmethod zmet, ints::metric metr, ints::aoloader& ao)
{
  // set J
  ao.request(ints::key::scr_xbb, true);

  switch (metr) {
    case ints::metric::coulomb: {
      ao.request(ints::key::coul_xx, false);
      ao.request(ints::key::coul_xx_inv, true);
      ao.request(ints::key::coul_xbb, true);
      break;
    }
    case ints::metric::erfc_coulomb: {
      ao.request(ints::key::erfc_xx, false);
      ao.request(ints::key::erfc_xx_inv, true);
      ao.request(ints::key::erfc_xbb, true);
      break;
    }
    case ints::metric::qr_fit: {
      ao.request(ints::key::ovlp_bb, false);
      ao.request(ints::key::coul_xx, true);
      ao.request(ints::key::ovlp_xx, false);
      ao.request(ints::key::ovlp_xx_inv, false);
      ao.request(ints::key::qr_xbb, true);
      break;
    }
    default: {
      throw std::runtime_error("Invalid metric for z builder");
    }
  }
}

#define ZBUILDER_LIST \
  (((zmethod), method), ((ints::aoloader&), aoloader), \
   ((util::optional<dbcsr::btype>), btype_intermeds), \
   ((ints::metric), metric))

class create_z_base {
 private:
  typedef create_z_base _create_base;

  MAKE_BUILDER_MEMBERS(create, CONCAT(Z_INIT_LIST, ZBUILDER_LIST))

 public:
  MAKE_BUILDER_SETS(create, CONCAT(Z_INIT_LIST, ZBUILDER_LIST))

  create_z_base()
  {
  }

  std::shared_ptr<Z> build()
  {
    CHECK_REQUIRED(CONCAT(Z_INIT_LIST, ZBUILDER_LIST))

    auto aoreg = c_aoloader->get_registry();

    dbcsr::sbtensor<3, double> eri3c2e_batched;

    switch (*c_metric) {
      case ints::metric::coulomb: {
        eri3c2e_batched =
            aoreg.get<dbcsr::sbtensor<3, double>>(ints::key::coul_xbb);
        break;
      }
      case ints::metric::erfc_coulomb: {
        eri3c2e_batched =
            aoreg.get<dbcsr::sbtensor<3, double>>(ints::key::erfc_xbb);
        break;
      }
      case ints::metric::qr_fit: {
        eri3c2e_batched =
            aoreg.get<dbcsr::sbtensor<3, double>>(ints::key::qr_xbb);
        break;
      }
      default: {
        throw std::runtime_error("Invalid metric for z builder.");
      }
    }

    std::shared_ptr<Z> zbuilder;

    if (*c_method == zmethod::llmp_full) {
      zbuilder =
          LLMP_FULL_Z::create()
              .set_world(c_set_world)
              .set_molecule(c_set_molecule)
              .print((c_print) ? *c_print : 0)
              .eri3c2e_batched(eri3c2e_batched)
              .intermeds(
                  (c_btype_intermeds) ? *c_btype_intermeds : dbcsr::btype::core)
              .build();
    }
    else if (*c_method == zmethod::llmp_mem) {
      zbuilder = LLMP_MEM_Z::create()
                     .set_world(c_set_world)
                     .set_molecule(c_set_molecule)
                     .print((c_print) ? *c_print : 0)
                     .eri3c2e_batched(eri3c2e_batched)
                     .build();
    }
    else if (*c_method == zmethod::ll_full) {
      zbuilder =
          LL_FULL_Z::create()
              .set_world(c_set_world)
              .set_molecule(c_set_molecule)
              .print((c_print) ? *c_print : 0)
              .intermeds(
                  (c_btype_intermeds) ? *c_btype_intermeds : dbcsr::btype::core)
              .eri3c2e_batched(eri3c2e_batched)
              .build();
    }

    return zbuilder;
  }
};

inline create_z_base create_z()
{
  return create_z_base();
}

}  // namespace mp

}  // namespace megalochem

#endif
