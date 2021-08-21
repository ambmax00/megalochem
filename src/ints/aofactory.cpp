#include "ints/aofactory.hpp"
#include <stdexcept>
#include <vector>
#include "ints/integrals.hpp"
#include "ints/screening.hpp"
#include "math/linalg/piv_cd.hpp"
#include "math/linalg/newton_schulz.hpp"
#include "utils/mpi_time.hpp"

extern "C" {
#include <cint.h>
}

#include <iostream>
#include <limits>

namespace megalochem {

namespace ints {

enum class op {
  invalid = 0,
  overlap = 1,
  coulomb = 2,
  kinetic = 3,
  nuclear = 4,
  emultipole = 5,
  erfc_coulomb = 6,
  _END = 7
};

enum class ctr {
  invalid = 0,
  c_2c1e = 1,
  c_2c2e = 2,
  c_3c1e = 3,
  c_3c2e = 4,
  c_4c2e = 5,
  c_4c1e = 6,
  _END = 7
};

constexpr int combine(op e1, ctr e2)
{
  const int pos1 = static_cast<int>(e1);
  const int pos2 = static_cast<int>(e2);
  const int size1 = static_cast<int>(op::_END);
  return pos1 + pos2 * size1;
}

class aofactory::impl {
 private:
  void reserve_3_partial(
      dbcsr::shared_tensor<3>& t_in,
      vec<vec<int>>& blkbounds,
      shared_screener s_scr)
  {
    auto scr = s_scr.get();

    auto blksizes = t_in->blk_sizes();

    size_t totblk = 0;

    auto blk_idx_loc = t_in->blks_local();

    auto idx_speed = t_in->idx_speed();

    const int dim0 = idx_speed[2];
    const int dim1 = idx_speed[1];
    const int dim2 = idx_speed[0];

    const size_t nblk0 = blkbounds[0][1] - blkbounds[0][0] + 1;
    const size_t nblk1 = blkbounds[1][1] - blkbounds[1][0] + 1;
    const size_t nblk2 = blkbounds[2][1] - blkbounds[2][0] + 1;

    const size_t maxblks = nblk0 * nblk1 * nblk2;

    int iblk[3];

    arrvec<int, 3> res;

    for (auto& r : res) r.reserve(maxblks);

    for (int i0 = 0; i0 != int(blk_idx_loc[dim0].size()); ++i0) {
      iblk[dim0] = blk_idx_loc[dim0][i0];
      if (iblk[dim0] < blkbounds[dim0][0] || iblk[dim0] > blkbounds[dim0][1])
        continue;

      for (int i1 = 0; i1 != int(blk_idx_loc[dim1].size()); ++i1) {
        iblk[dim1] = blk_idx_loc[dim1][i1];
        if (iblk[dim1] < blkbounds[dim1][0] || iblk[dim1] > blkbounds[dim1][1])
          continue;

        for (int i2 = 0; i2 != int(blk_idx_loc[dim2].size()); ++i2) {
          iblk[dim2] = blk_idx_loc[dim2][i2];

          if (iblk[dim2] < blkbounds[dim2][0] ||
              iblk[dim2] > blkbounds[dim2][1])
            continue;

          if (scr && scr->skip_block_xbb(iblk[0], iblk[1], iblk[2])) {
            continue;
          }

          ++totblk;

          res[0].push_back(iblk[0]);
          res[1].push_back(iblk[1]);
          res[2].push_back(iblk[2]);
        }
      }
    }

    auto x = m_cdfbas->cluster_sizes();
    auto b = m_cbas->cluster_sizes();

    double mem = 0.0;
    double mem_tot = 0.0;

    for (size_t i = 0; i != res[0].size(); ++i) {
      mem += x[res[0][i]] * b[res[1][i]] * b[res[2][i]] * 8;
    }

    for (int ip = 0; ip != m_world.size(); ++ip) {
      if (ip == m_world.rank()) {
        std::cout << "RANK: " << mem / 1e+9
                  << " GB will be reserved. Total number of blocks: " << totblk
                  << std::endl;
      }
      MPI_Barrier(m_world.comm());
    }

    MPI_Allreduce(&mem, &mem_tot, 1, MPI_DOUBLE, MPI_SUM, m_world.comm());
    if (m_world.rank() == 0)
      std::cout << "TOTAL MEM: " << mem_tot << std::endl;

    t_in->reserve(res);
  }

  void reserve_3_partial_idx(
      dbcsr::shared_tensor<3>& t_in, arrvec<int, 3>& idx, shared_screener s_scr)
  {
    auto scr = s_scr.get();

    arrvec<int, 3> newblks;

    for (auto x : idx[0]) {
      for (auto m : idx[1]) {
        for (auto n : idx[2]) {
          if (scr && scr->skip_block_xbb(x, m, n))
            continue;

          newblks[0].push_back(x);
          newblks[1].push_back(m);
          newblks[2].push_back(n);
        }
      }
    }

    t_in->reserve(newblks);
  }

  void reserve_4_partial(
      dbcsr::shared_tensor<4>& t_in,
      vec<vec<int>>& blkbounds,
      shared_screener s_scr)
  {
    auto scr = s_scr.get();

    auto blksizes = t_in->blk_sizes();

    auto blk_idx_loc = t_in->blks_local();

    auto idx_speed = t_in->idx_speed();

    const int dim0 = idx_speed[3];
    const int dim1 = idx_speed[2];
    const int dim2 = idx_speed[1];
    const int dim3 = idx_speed[0];

    const size_t nblk0 = blkbounds[0][1] - blkbounds[0][0] + 1;
    const size_t nblk1 = blkbounds[1][1] - blkbounds[1][0] + 1;
    const size_t nblk2 = blkbounds[2][1] - blkbounds[2][0] + 1;
    const size_t nblk3 = blkbounds[3][1] - blkbounds[3][0] + 1;

    const size_t maxblks = nblk0 * nblk1 * nblk2 * nblk3;

    int iblk[4];

    arrvec<int, 4> res;

    for (auto& r : res) r.reserve(maxblks);

    for (int i0 = 0; i0 != (int)blk_idx_loc[dim0].size(); ++i0) {
      iblk[dim0] = blk_idx_loc[dim0][i0];
      if (iblk[dim0] < blkbounds[dim0][0] || iblk[dim0] > blkbounds[dim0][1])
        continue;

      for (int i1 = 0; i1 != (int)blk_idx_loc[dim1].size(); ++i1) {
        iblk[dim1] = blk_idx_loc[dim1][i1];
        if (iblk[dim1] < blkbounds[dim1][0] || iblk[dim1] > blkbounds[dim1][1])
          continue;

        for (int i2 = 0; i2 != (int)blk_idx_loc[dim2].size(); ++i2) {
          iblk[dim2] = blk_idx_loc[dim2][i2];

          if (iblk[dim2] < blkbounds[dim2][0] ||
              iblk[dim2] > blkbounds[dim2][1])
            continue;

          for (int i3 = 0; i3 != (int)blk_idx_loc[dim3].size(); ++i3) {
            iblk[dim3] = blk_idx_loc[dim3][i3];

            if (iblk[dim3] < blkbounds[dim3][0] ||
                iblk[dim3] > blkbounds[dim3][1])
              continue;

            if (scr && scr->skip_block_bbbb(i0, i1, i2, i3))
              continue;

            res[0].push_back(iblk[0]);
            res[1].push_back(iblk[1]);
            res[2].push_back(iblk[2]);
            res[3].push_back(iblk[3]);
          }
        }
      }
    }

    t_in->reserve(res);
  }

 public:
  std::function<void(dbcsr::shared_tensor<3>&, vec<vec<int>>&)> get_generator(
      shared_screener s_scr)
  {
    using namespace std::placeholders;

    auto gen =
        std::bind(&aofactory::impl::compute_3_partial, this, _1, _2, s_scr);

    return gen;
  }

 protected:
  world m_world;

  dbcsr::cart m_cart;

  std::vector<desc::Atom> m_atoms;

  desc::shared_cluster_basis m_cbas, m_cdfbas, m_cbas2;

  std::vector<int> m_b_cint_offsets, m_x_cint_offsets, m_b2_cint_offsets;

  std::vector<int> m_cint_atm;
  std::vector<int> m_cint_bas;
  std::vector<double> m_cint_env;

  int m_cint_natoms;
  int m_cint_nbas;

  CINTIntegralFunction* m_intfunc;

  std::string m_intname;
  ctr m_ctr = ctr::invalid;
  op m_op = op::invalid;
  int m_max_l;

  std::vector<std::vector<int>> m_shell_offsets;
  std::vector<std::vector<int>> m_nshells;
  std::vector<std::vector<int>> m_tensor_sizes;

  double gaussian_int(int l, double alpha)
  {
    double l1 = (l + 1) * 0.5;
    double res = tgamma(l1) / (2.0 * pow(alpha, l1));
    return res;
  }

  double gto_norm(int l, double e)
  {
    return 1.0 / sqrt(gaussian_int(l * 2 + 2, 2 * e));
  }

  std::vector<double> gto_normalize(
      int l, std::vector<double> es, std::vector<double> cs)
  {
    auto cs_norm = cs;
    for (size_t i = 0; i != cs_norm.size(); ++i) {
      cs_norm[i] *= gto_norm(l, es[i]);
    }

    size_t nprim = es.size();
    std::vector<double> ee(nprim * nprim);

    for (size_t ia = 0; ia != nprim; ++ia) {
      for (size_t ib = 0; ib != nprim; ++ib) {
        ee[ia + ib * nprim] = gaussian_int(l * 2 + 2, es[ia] + es[ib]);
      }
    }

    double norm = 0.0;
    for (size_t p = 0; p != nprim; ++p) {
      for (size_t q = 0; q != nprim; ++q) {
        norm += cs_norm[p] * ee[p + q * nprim] * cs_norm[q];
      }
    }

    norm = 1.0 / sqrt(norm);

    for (size_t i = 0; i != nprim; ++i) { cs_norm[i] *= norm; }

    return cs_norm;
  }

 public:
  impl(desc::shared_molecule mol, world w) :
      m_world(w), m_cart(w.dbcsr_grid()), m_atoms(mol->atoms()),
      m_cbas(mol->c_basis()), m_cdfbas(mol->c_dfbasis()),
      m_cbas2(mol->c_basis2()), m_cint_natoms(0), m_cint_nbas(0), m_max_l(0)
  {
    init();
  }

  impl(
      world w,
      desc::shared_cluster_basis cbas,
      desc::shared_cluster_basis cdfbas,
      desc::shared_cluster_basis cbas2) :
      m_world(w),
      m_cart(w.dbcsr_grid()), m_cbas(cbas), m_cdfbas(cdfbas), m_cbas2(cbas2),
      m_cint_natoms(0), m_cint_nbas(0), m_max_l(0)
  {
    for (auto& cltr : *cbas) {
      for (auto shell : cltr.shells) {
        // std::cout << "CLUSTER" << std::endl;

        // check if coordinates inside
        auto it = std::find_if(
            m_atoms.begin(), m_atoms.end(), [&shell](const desc::Atom& a) {
              return (fabs(a.x - shell.O[0]) < 1e-12) &&
                  (fabs(a.y - shell.O[1]) < 1e-12) &&
                  (fabs(a.z - shell.O[2]) < 1e-12);
            });

        if (it == m_atoms.end()) {
          m_atoms.push_back(desc::Atom{shell.O[0], shell.O[1], shell.O[2], 0});
        }
      }
    }

    init();
  }

  void init()
  {
    // atoms
    m_cint_natoms = m_atoms.size();
    int off = PTR_ENV_START;

    m_cint_atm.resize(ATM_SLOTS * m_cint_natoms);

    m_cint_env.resize(off);
    m_cint_env[PTR_RANGE_OMEGA] = 0;

    for (int i = 0; i != m_cint_natoms; ++i) {
      m_cint_atm[i * ATM_SLOTS + CHARGE_OF] = m_atoms[i].atomic_number;
      m_cint_atm[i * ATM_SLOTS + PTR_COORD] = off;

      m_cint_env.push_back(m_atoms[i].x);
      m_cint_env.push_back(m_atoms[i].y);
      m_cint_env.push_back(m_atoms[i].z);

      off += 3;
    }

    // add unit shell
    std::vector<int> bas_unit(BAS_SLOTS);
    bas_unit[ATOM_OF] = 0;
    bas_unit[ANG_OF] = 0;
    bas_unit[NPRIM_OF] = 1;
    bas_unit[NCTR_OF] = 1;
    bas_unit[PTR_EXP] = off++;
    bas_unit[PTR_COEFF] = off++;

    m_cint_bas.insert(m_cint_bas.begin(), bas_unit.begin(), bas_unit.end());

    constexpr double two_sqrt_pi = 3.5449077018110320545963349666;

    m_cint_env.push_back(0.0);
    m_cint_env.push_back(two_sqrt_pi);

    auto add_basis = [this, &off](desc::cluster_basis& cbas) {
      for (auto& cltr : cbas) {
        // std::cout << "CLUSTER" << std::endl;

        for (auto& shell : cltr.shells) {
          // std::cout << shell << std::endl;

          m_max_l = std::max((size_t)m_max_l, shell.l);

          std::vector<int> bas_i(BAS_SLOTS);
          bas_i[ATOM_OF] = atom_of(shell, m_atoms);
          bas_i[ANG_OF] = shell.l;
          bas_i[NPRIM_OF] = shell.nprim();
          bas_i[NCTR_OF] = 1;
          bas_i[PTR_EXP] = off;

          for (size_t i = 0; i != shell.alpha.size(); ++i) {
            m_cint_env.push_back(shell.alpha[i]);
            ++off;
          }

          bas_i[PTR_COEFF] = off;

          auto coeff = gto_normalize(shell.l, shell.alpha, shell.coeff);

          for (size_t i = 0; i != coeff.size(); ++i) {
            m_cint_env.push_back(coeff[i]);
            ++off;
          }

          m_cint_bas.insert(m_cint_bas.end(), bas_i.begin(), bas_i.end());
        }
      }

      m_cint_nbas += cbas.nbf();
    };

    add_basis(*m_cbas);
    if (m_cdfbas)
      add_basis(*m_cdfbas);
    if (m_cbas2)
      add_basis(*m_cbas2);

    // std::cout << "MAX_L: " << m_max_l << std::endl;

    /*auto print = [](auto& v) {
            for (auto e : v) {
                    std::cout << e << " ";
            } std::cout << std::endl;
    };*/

    // print(m_atm);
    // print(m_bas);
    // print(m_env);

    // offsets

    off = 1;

    auto add_offsets = [&](int& off, std::vector<int> nshells) {
      std::vector<int> shell_offsets = nshells;

      for (size_t i = 0; i != nshells.size(); ++i) {
        shell_offsets[i] = off;
        off += nshells[i];
      }

      return shell_offsets;
    };

    m_b_cint_offsets = add_offsets(off, m_cbas->nshells());
    if (m_cdfbas) {
      m_x_cint_offsets = add_offsets(off, m_cdfbas->nshells());
    }
    if (m_cbas2) {
      m_b2_cint_offsets = add_offsets(off, m_cbas2->nshells());
    }
  }

  void set_operator(op i_op)
  {
    m_op = i_op;
  }

  void set_center(ctr i_ctr)
  {
    m_ctr = i_ctr;
  }

  void set_dim(std::string dim)
  {
    if (dim == "bb") {
      m_nshells = {m_cbas->nshells(), m_cbas->nshells()};
      m_shell_offsets = {m_b_cint_offsets, m_b_cint_offsets};
      m_tensor_sizes = {m_cbas->cluster_sizes(), m_cbas->cluster_sizes()};
    }
    else if (dim == "bb2") {
      m_nshells = {m_cbas->nshells(), m_cbas2->nshells()};
      m_shell_offsets = {m_b_cint_offsets, m_b2_cint_offsets};
      m_tensor_sizes = {m_cbas->cluster_sizes(), m_cbas2->cluster_sizes()};
    }
    else if (dim == "xx") {
      m_nshells = {m_cdfbas->nshells(), m_cdfbas->nshells()};
      m_shell_offsets = {m_x_cint_offsets, m_x_cint_offsets};
      m_tensor_sizes = {m_cdfbas->cluster_sizes(), m_cdfbas->cluster_sizes()};
    }
    else if (dim == "bbbb") {
      m_nshells = {
          m_cbas->nshells(),
          m_cbas->nshells(),
          m_cbas->nshells(),
          m_cbas->nshells(),
      };
      m_shell_offsets = {
          m_b_cint_offsets, m_b_cint_offsets, m_b_cint_offsets,
          m_b_cint_offsets};
      m_tensor_sizes = {
          m_cbas->cluster_sizes(), m_cbas->cluster_sizes(),
          m_cbas->cluster_sizes(), m_cbas->cluster_sizes()};
    }
    else if (dim == "xbb") {
      m_nshells = {m_cdfbas->nshells(), m_cbas->nshells(), m_cbas->nshells()};
      m_shell_offsets = {m_x_cint_offsets, m_b_cint_offsets, m_b_cint_offsets};
      m_tensor_sizes = {
          m_cdfbas->cluster_sizes(),
          m_cbas->cluster_sizes(),
          m_cbas->cluster_sizes(),
      };
    }
    else {
      throw std::runtime_error("Invalid dimension");
    }
  }

  void set_name(std::string istr)
  {
    m_intname = istr;
  }

  void setup_calc()
  {
    m_cint_env[PTR_RANGE_OMEGA] = 0.0;

    switch (combine(m_op, m_ctr)) {
      case combine(op::overlap, ctr::c_2c1e):
        m_intfunc = &int1e_ovlp_sph;
        break;

      case combine(op::kinetic, ctr::c_2c1e):
        m_intfunc = &int1e_kin_sph;
        break;

      case combine(op::nuclear, ctr::c_2c1e):
        m_intfunc = &int1e_nuc_sph;
        break;

      case combine(op::overlap, ctr::c_3c1e):
        m_intfunc = &int3c1e_sph;
        break;

      case combine(op::overlap, ctr::c_4c1e):
        m_intfunc = &int4c1e_sph;
        break;

      case combine(op::coulomb, ctr::c_2c2e):
        m_intfunc = &int2c2e_sph;
        break;

      case combine(op::coulomb, ctr::c_3c2e):
        m_cint_env[PTR_RANGE_OMEGA] = 0.0;
        m_intfunc = &int3c2e_sph;
        break;

      case combine(op::coulomb, ctr::c_4c2e):
        m_intfunc = &int2e_sph;
        break;

      case combine(op::erfc_coulomb, ctr::c_2c2e):
        m_cint_env[PTR_RANGE_OMEGA] = -global::omega;
        m_intfunc = &int2c2e_sph;
        break;

      case combine(op::erfc_coulomb, ctr::c_3c2e):
        m_cint_env[PTR_RANGE_OMEGA] = -global::omega;
        m_intfunc = &int3c2e_sph;
        break;

      case combine(op::erfc_coulomb, ctr::c_4c2e):
        m_cint_env[PTR_RANGE_OMEGA] = -global::omega;
        m_intfunc = &int2e_sph;
        break;

      case combine(op::emultipole, ctr::c_2c1e):
        m_intfunc = &int1e_r_sph;
        break;

      default:
        throw std::runtime_error("Operator/Centre combination not valid!");
    }
  }

  void finalize()
  {
    // ...
  }

  dbcsr::shared_matrix<double> compute()
  {
    bool sym = (m_tensor_sizes[0] == m_tensor_sizes[1]) ? true : false;

    dbcsr::type mtype =
        (sym) ? dbcsr::type::symmetric : dbcsr::type::no_symmetry;

    auto m_ints = dbcsr::matrix<double>::create()
                      .name(m_intname)
                      .set_cart(m_cart)
                      .row_blk_sizes(m_tensor_sizes[0])
                      .col_blk_sizes(m_tensor_sizes[1])
                      .matrix_type(mtype)
                      .build();

    if (sym) {
      m_ints->reserve_sym();
    }
    else {
      m_ints->reserve_all();
    }

    calc_ints(
        *m_ints, m_shell_offsets, m_nshells, m_intfunc, m_cint_atm.data(),
        m_cint_natoms, m_cint_bas.data(), m_cint_nbas, m_cint_env.data(),
        m_max_l);

    return m_ints;
  }
  
  void compute_2_reserved(dbcsr::shared_matrix<double>& m_ints) 
  {
    calc_ints(
        *m_ints, m_shell_offsets, m_nshells, m_intfunc, m_cint_atm.data(),
        m_cint_natoms, m_cint_bas.data(), m_cint_nbas, m_cint_env.data(),
        m_max_l);
  }

  std::array<dbcsr::shared_matrix<double>, 3> compute_xyz(std::array<int, 3> O)
  {
    auto ints_x = dbcsr::matrix<>::create()
                      .name(m_intname + "_x")
                      .set_cart(m_cart)
                      .row_blk_sizes(m_tensor_sizes[0])
                      .col_blk_sizes(m_tensor_sizes[1])
                      .matrix_type(dbcsr::type::symmetric)
                      .build();

    auto ints_y = dbcsr::matrix<>::create_template(*ints_x)
                      .name(m_intname + "_y")
                      .build();

    auto ints_z = dbcsr::matrix<>::create_template(*ints_x)
                      .name(m_intname + "_z")
                      .build();

    ints_x->reserve_sym();
    ints_y->reserve_sym();
    ints_z->reserve_sym();

    m_cint_env[PTR_COMMON_ORIG + 0] = O[0];
    m_cint_env[PTR_COMMON_ORIG + 1] = O[1];
    m_cint_env[PTR_COMMON_ORIG + 2] = O[2];

    calc_ints(
        *ints_x, *ints_y, *ints_z, m_shell_offsets, m_nshells, m_intfunc,
        m_cint_atm.data(), m_cint_natoms, m_cint_bas.data(), m_cint_nbas,
        m_cint_env.data(), m_max_l);

    std::array<dbcsr::shared_matrix<double>, 3> out = {ints_x, ints_y, ints_z};

    return out;
  }

  void compute_3_partial(
      dbcsr::shared_tensor<3>& t_in,
      vec<vec<int>>& blkbounds,
      shared_screener s_scr)
  {
    reserve_3_partial(t_in, blkbounds, s_scr);

    calc_ints(
        *t_in, m_shell_offsets, m_nshells, m_intfunc, m_cint_atm.data(),
        m_cint_natoms, m_cint_bas.data(), m_cint_nbas, m_cint_env.data(),
        m_max_l);
  }

  void compute_3_reserved(dbcsr::shared_tensor<3>& t_in)
  {
    calc_ints(
        *t_in, m_shell_offsets, m_nshells, m_intfunc, m_cint_atm.data(),
        m_cint_natoms, m_cint_bas.data(), m_cint_nbas, m_cint_env.data(),
        m_max_l);
  }

  void compute_3_partial_idx(
      dbcsr::shared_tensor<3>& t_in, arrvec<int, 3>& idx, shared_screener s_scr)
  {
    reserve_3_partial_idx(t_in, idx, s_scr);

    calc_ints(
        *t_in, m_shell_offsets, m_nshells, m_intfunc, m_cint_atm.data(),
        m_cint_natoms, m_cint_bas.data(), m_cint_nbas, m_cint_env.data(),
        m_max_l);
  }

  void compute_4_partial(
      dbcsr::shared_tensor<4>& t_in,
      vec<vec<int>>& blkbounds,
      shared_screener s_scr)
  {
    reserve_4_partial(t_in, blkbounds, s_scr);

    calc_ints(
        *t_in, m_shell_offsets, m_nshells, m_intfunc, m_cint_atm.data(),
        m_cint_natoms, m_cint_bas.data(), m_cint_nbas, m_cint_env.data(),
        m_max_l);
  }

  dbcsr::shared_matrix<double> compute_screen(
      std::string method, std::string dim)
  {
    auto rowsizes = (dim == "bbbb") ? m_cbas->nshells() : m_cdfbas->nshells();
    auto colsizes = (dim == "bbbb") ? m_cbas->nshells() : vec<int>{1};

    auto sym =
        (dim == "bbbb") ? dbcsr::type::symmetric : dbcsr::type::no_symmetry;

    auto m_ints = dbcsr::matrix<double>::create()
                      .name(m_intname)
                      .set_cart(m_cart)
                      .row_blk_sizes(rowsizes)
                      .col_blk_sizes(colsizes)
                      .matrix_type(sym)
                      .build();

    // reserve symmtric blocks
    int nblks = m_ints->nblkrows_total();

    if (sym == dbcsr::type::symmetric) {
      vec<int> resrows, rescols;

      for (int i = 0; i != nblks; ++i) {
        for (int j = 0; j != nblks; ++j) {
          if (m_ints->proc(i, j) == m_world.rank() && i <= j) {
            resrows.push_back(i);
            rescols.push_back(j);
          }
        }
      }

      m_ints->reserve_blocks(resrows, rescols);
    }
    else {
      m_ints->reserve_all();
    }

    if (dim == "bbbb" && method == "schwarz") {
      calc_ints_schwarz_mn(
          *m_ints, m_shell_offsets, m_nshells, m_intfunc, m_cint_atm.data(),
          m_cint_natoms, m_cint_bas.data(), m_cint_nbas, m_cint_env.data(),
          m_max_l);
    }
    else if (dim == "xx" && method == "schwarz") {
      calc_ints_schwarz_x(
          *m_ints, m_shell_offsets, m_nshells, m_intfunc, m_cint_atm.data(),
          m_cint_natoms, m_cint_bas.data(), m_cint_nbas, m_cint_env.data(),
          m_max_l);
    }
    else {
      throw std::runtime_error("Unknown screening method.");
    }

    // dbcsr::print(*m_ints);

    return m_ints;
  }
};

aofactory::aofactory(desc::shared_molecule mol, world w) :
    pimpl(new impl(mol, w))
{
}

aofactory::aofactory(
    world w,
    desc::shared_cluster_basis cbas,
    desc::shared_cluster_basis cdfbas,
    desc::shared_cluster_basis cbas2) :
    pimpl(new impl(w, cbas, cdfbas, cbas2))
{
}

aofactory::~aofactory()
{
  delete pimpl;
}

dbcsr::shared_matrix<double> aofactory::ao_overlap()
{
  pimpl->set_name("s_bb");
  pimpl->set_dim("bb");
  pimpl->set_center(ctr::c_2c1e);
  pimpl->set_operator(op::overlap);
  pimpl->setup_calc();
  return pimpl->compute();
}

dbcsr::shared_matrix<double> aofactory::ao_overlap2()
{
  pimpl->set_name("s_bb2");
  pimpl->set_dim("bb2");
  pimpl->set_center(ctr::c_2c1e);
  pimpl->set_operator(op::overlap);
  pimpl->setup_calc();
  return pimpl->compute();
}

dbcsr::shared_matrix<double> aofactory::ao_kinetic()
{
  pimpl->set_name("t_bb");
  pimpl->set_dim("bb");
  pimpl->set_center(ctr::c_2c1e);
  pimpl->set_operator(op::kinetic);
  pimpl->setup_calc();
  return pimpl->compute();
}

dbcsr::shared_matrix<double> aofactory::ao_nuclear()
{
  pimpl->set_name("v_bb");
  pimpl->set_dim("bb");
  pimpl->set_center(ctr::c_2c1e);
  pimpl->set_operator(op::nuclear);
  pimpl->setup_calc();
  return pimpl->compute();
}

dbcsr::shared_matrix<double> aofactory::ao_2c2e(metric m)
{
  op iop = op::invalid;
  if (m == metric::coulomb)
    iop = op::coulomb;
  if (m == metric::erfc_coulomb)
    iop = op::erfc_coulomb;

  pimpl->set_name("i_xx");
  pimpl->set_dim("xx");
  pimpl->set_center(ctr::c_2c2e);
  pimpl->set_operator(iop);
  pimpl->setup_calc();
  return pimpl->compute();
}

dbcsr::shared_matrix<double> aofactory::ao_auxoverlap()
{
  pimpl->set_name("s_xx");
  pimpl->set_dim("xx");
  pimpl->set_center(ctr::c_2c1e);
  pimpl->set_operator(op::overlap);
  pimpl->setup_calc();
  return pimpl->compute();
}

std::array<dbcsr::shared_matrix<double>, 3> aofactory::ao_emultipole(
    std::array<int, 3> O)
{
  pimpl->set_name("emult");
  pimpl->set_dim("bb");
  pimpl->set_center(ctr::c_2c1e);
  pimpl->set_operator(op::emultipole);
  pimpl->setup_calc();
  return pimpl->compute_xyz(O);
}

void aofactory::ao_2c2e_setup(metric m) 
{
  op iop = op::invalid;
  if (m == metric::coulomb)
    iop = op::coulomb;
  if (m == metric::erfc_coulomb)
    iop = op::erfc_coulomb;

  pimpl->set_center(ctr::c_2c2e);
  pimpl->set_dim("xx");
  pimpl->set_operator(iop);
  pimpl->setup_calc();
}

void aofactory::ao_3c2e_setup(metric m)
{
  op iop = op::invalid;
  if (m == metric::coulomb)
    iop = op::coulomb;
  if (m == metric::erfc_coulomb)
    iop = op::erfc_coulomb;

  pimpl->set_center(ctr::c_3c2e);
  pimpl->set_dim("xbb");
  pimpl->set_operator(iop);
  pimpl->setup_calc();
}

void aofactory::ao_3c1e_ovlp_setup()
{
  pimpl->set_center(ctr::c_3c1e);
  pimpl->set_dim("xbb");
  pimpl->set_operator(op::overlap);
  pimpl->setup_calc();
}

void aofactory::ao_eri_setup(metric m)
{
  op iop = op::invalid;
  if (m == metric::coulomb)
    iop = op::coulomb;
  if (m == metric::erfc_coulomb)
    iop = op::erfc_coulomb;

  pimpl->set_center(ctr::c_4c2e);
  pimpl->set_dim("bbbb");
  pimpl->set_operator(iop);
  pimpl->setup_calc();
}

void aofactory::ao_2c_fill(dbcsr::shared_matrix<double>& m_in)
{
  pimpl->compute_2_reserved(m_in);
}

void aofactory::ao_3c_fill(dbcsr::shared_tensor<3, double>& t_in)
{
  pimpl->compute_3_reserved(t_in);
}

void aofactory::ao_3c_fill(
    dbcsr::shared_tensor<3, double>& t_in,
    vec<vec<int>>& blkbounds,
    shared_screener scr)
{
  pimpl->compute_3_partial(t_in, blkbounds, scr);
}

void aofactory::ao_3c_fill_idx(
    dbcsr::shared_tensor<3, double>& t_in,
    arrvec<int, 3>& blkbounds,
    shared_screener scr)
{
  pimpl->compute_3_partial_idx(t_in, blkbounds, scr);
}

void aofactory::ao_4c_fill(
    dbcsr::shared_tensor<4, double>& t_in,
    vec<vec<int>>& blkbounds,
    shared_screener scr)
{
  pimpl->compute_4_partial(t_in, blkbounds, scr);
}

dbcsr::shared_matrix<double> aofactory::ao_schwarz()
{
  pimpl->set_name("Z_mn");
  pimpl->set_dim("bb");
  pimpl->set_center(ctr::c_4c2e);
  pimpl->set_operator(op::coulomb);
  pimpl->setup_calc();
  return pimpl->compute_screen("schwarz", "bbbb");
}

dbcsr::shared_matrix<double> aofactory::ao_3cschwarz()
{
  pimpl->set_name("Z_x");
  pimpl->set_dim("xx");
  pimpl->set_center(ctr::c_2c2e);
  pimpl->set_operator(op::coulomb);
  pimpl->setup_calc();
  return pimpl->compute_screen("schwarz", "xx");
}

dbcsr::shared_matrix<double> aofactory::ao_schwarz_ovlp()
{
  pimpl->set_name("Z_mn");
  pimpl->set_dim("bb");
  pimpl->set_center(ctr::c_4c1e);
  pimpl->set_operator(op::overlap);
  pimpl->setup_calc();
  return pimpl->compute_screen("schwarz", "bbbb");
}

dbcsr::shared_matrix<double> aofactory::ao_3cschwarz_ovlp()
{
  pimpl->set_name("Z_x");
  pimpl->set_dim("xx");
  pimpl->set_center(ctr::c_2c1e);
  pimpl->set_operator(op::overlap);
  pimpl->setup_calc();
  return pimpl->compute_screen("schwarz", "xx");
}

std::function<void(dbcsr::stensor<3>&, vec<vec<int>>&)>
aofactory::get_generator(shared_screener s_scr)
{
  return pimpl->get_generator(s_scr);
}

desc::shared_cluster_basis remove_lindep(
    world wrd,
    desc::shared_cluster_basis cbas,
    double cutoff,
    std::optional<std::string> opt_split,
    std::optional<int> opt_nsplit)
{
  util::mpi_log LOG(wrd.comm(), 0);
  util::mpi_time TIME(wrd.comm(), "Removing linear dependencies", 0);
  TIME.start();

  LOG.os<>("Removing linear dependencies in basis set...\n");

  ints::aofactory aofac(wrd, cbas);

  auto ovlp = aofac.ao_overlap();

  math::pivinc_cd pivcd(wrd, ovlp, 2);
  pivcd.compute(std::nullopt, cutoff);

  int prank = pivcd.rank();
  auto perm = pivcd.perm();
  
  LOG.os<>("Rank of overlap matrix: ", prank, '\n');

  // get all Shells in a single vector

  std::vector<desc::Shell> vshell;
  std::vector<int> shell_s2b;
  std::vector<int> shell_b2s;

  // make a mapping bas func -> bas shell

  int off = 0;
  int ishell = 0;

  for (auto& cltr : *cbas) {
    for (auto& shell : cltr.shells) {
      vshell.push_back(shell);
      shell_s2b.push_back(off);
      off += shell.size();

      for (size_t ii = 0; ii != shell.size(); ++ii) {
        shell_b2s.push_back(ishell);
      }

      ++ishell;
    }
  }

/*
  for (auto i : shell_s2b) {
          std::cout << i << " ";
  } std::cout << std::endl;

  for (auto i : shell_b2s) {
          std::cout << i << " ";
  } std::cout << std::endl;
*/
  // check which shells we are keeping

  std::vector<bool> keep_shell(vshell.size(), false);

  for (int ifunc = 0; ifunc != prank; ++ifunc) {
    int jshell = shell_b2s[perm[ifunc]];
    keep_shell[jshell] = true;
  }

  std::vector<desc::Shell> newvshell;

  for (size_t ishell = 0; ishell != vshell.size(); ++ishell) {
    if (keep_shell[ishell]) {
      newvshell.push_back(vshell[ishell]);
    }
    else {
      // LOG.os<>("Removing: \n", vshell[ishell], '\n');
    }
  }

  LOG.os<>(
      "Removed ", vshell.size() - newvshell.size(), " of ", vshell.size(),
      " shells.\n");

  auto newcbas =
      std::make_shared<desc::cluster_basis>(newvshell, opt_split, opt_nsplit);

  return newcbas;
}

}  // end namespace ints

}  // namespace megalochem
