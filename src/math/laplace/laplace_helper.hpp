#ifndef MATH_LAPLACE_HELPER
#define MATH_LAPLACE_HELPER

#include <dbcsr_matrix_ops.hpp>
#include "math/laplace/minimax.hpp"
#include "math/linalg/piv_cd.hpp"
#include "megalochem.hpp"
#include "utils/mpi_log.hpp"

namespace megalochem {

namespace math {

class laplace_helper {
 private:
  world m_world;
  util::mpi_log LOG;

  const int m_nlap;

  dbcsr::shared_matrix<double> m_c_bo, m_c_bv, m_s_sqrt, m_s_invsqrt;
  std::vector<double> m_eps_occ, m_eps_vir;

  std::vector<double> m_weights, m_xpoints;
  std::vector<dbcsr::shared_matrix<double>> m_p_occs, m_p_virs, m_l_occs,
      m_l_virs;

 public:
  laplace_helper(
      world w,
      int nlap,
      dbcsr::shared_matrix<double> c_bo,
      dbcsr::shared_matrix<double> c_bv,
      dbcsr::shared_matrix<double> s_sqrt,
      dbcsr::shared_matrix<double> s_invsqrt,
      std::vector<double> eps_occ,
      std::vector<double> eps_vir,
      int nprint) :
      m_world(w),
      LOG(c_bo->get_cart().comm(), nprint), m_nlap(nlap), m_c_bo(c_bo),
      m_c_bv(c_bv), m_s_sqrt(s_sqrt), m_s_invsqrt(s_invsqrt),
      m_eps_occ(eps_occ), m_eps_vir(eps_vir)
  {
  }

  laplace_helper(const laplace_helper& in) = default;
  laplace_helper& operator=(const laplace_helper&) = default;

  dbcsr::shared_matrix<double> get_scaled_coeff(
      char dim, int ilap, double wfactor, double xfactor)
  {
    auto& c_bm = (dim == 'O') ? m_c_bo : m_c_bv;
    auto& eps_m = (dim == 'O') ? m_eps_occ : m_eps_vir;

    auto c_bm_scaled = dbcsr::matrix<>::copy(*c_bm).build();
    auto eps_scaled = eps_m;

    double xpt = m_xpoints[ilap];
    double wght = m_weights[ilap];

    std::for_each(
        eps_scaled.begin(), eps_scaled.end(),
        [xpt, wght, wfactor, xfactor](double& eps) {
          eps = std::exp(wfactor * std::log(wght) + xfactor * eps * xpt);
        });

    c_bm_scaled->scale(eps_scaled, "right");

    return c_bm_scaled;
  }

  dbcsr::shared_matrix<double> get_density(dbcsr::shared_matrix<double> c_bm)
  {
    auto b = c_bm->row_blk_sizes();

    auto w = c_bm->get_cart();
    auto p_bb = dbcsr::matrix<>::create()
                    .set_cart(w)
                    .name("density matrix of " + c_bm->name())
                    .row_blk_sizes(b)
                    .col_blk_sizes(b)
                    .matrix_type(dbcsr::type::symmetric)
                    .build();

    dbcsr::multiply('N', 'T', 1.0, *c_bm, *c_bm, 0.0, *p_bb).perform();

    return p_bb;
  }

  dbcsr::shared_matrix<double> get_ortho_cholesky(
      dbcsr::shared_matrix<double> coeff, int nsplit)
  {
    int max_rank = coeff->nfullcols_total();

    auto coeff_ortho =
        dbcsr::matrix<>::create_template(*coeff).name("co_ortho").build();

    dbcsr::multiply('N', 'N', 1.0, *m_s_sqrt, *coeff, 0.0, *coeff_ortho)
        .perform();

    auto p_ortho = get_density(coeff_ortho);

    math::pivinc_cd chol(m_world, p_ortho, LOG.global_plev());
    chol.compute(max_rank);

    int rank = chol.rank();

    auto b = coeff->row_blk_sizes();
    auto u = dbcsr::split_range(rank, nsplit);

    LOG.os<1>("Cholesky decomposition rank: ", rank, '\n');

    static int i = 0;
    ++i;

    /*std::string filename_ortho = std::string(std::filesystem::current_path())
            + "/cholortho_" + coeff->name() + "_" + std::to_string(i);

    std::string filename = std::string(std::filesystem::current_path())
            + "/chol_" + dim + "_" + std::to_string(i);*/

    auto L_bu_ortho = chol.L(b, u);

    // util::plot(L_bu_ortho, 1e-5, filename_ortho);

    auto L_bu =
        dbcsr::matrix<>::create_template(*L_bu_ortho).name("L_bu").build();

    dbcsr::multiply('N', 'N', 1.0, *m_s_invsqrt, *L_bu_ortho, 0.0, *L_bu)
        .perform();

    // util::plot(L_bu, 1e-5, filename);

    return L_bu;
  }

  void compute(bool do_o, bool do_v, double offset, int nsplit)
  {
    // laplace
    LOG.os<1>("Laplace helper: launching computation.\n");
    LOG.os<1>("Computing laplace points.\n");

    double emin = m_eps_occ.front();
    double ehomo = m_eps_occ.back();
    double elumo = m_eps_vir.front();
    double emax = m_eps_vir.back();

    double ymin = 2 * (elumo - ehomo) + offset;
    double ymax = 2 * (emax - emin) + offset;

    LOG.os<1>(
        "eps_min/eps_homo/eps_lumo/eps_max ", emin, " ", ehomo, " ", elumo, " ",
        emax, '\n');
    LOG.os<1>("ymin/ymax ", ymin, " ", ymax, '\n');

    math::minimax lp(LOG.global_plev());

    if (m_world.rank() == 0) {
      lp.compute(m_nlap, ymin, ymax);
      m_weights = lp.weights();
      m_xpoints = lp.exponents();
    }
    else {
      m_weights.resize(m_nlap);
      m_xpoints.resize(m_nlap);
    }

    MPI_Bcast(m_weights.data(), m_nlap, MPI_DOUBLE, 0, m_world.comm());
    MPI_Bcast(m_xpoints.data(), m_nlap, MPI_DOUBLE, 0, m_world.comm());

    m_p_occs.clear();
    m_p_virs.clear();
    m_l_occs.clear();
    m_l_virs.clear();

    m_p_occs.resize(m_nlap, nullptr);
    m_p_virs.resize(m_nlap, nullptr);
    m_l_occs.resize(m_nlap, nullptr);
    m_l_virs.resize(m_nlap, nullptr);

    LOG.os<1>("Computing laplace matrices.\n");

    for (int ilap = 0; ilap != m_nlap; ++ilap) {
      auto c_bo_scaled = get_scaled_coeff('O', ilap, 0.125, 0.5);
      auto c_bv_scaled = get_scaled_coeff('V', ilap, 0.125, -0.5);

      m_p_occs[ilap] = get_density(c_bo_scaled);
      m_p_virs[ilap] = get_density(c_bv_scaled);

      if (do_o)
        m_l_occs[ilap] = get_ortho_cholesky(c_bo_scaled, nsplit);
      if (do_v)
        m_l_virs[ilap] = get_ortho_cholesky(c_bv_scaled, nsplit);
    }
  }

  std::vector<dbcsr::shared_matrix<double>> pseudo_densities_occ()
  {
    return m_p_occs;
  }

  std::vector<dbcsr::shared_matrix<double>> pseudo_densities_vir()
  {
    return m_p_virs;
  }

  std::vector<dbcsr::shared_matrix<double>> pseudo_cholesky_occ()
  {
    return m_l_occs;
  }

  std::vector<dbcsr::shared_matrix<double>> pseudo_cholesky_vir()
  {
    return m_l_virs;
  }

  std::vector<double> weights()
  {
    return m_weights;
  }

  std::vector<double> xpoints()
  {
    return m_xpoints;
  }

  ~laplace_helper()
  {
  }
};

}  // namespace math

}  // end namespace megalochem

#endif
