#include <Eigen/Core>
#include <Eigen/SVD>
#include "extern/lapack.hpp"
#include "fock/jkbuilder.hpp"
#include "ints/aofactory.hpp"
#include "ints/screening.hpp"
#include "math/linalg/SVD.hpp"

namespace megalochem {

namespace fock {

void DFROBUST_K::init()
{
  init_base();

  auto b = m_mol->dims().b();
  auto x = m_mol->dims().x();
  arrvec<int, 2> bb = {b, b};
  arrvec<int, 2> xx = {x, x};
  arrvec<int, 3> xbb = {x, b, b};

  m_spgrid2 = dbcsr::pgrid<2>::create(m_cart.comm()).build();
  auto spgrid3 = m_eri3c2e_batched->spgrid();

  m_K_01 = dbcsr::tensor<2>::create()
               .set_pgrid(*m_spgrid2)
               .name("K_01")
               .map1({0})
               .map2({1})
               .blk_sizes(bb)
               .build();

  m_p_bb = dbcsr::tensor<2>::create_template(*m_K_01)
               .name("p_bb_0_1")
               .map1({0})
               .map2({1})
               .build();

  m_v_xx_01 = dbcsr::tensor<2>::create()
                  .name("s_xx_01")
                  .set_pgrid(*m_spgrid2)
                  .map1({0})
                  .map2({1})
                  .blk_sizes(xx)
                  .build();

  m_cbar_xbb_01_2 = dbcsr::tensor<3>::create()
                        .name("cbar_xbb_01_2")
                        .set_pgrid(*spgrid3)
                        .map1({0, 1})
                        .map2({2})
                        .blk_sizes(xbb)
                        .build();

  m_cbar_xbb_02_1 = dbcsr::tensor<3>::create_template(*m_cbar_xbb_01_2)
                        .name("cbar_xbb_02_1")
                        .map1({0, 2})
                        .map2({1})
                        .build();

  m_cbar_xbb_0_12 = dbcsr::tensor<3>::create_template(*m_cbar_xbb_01_2)
                        .name("cbar_xbb_0_12")
                        .map1({0})
                        .map2({1, 2})
                        .build();

  m_cfit_xbb_01_2 = dbcsr::tensor<3>::create_template(*m_cbar_xbb_01_2)
                        .name("cfit_xbb_01_2")
                        .map1({0, 1})
                        .map2({2})
                        .build();

  m_cpq_xbb_0_12 = dbcsr::tensor<3>::create_template(*m_cbar_xbb_01_2)
                       .name("cpq_xbb_0_12")
                       .map1({0})
                       .map2({1, 2})
                       .build();

  m_cpq_xbb_02_1 = dbcsr::tensor<3>::create_template(*m_cbar_xbb_01_2)
                       .name("cpq_xbb_02_1")
                       .map1({0, 2})
                       .map2({1})
                       .build();
}

void DFROBUST_K::compute_K()
{
  TIME.start();

  dbcsr::copy_matrix_to_tensor(*m_v_xx, *m_v_xx_01);

  auto compute_K_single = [&](dbcsr::shared_matrix<double>& p_bb,
                              dbcsr::shared_matrix<double>& k_bb,
                              std::string x) {
    LOG.os<1>("Computing K " + x, '\n');

    // c_bar(X,n,l) = c_fit(X,n,s) * P(s,l)
    auto k_1 = dbcsr::tensor<2>::create_template(*m_K_01).name("k_1").build();

    auto k_2 = dbcsr::tensor<2>::create_template(*m_K_01).name("k_2").build();

    dbcsr::copy_matrix_to_tensor(*p_bb, *m_p_bb);

    m_eri3c2e_batched->decompress_init({0}, vec<int>{0, 2}, vec<int>{1});
    m_fitting_batched->decompress_init({0}, vec<int>{0}, vec<int>{1, 2});

    int nbbatches = m_eri3c2e_batched->nbatches(2);
    int nxbatches = m_eri3c2e_batched->nbatches(0);

    auto fullbbounds = m_eri3c2e_batched->full_bounds(1);

    for (int ix = 0; ix != nxbatches; ++ix) {
      auto xbounds = m_eri3c2e_batched->bounds(0, ix);

      // form cpq
      for (int iy = 0; iy != nxbatches; ++iy) {
        auto ybounds = m_eri3c2e_batched->bounds(0, iy);

        m_fitting_batched->decompress({iy});
        auto cfit_xbb_0_12 = m_fitting_batched->get_work_tensor();

        vec<vec<int>> xbds = {xbounds};

        vec<vec<int>> ybds = {ybounds};

        dbcsr::contract(1.0, *m_v_xx_01, *cfit_xbb_0_12, 0.0, *m_cpq_xbb_0_12)
            .bounds1(ybds)
            .bounds2(xbds)
            .perform("XY, Ynl -> Xnl");

        vec<vec<int>> xmnbds = {xbounds, fullbbounds, fullbbounds};

        dbcsr::copy(*m_cpq_xbb_0_12, *m_cpq_xbb_02_1)
            .bounds(xmnbds)
            .sum(true)
            .move_data(true)
            .perform();
      }

      m_fitting_batched->decompress({ix});
      m_eri3c2e_batched->decompress({ix});

      auto eri_xbb_02_1 = m_eri3c2e_batched->get_work_tensor();
      auto cfit_xbb_0_12 = m_fitting_batched->get_work_tensor();

      vec<vec<int>> xmnbds = {xbounds, fullbbounds, fullbbounds};

      dbcsr::copy(*cfit_xbb_0_12, *m_cfit_xbb_01_2).bounds(xmnbds).perform();

      for (int isig = 0; isig != nbbatches; ++isig) {
        // form cbar

        auto sigbounds = m_eri3c2e_batched->bounds(2, isig);

        vec<vec<int>> xnbds = {xbounds, fullbbounds};

        vec<vec<int>> sbds = {sigbounds};

        dbcsr::contract(1.0, *m_cfit_xbb_01_2, *m_p_bb, 0.0, *m_cbar_xbb_01_2)
            .bounds2(xnbds)
            .bounds3(sbds)
            .filter(dbcsr::global::filter_eps)
            .perform("Xnl, ls -> Xns");

        // form k_1

        vec<vec<int>> xmsbds = {xbounds, fullbbounds, sigbounds};

        dbcsr::copy(*m_cbar_xbb_01_2, *m_cbar_xbb_02_1)
            .bounds(xmsbds)
            .move_data(true)
            .perform();

        vec<vec<int>> xsbds = {xbounds, sigbounds};

        dbcsr::contract(1.0, *eri_xbb_02_1, *m_cbar_xbb_02_1, 1.0, *k_1)
            .bounds1(xsbds)
            .perform("Xns, Xms -> mn");

        // form k_2

        dbcsr::contract(1.0, *m_cpq_xbb_02_1, *m_cbar_xbb_02_1, 1.0, *k_2)
            .bounds1(xsbds)
            .perform("Xns, Xms -> mn");

        m_cbar_xbb_02_1->clear();

      }  // end loop sig

      m_cpq_xbb_02_1->clear();
      m_cfit_xbb_01_2->clear();

    }  // end loop x

    k_2->scale(-0.5);

    dbcsr::copy(*k_1, *m_K_01).perform();
    dbcsr::copy(*k_2, *m_K_01).sum(true).perform();

    dbcsr::copy(*m_K_01, *k_1).perform();
    dbcsr::copy(*k_1, *m_K_01).order(vec<int>{1, 0}).sum(true).perform();

    dbcsr::copy_tensor_to_matrix(*m_K_01, *k_bb);

    k_bb->scale(-1.0);

    m_K_01->clear();
  };  // end lambda

  compute_K_single(m_p_A, m_K_A, "A");

  if (m_K_B)
    compute_K_single(m_p_B, m_K_B, "B");

  if (LOG.global_plev() >= 2) {
    dbcsr::print(*m_K_A);
    if (m_K_B)
      dbcsr::print(*m_K_B);
  }

  m_v_xx_01->clear();

  /*
  auto& time_reo_int1 = TIME.sub("Reordering integrals (1)");
  auto& time_fetch_ints = TIME.sub("Fetching ints");
  auto& time_reo_cbar = TIME.sub("Reordering c_bar");
  auto& time_reo_ctil = TIME.sub("Reordering c_tilde");
  auto& time_form_cbar = TIME.sub("Forming c_bar");
  auto& time_form_ctil = TIME.sub("Forming c_til");
  auto& time_copy_cfit = TIME.sub("Copying c_fit");
  auto& time_form_K1 = TIME.sub("Forming K1");
  auto& time_form_K2 = TIME.sub("Forming K2");
  */

  TIME.finish();
}

}  // namespace fock

}  // namespace megalochem
