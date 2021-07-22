#include <dbcsr_tensor_ops.hpp>
#include "fock/fock_defaults.hpp"
#include "fock/jkbuilder.hpp"
#include "ints/fitting.hpp"
#include "math/linalg/LLT.hpp"
#include "math/linalg/SVD.hpp"
#include "math/linalg/piv_cd.hpp"

namespace megalochem {

namespace fock {

void DF_J::init()
{
  init_base();

  auto b = m_mol->dims().b();
  auto x = m_mol->dims().x();
  vec<int> d = {1};

  int nbf = std::accumulate(b.begin(), b.end(), 0);
  int xnbf = std::accumulate(x.begin(), x.end(), 0);

  arrvec<int, 3> bbd = {b, b, d};
  arrvec<int, 2> xd = {x, d};
  arrvec<int, 2> xx = {x, x};

  std::array<int, 2> tsizes2 = {xnbf, 1};
  std::array<int, 3> tsizes3 = {nbf, nbf, 1};

  m_spgrid2 = dbcsr::pgrid<2>::create(m_cart.comm()).build();

  m_spgrid_xd =
      dbcsr::pgrid<2>::create(m_cart.comm()).tensor_dims(tsizes2).build();

  m_spgrid_bbd =
      dbcsr::pgrid<3>::create(m_cart.comm()).tensor_dims(tsizes3).build();

  m_gp_xd = dbcsr::tensor<2>::create()
                .name("gp_xd")
                .set_pgrid(*m_spgrid_xd)
                .map1({0})
                .map2({1})
                .blk_sizes(xd)
                .build();

  m_gq_xd = dbcsr::tensor<2>::create_template(*m_gp_xd).name("gq_xd").build();

  m_J_bbd = dbcsr::tensor<3>::create()
                .name("J_bbd")
                .set_pgrid(*m_spgrid_bbd)
                .map1({0, 1})
                .map2({2})
                .blk_sizes(bbd)
                .build();

  m_ptot_bbd =
      dbcsr::tensor<3>::create_template(*m_J_bbd).name("ptot_bbd").build();

  m_v_inv_01 = dbcsr::tensor<2>::create()
                   .name("inv")
                   .set_pgrid(*m_spgrid2)
                   .blk_sizes(xx)
                   .map1({0})
                   .map2({1})
                   .build();
}

void DF_J::compute_J()
{
  auto& con1 = TIME.sub("first contraction");
  auto& con2 = TIME.sub("second contraction");
  auto& fetch1 = TIME.sub("fetch ints (1)");
  auto& fetch2 = TIME.sub("fetch ints (2)");
  auto& reoint = TIME.sub("Reordering ints");

  TIME.start();

  // copy over density

  auto ptot = dbcsr::matrix<>::create_template(*m_p_A).name("ptot").build();

  if (m_p_A && !m_p_B) {
    ptot->copy_in(*m_p_A);
    ptot->scale(2.0);
    bool sym = ptot->has_symmetry();
    // if (!sym) std::cout << "HAS NO SYMMETRY" << std::endl;
    dbcsr::copy_matrix_to_3Dtensor_new(*ptot, *m_ptot_bbd, sym);
    // dbcsr::print(ptot);
    ptot->clear();
  }
  else {
    ptot->copy_in(*m_p_A);
    ptot->add(1.0, 1.0, *m_p_B);
    bool sym = ptot->has_symmetry();
    dbcsr::copy_matrix_to_3Dtensor_new<double>(*ptot, *m_ptot_bbd, sym);
    ptot->clear();
  }

  m_ptot_bbd->filter(dbcsr::global::filter_eps);

  // m_gp_xd->batched_contract_init();
  // m_ptot_bbd->batched_contract_init();

  reoint.start();
  m_eri3c2e_batched->decompress_init({2}, vec<int>{0}, vec<int>{1, 2});
  reoint.finish();

  int nbatches = m_eri3c2e_batched->nbatches(2);

  for (int inu = 0; inu != nbatches; ++inu) {
    fetch1.start();
    m_eri3c2e_batched->decompress({inu});
    auto eri_0_12 = m_eri3c2e_batched->get_work_tensor();
    fetch1.finish();

    con1.start();

    vec<vec<int>> bounds1 = {
        m_eri3c2e_batched->full_bounds(1), m_eri3c2e_batched->bounds(2, inu)};

    dbcsr::contract(1.0, *eri_0_12, *m_ptot_bbd, 1.0, *m_gp_xd)
        .bounds1(bounds1)
        .filter(dbcsr::global::filter_eps)
        .perform("XMN, MN_ -> X_");

    con1.finish();
  }

  m_eri3c2e_batched->decompress_finalize();

  // m_gp_xd->batched_contract_finalize();
  // m_ptot_bbd->batched_contract_finalize();

  LOG.os<1>("X_, XY -> Y_\n");

  dbcsr::copy_matrix_to_tensor(*m_v_inv, *m_v_inv_01);

  dbcsr::contract(1.0, *m_gp_xd, *m_v_inv_01, 0.0, *m_gq_xd)
      .filter(dbcsr::global::filter_eps)
      .perform("X_, XY -> Y_");

  m_v_inv_01->clear();
  // dbcsr::print(*m_gq_xd);

  // dbcsr::print(*m_inv);

  // m_J_bbd->batched_contract_init();
  // m_gq_xd->batched_contract_init();

  m_eri3c2e_batched->decompress_init({2}, vec<int>{0}, vec<int>{1, 2});

  for (int inu = 0; inu != nbatches; ++inu) {
    fetch2.start();
    m_eri3c2e_batched->decompress({inu});
    auto eri_0_12 = m_eri3c2e_batched->get_work_tensor();
    fetch2.finish();

    con2.start();

    vec<vec<int>> bounds3 = {
        m_eri3c2e_batched->full_bounds(1), m_eri3c2e_batched->bounds(2, inu)};

    dbcsr::contract(1.0, *m_gq_xd, *eri_0_12, 1.0, *m_J_bbd)
        .bounds3(bounds3)
        .filter(dbcsr::global::filter_eps / nbatches)
        .perform("X_, XMN -> MN_");

    con2.finish();
  }

  m_eri3c2e_batched->decompress_finalize();

  // m_J_bbd->batched_contract_finalize();
  // m_gq_xd->batched_contract_finalize();

  LOG.os<1>("Copy over...\n");

  dbcsr::copy_3Dtensor_to_matrix_new(*m_J_bbd, *m_J);

  m_J_bbd->clear();
  m_gp_xd->clear();
  m_gq_xd->clear();

  if (LOG.global_plev() >= 2) {
    dbcsr::print(*m_J);
  }

  TIME.finish();
}

void DFMO_K::init()
{
  init_base();

  auto b = m_mol->dims().b();
  auto x = m_mol->dims().x();
  arrvec<int, 2> bb = {b, b};
  arrvec<int, 2> xx = {x, x};

  m_spgrid2 = dbcsr::pgrid<2>::create(m_cart.comm()).build();

  m_K_01 = dbcsr::tensor<2>::create()
               .set_pgrid(*m_spgrid2)
               .name("K_01")
               .map1({0})
               .map2({1})
               .blk_sizes(bb)
               .build();

  m_v_invsqrt_01 = dbcsr::tensor<2>::create()
                       .set_pgrid(*m_spgrid2)
                       .name("s_xx_invsqrt")
                       .map1({0})
                       .map2({1})
                       .blk_sizes(xx)
                       .build();
  dbcsr::copy_matrix_to_tensor(*m_v_invsqrt, *m_v_invsqrt_01);
}

void DFMO_K::compute_K()
{
  TIME.start();

  auto b = m_mol->dims().b();
  auto X = m_mol->dims().x();

  auto compute_K_single = [&](dbcsr::shared_matrix<double>& c_bm,
                              dbcsr::shared_matrix<double>& k_bb,
                              std::string x) {
    auto& reo0 = TIME.sub("Reordering ints (1) " + x);
    auto& fetch1 = TIME.sub("Fetch ints " + x);
    // auto& retints = TIME.sub("Reordering ints (2) " + x);
    auto& con1 = TIME.sub("Contraction (1) " + x);
    auto& con2 = TIME.sub("Contraction (2) " + x);
    auto& con3 = TIME.sub("Contraction (3) " + x);
    auto& reo1 = TIME.sub("Reordering (1) " + x);
    auto& reo2 = TIME.sub("Reordering (2) " + x);

    vec<int> o, m, m_off;
    k_bb->clear();

    if (m_SAD_iter) {
      m = c_bm->col_blk_sizes();
      o = m;
    }
    else {
      m = c_bm->col_blk_sizes();
      o = (x == "A") ? m_mol->dims().oa() : m_mol->dims().ob();
    }

    // split it

    int occ_nbatches = m_occ_nbatches;
    vec<vec<int>> o_bounds = dbcsr::make_blk_bounds(o, occ_nbatches);

    vec<int> o_offsets(o.size());
    int off = 0;

    for (size_t i = 0; i != o.size(); ++i) {
      o_offsets[i] = off;
      off += o[i];
    }

    for (size_t i = 0; i != o_bounds.size(); ++i) {
      o_bounds[i][0] = o_offsets[o_bounds[i][0]];
      o_bounds[i][1] = o_offsets[o_bounds[i][1]] + o[o_bounds[i][1]] - 1;
    }

    if (LOG.global_plev() >= 1) {
      LOG.os<1>("OCC bounds: ");
      for (auto p : o_bounds) { LOG.os<1>(p[0], " -> ", p[1]); }
      LOG.os<1>('\n');
    }

    arrvec<int, 2> bm = {b, m};
    arrvec<int, 3> xmb = {X, m, b};

    int nocc = std::accumulate(m.begin(), m.end(), 0);
    int nbf = std::accumulate(b.begin(), b.end(), 0);
    int xnbf = std::accumulate(x.begin(), x.end(), 0);

    std::array<int, 2> tsizes2 = {nbf, nocc};
    std::array<int, 3> tsizes3 = {xnbf, nocc, nbf};

    dbcsr::shared_pgrid<2> grid2 =
        dbcsr::pgrid<2>::create(m_cart.comm()).tensor_dims(tsizes2).build();

    dbcsr::shared_pgrid<3> grid3 =
        dbcsr::pgrid<3>::create(m_cart.comm()).tensor_dims(tsizes3).build();

    m_c_bm = dbcsr::tensor<2>::create()
                 .set_pgrid(*grid2)
                 .name("c_bm_" + x + "_0_1")
                 .map1({0})
                 .map2({1})
                 .blk_sizes(bm)
                 .build();

    dbcsr::copy_matrix_to_tensor(*c_bm, *m_c_bm);

    m_HT1_xmb_02_1 = dbcsr::tensor<3>::create()
                         .name("HT1_xmb_02_1_" + x)
                         .set_pgrid(*grid3)
                         .map1({0, 2})
                         .map2({1})
                         .blk_sizes(xmb)
                         .build();

    m_HT1_xmb_0_12 = dbcsr::tensor<3>::create_template(*m_HT1_xmb_02_1)
                         .name("HT1_xmb_0_12_" + x)
                         .map1({0})
                         .map2({1, 2})
                         .build();

    m_HT2_xmb_0_12 = dbcsr::tensor<3>::create_template(*m_HT1_xmb_02_1)
                         .name("HT2_xmb_0_12_" + x)
                         .map1({0})
                         .map2({1, 2})
                         .build();

    m_HT2_xmb_01_2 = dbcsr::tensor<3>::create_template(*m_HT1_xmb_02_1)
                         .name("HT2_xmb_01_2_" + x)
                         .map1({0, 1})
                         .map2({2})
                         .build();

    int64_t nze_HTI = 0;

    auto full = m_HT1_xmb_02_1->nfull_total();
    int64_t nze_HTI_tot =
        (int64_t)full[0] * (int64_t)full[1] * (int64_t)full[2];

    for (int iocc = 0; iocc != (int)o_bounds.size(); ++iocc) {
      LOG.os<1>(
          "IOCC = ", iocc, " ", o_bounds[iocc][0], " -> ", o_bounds[iocc][1],
          '\n');

      vec<vec<int>> o_tbounds = {o_bounds[iocc]};

      reo0.start();
      m_eri3c2e_batched->decompress_init({2}, vec<int>{0, 2}, vec<int>{1});
      reo0.finish();

      // m_c_bm->batched_contract_init();
      // m_HT1_xmb_02_1->batched_contract_init();

      for (int inu = 0; inu != m_eri3c2e_batched->nbatches(2); ++inu) {
        // std::cout << "MBATCH: " << M << std::endl;

        fetch1.start();
        m_eri3c2e_batched->decompress({inu});
        auto eri_xbb_02_1 = m_eri3c2e_batched->get_work_tensor();
        fetch1.finish();

        vec<vec<int>> xn_bounds = {
            m_eri3c2e_batched->full_bounds(0),
            m_eri3c2e_batched->bounds(2, inu)};

        con1.start();
        dbcsr::contract(1.0, *eri_xbb_02_1, *m_c_bm, 1.0, *m_HT1_xmb_02_1)
            .bounds2(xn_bounds)
            .bounds3(o_tbounds)
            .filter(dbcsr::global::filter_eps / m_eri3c2e_batched->nbatches(2))
            .perform("XMN, Mi -> XiN");
        con1.finish();
      }

      nze_HTI += m_HT1_xmb_02_1->num_nze_total();

      // m_c_bm->batched_contract_finalize();
      // m_HT1_xmb_02_1->batched_contract_finalize();
      m_eri3c2e_batched->decompress_finalize();

      // end for M
      reo1.start();
      dbcsr::copy(*m_HT1_xmb_02_1, *m_HT1_xmb_0_12).move_data(true).perform();
      reo1.finish();

      vec<vec<int>> nu_o_bounds = {
          o_bounds[iocc], m_eri3c2e_batched->full_bounds(2)};

      con2.start();
      dbcsr::contract(
          1.0, *m_HT1_xmb_0_12, *m_v_invsqrt_01, 0.0, *m_HT2_xmb_0_12)
          .bounds2(nu_o_bounds)
          .filter(dbcsr::global::filter_eps)
          .perform("XiN, XY -> YiN");
      con2.finish();
      m_HT1_xmb_0_12->clear();

      reo2.start();
      dbcsr::copy(*m_HT2_xmb_0_12, *m_HT2_xmb_01_2).move_data(true).perform();
      reo2.finish();

      auto HT2_xmb_01_2_copy =
          dbcsr::tensor<3>::create_template(*m_HT2_xmb_01_2)
              .name("HT2_xmb_01_2_copy")
              .build();

      dbcsr::copy(*m_HT2_xmb_01_2, *HT2_xmb_01_2_copy).perform();

      // dbcsr::print(*m_HT2_xmb_01_2);
      // dbcsr::print(*HT2_xmb_01_2_copy);

      LOG.os<1>("Computing K_mn = HT_xim * HT_xin\n");

      // m_K_01->batched_contract_init();
      // m_HT2_xmb_01_2->batched_contract_init();
      // HT2_xmb_01_2_copy->batched_contract_init();

      for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
        vec<vec<int>> x_o_bounds = {
            m_eri3c2e_batched->bounds(0, ix), o_bounds[iocc]};

        con3.start();
        dbcsr::contract(1.0, *m_HT2_xmb_01_2, *HT2_xmb_01_2_copy, 1.0, *m_K_01)
            .bounds1(x_o_bounds)
            .filter(dbcsr::global::filter_eps / m_eri3c2e_batched->nbatches(0))
            .perform("XiM, XiN -> MN");
        con3.finish();
      }

      // m_K_01->batched_contract_finalize();
      // m_HT2_xmb_01_2->batched_contract_finalize();
      // HT2_xmb_01_2_copy->batched_contract_finalize();

      m_HT2_xmb_01_2->clear();
      HT2_xmb_01_2_copy->clear();

    }  // end for I

    double HTI_occupancy = (double)nze_HTI / (double)nze_HTI_tot;
    LOG.os<1>("Occupancy of HTI: ", HTI_occupancy * 100, "%\n");

    dbcsr::copy_tensor_to_matrix(*m_K_01, *k_bb);
    m_K_01->clear();
    k_bb->scale(-1.0);

    m_HT1_xmb_02_1->destroy();
    m_HT1_xmb_0_12->destroy();
    m_HT2_xmb_01_2->destroy();
    m_HT2_xmb_0_12->destroy();
    m_c_bm->destroy();

    grid2->destroy();
    grid3->destroy();
  };  // end lambda function

  compute_K_single(m_c_A, m_K_A, "A");

  if (m_K_B)
    compute_K_single(m_c_B, m_K_B, "B");

  if (LOG.global_plev() >= 2) {
    dbcsr::print(*m_K_A);
    if (m_K_B)
      dbcsr::print(*m_K_B);
  }

  TIME.finish();
}

void DFAO_K::init()
{
  init_base();

  auto b = m_mol->dims().b();
  auto x = m_mol->dims().x();

  arrvec<int, 3> xbb = {x, b, b};

  m_spgrid3_xbb = m_eri3c2e_batched->spgrid();

  // ========== END ==========

  arrvec<int, 2> bb = {b, b};

  m_spgrid2 = dbcsr::pgrid<2>::create(m_cart.comm()).build();

  m_cbar_xbb_01_2 = dbcsr::tensor<3>::create()
                        .name("Cbar_xbb_01_2")
                        .set_pgrid(*m_spgrid3_xbb)
                        .blk_sizes(xbb)
                        .map1({0, 1})
                        .map2({2})
                        .build();

  m_cbar_xbb_1_02 = dbcsr::tensor<3>::create_template(*m_cbar_xbb_01_2)
                        .name("Cbar_xbb_1_02")
                        .map1({1})
                        .map2({0, 2})
                        .build();

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
}

void DFAO_K::compute_K()
{
  TIME.start();

  auto compute_K_single = [&](dbcsr::shared_matrix<double>& p_bb,
                              dbcsr::shared_matrix<double>& k_bb,
                              std::string x) {
    LOG.os<1>("Computing exchange part (", x, ")\n");

    dbcsr::copy_matrix_to_tensor(*p_bb, *m_p_bb);
    m_p_bb->filter(dbcsr::global::filter_eps);
    // dbcsr::print(*c_bm);
    // dbcsr::print(*m_c_bm);

    // LOOP OVER X

    auto& reo_int = TIME.sub("Reordering ints " + x);
    auto& reo_1_batch = TIME.sub("Reordering (1)/batch " + x);
    auto& con_1_batch = TIME.sub("Contraction (1)/batch " + x);
    auto& con_2_batch = TIME.sub("Contraction (2)/batch " + x);
    auto& fetch = TIME.sub("Fetching integrals/batch " + x);
    auto& fetch2 = TIME.sub("Fetching fitting coeffs/batch " + x);
    // auto& retint = TIME.sub("Returning integrals/batch " + x);

    m_fitting_batched->decompress_init({2, 0}, vec<int>{1}, vec<int>{0, 2});

    // m_K_01->batched_contract_init();
    // m_cbar_xbb_01_2->batched_contract_init();
    // m_cbar_xbb_1_02->batched_contract_init();

    int64_t nze_cbar = 0;
    auto full = m_cbar_xbb_01_2->nfull_total();
    int64_t nze_cbar_tot = (int64_t)full[0] * (int64_t)full[1] *
      (int64_t)full[2];

    reo_int.start();
    m_eri3c2e_batched->decompress_init({0}, vec<int>{0, 1}, vec<int>{2});
    reo_int.finish();

    for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
      // fetch integrals
      fetch.start();
      m_eri3c2e_batched->decompress({ix});
      auto eri_01_2 = m_eri3c2e_batched->get_work_tensor();
      fetch.finish();

      // m_p_bb->batched_contract_init();

      for (int inu = 0; inu != m_eri3c2e_batched->nbatches(2); ++inu) {
        vec<vec<int>> xm_bounds = {
            m_eri3c2e_batched->bounds(0, ix),
            m_eri3c2e_batched->full_bounds(1)};

        vec<vec<int>> n_bounds = {m_eri3c2e_batched->bounds(2, inu)};

        con_1_batch.start();
        dbcsr::contract(1.0, *eri_01_2, *m_p_bb, 0.0, *m_cbar_xbb_01_2)
            .bounds2(xm_bounds)
            .bounds3(n_bounds)
            .filter(dbcsr::global::filter_eps)
            .perform("XMN, NL -> XML");
        con_1_batch.finish();

        nze_cbar += m_cbar_xbb_01_2->num_nze_total();

        // m_cbar_xbb_01_2->filter(dbcsr::global::filter_eps);

        vec<vec<int>> copy_bounds = {
            m_eri3c2e_batched->bounds(0, ix), m_eri3c2e_batched->full_bounds(1),
            m_eri3c2e_batched->bounds(2, inu)};

        reo_1_batch.start();
        dbcsr::copy(*m_cbar_xbb_01_2, *m_cbar_xbb_1_02)
            .bounds(copy_bounds)
            .move_data(true)
            .perform();
        reo_1_batch.finish();

        // dbcsr::print(*m_cbar_xbb_02_1);

        // dbcsr::print(*m_cbar_xbb_02_1);

        // get c_xbb
        fetch2.start();
        m_fitting_batched->decompress({inu, ix});
        auto c_xbb_1_02 = m_fitting_batched->get_work_tensor();
        fetch2.finish();

        // dbcsr::print(*c_xbb_1_02);

        vec<vec<int>> xs_bounds = {
            m_eri3c2e_batched->bounds(0, ix),
            m_eri3c2e_batched->bounds(2, inu)};

        con_2_batch.start();
        dbcsr::contract(1.0, *c_xbb_1_02, *m_cbar_xbb_1_02, 1.0, *m_K_01)
            .bounds1(xs_bounds)
            .filter(dbcsr::global::filter_eps / m_eri3c2e_batched->nbatches(2))
            .perform("XNS, XMS -> MN");
        con_2_batch.finish();

        m_cbar_xbb_1_02->clear();
      }

      // m_p_bb->batched_contract_finalize();
    }

    m_eri3c2e_batched->decompress_finalize();
    m_fitting_batched->decompress_finalize();

    // m_cbar_xbb_01_2->batched_contract_finalize();
    // m_cbar_xbb_1_02->batched_contract_finalize();

    double occ_cbar = (double) nze_cbar / (double) nze_cbar_tot;
    LOG.os<1>("Occupancy of cbar: ", occ_cbar, "%\n");

    // m_K_01->batched_contract_finalize();

    dbcsr::copy_tensor_to_matrix(*m_K_01, *k_bb);
    m_K_01->clear();
    m_p_bb->clear();
    k_bb->scale(-1.0);

    LOG.os<1>("Done with exchange.\n");
  };  // end lambda function

  compute_K_single(m_p_A, m_K_A, "A");

  if (m_K_B)
    compute_K_single(m_p_B, m_K_B, "B");

  if (LOG.global_plev() >= 2) {
    dbcsr::print(*m_K_A);
    if (m_K_B)
      dbcsr::print(*m_K_B);
  }

  TIME.finish();
}

void DFMEM_K::init()
{
  init_base();

  auto b = m_mol->dims().b();
  auto x = m_mol->dims().x();

  arrvec<int, 3> xbb = {x, b, b};
  arrvec<int, 2> bb = {b, b};
  arrvec<int, 2> xx = {x, x};

  m_spgrid3_xbb = m_eri3c2e_batched->spgrid();
  m_spgrid2 = dbcsr::pgrid<2>::create(m_cart.comm()).build();

  m_eri_xbb_0_12 = dbcsr::tensor<3>::create()
                        .name("eri_xbb_0_12")
                        .set_pgrid(*m_spgrid3_xbb)
                        .blk_sizes(xbb)
                        .map1({0})
                        .map2({1,2})
                        .build();

  m_c_xbb_02_1 = dbcsr::tensor<3>::create_template(*m_eri_xbb_0_12)
                        .name("c_xbb_02_1")
                        .map1({0,2})
                        .map2({1})
                        .build();
  
  m_c_xbb_0_12 = dbcsr::tensor<3>::create_template(*m_eri_xbb_0_12)
                        .name("c_xbb_0_12")
                        .map1({0})
                        .map2({1,2})
                        .build();

  m_cbar_xbb_02_1 = dbcsr::tensor<3>::create_template(*m_eri_xbb_0_12)
                     .name("cbar_xbb_02_1")
                     .map1({0,2})
                     .map2({1})
                     .build();

  m_cbar_xbb_01_2 = dbcsr::tensor<3>::create_template(*m_eri_xbb_0_12)
                       .name("cbar_xbb_01_2")
                       .map1({0,1})
                       .map2({2})
                       .build();

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
                  .name("v_xx_01")
                  .set_pgrid(*m_spgrid2)
                  .blk_sizes(xx)
                  .map1({0})
                  .map2({1})
                  .build();
}

void DFMEM_K::compute_K()
{
  TIME.start();

  dbcsr::copy_matrix_to_tensor(*m_v_xx, *m_v_xx_01);
  m_v_xx_01->filter(dbcsr::global::filter_eps);
  auto x = m_mol->dims().x();
  auto b = m_mol->dims().b();
  
  int nblks = 0;
  
  auto get_blocks = [&](auto cbar) {
    Eigen::MatrixXi blocks_local = Eigen::MatrixXi::Zero(x.size(),b.size());
    Eigen::MatrixXi blocks_global = Eigen::MatrixXi::Zero(x.size(),b.size());
    dbcsr::iterator_t<3> iter3(*cbar);
    iter3.start();
    while (iter3.blocks_left()) {
      iter3.next();
      blocks_local(iter3.idx()[0],iter3.idx()[2]) = 1;
    }
    iter3.stop();
    MPI_Allreduce(blocks_local.data(), blocks_global.data(), blocks_global.size(),
      MPI_INT, MPI_LOR, m_world.comm());
      
    for (int ii = 0; ii != blocks_global.size(); ++ii) {
      nblks += (blocks_global.data()[ii] == 0) ? 0 : 1;
    }
      
    return blocks_global;
  };

  auto compute_K_single = [&](dbcsr::shared_matrix<double>& p_bb,
                              dbcsr::shared_matrix<double>& k_bb,
                              std::string xstr) {
    LOG.os<1>("Computing exchange part (", xstr, ")\n");

    dbcsr::copy_matrix_to_tensor(*p_bb, *m_p_bb);
    m_p_bb->filter(dbcsr::global::filter_eps);
    // dbcsr::print(*c_bm);
    // dbcsr::print(*m_c_bm);

    auto& reo_int = TIME.sub("Reordering ints " + xstr);
    auto& reo_1 = TIME.sub("Reordering (1)/batch " + xstr);
    auto& reo_2 = TIME.sub("Reordering (2)/batch " + xstr);
    auto& reo_3 = TIME.sub("Reordering (3)/batch " + xstr);
    auto& con_1 = TIME.sub("Contraction (1)/batch " + xstr);
    auto& con_2 = TIME.sub("Contraction (2)/batch " + xstr);
    auto& con_3 = TIME.sub("Contraction (3)/batch " + xstr);
    auto& fetch = TIME.sub("Fetching coeffs/batch " + xstr);
    // auto& retint = TIME.sub("Returning integrals/batch " + x);

    reo_int.start();
    m_eri3c2e_batched->decompress_init({0}, vec<int>{0,1}, vec<int>{2});
    reo_int.finish();

    int nxbatches = m_eri3c2e_batched->nbatches(0);

    for (int ix = 0; ix != nxbatches; ++ix) {
      LOG.os<1>("BATCH X: ", ix, '\n');
      
      m_eri3c2e_batched->decompress({ix});
      auto eri_xbb_01_2 = m_eri3c2e_batched->get_work_tensor();
      
      for (int isig = 0; isig != m_eri3c2e_batched->nbatches(2); ++isig) {
        LOG.os<1>("BATCH SIG: ", isig, '\n');
        
        vec<vec<int>> xmbds = {
            m_eri3c2e_batched->bounds(0, ix),
            m_eri3c2e_batched->full_bounds(1)};

        vec<vec<int>> sbds = {m_eri3c2e_batched->bounds(2, isig)};

        con_2.start();
        dbcsr::contract(1.0, *eri_xbb_01_2, *m_p_bb, 0.0, *m_cbar_xbb_01_2)
            .bounds2(xmbds)
            .bounds3(sbds)
            .filter(dbcsr::global::filter_eps)
            .perform("Xmr, rs -> Xms");
        con_2.finish();
        
        auto xs_blocks = get_blocks(m_cbar_xbb_01_2);

        vec<vec<int>> xbds = {m_eri3c2e_batched->bounds(0, ix)};
        vec<vec<int>> nsbds = {
          m_eri3c2e_batched->full_bounds(1),
          m_eri3c2e_batched->bounds(2,isig)
        };

        con_1.start();
        dbcsr::contract(1.0, *m_v_xx_01, *eri_xbb_01_2, 0.0, *m_c_xbb_0_12)
            .bounds2(xbds)
            .bounds3(nsbds)
            .filter(dbcsr::global::filter_eps)
            .perform("XY, Yns -> Xns");
        con_1.finish();

        vec<vec<int>> cpybds = {
            m_eri3c2e_batched->bounds(0, ix), 
            m_eri3c2e_batched->full_bounds(1),
            m_eri3c2e_batched->bounds(2, isig)};

        reo_2.start();
        dbcsr::copy(*m_cbar_xbb_01_2, *m_cbar_xbb_02_1)
            .bounds(cpybds)
            .move_data(true)
            .perform();
        reo_2.finish();

        reo_3.start();
        dbcsr::copy(*m_c_xbb_0_12, *m_c_xbb_02_1).bounds(cpybds).perform();
        reo_3.finish();

        vec<vec<int>> xsbds = {
            m_eri3c2e_batched->bounds(0, ix),
            m_eri3c2e_batched->bounds(2, isig)
        };

        con_3.start();
        dbcsr::contract(1.0, *m_c_xbb_02_1, *m_cbar_xbb_02_1, 1.0, *m_K_01)
            .bounds1(xsbds)
            .filter(dbcsr::global::filter_eps / nxbatches)
            .perform("Xns, Xms -> mn");
        con_3.finish();
        
        m_c_xbb_02_1->clear();
        m_cbar_xbb_02_1->clear();
        
      }
    }

    m_eri3c2e_batched->decompress_finalize();
    
    std::cout << x.size() << " " << b.size() << std::endl;
    std::cout << "NBLKS: " << nblks << "/" << x.size()*b.size() << std::endl;

    // m_K_01->batched_contract_finalize();

    dbcsr::copy_tensor_to_matrix(*m_K_01, *k_bb);
    m_K_01->clear();
    m_p_bb->clear();
    k_bb->scale(-1.0);

    LOG.os<1>("Done with exchange.\n");
  };  // end lambda function

  compute_K_single(m_p_A, m_K_A, "A");

  if (m_K_B)
    compute_K_single(m_p_B, m_K_B, "B");

  if (LOG.global_plev() >= 2) {
    dbcsr::print(*m_K_A);
    if (m_K_B)
      dbcsr::print(*m_K_B);
  }

  m_v_xx_01->clear();

  TIME.finish();
}

void DFLMO_K::init()
{
  init_base();

  auto b = m_mol->dims().b();
  auto x = m_mol->dims().x();

  arrvec<int, 2> bb = {b, b};
  arrvec<int, 2> xx = {x, x};

  m_spgrid2 = dbcsr::pgrid<2>::create(m_cart.comm()).build();

  m_K_01 = dbcsr::tensor<2>::create()
               .set_pgrid(*m_spgrid2)
               .name("K_01")
               .map1({0})
               .map2({1})
               .blk_sizes(bb)
               .build();

  m_v_xx_01 = dbcsr::tensor<2>::create()
                  .set_pgrid(*m_spgrid2)
                  .name("K_01")
                  .map1({0})
                  .map2({1})
                  .blk_sizes(xx)
                  .build();
}

void DFLMO_K::compute_K()
{
  TIME.start();

  dbcsr::copy_matrix_to_tensor(*m_v_xx, *m_v_xx_01);
  m_v_xx_01->filter(dbcsr::global::filter_eps);

  auto compute_K_single_sym = [&](dbcsr::shared_matrix<double>& c_bm,
                                  dbcsr::shared_matrix<double>& k_bb,
                                  std::string X) {
    LOG.os<1>("Computing exchange part (", X, ")\n");

    auto& time_reo1 = TIME.sub("First reordering");
    auto& time_reo2 = TIME.sub("Second reordering");
    auto& time_reo3 = TIME.sub("Third reordering");
    auto& time_htint = TIME.sub("Forming half-transformed integrals");
    auto& time_htfit = TIME.sub("Contracting with v_xx");
    auto& time_formk = TIME.sub("Final contraction");
    auto& time_ints = TIME.sub("Fetching ints");

    LOG.os<1>("Setting up tensors\n");

    int nocc = c_bm->nfullcols_total();
    int nbas = m_mol->c_basis()->nbf();
    int nxbas = m_mol->c_dfbasis()->nbf();

    auto x = m_mol->dims().x();
    auto b = m_mol->dims().b();
    auto o = c_bm->col_blk_sizes();

    auto o_bounds = dbcsr::make_blk_bounds(o, m_occ_nbatches);
    int nobatches = o_bounds.size();

    vec<int> o_offsets(o.size());
    int off = 0;

    for (size_t i = 0; i != o.size(); ++i) {
      o_offsets[i] = off;
      off += o[i];
    }

    for (size_t i = 0; i != o_bounds.size(); ++i) {
      o_bounds[i][0] = o_offsets[o_bounds[i][0]];
      o_bounds[i][1] = o_offsets[o_bounds[i][1]] + o[o_bounds[i][1]] - 1;
    }

    std::array<int, 3> dims3 = {nxbas, nbas, nocc};

    arrvec<int, 3> xbm = {x, b, o};
    arrvec<int, 2> bm = {b, o};

    auto spgrid2_bm = dbcsr::pgrid<2>::create(m_cart.comm())
                          //.tensor_dims(dims2)
                          .build();

    LOG.os<1>("9\n");

    auto spgrid3_xbm =
        dbcsr::pgrid<3>::create(m_cart.comm()).tensor_dims(dims3).build();

    auto c_bm_01 = dbcsr::tensor<2>::create()
                       .name("c_bm_01")
                       .set_pgrid(*spgrid2_bm)
                       .blk_sizes(bm)
                       .map1({0})
                       .map2({1})
                       .build();

    auto ht_xbm_0_12 = dbcsr::tensor<3>::create()
                           .name("ht_xbm_0_12")
                           .set_pgrid(*spgrid3_xbm)
                           .blk_sizes(xbm)
                           .map1({0})
                           .map2({1, 2})
                           .build();

    auto ht_xbm_01_2 = dbcsr::tensor<3>::create()
                           .name("ht_xbm_01_2")
                           .set_pgrid(*spgrid3_xbm)
                           .blk_sizes(xbm)
                           .map1({0, 1})
                           .map2({2})
                           .build();

    auto ht_xbm_02_1 = dbcsr::tensor<3>::create()
                           .name("ht_xbm_02_1")
                           .set_pgrid(*spgrid3_xbm)
                           .blk_sizes(xbm)
                           .map1({0, 2})
                           .map2({1})
                           .build();

    auto htfit_xbm_0_12 = dbcsr::tensor<3>::create()
                              .name("htfit_xbm_0_12")
                              .set_pgrid(*spgrid3_xbm)
                              .blk_sizes(xbm)
                              .map1({0})
                              .map2({1, 2})
                              .build();

    auto htfit_xbm_02_1 = dbcsr::tensor<3>::create()
                              .name("ht_xbm_02_1")
                              .set_pgrid(*spgrid3_xbm)
                              .blk_sizes(xbm)
                              .map1({0, 2})
                              .map2({1})
                              .build();

    dbcsr::copy_matrix_to_tensor(*c_bm, *c_bm_01);
    c_bm_01->filter(dbcsr::global::filter_eps);

    int nxbatches = m_eri3c2e_batched->nbatches(0);
    int nbbatches = m_eri3c2e_batched->nbatches(2);

    for (int iocc = 0; iocc != nobatches; ++iocc) {
      LOG.os<1>("Occ batch ", iocc, '\n');

      auto obds = o_bounds[iocc];

      time_ints.start();
      m_eri3c2e_batched->decompress_init({2}, vec<int>{0, 1}, vec<int>{2});
      time_ints.finish();

      LOG.os<1>("Forming HT integrals...\n");
      time_htint.start();

      for (int inu = 0; inu != nbbatches; ++inu) {
        LOG.os<1>("-- Batch ", inu, '\n');

        m_eri3c2e_batched->decompress({inu});
        auto eri3c2e_01_2 = m_eri3c2e_batched->get_work_tensor();

        vec<vec<int>> nu_bounds = {m_eri3c2e_batched->bounds(2, inu)};

        vec<vec<int>> i_bounds = {obds};

        dbcsr::contract(1.0, *eri3c2e_01_2, *c_bm_01, 1.0, *ht_xbm_01_2)
            .bounds1(nu_bounds)
            .bounds3(i_bounds)
            .filter(dbcsr::global::filter_eps / nbbatches)
            .perform("Xmn, ni -> Xmi");
      }

      time_htint.finish();

      time_reo1.start();
      dbcsr::copy(*ht_xbm_01_2, *ht_xbm_0_12).move_data(true).perform();
      time_reo1.finish();

      m_eri3c2e_batched->decompress_finalize();

      LOG.os<1>("Forming K...\n");
      for (int ix = 0; ix != nxbatches; ++ix) {
        LOG.os<1>("-- Batch ", ix, '\n');

        vec<vec<int>> x_bounds = {m_eri3c2e_batched->bounds(0, ix)};

        vec<vec<int>> mi_bounds = {m_eri3c2e_batched->full_bounds(1), obds};

        time_htfit.start();
        dbcsr::contract(1.0, *m_v_xx_01, *ht_xbm_0_12, 1.0, *htfit_xbm_0_12)
            .bounds2(x_bounds)
            .bounds3(mi_bounds)
            .filter(dbcsr::global::filter_eps)
            .perform("XY, Ymi -> Xmi");
        time_htfit.finish();

        vec<vec<int>> xni_bounds = {
            m_eri3c2e_batched->bounds(0, ix), m_eri3c2e_batched->full_bounds(1),
            obds};

        time_reo2.start();
        dbcsr::copy(*htfit_xbm_0_12, *htfit_xbm_02_1)
            .bounds(xni_bounds)
            .move_data(true)
            .perform();
        time_reo2.finish();

        time_reo3.start();
        dbcsr::copy(*ht_xbm_0_12, *ht_xbm_02_1).bounds(xni_bounds).perform();
        time_reo3.finish();

        vec<vec<int>> xi_bounds = {m_eri3c2e_batched->bounds(0, ix), obds};

        time_formk.start();
        dbcsr::contract(1.0, *ht_xbm_02_1, *htfit_xbm_02_1, 1.0, *m_K_01)
            .bounds1(xi_bounds)
            .move(true)
            .filter(dbcsr::global::filter_eps / nxbatches)
            .perform("Xmi, Xni -> nm");
        time_formk.finish();
      }

      ht_xbm_0_12->clear();
    }

    dbcsr::copy_tensor_to_matrix(*m_K_01, *k_bb);
    m_K_01->clear();
    k_bb->scale(-1.0);

    LOG.os<1>("Done with exchange.\n");
  };  // end lambda function sym

  auto compute_K_single = [&](dbcsr::shared_matrix<double>& u_bm,
                              dbcsr::shared_matrix<double>& v_mb,
                              dbcsr::shared_matrix<double>& k_bb,
                              std::string X) {
    auto& time_reo1 = TIME.sub("First reordering");
    auto& time_reo2 = TIME.sub("Second reordering");
    auto& time_reo3 = TIME.sub("Third reordering");
    auto& time_htuint = TIME.sub("Forming half-transformed u integrals");
    auto& time_htvint = TIME.sub("Forming half-transformed v integrals");
    auto& time_htvfit = TIME.sub("Contracting with v_xx");
    auto& time_ints = TIME.sub("Fetching ints");

    LOG.os<1>("Computing exchange part (", X, "), NON SYMMETRIC\n");

    int nocc = u_bm->nfullcols_total();
    int nbas = m_mol->c_basis()->nbf();
    int nxbas = m_mol->c_basis()->nbf();

    auto x = m_mol->dims().x();
    auto b = m_mol->dims().b();
    auto o = u_bm->col_blk_sizes();

    LOG.os<1>("OCCS:\n");
    for (auto i : o) { LOG.os<1>(i, '\n'); }

    auto o_bounds = dbcsr::make_blk_bounds(o, m_occ_nbatches);
    int nobatches = o_bounds.size();

    vec<int> o_offsets(o.size());
    int off = 0;

    for (size_t i = 0; i != o.size(); ++i) {
      o_offsets[i] = off;
      off += o[i];
    }

    for (size_t i = 0; i != o_bounds.size(); ++i) {
      o_bounds[i][0] = o_offsets[o_bounds[i][0]];
      o_bounds[i][1] = o_offsets[o_bounds[i][1]] + o[o_bounds[i][1]] - 1;
    }

    std::array<int, 3> dims3 = {nxbas, nbas, nocc};

    arrvec<int, 3> xbm = {x, b, o};
    arrvec<int, 2> bm = {b, o};
    arrvec<int, 2> mb = {o, b};

    auto spgrid2_bm = dbcsr::pgrid<2>::create(m_cart.comm())
                          //.tensor_dims(dims2)
                          .build();

    auto spgrid2_mb = dbcsr::pgrid<2>::create(m_cart.comm())
                          //.tensor_dims(dims2t)
                          .build();

    auto spgrid3_xbm =
        dbcsr::pgrid<3>::create(m_cart.comm()).tensor_dims(dims3).build();

    auto u_bm_01 = dbcsr::tensor<2>::create()
                       .name("u_bm_01")
                       .set_pgrid(*spgrid2_bm)
                       .blk_sizes(bm)
                       .map1({0})
                       .map2({1})
                       .build();

    auto vt_mb_01 = dbcsr::tensor<2>::create()
                        .name("vt_mb_01")
                        .set_pgrid(*spgrid2_mb)
                        .blk_sizes(mb)
                        .map1({0})
                        .map2({1})
                        .build();

    auto htu_xbm_01_2 = dbcsr::tensor<3>::create()
                            .name("htu_xbm_01_2")
                            .set_pgrid(*spgrid3_xbm)
                            .blk_sizes(xbm)
                            .map1({0, 1})
                            .map2({2})
                            .build();

    auto htu_xbm_02_1 = dbcsr::tensor<3>::create()
                            .name("htu_xbm_02_1")
                            .set_pgrid(*spgrid3_xbm)
                            .blk_sizes(xbm)
                            .map1({0, 2})
                            .map2({1})
                            .build();

    auto htv_xbm_0_12 = dbcsr::tensor<3>::create()
                            .name("htv_xbm_0_12")
                            .set_pgrid(*spgrid3_xbm)
                            .blk_sizes(xbm)
                            .map1({0})
                            .map2({1, 2})
                            .build();

    auto htv_xbm_01_2 = dbcsr::tensor<3>::create()
                            .name("htv_xbm_01_2")
                            .set_pgrid(*spgrid3_xbm)
                            .blk_sizes(xbm)
                            .map1({0, 1})
                            .map2({2})
                            .build();

    auto htvfit_xbm_02_1 = dbcsr::tensor<3>::create()
                               .name("htvfit_xbm_02_1")
                               .set_pgrid(*spgrid3_xbm)
                               .blk_sizes(xbm)
                               .map1({0, 2})
                               .map2({1})
                               .build();

    auto htvfit_xbm_0_12 = dbcsr::tensor<3>::create()
                               .name("htvfit_xbm_0_12")
                               .set_pgrid(*spgrid3_xbm)
                               .blk_sizes(xbm)
                               .map1({0})
                               .map2({1, 2})
                               .build();

    dbcsr::copy_matrix_to_tensor(*u_bm, *u_bm_01);
    dbcsr::copy_matrix_to_tensor(*v_mb, *vt_mb_01);

    u_bm_01->filter(dbcsr::global::filter_eps);
    vt_mb_01->filter(dbcsr::global::filter_eps);

    int nxbatches = m_eri3c2e_batched->nbatches(0);
    int nbbatches = m_eri3c2e_batched->nbatches(2);

    for (int iocc = 0; iocc != nobatches; ++iocc) {
      LOG.os<1>("Occ batch ", iocc, '\n');

      auto obds = o_bounds[iocc];

      m_eri3c2e_batched->decompress_init({2}, vec<int>{0, 1}, vec<int>{2});

      // htv_xbm_01_2->batched_contract_init();
      // htu_xbm_01_2->batched_contract_init();

      LOG.os<1>("Forming HT integrals...\n");
      for (int inu = 0; inu != nbbatches; ++inu) {
        LOG.os<1>("-- Batch ", inu, '\n');

        time_ints.start();
        m_eri3c2e_batched->decompress({inu});
        auto eri3c2e_01_2 = m_eri3c2e_batched->get_work_tensor();
        time_ints.finish();

        vec<vec<int>> nu_bounds = {m_eri3c2e_batched->bounds(2, inu)};

        vec<vec<int>> i_bounds = {obds};

        time_htuint.start();
        dbcsr::contract(1.0, *eri3c2e_01_2, *vt_mb_01, 1.0, *htv_xbm_01_2)
            .bounds1(nu_bounds)
            .bounds3(i_bounds)
            .filter(dbcsr::global::filter_eps / nbbatches)
            .perform("Xmn, in -> Xmi");
        time_htuint.finish();

        time_htvint.start();
        dbcsr::contract(1.0, *eri3c2e_01_2, *u_bm_01, 1.0, *htu_xbm_01_2)
            .bounds1(nu_bounds)
            .bounds3(i_bounds)
            .filter(dbcsr::global::filter_eps / nbbatches)
            .perform("Xmn, ni -> Xmi");
        time_htvint.finish();
      }

      m_eri3c2e_batched->decompress_finalize();
      // htv_xbm_01_2->batched_contract_finalize();
      // htu_xbm_01_2->batched_contract_finalize();

      time_reo1.start();
      dbcsr::copy(*htv_xbm_01_2, *htv_xbm_0_12).move_data(true).perform();
      time_reo1.finish();

      time_reo2.start();
      dbcsr::copy(*htu_xbm_01_2, *htu_xbm_02_1).move_data(true).perform();
      time_reo2.finish();

      LOG.os<1>("Forming K...\n");
      for (int ix = 0; ix != nxbatches; ++ix) {
        LOG.os<1>("-- Batch ", ix, '\n');

        LOG.os<1>("Forming HTV fit\n");
        vec<vec<int>> x_bounds = {m_eri3c2e_batched->bounds(0, ix)};

        vec<vec<int>> mi_bounds = {m_eri3c2e_batched->full_bounds(1), obds};

        time_htvfit.start();
        dbcsr::contract(1.0, *m_v_xx_01, *htv_xbm_0_12, 0.0, *htvfit_xbm_0_12)
            .bounds2(x_bounds)
            .bounds3(mi_bounds)
            .filter(dbcsr::global::filter_eps)
            .perform("XY, Ymi -> Xmi");
        time_htvfit.finish();

        vec<vec<int>> xni_bounds = {
            m_eri3c2e_batched->bounds(0, ix), m_eri3c2e_batched->full_bounds(1),
            obds};

        time_reo3.start();
        dbcsr::copy(*htvfit_xbm_0_12, *htvfit_xbm_02_1)
            .bounds(xni_bounds)
            .move_data(true)
            .perform();
        time_reo3.finish();

        vec<vec<int>> xi_bounds = {m_eri3c2e_batched->bounds(0, ix), obds};

        dbcsr::contract(1.0, *htu_xbm_02_1, *htvfit_xbm_02_1, 1.0, *m_K_01)
            .bounds1(xi_bounds)
            .filter(dbcsr::global::filter_eps / nxbatches)
            .perform("Xmi, Xni -> mn");

        htvfit_xbm_02_1->clear();
      }

      htv_xbm_0_12->clear();
      htu_xbm_02_1->clear();
    }

    dbcsr::copy_tensor_to_matrix(*m_K_01, *k_bb);
    m_K_01->clear();
    k_bb->scale(-1.0);

    LOG.os<1>("Done with exchange.\n");
  };  // end lambda function asym

  if (m_sym) {
    compute_K_single_sym(m_c_A, m_K_A, "A");
    if (m_K_B)
      compute_K_single_sym(m_c_B, m_K_B, "B");
  }
  else {
    compute_K_single(m_u_A, m_v_A, m_K_A, "A");
    if (m_K_B)
      compute_K_single(m_u_B, m_v_B, m_K_B, "B");
  }

  if (LOG.global_plev() >= 2) {
    dbcsr::print(*m_K_A);
    if (m_K_B)
      dbcsr::print(*m_K_B);
  }

  m_v_xx_01->clear();

  TIME.finish();
}

}  // namespace fock

}  // end namespace megalochem
