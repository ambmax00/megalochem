#include "adc/adc_mvp.hpp"
#include "adc/adcmod.hpp"
#include "locorb/locorb.hpp"
#include "math/linalg/LLT.hpp"
#include "math/linalg/SVD.hpp"
#include "math/linalg/piv_cd.hpp"
#include "math/solvers/davidson.hpp"
#include "math/solvers/hermitian_eigen_solver.hpp"

#include <dbcsr_conversions.hpp>

namespace megalochem {

namespace adc {

eigenpair adcmod::guess()
{
  LOG.os<>("Setting up guess vectors...\n");

  std::vector<dbcsr::shared_matrix<double>> dav_eigvecs;
  std::vector<double> dav_eigvals;

  if (m_guess == "hf") {
    LOG.os<>(
        "Generating guesses using molecular orbital energy differences.\n");

    auto o = m_wfn->mol->dims().oa();
    auto v = m_wfn->mol->dims().va();

    auto eps_occ = *m_wfn->hf_wfn->eps_occ_A();
    auto eps_vir = *m_wfn->hf_wfn->eps_vir_A();

    auto d_ia = compute_diag_0(o, v, eps_occ, eps_vir);

    auto eigen_ia = dbcsr::matrix_to_eigen(*d_ia);

    std::vector<int> index(eigen_ia.size(), 0);
    for (size_t i = 0; i != index.size(); ++i) { index[i] = i; }

    std::sort(index.begin(), index.end(), [&](const int& a, const int& b) {
      return (eigen_ia.data()[a] < eigen_ia.data()[b]);
    });

    dav_eigvecs.resize(m_nguesses);
    dav_eigvals.resize(m_nroots);

    int nocc = m_wfn->mol->nocc_alpha();
    int nvir = m_wfn->mol->nvir_alpha();

    for (int i = 0; i != m_nguesses; ++i) {
      LOG.os<>("Guess ", i, '\n');

      Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(nocc, nvir);
      mat.data()[index[i]] = 1.0;

      std::string name = "guess_" + std::to_string(i);

      auto guessmat = dbcsr::eigen_to_matrix(
          mat, m_cart, name, o, v, dbcsr::type::no_symmetry);

      dav_eigvecs[i] = guessmat;
    }

    LOG.os<1>("Initial guess excitation energies:\n");
    for (int ii = 0; ii != m_nroots; ++ii) {
      dav_eigvals[ii] = eigen_ia.data()[index[ii]];
      LOG.os<1>(dav_eigvals[ii], " ");
    }

    LOG.os<1>('\n');
  }
  else if (m_guess == "adc") {
    LOG.os<>("Taking eigenvalues and eigenvectors from a previous computation "
             "as guess.\n");

    dav_eigvals = m_wfn->adc_wfn->davidson_eigenvalues();
    dav_eigvecs = m_wfn->adc_wfn->davidson_eigenvectors();
  }
  else {
    throw std::runtime_error("Unknwon guess method");
  }

  return {dav_eigvals, dav_eigvecs};
}

eigenpair adcmod::run_adc1_local(eigenpair& epairs)
{
  int istart = (m_block) ? 0 : m_nroots - 1;

  eigenpair out;

  out.eigvecs.resize(m_nroots);
  out.eigvals.resize(m_nroots);

  for (int istate = 0; istate != m_nroots; ++istate) {
    out.eigvecs[istate] =
        dbcsr::matrix<double>::copy(*epairs.eigvecs[istate]).build();
    out.eigvals[istate] = epairs.eigvals[istate];
  }

  LOG.os<>("==== Starting Local ADC(1) Computation ====\n\n");

  for (int istate = istart; istate != m_nroots; ++istate) {
    LOG.os<>("Computing State Nr. ", istate, '\n');

    canon_lmo lmo_info;

    if (m_local_method == "pao") {
      lmo_info = get_canon_pao(epairs.eigvecs[istate]);
    }
    else if (m_local_method == "nto") {
      lmo_info = get_canon_nto(epairs.eigvecs[istate]);
    }
    else {
      throw std::runtime_error("Invalid method for localization.");
    }

    auto ladc1_mvp = create_adc1(lmo_info);

    auto l_bo = lmo_info.c_ao_lmo_bo;
    auto l_bv = lmo_info.c_ao_lmo_bv;

    auto o = l_bo->col_blk_sizes();
    auto v = l_bv->col_blk_sizes();
    auto epso = lmo_info.eps_occ;
    auto epsv = lmo_info.eps_vir;

    auto d_ia = compute_diag_0(o, v, epso, epsv);

    eigenpair local_epairs;

    for (auto p : epairs.eigvecs) {
      auto m = u_transform(
          p, 'N', lmo_info.u_lmo_cmo_oo, 'T', lmo_info.u_lmo_cmo_vv);
      local_epairs.eigvecs.push_back(m);
    }

    math::davidson<MVP> dav(m_world, LOG.global_plev());

    dav.set_factory(ladc1_mvp);
    dav.set_diag(d_ia);
    dav.pseudo(false);
    dav.balancing(m_balanced);
    dav.block(false);
    dav.conv(m_conv);
    dav.maxiter(m_dav_max_iter);

    auto& t_davidson = TIME.sub("Davidson diagonalization");

    LOG.os<>("Starting Davidson procedure.\n");

    t_davidson.start();
    dav.compute(local_epairs.eigvecs, m_nroots);
    t_davidson.finish();

    ladc1_mvp->print_info();

    LOG.os<>("Finished Davidson procedure.\n");

    auto adc1_dav_eigvecs = dav.ritz_vectors();
    auto adc1_dav_eigvals = dav.eigvals();

    auto cmo_eigvec = u_transform(
        adc1_dav_eigvecs[istate], 'T', lmo_info.u_lmo_cmo_oo, 'N',
        lmo_info.u_lmo_cmo_vv);

    out.eigvecs[istate] = cmo_eigvec;
    out.eigvals[istate] = adc1_dav_eigvals[istate];
  }

  LOG.os<>("==== Finished Local ADC(1) Computation ====\n\n");

  LOG.os<>("ADC(1) Excitation energies:\n");
  for (int iroot = istart; iroot != m_nroots; ++iroot) {
    LOG.os<>("Excitation nr. ", iroot + 1, " : ", out.eigvals[iroot], '\n');
  }

  return out;
}

eigenpair adcmod::run_adc1(eigenpair& epairs)
{
  auto adc1_mvp = create_adc1();

  math::davidson<MVP> dav(m_world.comm(), LOG.global_plev());

  auto o = m_wfn->mol->dims().oa();
  auto v = m_wfn->mol->dims().va();
  auto epso = m_wfn->hf_wfn->eps_occ_A();
  auto epsv = m_wfn->hf_wfn->eps_vir_A();

  auto d_ia = compute_diag_0(o, v, *epso, *epsv);

  dav.set_factory(adc1_mvp);
  dav.set_diag(d_ia);
  dav.pseudo(false);
  dav.balancing(m_balanced);
  dav.block(m_block);
  dav.conv(m_conv);
  dav.maxiter(m_dav_max_iter);

  auto& t_davidson = TIME.sub("Davidson diagonalization");

  LOG.os<>("==== Starting ADC(1) Computation ====\n\n");

  t_davidson.start();
  dav.compute(epairs.eigvecs, m_nroots);
  t_davidson.finish();

  adc1_mvp->print_info();

  auto adc1_dav_eigvecs = dav.ritz_vectors();
  auto adc1_dav_eigvals = dav.eigvals();

  LOG.os<>("==== Finished ADC(1) Computation ====\n\n");

  int istart = (m_block) ? 0 : m_nroots - 1;

  LOG.os<>("ADC(1) Excitation energies:\n");
  for (int iroot = istart; iroot != m_nroots; ++iroot) {
    LOG.os<>(
        "Excitation nr. ", iroot + 1, " : ", adc1_dav_eigvals[iroot], '\n');
  }

  return {adc1_dav_eigvals, adc1_dav_eigvecs};
}

eigenpair adcmod::run_adc2_local(eigenpair& epairs)
{
  eigenpair adc2_epair;

  adc2_epair.eigvecs.resize(m_nroots);
  adc2_epair.eigvals.resize(m_nroots);

  for (int istate = 0; istate != m_nroots; ++istate) {
    adc2_epair.eigvecs[istate] =
        dbcsr::matrix<double>::copy(*epairs.eigvecs[istate]).build();
    adc2_epair.eigvals[istate] = epairs.eigvals[istate];
  }

  int istart = (m_block) ? 0 : m_nroots - 1;

  LOG.os<>("==== Starting ADC(2) Computation ====\n\n");

  for (int iroot = istart; iroot != m_nroots; ++iroot) {
    LOG.os<>("============================================\n");
    LOG.os<>("    Computing excited state nr. ", iroot + 1, '\n');
    LOG.os<>("============================================\n\n");

    canon_lmo lmo_info;

    if (m_local_method == "pao") {
      lmo_info = get_canon_pao(epairs.eigvecs[iroot]);
    }
    else if (m_local_method == "nto") {
      lmo_info = get_canon_nto(epairs.eigvecs[iroot]);
    }
    else {
      throw std::runtime_error("Invalid method for localization.");
    }

    auto ladc2_mvp = create_adc2(lmo_info);

    auto l_bo = lmo_info.c_ao_lmo_bo;
    auto l_bv = lmo_info.c_ao_lmo_bv;

    auto o = l_bo->col_blk_sizes();
    auto v = l_bv->col_blk_sizes();
    auto epso = lmo_info.eps_occ;
    auto epsv = lmo_info.eps_vir;

    auto d_ia = compute_diag_0(o, v, epso, epsv);

    eigenpair local_epairs;

    for (auto& p : epairs.eigvecs) {
      auto m = u_transform(
          p, 'N', lmo_info.u_lmo_cmo_oo, 'T', lmo_info.u_lmo_cmo_vv);
      local_epairs.eigvecs.push_back(m);
    }

    LOG.os<>("Setting up ADC(2) MVP builder.\n");

    auto adc2_mvp = create_adc2(lmo_info);

    math::diis_davidson<MVP> mdav(m_world.comm(), LOG.global_plev());

    mdav.set_factory(adc2_mvp);
    mdav.set_diag(d_ia);
    mdav.macro_maxiter(m_diis_max_iter);
    mdav.macro_conv(m_conv);
    mdav.balancing(m_balanced);
    mdav.micro_maxiter(m_dav_max_iter);

    mdav.compute(local_epairs.eigvecs, iroot + 1, epairs.eigvals[iroot]);

    auto evec = mdav.ritz_vectors()[iroot];
    auto evec_cmo = u_transform(
        evec, 'T', lmo_info.u_lmo_cmo_oo, 'N', lmo_info.u_lmo_cmo_vv);

    adc2_epair.eigvals[iroot] = mdav.eigval()[iroot];
    adc2_epair.eigvecs[iroot] = evec_cmo;
  }

  LOG.os<>("==== Finished ADC(2) Computation ====\n\n");

  LOG.os<>("ADC(2) Excitation energies:\n");
  for (int iroot = istart; iroot != m_nroots; ++iroot) {
    LOG.os<>(
        "Excitation nr. ", iroot + 1, " : ", adc2_epair.eigvals[iroot], '\n');
  }

  return adc2_epair;
}

eigenpair adcmod::run_adc2(eigenpair& epairs)
{
  math::diis_davidson<MVP> mdav(m_world.comm(), LOG.global_plev());

  eigenpair adc2_epair;

  adc2_epair.eigvecs.resize(m_nroots);
  adc2_epair.eigvals.resize(m_nroots);

  for (int istate = 0; istate != m_nroots; ++istate) {
    adc2_epair.eigvecs[istate] =
        dbcsr::matrix<double>::copy(*epairs.eigvecs[istate]).build();
    adc2_epair.eigvals[istate] = epairs.eigvals[istate];
  }

  auto o = m_wfn->mol->dims().oa();
  auto v = m_wfn->mol->dims().va();
  auto epso = m_wfn->hf_wfn->eps_occ_A();
  auto epsv = m_wfn->hf_wfn->eps_vir_A();

  auto d_ia = compute_diag_0(o, v, *epso, *epsv);

  mdav.macro_maxiter(m_diis_max_iter);
  mdav.macro_conv(m_conv);

  mdav.set_diag(d_ia);
  mdav.balancing(m_balanced);
  mdav.micro_maxiter(m_dav_max_iter);

  int istart = (m_block) ? 0 : m_nroots - 1;

  std::shared_ptr<MVP> adc2_mvp;

  LOG.os<>("==== Starting ADC(2) Computation ====\n\n");

  LOG.os<>("=== Setting up MVP builder ===\n");

  adc2_mvp = create_adc2();
  mdav.set_factory(adc2_mvp);

  for (int iroot = istart; iroot != m_nroots; ++iroot) {
    LOG.os<>("============================================\n");
    LOG.os<>("    Computing excited state nr. ", iroot + 1, '\n');
    LOG.os<>("============================================\n\n");

    LOG.os<>("Setting up ADC(2) MVP builder.\n");

    mdav.compute(epairs.eigvecs, iroot + 1, epairs.eigvals[iroot]);

    adc2_epair.eigvals[iroot] = mdav.eigval()[iroot];
    adc2_epair.eigvecs[iroot] = mdav.ritz_vectors()[iroot];
  }

  LOG.os<>("==== Finished ADC(2) Computation ====\n\n");

  LOG.os<>("ADC(2) Excitation energies:\n");
  for (int iroot = istart; iroot != m_nroots; ++iroot) {
    LOG.os<>(
        "Excitation nr. ", iroot + 1, " : ", adc2_epair.eigvals[iroot], '\n');
  }

  return adc2_epair;
}

desc::shared_wavefunction adcmod::compute()
{
  // Generate guesses

  LOG.os<>("--- Starting Computation ---\n\n");

  auto guess_pairs = guess();

  eigenpair out;

  switch (m_adcmethod) {
    case adcmethod::ri_ao_adc1:
      out = (m_local) ? run_adc1_local(guess_pairs) : run_adc1(guess_pairs);
      break;
    case adcmethod::sos_cd_ri_adc2:
      out = (m_local) ? run_adc2_local(guess_pairs) : run_adc2(guess_pairs);
      break;
  }

  auto wfn_out = std::make_shared<desc::wavefunction>();

  wfn_out->mol = m_wfn->mol;
  wfn_out->hf_wfn = m_wfn->hf_wfn;
  wfn_out->mp_wfn = m_wfn->mp_wfn;
  wfn_out->adc_wfn = std::make_shared<desc::adc_wavefunction>(
      m_block, out.eigvals, out.eigvecs);

  TIME.print_info();

  return wfn_out;
}

std::vector<bool> adcmod::get_significant_blocks(
    dbcsr::shared_matrix<double> u_ia, double theta)
{
  auto c_bo = m_wfn->hf_wfn->c_bo_A();
  auto c_bv = m_wfn->hf_wfn->c_bv_A();

  auto u_bb_a = u_transform(u_ia, 'N', c_bo, 'T', c_bv);

  auto b = m_wfn->mol->dims().b();
  auto atoms = m_wfn->mol->atoms();
  int natoms = atoms.size();

  auto retvec = b;
  std::iota(retvec.begin(), retvec.end(), 0);

  std::vector<int> retmol(natoms);
  std::iota(retmol.begin(), retmol.end(), 0);

  // return std::make_tuple(retmol,retvec);

  auto dist = dbcsr::default_dist(b.size(), m_cart.size(), b);

  double norm = u_bb_a->norm(dbcsr_norm_frobenius);

  u_bb_a->scale(1.0 / norm);

  u_bb_a->replicate_all();

  std::vector<double> occ_norms(b.size(), 0.0);
  std::vector<double> vir_norms(b.size(), 0.0);
  std::vector<int> idx_occ(b.size());
  std::vector<int> idx_vir(b.size());

  std::iota(idx_occ.begin(), idx_occ.end(), 0);
  std::iota(idx_vir.begin(), idx_vir.end(), 0);

  auto blkmap = m_wfn->mol->c_basis()->block_to_atom(atoms);

  // loop over blocks rows/cols
  for (int iblk = 0; iblk != (int)b.size(); ++iblk) {
    if (dist[iblk] == m_cart.rank()) {
      int blksize_i = b[iblk];

      std::vector<double> blknorms_r(blksize_i, 0.0);
      std::vector<double> blknorms_c(blksize_i, 0.0);

      for (int jblk = 0; jblk != (int)b.size(); ++jblk) {
        bool foundr = false;
        bool foundc = false;

        auto blkr_p = u_bb_a->get_block_p(iblk, jblk, foundr);
        auto blkc_p = u_bb_a->get_block_p(jblk, iblk, foundc);

        int blksize_j = b[jblk];

        if (foundr) {
          for (int i = 0; i != blksize_i; ++i) {
            for (int j = 0; j != blksize_j; ++j) {
              blknorms_r[i] += pow(blkr_p(i, j), 2.0);
            }
          }
        }

        if (foundc) {
          for (int i = 0; i != blksize_i; ++i) {
            for (int j = 0; j != blksize_j; ++j) {
              blknorms_c[i] += pow(blkc_p(j, i), 2.0);
            }
          }
        }
      }

      occ_norms[iblk] =
          std::accumulate(blknorms_r.begin(), blknorms_r.end(), 0.0);
      vir_norms[iblk] =
          std::accumulate(blknorms_c.begin(), blknorms_c.end(), 0.0);
    }
  }

  // communicate to all processes
  MPI_Allreduce(
      MPI_IN_PLACE, occ_norms.data(), b.size(), MPI_DOUBLE, MPI_SUM,
      m_world.comm());

  MPI_Allreduce(
      MPI_IN_PLACE, vir_norms.data(), b.size(), MPI_DOUBLE, MPI_SUM,
      m_world.comm());

  LOG.os<2>("NORMS ALL (OCC): \n");
  for (auto v : occ_norms) { LOG.os<2>(v, " "); }
  LOG.os<2>('\n');

  LOG.os<2>("NORMS ALL (VIR): \n");
  for (auto v : vir_norms) { LOG.os<2>(v, " "); }
  LOG.os<2>('\n');

  std::sort(
      idx_occ.begin(), idx_occ.end(), [&occ_norms](const int a, const int b) {
        return (occ_norms[a] > occ_norms[b]);
      });

  std::sort(
      idx_vir.begin(), idx_vir.end(), [&vir_norms](const int a, const int b) {
        return (vir_norms[a] > vir_norms[b]);
      });

  double totnorm_occ = 0.0;
  double totnorm_vir = 0.0;

  std::vector<bool> atoms_check(natoms, false);

  for (auto idx : idx_occ) {
    // std::cout << totnorm << std::endl;
    if (totnorm_occ < theta) {
      totnorm_occ += occ_norms[idx];
      int iatom = blkmap[idx];
      atoms_check[iatom] = true;
    }
    else {
      break;
    }
  }

  for (auto idx : idx_vir) {
    // std::cout << totnorm << std::endl;
    if (totnorm_vir < theta) {
      totnorm_vir += vir_norms[idx];
      int iatom = blkmap[idx];
      atoms_check[iatom] = true;
    }
    else {
      break;
    }
  }

  LOG.os<1>("ATOM CHECK: \n");
  for (auto v : atoms_check) { LOG.os<1>(v, " "); }
  LOG.os<1>('\n');

  return atoms_check;
}

adcmod::canon_lmo adcmod::get_canon_nto(dbcsr::shared_matrix<double> u_ia)
{
  LOG.os<>(
      "Computing canonicalized NTO coefficient matrices with eps = ", m_cutoff,
      '\n');

  auto c_bo = m_wfn->hf_wfn->c_bo_A();
  auto c_bv = m_wfn->hf_wfn->c_bv_A();
  auto eps_o = *m_wfn->hf_wfn->eps_occ_A();
  auto eps_v = *m_wfn->hf_wfn->eps_vir_A();

  // Preliminaries
  auto b = c_bo->row_blk_sizes();
  auto o = c_bo->col_blk_sizes();
  auto v = c_bv->col_blk_sizes();

  // Step 1: SVD decomposition of u_ia'

  LOG.os<>("-- Computing SVD decomposition of u_ia\n");

  auto u_ia_copy = dbcsr::matrix<>::copy(*u_ia).build();

  double norm = 1.0 / sqrt(u_ia->dot(*u_ia));
  u_ia_copy->scale(norm);

  math::SVD svd_decomp(m_world, u_ia_copy, 'V', 'V', 1);
  svd_decomp.compute(m_cutoff);
  int rank = svd_decomp.rank();

  auto r = dbcsr::split_range(rank, o[0]);

  auto u_ir = svd_decomp.U(o, r);
  auto vt_rv = svd_decomp.Vt(r, v);

  // Step 2: Orthogonalize MOs

  LOG.os<>("-- Orthogonalizing new MOs\n");

  math::SVD svd_o(m_world, u_ir, 'V', 'V', 1);
  math::SVD svd_v(m_world, vt_rv, 'V', 'V', 1);

  svd_o.compute(m_ortho_eps);
  svd_v.compute(m_ortho_eps);

  auto t = dbcsr::split_range(svd_o.rank(), o[0]);
  auto s = dbcsr::split_range(svd_v.rank(), o[1]);

  auto uortho_ot = svd_o.U(o, t);
  auto vtortho_sv = svd_v.Vt(s, v);

  // Step 3: Form fock matrices

  LOG.os<>("-- Forming Fock matrices\n");

  auto wrd = c_bo->get_cart();

  auto form_fock = [&](std::vector<int> m, std::vector<double> eps_m) {
    auto f = dbcsr::matrix<>::create()
                 .set_cart(wrd)
                 .name("fock")
                 .row_blk_sizes(m)
                 .col_blk_sizes(m)
                 .matrix_type(dbcsr::type::no_symmetry)
                 .build();

    f->reserve_diag_blocks();
    f->set_diag(eps_m);

    return f;
  };

  auto f_oo = form_fock(o, eps_o);
  auto f_vv = form_fock(v, eps_v);

  auto f_tt = u_transform(f_oo, 'T', uortho_ot, 'N', uortho_ot);
  auto f_ss = u_transform(f_vv, 'N', vtortho_sv, 'T', vtortho_sv);

  // Step 4 : Canonicalize

  LOG.os<>("-- Canonicalizing NTOs\n");

  math::hermitian_eigen_solver hsolver_o(m_world, f_tt, 'V', false);
  math::hermitian_eigen_solver hsolver_v(m_world, f_ss, 'V', false);

  hsolver_o.compute();
  hsolver_v.compute();

  auto canon_tt = hsolver_o.eigvecs();
  auto canon_ss = hsolver_v.eigvecs();

  auto eps_t = hsolver_o.eigvals();
  auto eps_s = hsolver_v.eigvals();

  auto trans_to = dbcsr::matrix<>::create()
                      .name("trans ot")
                      .set_cart(wrd)
                      .row_blk_sizes(t)
                      .col_blk_sizes(o)
                      .matrix_type(dbcsr::type::no_symmetry)
                      .build();

  auto trans_sv = dbcsr::matrix<>::create()
                      .name("trans vs")
                      .set_cart(wrd)
                      .row_blk_sizes(s)
                      .col_blk_sizes(v)
                      .matrix_type(dbcsr::type::no_symmetry)
                      .build();

  auto c_bt = dbcsr::matrix<>::create()
                  .name("SVD c_bo")
                  .set_cart(wrd)
                  .row_blk_sizes(b)
                  .col_blk_sizes(t)
                  .matrix_type(dbcsr::type::no_symmetry)
                  .build();

  auto c_bs = dbcsr::matrix<>::create()
                  .name("SVD c_bv")
                  .set_cart(wrd)
                  .row_blk_sizes(b)
                  .col_blk_sizes(s)
                  .matrix_type(dbcsr::type::no_symmetry)
                  .build();

  LOG.os<>("-- Forming final NTO coefficient matrices\n");

  dbcsr::multiply('T', 'T', 1.0, *canon_tt, *uortho_ot, 0.0, *trans_to)
      .perform();
  dbcsr::multiply('T', 'N', 1.0, *canon_ss, *vtortho_sv, 0.0, *trans_sv)
      .perform();

  dbcsr::multiply('N', 'T', 1.0, *c_bo, *trans_to, 0.0, *c_bt).perform();
  dbcsr::multiply('N', 'T', 1.0, *c_bv, *trans_sv, 0.0, *c_bs).perform();

  auto print = [&](auto v) {
    for (auto d : v) { LOG.os<>(d, " "); }
    LOG.os<>('\n');
  };

  LOG.os<>("Occupied energies.\n");
  print(eps_o);
  print(eps_t);

  LOG.os<>("Virtual energies.\n");
  print(eps_v);
  print(eps_s);

  int no_t = c_bo->nfullcols_total();
  int nv_t = c_bv->nfullcols_total();
  int no = c_bt->nfullcols_total();
  int nv = c_bs->nfullcols_total();

  auto p_bb = dbcsr::matrix<double>::create()
                  .name("p_bb")
                  .set_cart(m_world.dbcsr_grid())
                  .row_blk_sizes(b)
                  .col_blk_sizes(b)
                  .matrix_type(dbcsr::type::symmetric)
                  .build();

  auto p_bb_loc = dbcsr::matrix<double>::create()
                      .name("p_bb")
                      .set_cart(m_world.dbcsr_grid())
                      .row_blk_sizes(b)
                      .col_blk_sizes(b)
                      .matrix_type(dbcsr::type::symmetric)
                      .build();

  dbcsr::multiply('N', 'T', 1.0, *c_bo, *c_bo, 0.0, *p_bb).perform();

  dbcsr::multiply('N', 'T', 1.0, *c_bt, *c_bt, 0.0, *p_bb_loc).perform();

  p_bb->filter(1e-4);
  p_bb_loc->filter(1e-4);

  LOG.os<>("SPARSITY: ", p_bb->occupation(), " ", p_bb_loc->occupation(), '\n');

  LOG.os<>(
      "DIMENSIONS REDUCED FROM: ", no_t, "/", nv_t, " -> ", no, "/", nv, '\n');

  return canon_lmo{c_bt, c_bs, trans_to, trans_sv, eps_t, eps_s};
}

adcmod::canon_lmo adcmod::get_canon_pao(dbcsr::shared_matrix<double> u_ia)
{
  LOG.os<>("Computing restricted canonical molecular orbitals.\n");

  // =================================================================
  // ======== STEP 1: localize orbitals ==============================
  // =================================================================

  LOG.os<>("Computing PAOs\n");

  locorb::mo_localizer moloc(m_world, m_wfn->mol);

  ints::aofactory aofac(m_wfn->mol, m_world);
  auto s_bb = aofac.ao_overlap();

  auto c_bo = m_wfn->hf_wfn->c_bo_A();
  auto c_bv = m_wfn->hf_wfn->c_bv_A();

  auto [opao_bm, c2pao_mo] = moloc.compute_pao(c_bo, s_bb);
  auto [vpao_bn, c2pao_nv] = moloc.compute_pao(c_bv, s_bb);

  // =================================================================
  // ============ STEP 2 : TRANSFORM U ===============================
  // =================================================================

  LOG.os<>("Transforming guess vector\n");

  auto u_mn = u_transform(u_ia, 'N', c2pao_mo, 'T', c2pao_nv);

  double norm = u_mn->norm(dbcsr_norm_frobenius);

  u_mn->scale(1.0 / norm);

  // =================================================================
  // ================ STEP 3 : GET NORMS =============================
  // =================================================================

  LOG.os<>("Extracting norms form transformed guess vector\n");

  int noccs = c_bo->nfullcols_total();
  int nvirs = c_bv->nfullcols_total();
  int n_pao_occs = u_mn->nfullrows_total();
  int n_pao_virs = u_mn->nfullcols_total();

  std::vector<double> onorms(n_pao_occs, 0.0), vnorms(n_pao_virs, 0.0);
  std::vector<int> oidx(n_pao_occs, 0), vidx(n_pao_virs, 0);

  std::iota(oidx.begin(), oidx.end(), 0);
  std::iota(vidx.begin(), vidx.end(), 0);

  dbcsr::iterator iter(*u_mn);
  iter.start();

  while (iter.blocks_left()) {
    iter.next_block();

    int o_size = iter.row_size();
    int v_size = iter.col_size();

    int o_off = iter.row_offset();
    int v_off = iter.col_offset();

    for (int iv = 0; iv != v_size; ++iv) {
      for (int io = 0; io != o_size; ++io) {
        onorms[o_off + io] += std::pow(iter(io, iv), 2.0);
        vnorms[v_off + iv] += std::pow(iter(io, iv), 2.0);
      }
    }
  }

  // communicate to all processes
  MPI_Allreduce(
      MPI_IN_PLACE, onorms.data(), n_pao_occs, MPI_DOUBLE, MPI_SUM,
      m_world.comm());

  MPI_Allreduce(
      MPI_IN_PLACE, vnorms.data(), n_pao_virs, MPI_DOUBLE, MPI_SUM,
      m_world.comm());

  LOG.os<2>("NORMS ALL (OCC): \n");
  for (auto v : onorms) { LOG.os<2>(v, " "); }
  LOG.os<2>('\n');

  LOG.os<2>("NORMS ALL (VIR): \n");
  for (auto v : vnorms) { LOG.os<2>(v, " "); }
  LOG.os<2>('\n');

  // =================================================================
  // ========== STEP 4 : GET SIGNIFICANT ATOMS =======================
  // =================================================================

  LOG.os<>("Finding significant atoms\n");

  double threshold = 1 - m_cutoff;

  std::sort(oidx.begin(), oidx.end(), [&onorms](const int a, const int b) {
    return (onorms[a] > onorms[b]);
  });

  std::sort(vidx.begin(), vidx.end(), [&vnorms](const int a, const int b) {
    return (vnorms[a] > vnorms[b]);
  });

  double totnorm_occ = 0.0;
  double totnorm_vir = 0.0;

  int nocc_restricted = 0;
  int nvir_restricted = 0;

  while (totnorm_occ < threshold && nocc_restricted < n_pao_occs) {
    totnorm_occ += onorms[oidx[nocc_restricted++]];
  }

  while (totnorm_vir < threshold && nvir_restricted < n_pao_virs) {
    totnorm_vir += vnorms[vidx[nvir_restricted++]];
  }

  // get atom centre of PAOs

  auto cbas = m_wfn->mol->c_basis();
  auto atoms = m_wfn->mol->atoms();
  auto blkmap = cbas->block_to_atom(atoms);

  std::vector<int> func_to_atom_occ;
  std::vector<int> func_to_atom_vir;

  for (size_t iblk = 0; iblk != cbas->size(); ++iblk) {
    int iatom = blkmap[iblk];
    int nbf = desc::nbf(cbas->at(iblk).shells);

    std::vector<int> map(nbf, iatom);
    func_to_atom_occ.insert(func_to_atom_occ.end(), map.begin(), map.end());
    func_to_atom_vir.insert(func_to_atom_vir.end(), map.begin(), map.end());
  }

  std::vector<bool> use_atom(atoms.size(), false);

  for (int io = 0; io != nocc_restricted; ++io) {
    int iatom = func_to_atom_occ[oidx[io]];
    use_atom[iatom] = true;
  }

  for (int iv = 0; iv != nvir_restricted; ++iv) {
    int iatom = func_to_atom_vir[vidx[iv]];
    use_atom[iatom] = true;
  }

  LOG.os<1>("Atoms used:\n");
  for (size_t ii = 0; ii != use_atom.size(); ++ii) {
    if (use_atom[ii])
      LOG.os<1>(ii, " ");
  }
  LOG.os<1>('\n');

  // =================================================================
  // ========== STEP 5 : PROJECT ONTO SMALLER BASIS SET ==============
  // =================================================================

  LOG.os<>("Projecting MOs onto local basis set\n");

  std::vector<desc::Shell> vshell_sub;

  for (size_t ii = 0; ii != cbas->size(); ++ii) {
    int iatom = blkmap[ii];
    if (use_atom[iatom]) {
      vshell_sub.insert(
          vshell_sub.end(), cbas->at(ii).shells.begin(),
          cbas->at(ii).shells.end());
    }
  }

  auto cbas_sub =
      std::make_shared<desc::cluster_basis>(vshell_sub, "atomic", 8);

  auto p = cbas_sub->cluster_sizes();

  ints::aofactory aofac_pp(m_world, cbas_sub);
  ints::aofactory aofac_bp(m_world, cbas, nullptr, cbas_sub);

  auto s_pp = aofac_pp.ao_overlap();
  auto s_bp = aofac_bp.ao_overlap2();

  math::LLT llt_pp(m_world, s_pp, true);
  math::LLT llt_bb(m_world, s_bb, true);

  llt_pp.compute();
  llt_bb.compute();

  auto b = m_wfn->mol->dims().b();
  auto o = m_wfn->mol->dims().oa();
  auto v = m_wfn->mol->dims().va();

  auto s_inv_pp = llt_pp.inverse(p);
  auto s_sqrt_bb = llt_bb.L(b);

  auto temp_po = dbcsr::matrix<double>::create()
                     .set_cart(m_world.dbcsr_grid())
                     .name("temp_po")
                     .row_blk_sizes(p)
                     .col_blk_sizes(o)
                     .matrix_type(dbcsr::type::no_symmetry)
                     .build();

  auto temp_pv = dbcsr::matrix<double>::create()
                     .set_cart(m_world.dbcsr_grid())
                     .name("temp_pv")
                     .row_blk_sizes(p)
                     .col_blk_sizes(v)
                     .matrix_type(dbcsr::type::no_symmetry)
                     .build();

  auto cortho_bo =
      dbcsr::matrix<double>::create_template(*c_bo).name("cortho_bo").build();

  auto cortho_bv =
      dbcsr::matrix<double>::create_template(*c_bv).name("cortho_bv").build();

  auto cortho_po = dbcsr::matrix<double>::create_template(*temp_po)
                       .name("cortho_po")
                       .build();

  auto cortho_pv = dbcsr::matrix<double>::create_template(*temp_pv)
                       .name("cortho_pv")
                       .build();

  dbcsr::multiply('T', 'N', 1.0, *s_sqrt_bb, *c_bo, 0.0, *cortho_bo).perform();
  dbcsr::multiply('T', 'N', 1.0, *s_sqrt_bb, *c_bv, 0.0, *cortho_bv).perform();

  dbcsr::multiply('T', 'N', 1.0, *s_bp, *cortho_bo, 0.0, *temp_po).perform();
  dbcsr::multiply('T', 'N', 1.0, *s_bp, *cortho_bv, 0.0, *temp_pv).perform();

  dbcsr::multiply('N', 'N', 1.0, *s_inv_pp, *temp_po, 1.0, *cortho_po)
      .perform();
  dbcsr::multiply('N', 'N', 1.0, *s_inv_pp, *temp_pv, 1.0, *cortho_pv)
      .perform();

  // =================================================================
  // ==========   STEP 6 : PERFORM AN SVD    =========================
  // =================================================================

  LOG.os<>("Performing SVD on truncated MO coefficients\n");

  math::SVD svd_occ(m_world, cortho_po, 'V', 'V', 0);
  math::SVD svd_vir(m_world, cortho_pv, 'V', 'V', 0);

  svd_occ.compute(m_ortho_eps);
  svd_vir.compute(m_ortho_eps);

  int nocc_red = svd_occ.rank();
  int nvir_red = svd_vir.rank();

  auto o_red = dbcsr::split_range(nocc_red, m_wfn->mol->mo_split());
  auto v_red = dbcsr::split_range(nvir_red, m_wfn->mol->mo_split());

  auto trans_red_cmo_oo = svd_occ.Vt(o_red, o);
  auto trans_red_cmo_vv = svd_vir.Vt(v_red, v);

  auto s_o = svd_occ.s();
  auto s_v = svd_vir.s();

  auto print = [&](auto v) {
    for (auto ele : v) { LOG.os<>(ele, " "); }
    LOG.os<>('\n');
  };

  LOG.os<>("SVD occ values\n");
  print(s_o);
  LOG.os<>("SVD vir values\n");
  print(s_v);

  // red0cmo_oo->scale(s_o, "left");
  // red0cmo_vv->scale(s_v, "left");

  // dbcsr::print(*trans_red_cmo_oo);
  // exit(0);

  LOG.os<>(
      "Reduced MO space from ", noccs, "/", nvirs, " to ", nocc_red, "/",
      nvir_red, '\n');

  // =================================================================
  // ========== STEP 6 : CANONICALIZE ================================
  // =================================================================

  LOG.os<>("Cononicalizing restricted localized molecular orbitals.\n");

  // std::cout << "1" << std::endl;

  auto f_oo = dbcsr::matrix<double>::create()
                  .set_cart(m_world.dbcsr_grid())
                  .name("f_oo")
                  .row_blk_sizes(o)
                  .col_blk_sizes(o)
                  .matrix_type(dbcsr::type::no_symmetry)
                  .build();

  auto f_vv = dbcsr::matrix<double>::create()
                  .set_cart(m_world.dbcsr_grid())
                  .name("f_vv")
                  .row_blk_sizes(v)
                  .col_blk_sizes(v)
                  .matrix_type(dbcsr::type::no_symmetry)
                  .build();

  f_oo->reserve_diag_blocks();
  f_vv->reserve_diag_blocks();

  // std::cout << "2" << std::endl;

  auto eps_occ = *m_wfn->hf_wfn->eps_occ_A();
  auto eps_vir = *m_wfn->hf_wfn->eps_vir_A();

  f_oo->set_diag(*m_wfn->hf_wfn->eps_occ_A());
  f_vv->set_diag(*m_wfn->hf_wfn->eps_vir_A());

  auto f_mm = u_transform(f_oo, 'N', trans_red_cmo_oo, 'T', trans_red_cmo_oo);
  auto f_nn = u_transform(f_vv, 'N', trans_red_cmo_vv, 'T', trans_red_cmo_vv);

  // std::cout << "3" << std::endl;

  math::hermitian_eigen_solver hermocc(m_world, f_mm, 'V', true);
  math::hermitian_eigen_solver hermvir(m_world, f_nn, 'V', true);

  hermocc.compute();
  hermvir.compute();

  auto eps_m = hermocc.eigvals();
  auto eps_n = hermvir.eigvals();

  auto trans_red_spade_oo = hermocc.eigvecs();
  auto trans_red_spade_vv = hermvir.eigvecs();

  // std::cout << "4" << std::endl;

  auto trans_spade_cmo_oo = dbcsr::matrix<double>::create()
                                .name("trans")
                                .set_cart(m_world.dbcsr_grid())
                                .row_blk_sizes(o_red)
                                .col_blk_sizes(o)
                                .matrix_type(dbcsr::type::no_symmetry)
                                .build();

  auto trans_spade_cmo_vv = dbcsr::matrix<double>::create()
                                .name("trans")
                                .set_cart(m_world.dbcsr_grid())
                                .row_blk_sizes(v_red)
                                .col_blk_sizes(v)
                                .matrix_type(dbcsr::type::no_symmetry)
                                .build();

  // std::cout << "5" << std::endl;

  dbcsr::multiply(
      'T', 'N', 1.0, *trans_red_spade_oo, *trans_red_cmo_oo, 0.0,
      *trans_spade_cmo_oo)
      .perform();

  dbcsr::multiply(
      'T', 'N', 1.0, *trans_red_spade_vv, *trans_red_cmo_vv, 0.0,
      *trans_spade_cmo_vv)
      .perform();

  auto l_bo = dbcsr::matrix<double>::create()
                  .set_cart(m_world.dbcsr_grid())
                  .name("localized restricted OLMOs")
                  .row_blk_sizes(b)
                  .col_blk_sizes(o_red)
                  .matrix_type(dbcsr::type::no_symmetry)
                  .build();

  auto l_bv = dbcsr::matrix<double>::create()
                  .set_cart(m_world.dbcsr_grid())
                  .name("localized restricted OLMOs")
                  .row_blk_sizes(b)
                  .col_blk_sizes(v_red)
                  .matrix_type(dbcsr::type::no_symmetry)
                  .build();

  // std::cout << "7" << std::endl;

  dbcsr::multiply('N', 'T', 1.0, *c_bo, *trans_spade_cmo_oo, 0.0, *l_bo)
      .perform();

  dbcsr::multiply('N', 'T', 1.0, *c_bv, *trans_spade_cmo_vv, 0.0, *l_bv)
      .perform();

  LOG.os<>("Finished!");

  LOG.os<>("Old occ energies:\n");
  print(eps_occ);

  LOG.os<>("Old vir energies:\n");
  print(eps_vir);

  LOG.os<>("New occ energies:\n");
  print(eps_m);

  LOG.os<>("New vir energies:\n");
  print(eps_n);

  // check sparsity

  auto p_bb = dbcsr::matrix<double>::create()
                  .name("p_bb")
                  .set_cart(m_world.dbcsr_grid())
                  .row_blk_sizes(b)
                  .col_blk_sizes(b)
                  .matrix_type(dbcsr::type::symmetric)
                  .build();

  auto p_bb_loc = dbcsr::matrix<double>::create()
                      .name("p_bb")
                      .set_cart(m_world.dbcsr_grid())
                      .row_blk_sizes(b)
                      .col_blk_sizes(b)
                      .matrix_type(dbcsr::type::symmetric)
                      .build();

  dbcsr::multiply('N', 'T', 1.0, *c_bo, *c_bo, 0.0, *p_bb).perform();

  dbcsr::multiply('N', 'T', 1.0, *l_bo, *l_bo, 0.0, *p_bb_loc).perform();

  p_bb->filter(1e-4);
  p_bb_loc->filter(1e-4);

  LOG.os<>("SPARSITY: ", p_bb->occupation(), " ", p_bb_loc->occupation(), '\n');

  return canon_lmo{l_bo,  l_bv, trans_spade_cmo_oo, trans_spade_cmo_vv,
                   eps_m, eps_n};
}

std::vector<bool> adcmod::get_significant_atoms(
    dbcsr::shared_matrix<double> u_ia)
{
  LOG.os<>("Getting significant atoms\n");

  // =================================================================
  // ======== STEP 1: localize orbitals ==============================
  // =================================================================

  LOG.os<>("Computing PAOs\n");

  auto b = m_wfn->mol->dims().b();

  locorb::mo_localizer moloc(m_world, m_wfn->mol);

  ints::aofactory aofac(m_wfn->mol, m_world);
  auto s_bb = aofac.ao_overlap();

  auto c_bo = m_wfn->hf_wfn->c_bo_A();
  auto c_bv = m_wfn->hf_wfn->c_bv_A();

  auto [opao_bm, c2pao_mo] = moloc.compute_pao(c_bo, s_bb);
  auto [vpao_bn, c2pao_nv] = moloc.compute_pao(c_bv, s_bb);

  // =================================================================
  // ============ STEP 2 : TRANSFORM U ===============================
  // =================================================================

  LOG.os<>("Transforming guess vector\n");

  auto u_mn = u_transform(u_ia, 'N', c2pao_mo, 'T', c2pao_nv);

  double norm = u_mn->norm(dbcsr_norm_frobenius);

  u_mn->scale(1.0 / norm);

  // =================================================================
  // ================ STEP 3 : GET NORMS =============================
  // =================================================================

  LOG.os<>("Extracting norms form transformed guess vector\n");

  int n_pao_occs = u_mn->nfullrows_total();
  int n_pao_virs = u_mn->nfullcols_total();

  std::vector<double> onorms(n_pao_occs, 0.0), vnorms(n_pao_virs, 0.0);
  std::vector<int> oidx(n_pao_occs, 0), vidx(n_pao_virs, 0);

  std::iota(oidx.begin(), oidx.end(), 0);
  std::iota(vidx.begin(), vidx.end(), 0);

  dbcsr::iterator iter(*u_mn);
  iter.start();

  while (iter.blocks_left()) {
    iter.next_block();

    int o_size = iter.row_size();
    int v_size = iter.col_size();

    int o_off = iter.row_offset();
    int v_off = iter.col_offset();

    for (int iv = 0; iv != v_size; ++iv) {
      for (int io = 0; io != o_size; ++io) {
        onorms[o_off + io] += std::pow(iter(io, iv), 2.0);
        vnorms[v_off + iv] += std::pow(iter(io, iv), 2.0);
      }
    }
  }

  // communicate to all processes
  MPI_Allreduce(
      MPI_IN_PLACE, onorms.data(), n_pao_occs, MPI_DOUBLE, MPI_SUM,
      m_world.comm());

  MPI_Allreduce(
      MPI_IN_PLACE, vnorms.data(), n_pao_virs, MPI_DOUBLE, MPI_SUM,
      m_world.comm());

  LOG.os<2>("NORMS ALL (OCC): \n");
  for (auto v : onorms) { LOG.os<2>(v, " "); }
  LOG.os<2>('\n');

  LOG.os<2>("NORMS ALL (VIR): \n");
  for (auto v : vnorms) { LOG.os<2>(v, " "); }
  LOG.os<2>('\n');

  // =================================================================
  // ========== STEP 4 : GET SIGNIFICANT ATOMS =======================
  // =================================================================

  LOG.os<>("Finding significant atoms\n");

  double threshold = 1 - m_cutoff;

  std::sort(oidx.begin(), oidx.end(), [&onorms](const int a, const int b) {
    return (onorms[a] > onorms[b]);
  });

  std::sort(vidx.begin(), vidx.end(), [&vnorms](const int a, const int b) {
    return (vnorms[a] > vnorms[b]);
  });

  double totnorm_occ = 0.0;
  double totnorm_vir = 0.0;

  int nocc_restricted = 0;
  int nvir_restricted = 0;

  while (totnorm_occ < threshold && nocc_restricted < n_pao_occs) {
    totnorm_occ += onorms[oidx[nocc_restricted++]];
  }

  while (totnorm_vir < threshold && nvir_restricted < n_pao_virs) {
    totnorm_vir += vnorms[vidx[nvir_restricted++]];
  }

  // get atom centre of PAOs

  auto cbas = m_wfn->mol->c_basis();
  auto atoms = m_wfn->mol->atoms();
  auto blkmap = cbas->block_to_atom(atoms);

  std::vector<int> func_to_atom_occ;
  std::vector<int> func_to_atom_vir;

  for (size_t iblk = 0; iblk != cbas->size(); ++iblk) {
    int iatom = blkmap[iblk];
    int nbf = desc::nbf(cbas->at(iblk).shells);

    std::vector<int> map(nbf, iatom);
    func_to_atom_occ.insert(func_to_atom_occ.end(), map.begin(), map.end());
    func_to_atom_vir.insert(func_to_atom_vir.end(), map.begin(), map.end());
  }

  std::vector<bool> use_atom(atoms.size(), false);

  for (int io = 0; io != nocc_restricted; ++io) {
    int iatom = func_to_atom_occ[oidx[io]];
    use_atom[iatom] = true;
  }

  for (int iv = 0; iv != nvir_restricted; ++iv) {
    int iatom = func_to_atom_vir[vidx[iv]];
    use_atom[iatom] = true;
  }

  LOG.os<1>("Atoms used:\n");
  for (size_t ii = 0; ii != use_atom.size(); ++ii) {
    if (use_atom[ii])
      LOG.os<1>(ii, " ");
  }
  LOG.os<1>('\n');

  return use_atom;
}

/*
adcmod::canon_lmo_mol adcmod::get_restricted_cmos2(dbcsr::shared_matrix<double>
u_ia) {

        LOG.os<>("Computing restricted canonical molecular orbitals.\n");

        // =================================================================
        // ======== STEP 1: localize orbitals ==============================
        // =================================================================

        // =================================================================
        // ============ STEP 2 : TRANSFORM U ===============================
        // =================================================================

    // =================================================================
        // ================ STEP 3 : GET NORMS =============================
        // =================================================================

        // =================================================================
        // ========== STEP 4 : GET SIGNIFICANT ATOMS =======================
        // =================================================================

        auto use_atom = get_significant_atoms(u_ia);

        // =================================================================
        // ========== STEP 5 : PROJECT ONTO SUBSCAPCE ==============
        // =================================================================

        LOG.os<>("Projection onto subspace basis\n");

        auto mol = m_wfn->mol;
        auto cbas = mol->c_basis();

        auto c_bo = m_wfn->hf_wfn->c_bo_A();
        auto c_bv = m_wfn->hf_wfn->c_bv_A();

        int noccs = c_bo->nfullcols_total();
        int nvirs = c_bv->nfullcols_total();

        auto blkmap = cbas->block_to_atom(mol->atoms());

        std::vector<desc::Shell> vshell_sub;

        for (int ii = 0; ii != cbas->size(); ++ii) {
                int iatom = blkmap[ii];
                if (use_atom[iatom]) {
                        vshell_sub.insert(vshell_sub.end(),
cbas->at(ii).begin(), cbas->at(ii).end());
                }
        }

        auto cbas_sub = std::make_shared<desc::cluster_basis>(
                vshell_sub, cbas->split_method(), cbas->nsplit());

        auto b = cbas->cluster_sizes();
        auto p = cbas_sub->cluster_sizes();
        auto o = m_wfn->mol->dims().oa();
        auto v = m_wfn->mol->dims().va();

        ints::aofactory aofac_pp(m_world, cbas_sub);
        ints::aofactory aofac_bp(m_world, cbas, nullptr, cbas_sub);

        auto s_bb = aofac_bp.ao_overlap();
        auto s_pp = aofac_pp.ao_overlap();
        auto s_bp = aofac_bp.ao_overlap2();

        math::LLT llt_pp(m_world, s_pp, true);
        math::LLT llt_bb(m_world, s_bb, true);

        llt_pp.compute();
        llt_bb.compute();

        auto s_inv_pp = llt_pp.inverse(p);
        auto s_sqrt_bb = llt_bb.L(b);
        auto s_invsqrt_pp = llt_pp.L_inv(p);

        auto temp_po = dbcsr::matrix<double>::create()
                .set_cart(m_world.dbcsr_grid())
                .name("temp_po")
                .row_blk_sizes(p)
                .col_blk_sizes(o)
                .matrix_type(dbcsr::type::no_symmetry)
                .build();

        auto temp_pv = dbcsr::matrix<double>::create()
                .set_cart(m_world.dbcsr_grid())
                .name("temp_pv")
                .row_blk_sizes(p)
                .col_blk_sizes(v)
                .matrix_type(dbcsr::type::no_symmetry)
                .build();

        auto cortho_bo = dbcsr::matrix<double>::create_template(*c_bo)
                .name("cortho_bo")
                .build();

        auto cortho_bv = dbcsr::matrix<double>::create_template(*c_bv)
                .name("cortho_bv")
                .build();

        auto cortho_po = dbcsr::matrix<double>::create_template(*temp_po)
                .name("cortho_po")
                .build();

        auto cortho_pv = dbcsr::matrix<double>::create_template(*temp_pv)
                .name("cortho_pv")
                .build();

        dbcsr::multiply('T', 'N', 1.0, *s_sqrt_bb, *c_bo, 0.0,
*cortho_bo).perform(); dbcsr::multiply('T', 'N', 1.0, *s_sqrt_bb, *c_bv, 0.0,
*cortho_bv).perform();

        dbcsr::multiply('T', 'N', 1.0, *s_bp, *cortho_bo, 0.0,
*temp_po).perform(); dbcsr::multiply('T', 'N', 1.0, *s_bp, *cortho_bv, 0.0,
*temp_pv).perform();

        dbcsr::multiply('N', 'N', 1.0, *s_inv_pp, *temp_po, 1.0, *cortho_po)
                .perform();
        dbcsr::multiply('N', 'N', 1.0, *s_inv_pp, *temp_pv, 1.0, *cortho_pv)
                .perform();

        LOG.os<>("Reduced active basis set from ", cbas->nbf(), " to ",
cbas_sub->nbf(), '\n');

        // =================================================================
        // ==========   STEP 6 : PERFORM AN SVD    =========================
        // =================================================================

        LOG.os<>("Performing SVD on truncated MO coefficients\n");

        math::SVD svd_occ(m_world, cortho_po, 'V', 'V', 0);
        math::SVD svd_vir(m_world, cortho_pv, 'V', 'V', 0);

        svd_occ.compute(1e-12);
        svd_vir.compute(1e-12);

        int nocc_red = svd_occ.rank();
        int nvir_red = svd_vir.rank();

        auto o_red = dbcsr::split_range(nocc_red, m_wfn->mol->mo_split());
        auto v_red = dbcsr::split_range(nvir_red, m_wfn->mol->mo_split());

        auto trans_red_cmo_oo = svd_occ.Vt(o_red, o);
        auto trans_red_cmo_vv = svd_vir.Vt(v_red, v);

        auto u_po = svd_occ.U(p,o_red);
        auto u_pv = svd_vir.U(p,v_red);

        auto s_o = svd_occ.s();
        auto s_v = svd_vir.s();

        auto print = [&](auto v) {
                for (auto ele : v) {
                        LOG.os<>(ele, " ");
                } LOG.os<>('\n');
        };

        LOG.os<>("SVD occ values\n");
        print(s_o);
        LOG.os<>("SVD vir values\n");
        print(s_v);

        //dbcsr::print(*trans_red_cmo_oo);

        LOG.os<>("Reduced MO space from ", noccs, "/", nvirs, " to ",
                nocc_red, "/", nvir_red, '\n');

        // =================================================================
        // ========== STEP 6 : CANONICALIZE ================================
        // =================================================================

        LOG.os<>("Cononicalizing restricted localized molecular orbitals.\n");

        //std::cout << "1" << std::endl;

        auto f_oo = dbcsr::matrix<double>::create()
                .set_cart(m_world.dbcsr_grid())
                .name("f_oo")
                .row_blk_sizes(o)
                .col_blk_sizes(o)
                .matrix_type(dbcsr::type::no_symmetry)
                .build();

        auto f_vv = dbcsr::matrix<double>::create()
                .set_cart(m_world.dbcsr_grid())
                .name("f_vv")
                .row_blk_sizes(v)
                .col_blk_sizes(v)
                .matrix_type(dbcsr::type::no_symmetry)
                .build();

        f_oo->reserve_diag_blocks();
        f_vv->reserve_diag_blocks();

        //std::cout << "2" << std::endl;

        auto eps_occ = *m_wfn->hf_wfn->eps_occ_A();
        auto eps_vir = *m_wfn->hf_wfn->eps_vir_A();

        f_oo->set_diag(*m_wfn->hf_wfn->eps_occ_A());
        f_vv->set_diag(*m_wfn->hf_wfn->eps_vir_A());

        auto f_mm = u_transform(f_oo, 'N', trans_red_cmo_oo, 'T',
trans_red_cmo_oo); auto f_nn = u_transform(f_vv, 'N', trans_red_cmo_vv, 'T',
trans_red_cmo_vv);

        //std::cout << "3" << std::endl;

        math::hermitian_eigen_solver hermocc(m_world, f_mm, 'V', true);
        math::hermitian_eigen_solver hermvir(m_world, f_nn, 'V', true);

        hermocc.compute();
        hermvir.compute();

        auto eps_m = hermocc.eigvals();
        auto eps_n = hermvir.eigvals();

        auto trans_red_spade_oo = hermocc.eigvecs();
        auto trans_red_spade_vv = hermvir.eigvecs();

        //dbcsr::print(*trans_red_spade_oo);

        //std::cout << "4" << std::endl;

        auto trans_spade_cmo_oo = dbcsr::matrix<double>::create()
                .name("trans")
                .set_cart(m_world.dbcsr_grid())
                .row_blk_sizes(o_red)
                .col_blk_sizes(o)
                .matrix_type(dbcsr::type::no_symmetry)
                .build();

        auto trans_spade_cmo_vv = dbcsr::matrix<double>::create()
                .name("trans")
                .set_cart(m_world.dbcsr_grid())
                .row_blk_sizes(v_red)
                .col_blk_sizes(v)
                .matrix_type(dbcsr::type::no_symmetry)
                .build();

        //std::cout << "5" << std::endl;

        dbcsr::multiply('T', 'N', 1.0, *trans_red_spade_oo, *trans_red_cmo_oo,
                0.0, *trans_spade_cmo_oo).perform();

        dbcsr::multiply('T', 'N', 1.0, *trans_red_spade_vv, *trans_red_cmo_vv,
                0.0, *trans_spade_cmo_vv).perform();

        //dbcsr::print(*trans_spade_cmo_oo);

        auto lortho_po = dbcsr::matrix<double>::create()
                .set_cart(m_world.dbcsr_grid())
                .name("localized restricted OLMOs")
                .row_blk_sizes(p)
                .col_blk_sizes(o_red)
                .matrix_type(dbcsr::type::no_symmetry)
                .build();

        auto lortho_pv = dbcsr::matrix<double>::create()
                .set_cart(m_world.dbcsr_grid())
                .name("localized restricted OLMOs")
                .row_blk_sizes(p)
                .col_blk_sizes(v_red)
                .matrix_type(dbcsr::type::no_symmetry)
                .build();

        auto l_po = dbcsr::matrix<double>::create()
                .set_cart(m_world.dbcsr_grid())
                .name("localized restricted OLMOs")
                .row_blk_sizes(p)
                .col_blk_sizes(o)
                .matrix_type(dbcsr::type::no_symmetry)
                .build();

        auto l_pv = dbcsr::matrix<double>::create()
                .set_cart(m_world.dbcsr_grid())
                .name("localized restricted OLMOs")
                .row_blk_sizes(p)
                .col_blk_sizes(v)
                .matrix_type(dbcsr::type::no_symmetry)
                .build();

        //std::cout << "7" << std::endl;

        dbcsr::multiply('N', 'T', 1.0, *cortho_po, *trans_spade_cmo_oo, 0.0,
*lortho_po) .perform();

        dbcsr::multiply('N', 'T', 1.0, *cortho_pv, *trans_spade_cmo_vv, 0.0,
*lortho_pv) .perform();

        dbcsr::multiply('T', 'N', 1.0, *s_invsqrt_pp, *cortho_po, 0.0, *l_po)
                .perform();

        dbcsr::multiply('T', 'N', 1.0, *s_invsqrt_pp, *cortho_pv, 0.0, *l_pv)
                .perform();

        //dbcsr::print(*lortho_po);
        //dbcsr::print(*cortho_bo);

        dbcsr::multiply('N', 'N', 1.0, *s_invsqrt_pp, *u_po, 0.0, *l_po)
                .perform();

        dbcsr::multiply('N', 'N', 1.0, *s_invsqrt_pp, *u_pv, 0.0, *l_pv)
                .perform();

        //dbcsr::print(*l_po);

        LOG.os<>("Finished!");

        LOG.os<>("Old occ energies:\n");
        //print(eps_occ);

        LOG.os<>("Old vir energies:\n");
//	print(eps_vir);

        LOG.os<>("New occ energies:\n");
        //print(eps_m);

        LOG.os<>("New vir energies:\n");
        //print(eps_n);

        std::vector<int> atom_list;

        for (int ii = 0; ii != m_wfn->mol->atoms().size(); ++ii) {
                if (use_atom[ii]) atom_list.push_back(ii);
        }

        int noa = l_po->nfullcols_total();
        int nob = noa;
        int nva = l_pv->nfullcols_total();
        int nvb = nva;

        auto loc_mol = mol->fragment(noa, nob, nva, nvb, atom_list);

        //dbcsr::print(*c_bo);

        Eigen::MatrixXd oo_I = Eigen::MatrixXd::Identity(noa, noa);
        Eigen::MatrixXd vv_I = Eigen::MatrixXd::Identity(nva, nva);

        auto t_oo =dbcsr::eigen_to_matrix(oo_I, m_world.dbcsr_grid(), "oo",
                o, o, dbcsr::type::no_symmetry);

        auto t_vv =dbcsr::eigen_to_matrix(vv_I, m_world.dbcsr_grid(), "vv",
                v, v, dbcsr::type::no_symmetry);

        return canon_lmo_mol{loc_mol, l_po, l_pv, t_oo, t_vv,
                *m_wfn->hf_wfn->eps_occ_A(), *m_wfn->hf_wfn->eps_vir_A()};

}*/

}  // end namespace adc

}  // namespace megalochem
