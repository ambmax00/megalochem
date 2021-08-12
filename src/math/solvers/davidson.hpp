#ifndef MATH_DAVIDSON_H
#define MATH_DAVIDSON_H

#include <Eigen/Eigenvalues>
#include <dbcsr_conversions.hpp>
#include <dbcsr_matrix_ops.hpp>
#include <optional>
#include <stdexcept>
#include <vector>
#include "math/solvers/diis.hpp"
#include "megalochem.hpp"
#include "utils/mpi_log.hpp"

namespace megalochem {

namespace math {

using smat = dbcsr::shared_matrix<double>;

template <class MVFactory>
class davidson {
 private:
  world m_world;

  util::mpi_log LOG;
  smat m_diag;  // diagonal of matrix
  std::shared_ptr<MVFactory> m_fac;  // Matrix vector product engine

  int m_nroots;  // requested number of eigenvalues
  int m_subspace;  // current size of subspace

  bool m_pseudo;  // whether this is a pseudo-eval problem
  bool m_converged;  // wether procedure has converged
  bool m_block;  // wether we optimize all roots at once (Liu)
  bool m_balancing;  // use modified davidson by Parrish et al.

  std::vector<double> m_eigvals;  // eignvalues
  std::vector<double> m_errs;  // errors of the roots
  std::vector<double> m_vnorms;  // norms of trial vectors

  std::vector<smat> m_vecs;  // b vectors
  std::vector<smat> m_sigmas;  // Ab vectors
  std::vector<smat> m_ritzvecs;  // ritz vectors at end of computation

  double m_conv;
  int m_maxiter;
  int m_maxsubspace;

 public:
  davidson& set_factory(std::shared_ptr<MVFactory>& fac)
  {
    m_fac = fac;
    return *this;
  }

  davidson& set_diag(smat& in)
  {
    m_diag = in;
    return *this;
  }

  davidson& pseudo(bool is_pseudo)
  {
    m_pseudo = is_pseudo;
    return *this;
  }

  davidson& conv(double c)
  {
    m_conv = c;
    return *this;
  }

  davidson& maxiter(int maxi)
  {
    m_maxiter = maxi;
    return *this;
  }

  davidson& block(bool is_block)
  {
    m_block = is_block;
    return *this;
  }

  davidson& balancing(bool do_balancing)
  {
    m_balancing = do_balancing;
    return *this;
  }

  davidson(world w, int nprint) :
      m_world(w), LOG(w.comm(), nprint), m_diag(nullptr), m_fac(nullptr),
      m_nroots(1), m_subspace(0), m_pseudo(false), m_converged(false),
      m_block(true), m_balancing(true), m_conv(1e-5), m_maxiter(100),
      m_maxsubspace(30)
  {
  }

  void compute(
      std::vector<smat>& guess,
      int nroots,
      std::optional<double> omega = std::nullopt)
  {
    // if (m_balancing) throw std::runtime_error("BALANCING not yet completed.
    // -> NOrmalization!!");

    LOG.os<>("Launching davidson diagonalization.\n");
    LOG.os<>("Convergence: ", m_conv, '\n');

    if (!omega && m_pseudo)
      throw std::runtime_error(
          "Davidson solver initialized as pseudo-eigenvalue problem, but no "
          "omega given.");

    double prev_omega(0.0), current_omega(0.0), init_omega(0.0);
    if (omega) {
      prev_omega = *omega;
      init_omega = *omega;
    }

    // copy guesses
    m_vecs.clear();

    for (auto ptr : guess) {
      auto copy = dbcsr::matrix<>::copy(*ptr).build();
      m_vecs.push_back(copy);
    }

    LOG.os<1>("GUESS SIZE: ", m_vecs.size(), '\n');

    m_nroots = nroots;
    int nvecs_max = m_maxsubspace * m_nroots;

    LOG.os<1>("Max subspace: ", nvecs_max, '\n');

    if (m_nroots > (int)m_vecs.size()) {
      std::string msg = std::to_string(m_nroots) +
          " roots requested, but only " + std::to_string(m_vecs.size()) +
          " guesses given.\n";
      throw std::runtime_error(msg);
    }

    m_errs.clear();
    m_errs.resize(m_nroots);

    Eigen::MatrixXd Asub;
    Eigen::VectorXd Svec;
    Eigen::VectorXd evals;
    Eigen::MatrixXd evecs;
    Eigen::VectorXd prev_evals(0);
    Eigen::MatrixXd prev_evecs(0, 0);

    m_sigmas.clear();

    for (int ITER = 0; ITER != m_maxiter; ++ITER) {
      LOG.os<>("DAVIDSON ITERATION: ", ITER, '\n');

      // current subspace size
      m_subspace = m_vecs.size();
      int prev_subspace = m_sigmas.size();

      LOG.os<>("SUBSPACE: ", m_subspace, " PREV: ", prev_subspace, '\n');

      LOG.os<>("Computing MV products.\n");
      for (int i = prev_subspace; i != m_subspace; ++i) {
        // std::cout << "GUESS VECTOR: " << i << std::endl;
        if (LOG.global_plev() > 2) {
          LOG.os<>("GUESS VECTOR: \n");
          dbcsr::print(*m_vecs[i]);
        }

        auto Av_i = (m_pseudo) ? m_fac->compute(m_vecs[i], *omega) :
                                 m_fac->compute(m_vecs[i]);
        m_sigmas.push_back(Av_i);

        // std::cout << "SIGMA VECTOR: " << i << std::endl;
        if (LOG.global_plev() > 2) {
          LOG.os<>("SIGMA VECTOR: \n");
          dbcsr::print(*Av_i);
        }

        double vnorm = std::sqrt(m_vecs[i]->dot(*m_vecs[i]));
        m_vnorms.push_back(vnorm);
      }

      // form subspace matrix and overlap matrix
      Asub.conservativeResize(m_subspace, m_subspace);
      Svec.conservativeResize(m_subspace);

      for (int i = prev_subspace; i != m_subspace; ++i) {
        for (int j = 0; j != i + 1; ++j) {
          double vsig;

          if (m_balancing) {
            vsig = (m_vnorms[j] > m_vnorms[i]) ? m_vecs[i]->dot(*m_sigmas[j]) :
                                                 m_vecs[j]->dot(*m_sigmas[i]);
          }
          else {
            vsig = m_vecs[i]->dot(*m_sigmas[j]);
          }

          if (i == j)
            Svec(i) = m_vecs[i]->dot(*m_vecs[j]);

          Asub(i, j) = vsig;
          Asub(j, i) = vsig;
        }
      }

      LOG.os<2>("SUBSPACE MATRIX:\n");
      LOG.os<2>(Asub, '\n');

      LOG.os<2>("OVERLAP MATRIX DIAGONAL:\n");
      LOG.os<2>(Svec, '\n');

      // diagonalize it
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;

      if (m_balancing) {
        // compute sqrt inv of overlap
        Eigen::VectorXd Svec_invsqrt = Svec;

        for (int i = 0; i != m_subspace; ++i) {
          Svec_invsqrt(i) = 1.0 / std::sqrt(Svec(i));
        }

        Eigen::MatrixXd A_prime =
            Svec_invsqrt.asDiagonal() * Asub * Svec_invsqrt.asDiagonal();

        LOG.os<1>("SUBSPACE MATRIX TRANSFORMED: \n", A_prime, '\n');

        es.compute(A_prime);
        Eigen::MatrixXd evecs_prime = es.eigenvectors();

        evecs = Svec_invsqrt.asDiagonal() * evecs_prime;
        evals = es.eigenvalues();
      }
      else {
        es.compute(Asub);
        evals = es.eigenvalues();
        evecs = es.eigenvectors();
      }

      for (int i = 0; i != m_subspace; ++i) {
        // get min and max coeff
        double minc = evecs.col(i).minCoeff();
        double maxc = evecs.col(i).maxCoeff();
        double maxabs = (std::fabs(maxc) > std::fabs(minc)) ? maxc : minc;

        if (maxabs < 0)
          evecs.col(i) *= -1;
      }

      /* if zero eigenvalues, collapse subspace
      bool has_zero = false;
      for (int i = 0; i != */

      if (LOG.global_plev() >= 1) {
        LOG.os<1>("EIGENVALUES: \n");
        for (int i = 0; i != evals.size(); ++i) { LOG.os<1>(evals[i], " "); }
        LOG.os<1>('\n');
      }

      for (int i = 0; i != evals.size(); ++i) {
        if (evals[i] < 1e-12)
          throw std::runtime_error("EIGENVALUE ZERO OR NEGATIVE.\n");
      }

      LOG.os<2>("EIGENVECTORS: \n", evecs, '\n');

      prev_evals = evals;
      prev_evecs = evecs;

      // compute residual
      // r_k = sum_i U_ik sigma_i - sum_i U_ik lambda_k b_i

      LOG.os<1>("Computing residual.\n");

      // ======= Compute all residuals r_k ========

      std::vector<smat> residuals(m_nroots);
      smat temp =
          dbcsr::matrix<>::create_template(*m_vecs[0]).name("temp").build();
      double max_err = 0;

      int start_root = (m_block) ? 0 : m_nroots - 1;

      for (int iroot = start_root; iroot != m_nroots; ++iroot) {
        // int iroot = m_nroots - 1;

        smat r_k =
            dbcsr::matrix<>::create_template(*m_vecs[0]).name("temp").build();

        for (int i = 0; i != m_subspace; ++i) {
          temp->copy_in(*m_sigmas[i]);
          temp->scale(evecs(i, iroot));

          r_k->add(1.0, 1.0, *temp);

          temp->clear();
          temp->copy_in(*m_vecs[i]);

          temp->scale(-evals(iroot) * evecs(i, iroot));

          r_k->add(1.0, 1.0, *temp);

          temp->clear();
        }

        // because we are probably dealing with an approximation to
        // the real diagonal of the matrix, we are checking both the
        // residual and the eigenvectors of the subspace matrix
        // double alpha_ki = (ITER == 0) ? std::numeric_limits<double>::max()
        //		: fabs(evecs(m_subspace-1,iroot));

        double alpha_ki = std::fabs(evecs(m_subspace-1,iroot));
        double rnorm = r_k->norm(dbcsr_norm_frobenius);

        m_errs[iroot] =  std::min(rnorm, alpha_ki);
        // alpha_ki
        //);
        
        LOG.os<>("Root ", iroot, " alpha_ki = ", alpha_ki, " res. norm = ", rnorm, '\n');

        max_err = std::max(max_err, m_errs[iroot]);

        residuals[iroot] = r_k;
      }

      temp->release();

      // if (LOG.global_plev() >= 2) {
      //	dbcsr::print(*r_k);
      //}

      LOG.os<>("Root errors: \n");
      for (int iroot = start_root; iroot != m_nroots; ++iroot) {
        LOG.os<>("Root ", iroot, ": ", m_errs[iroot], '\n');
      }
      LOG.os<>("Max Error: ", max_err, '\n');

      // Convergence criteria
      // Normal davidson: largest norm of residuals is below certain threshold
      // pseudo davidson: largest norm of residual is higher than |omega -
      // omega'|

      if (m_pseudo) {
        current_omega = evals(m_nroots - 1);
        LOG.os<1>(
            "EVALS: ", *omega, " ", evals(m_nroots - 1),
            " Err: ", std::fabs(current_omega - prev_omega), '\n');
      }

      m_converged = (m_pseudo) ?
          ((m_errs[m_nroots - 1] < m_conv) ||
           m_errs[m_nroots - 1] < std::fabs(init_omega - current_omega)) :
          (max_err < m_conv);

      prev_omega = current_omega;

      if (m_converged)
        break;

      // correction
      // Only Diagonal-Preconditioned-Residue for now (DPR)
      // D_ia = (lamda_k - A_iaia)^-1

      LOG.os<1>("Computing correction vectors.\n");
      int nvecs_prev = m_vecs.size();

      for (int iroot = start_root; iroot != m_nroots; ++iroot) {
        // int iroot = m_nroots - 1;

        LOG.os<1>("Computing for root ", iroot, '\n');
        LOG.os<1>("Computing preconditioner.\n");

        smat D = dbcsr::matrix<>::create_template(*residuals[iroot])
                     .name("D")
                     .build();

        D->reserve_all();

        D->set(evals(iroot));

        D->add(1.0, -1.0, *m_diag);

        D->apply(dbcsr::func::inverse);

        if (LOG.global_plev() > 2) {
          dbcsr::print(*D);
        }

        // form new vector
        smat d_k = dbcsr::matrix<>::create_template(*residuals[iroot])
                       .name("d_k")
                       .build();

        // d(k)_ia = D(k)_ia * q(k)_ia

        d_k->hadamard_product(*residuals[iroot], *D);

        double dnorm = std::sqrt(d_k->dot(*d_k));

        if (!m_balancing)
          d_k->scale(1.0 / dnorm);

        residuals[iroot]->release();
        D->release();

        // GRAM SCHMIDT
        // b_new = d_k - sum_j proj_bi(d_k)
        // where proj_b(v) = dot(b,v)/dot(b,b)

        smat bnew = dbcsr::matrix<>::create_template(*d_k)
                        .name("b_" + std::to_string(m_subspace))
                        .build();

        smat temp2 =
            dbcsr::matrix<>::create_template(*d_k).name("temp2").build();

        // dbcsr::copy(d_k, bnew).perform();
        bnew->copy_in(*d_k);

        LOG.os<1>("Computing new guess vector.\n");
        for (size_t i = 0; i != m_vecs.size(); ++i) {
          // dbcsr::copy(*m_vecs[i], temp2).perform();
          temp2->copy_in(*m_vecs[i]);

          double proj = (d_k->dot(*m_vecs[i])) / (m_vecs[i]->dot(*m_vecs[i]));

          temp2->scale(-proj);

          bnew->add(1.0, 1.0, *temp2);
        }

        temp2->release();

        // normalize
        double bnorm = std::sqrt(bnew->dot(*bnew));

        LOG.os<1>("Norm quotient: ", bnorm / dnorm, '\n');

        if (!m_balancing)
          bnew->scale(1.0 / bnorm);

        if (LOG.global_plev() > 2) {
          dbcsr::print(*bnew);
        }

        if (bnorm / dnorm > 1e-3)
          m_vecs.push_back(bnew);
      }

      if ((int)m_vecs.size() == nvecs_prev) {
        throw std::runtime_error(
            "Davidson: linear dependence in trial vector space");
      }

      if ((int)m_vecs.size() > nvecs_max) {
        LOG.os<1>("Subspace too large. \n");
        collapse(evecs);
      }
    }

    m_eigvals.resize(m_nroots);
    std::copy(evals.data(), evals.data() + m_nroots, m_eigvals.data());

    smat temp3 =
        dbcsr::matrix<>::create_template(*m_vecs[0]).name("temp3").build();

    m_ritzvecs.clear();

    LOG.os<1>("Forming Ritz vectors.\n");
    // x_k = sum_i ^M U_ik * b_i
    collapse(evecs);
    m_ritzvecs = m_vecs;

    // std::cout << "FINISHED COMPUTATION" << std::endl;
    LOG.os<>("Finished computation.\n");
  }

  void collapse(Eigen::MatrixXd& evecs)
  {
    LOG.os<1>("Collapsing subspace...\n");

    std::vector<smat> new_vecs;
    smat tempx =
        dbcsr::matrix<>::create_template(*m_vecs[0]).name("tempx").build();

    for (int k = 0; k != m_nroots; ++k) {
      smat x_k = dbcsr::matrix<>::create_template(*m_vecs[0])
                     .name("new_guess_" + std::to_string(k))
                     .build();

      for (int i = 0; i < m_subspace; ++i) {
        tempx->copy_in(*m_vecs[i]);
        tempx->scale(evecs(i, k));

        x_k->add(1.0, 1.0, *tempx);
      }

      // normalize ?
      double norm = std::sqrt(x_k->dot(*x_k));
      if (m_balancing)
        x_k->scale(1.0 / norm);

      new_vecs.push_back(x_k);
    }

    m_vecs = new_vecs;
    m_sigmas.clear();
    m_vnorms.clear();
  }

  std::vector<smat> ritz_vectors()
  {
    return m_ritzvecs;
  }

  std::vector<double> residual_norms()
  {
    return m_errs;
  }

  std::vector<double> eigvals()
  {
    return m_eigvals;
  }

};  // end class

template <class MVFactory>
class diis_davidson {
 private:
  world m_world;
  util::mpi_log LOG;

  int m_macro_maxiter = 30;
  int m_diis_maxiter = 50;
  double m_macro_conv = 1e-5;
  double m_cdiis2_threshhold = 1e-15;

  std::shared_ptr<MVFactory> m_fac;
  smat m_diag;
  davidson<MVFactory> m_dav;

 public:
  diis_davidson& macro_maxiter(int maxi)
  {
    m_macro_maxiter = maxi;
    return *this;
  }

  diis_davidson& macro_conv(double macro)
  {
    m_macro_conv = macro;
    return *this;
  }

  diis_davidson& set_factory(std::shared_ptr<MVFactory>& fac)
  {
    m_fac = fac;
    m_dav.set_factory(fac);
    return *this;
  }

  diis_davidson& set_diag(smat& in)
  {
    m_diag = in;
    m_dav.set_diag(in);
    return *this;
  }

  diis_davidson& micro_maxiter(int maxi)
  {
    m_dav.maxiter(maxi);
    return *this;
  }

  diis_davidson& balancing(bool do_balancing)
  {
    m_dav.balancing(do_balancing);
    return *this;
  }

  diis_davidson(world w, int nprint) :
      m_world(w), LOG(w.comm(), nprint), m_dav(m_world, nprint)
  {
    m_dav.pseudo(true);
    m_dav.block(false);
  }

  void compute(std::vector<smat>& guess, int nroot, double omega)
  {
    std::vector<smat> current_guess = guess;
    double current_omega = omega;

    LOG.os<>("========== STARTING DIIS-DAVIDSON ================\n");
    LOG.os<>("======== PERFORMING PSEUDO-DAVDISON ==============\n");

    double pseudo_dav_conv = std::max(1e-3, m_macro_conv);
    m_dav.conv(pseudo_dav_conv);

    for (int ii = 0; ii != m_macro_maxiter; ++ii) {
      LOG.os<>("=== PSEUDO-DAVIDSON MACROITERATION: ", ii, "\n");

      double old_omega = current_omega;

      m_dav.compute(current_guess, nroot, current_omega);

      current_guess = m_dav.ritz_vectors();
      current_omega = m_dav.eigvals()[nroot - 1];

      auto resnorms = m_dav.residual_norms();

      double err = std::fabs(current_omega - old_omega);

      LOG.os<>(
          "=== MACRO ITERATION ERROR EIGENVALUE/RESIDUAL: ", err, "/ ",
          resnorms[nroot - 1], '\n');
      LOG.os<>("=== EIGENVALUE: ", current_omega, '\n');

      if (resnorms[nroot - 1] < pseudo_dav_conv && err < pseudo_dav_conv)
        break;
    }

    auto b_ov = m_dav.ritz_vectors()[nroot - 1];

    LOG.os<>("============ PSEUDO-DAVIDSON CONVERGED ===========\n");

    LOG.os<>("================= PERFORMING DIIS=================\n");
    math::diis_helper<2> diis(m_world, 0, 1, 8, LOG.global_plev());

    for (int iiter = 0; iiter != m_diis_maxiter; ++iiter) {
      // compute new matrix vector product
      auto sig_ov = m_fac->compute(b_ov, current_omega);

      // compute omega
      double old_omega = current_omega;
      current_omega = (b_ov->dot(*sig_ov)) / (b_ov->dot(*b_ov));

      LOG.os<>(
          "OMEGA: ", current_omega, " ", std::fabs(current_omega - old_omega),
          '\n');

      // compute residual
      // r(i) = (sig(i) - omega(i+1) * u(i))/||u(i)||
      auto r_ov = dbcsr::matrix<>::copy(*sig_ov).name("r_ov").build();

      r_ov->add(1.0, -current_omega, *b_ov);
      r_ov->scale(1.0 / std::sqrt(b_ov->dot(*b_ov)));

      double r_norm = r_ov->norm(dbcsr_norm_frobenius);

      LOG.os<>("==== RESIDUAL NORM: ", r_norm, '\n');

      if (r_norm < m_macro_conv)
        break;

      // compute update
      auto u_ov = dbcsr::matrix<>::create_template(*sig_ov)
                      .name("update vector")
                      .build();

      auto div =
          dbcsr::matrix<>::create_template(*sig_ov).name("divisor").build();

      div->reserve_all();
      div->set(current_omega);
      div->add(1.0, -1.0, *m_diag);

      div->apply(dbcsr::func::inverse);
      u_ov->hadamard_product(*r_ov, *div);
      // u_ov->scale(1.0/sqrt(u_ov->dot(*u_ov)));

      // store vectors
      auto trial = dbcsr::matrix<double>::copy(*b_ov).build();

      trial->add(1.0, 1.0, *u_ov);

      diis.compute_extrapolation_parameters(trial, u_ov, iiter);

      diis.extrapolate(trial, iiter);

      // trial->scale(1.0/sqrt(trial->dot(*trial)));

      b_ov = trial;
    }

    LOG.os<>("DIIS CONVERGED!!\n");
  }

  std::vector<smat> ritz_vectors()
  {
    return m_dav.ritz_vectors();
  }

  std::vector<double> eigval()
  {
    return m_dav.eigvals();
  }

};  // end class

}  // namespace math

}  // namespace megalochem

#endif
