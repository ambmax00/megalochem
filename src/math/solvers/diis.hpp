#ifndef DIIS_HELPER_H
#define DIIS_HELPER_H

#include <Eigen/QR>
#include <cassert>
#include <deque>
#include <vector>

#include <dbcsr_matrix_ops.hpp>
#include "megalochem.hpp"
#include "utils/mpi_log.hpp"

namespace megalochem {

namespace math {

using smat_d = dbcsr::shared_matrix<double>;

template <int N>
class diis_helper {
 private:
  world m_world;

  std::deque<smat_d> m_delta;
  std::deque<smat_d> m_trialvecs;
  smat_d m_last_ele;

  Eigen::MatrixXd m_B;
  Eigen::MatrixXd m_coeffs;

  const int m_max;
  const int m_min;
  const int m_start;
  const int m_print;

  util::mpi_log LOG;

 public:
  diis_helper(world w, int start, int min, int max, int print = 0) :
      m_world(w), m_B(0, 0), m_coeffs(0, 0), m_max(max), m_min(min),
      m_start(start), m_print(print), LOG(w.comm(), m_print){};

  void compute_extrapolation_parameters(smat_d& T, smat_d& err, int iter)
  {
    if (iter >= m_start) {
      // std::cout << "Iteration: " << iter << std::endl;
      LOG.os<2>("Number of error vectors stored: ", m_delta.size(), '\n');
      LOG.os<2>("Number of trial vectors stored: ", m_trialvecs.size(), '\n');
      // std::cout << "Size of delta: " << m_delta.size() << std::endl;

      bool reduce = false;
      if ((int)m_delta.size() >= m_max)
        reduce = true;

      auto err_copy = dbcsr::matrix<>::copy(*err).build();

      m_delta.push_back(err_copy);

      // determine error vector with max RMS
      auto to_erase = std::max_element(
          m_delta.begin(), m_delta.end(), [&](smat_d& e1, smat_d& e2) -> bool {
            return e1->norm(dbcsr_norm_frobenius) <
                e2->norm(dbcsr_norm_frobenius);
          });

      size_t max_pos = to_erase - m_delta.begin();
      LOG.os<2>("Max element found at position: ", max_pos, '\n');

      if (reduce)
        m_delta.erase(to_erase);

      // make a copy, put it into trialvecs
      auto m_in = dbcsr::matrix<>::create_template(*T)
                      .name("Trial Vec " + iter)
                      .build();
      m_in->copy_in(*T);

      m_trialvecs.push_back(m_in);

      if (reduce)
        m_trialvecs.erase(m_trialvecs.begin() + max_pos);

      // std::cout << "Trail vectors stored: " << m_trialvecs.size() <<
      // std::endl; std::cout << "Size of delta: " << m_delta.size() <<
      // std::endl;

      int nerr = m_delta.size();

      // compute new entries
      Eigen::VectorXd v(nerr);

      for (int i = 0; i != v.size(); ++i) {
        auto& ei = m_delta[i];
        auto& efin = m_delta[nerr - 1];

        // std::cout << "ei" << std::endl;
        // std::cout << ei << std::endl;

        v(i) = ei->dot(*efin);
      }

      // std::cout << v << std::endl;

      if (reduce) {
        LOG.os<2>("B before resizing...\n", m_B, '\n');
        // reomve max element column and row
        // first the row

        // std::cout  << max_pos << " " << 0 << " " << m_B.rows() - max_pos << "
        // " << m_B.cols() << std::endl; std::cout << max_pos + 1 << " " << 0 <<
        // " " << m_B.rows() - max_pos << " " <<  m_B.cols() << std::endl;

        m_B.block(max_pos, 0, m_B.rows() - max_pos - 1, m_B.cols()) =
            m_B.block(max_pos + 1, 0, m_B.rows() - max_pos - 1, m_B.cols());

        // std::cout << 0 << " " << max_pos << " " << m_B.rows() << " " <<
        // m_B.cols() - max_pos << std::endl; std::cout << 0 << max_pos + 1 << "
        // " << m_B.rows() << " " << m_B.cols() - max_pos << std::endl;

        m_B.block(0, max_pos, m_B.rows(), m_B.cols() - max_pos - 1) =
            m_B.block(0, max_pos + 1, m_B.rows(), m_B.cols() - max_pos - 1);

        // std::cout << "Reducing!" << std::endl;
        // Eigen::MatrixXd Bcrop = m_B.bottomRightCorner(nerr-1,nerr-1);
        // std::cout << Bcrop << std::endl;
        m_B.conservativeResize(nerr, nerr);

        LOG.os<2>(1, "B after resizing...\n", m_B, '\n');
        // m_B = Bcrop;
      }
      else {
        // std::cout << "Resizing..." << std::endl;
        if (nerr != 0)
          m_B.conservativeResize(nerr, nerr);
      }

      for (int i = 0; i != nerr; ++i) {
        m_B(nerr - 1, i) = v(i);
        m_B(i, nerr - 1) = v(i);
      }

      LOG.os<2>("New B: ", '\n', m_B, '\n');
      // std::cout << "Here is m_B" << std::endl;
      // std::cout << m_B << std::endl;

      if (m_delta.size() != 0) {
        // ADD lagrange stuff
        Eigen::MatrixXd Bsolve(nerr + 1, nerr + 1);

        Bsolve.block(0, 0, nerr, nerr) = m_B.block(0, 0, nerr, nerr);

        for (int i = 0; i != nerr; ++i) {
          Bsolve(nerr, i) = -1;
          Bsolve(i, nerr) = -1;
        }

        Bsolve(nerr, nerr) = 0;

        LOG.os<2>("B solve: ", '\n', Bsolve, '\n');
        // std::cout << "B solve" << std::endl;
        // std::cout << Bsolve << std::endl;

        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(nerr + 1, 1);
        C(nerr, 0) = -1;

        // std::cout << "C is" << std::endl;
        // std::cout << C << std::endl;

        // Solve Bsolve * X = C
        Eigen::MatrixXd X = Bsolve.colPivHouseholderQr().solve(C);

        assert(C.isApprox(Bsolve * X));

        // std::cout << "Solution:" << std::endl;
        // std::cout << X << std::endl;

        m_coeffs = X.block(0, 0, nerr, 1);

        LOG.os<2>("New coefficients: ", '\n', m_coeffs, '\n');
        // std::cout << "m_coeffs: " << std::endl;
        // std::cout << m_coeffs_ << std::endl;
      }
    }
  }

  void extrapolate(smat_d& trial, int iter)
  {
    extrapolate(trial, m_coeffs, iter);
  }

  void extrapolate(smat_d& trial, Eigen::MatrixXd& coeffs, int iter)
  {
    static bool first = true;

    if (coeffs.size() != (int)m_trialvecs.size())
      throw std::runtime_error("DIIS: Wrong dimensions.");

    if (iter >= m_start && (int)m_trialvecs.size() >= m_min) {
      if (first) {
        LOG.os<>("Starting DIIS...\n");
        first = false;
      }

      LOG.os<2>("Extrapolating...\n");
      LOG.os<2>("Extrapolation factors:\n");

      for (int i = 0; i != m_coeffs.size(); ++i) { LOG.os<2>(coeffs(i), " "); }
      LOG.os<2>('\n');

      // dbcsr::print(*trial);
      trial->clear();

      // do M = c1 * T1 + c2 * T2 + ...
      for (int i = 0; i != coeffs.size(); ++i) {
        // dbcsr::print(*m_trialvecs[i]);
        trial->add(1.0, coeffs(i), *m_trialvecs[i]);
      }

      // std::cout << "Extrapolated M" << std::endl;
      // std::cout << M << std::endl;

      // exchange last element in trailvecs for this one
      // Some sources say we should, some say we shouldn't?
      // m_trialvecs.pop_back();
      // m_trialvecs.push_back(M);
    }
  }

  Eigen::MatrixXd& coeffs()
  {
    return m_coeffs;
  }

};  // end class

}  // end namespace math

}  // namespace megalochem

#endif
