#ifndef MATH_RCM_H
#define MATH_RCM_H

#include <Eigen/Core>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace megalochem {

namespace math {

/* RCM: takes in a set of coordinates, and a function, then reduces
 * the bandwidth of the connectivity matrix, and spits out a vector
 * with reordered indices (also offers a reordering tool (?))
 */

template <class Point>
class rcm {
  using func = std::function<double(Point, Point)>;

 private:
  const std::vector<Point> m_points;
  const func m_distfunc;
  const double m_cutoff;
  const int m_dim;

  std::vector<int> m_index;
  Eigen::MatrixXd m_distmat;
  Eigen::MatrixXi m_conmat;

  void compute_distmat()
  {
    for (int p1 = 0; p1 != m_dim; ++p1) {
      for (int p2 = p1; p2 != m_dim; ++p2) {
        m_distmat(p1, p2) = m_distfunc(m_points[p1], m_points[p2]);
        m_distmat(p2, p1) = m_distmat(p1, p2);
      }
    }
  }

  void compute_conmat()
  {
    for (int i = 0; i != m_dim * m_dim; ++i) {
      m_conmat.data()[i] = fabs(m_distmat.data()[i]) >= m_cutoff ? 0 : 1;
    }
  }

  int degree(int node)
  {
    int d = 0;

    for (int i = 0; i != m_dim; ++i) { d += m_conmat(node, i); }

    return d - 1;
  }

  int min_degree(std::vector<int>& R)
  {
    // find point with minimum degree (max dist), which is not yet in R

    double max_dist = 0;
    int node1 = 0;
    int node2 = 0;

    for (int i = 0; i < m_dim; ++i) {
      for (int j = i + 1; j < m_dim; ++j) {
        if (m_distmat(i, j) > max_dist) {
          auto iter_i = std::find(R.begin(), R.end(), i);
          auto iter_j = std::find(R.begin(), R.end(), j);

          if (iter_i != R.end() && iter_j != R.end())
            continue;

          max_dist = m_distmat(i, j);
          node1 = i;
          node2 = j;
        }
      }
    }

    // std::cout << "MAX dist at: " << node1 << " " << node2 << " " << max_dist
    // << std::endl;

    // degree of the nodes
    int deg1 = degree(node1);
    int deg2 = degree(node2);

    // std::cout << "Degrees: " << deg1 << " " << deg2 << std::endl;

    // Return the node with the least amount of bonds (If applicable), and
    // which has not yet been included in R

    auto iter_1 = std::find(R.begin(), R.end(), node1);
    auto iter_2 = std::find(R.begin(), R.end(), node2);

    if ((deg1 > deg2 || iter_1 != R.end()) && iter_2 == R.end()) {
      return node2;
    }
    else {
      return node1;
    }
  }

  void add_neighbours(int P, std::deque<int>& Q)
  {
    std::vector<int> q;

    for (int i = 0; i != m_dim; ++i) {
      if ((m_conmat(P, i) == 1) && (i != P)) {
        q.push_back(i);
      }
    }

    // SORT IT ACCORDING TO DEGREE
    auto sortdeg = [&](const int q1, const int q2) -> bool {
      return degree(q1) > degree(q2);
    };

    std::sort(q.begin(), q.end(), sortdeg);

    Q.insert(Q.end(), q.begin(), q.end());
  }

 public:
  rcm(std::vector<Point> t_points, double t_cutoff, func t_distfunc) :
      m_points(t_points), m_distfunc(t_distfunc), m_cutoff(t_cutoff),
      m_dim((int)t_points.size()), m_index(t_points.size()),
      m_distmat(Eigen::MatrixXd::Zero(m_dim, m_dim)),
      m_conmat(Eigen::MatrixXi::Zero(m_dim, m_dim))
  {
  }

  ~rcm(){};

  void compute()
  {
    compute_distmat();
    compute_conmat();

    // std::cout << m_distmat << std::endl;
    // std::cout << m_conmat << std::endl;

    // PREPARE EMTPY QUEUE AND RESULT ARRAY
    std::deque<int> Q;
    std::vector<int> R;

    while ((int)R.size() != m_dim) {
      // find object with minimum degree which is not yet in R
      int P = min_degree(R);

      // add p to R
      R.push_back(P);

      // std::cout << "R" << std::endl;
      // for (auto r : R) {
      //	std::cout << r << " ";
      //} //std::cout << std::endl;

      // ADD TO THE QUEUE ALL ADJACENT NODES TO P
      add_neighbours(P, Q);

      /*std::cout << "Q" << std::endl;
      for (auto r : Q) {
              std::cout << r << " ";
      } std::cout << std::endl;*/

      while (Q.size()) {
        // extract first node
        int Q0 = Q[0];

        // check if in R
        auto iter_Q0 = std::find(R.begin(), R.end(), Q0);

        // if not there, remove and continue
        if (iter_Q0 != R.end()) {
          Q.pop_front();
          continue;
        }

        // else add Q0 to R, and neighbours of Q0 to Q
        R.push_back(Q0);
        add_neighbours(Q0, Q);

        Q.pop_front();

        /*std::cout << "Q" << std::endl;
        for (auto r : Q) {
                std::cout << r << " ";
        } std::cout << std::endl;*/
      }

      if ((int)R.size() == m_dim) {
        break;
      }
    }

    std::reverse(R.begin(), R.end());

    /*std::cout << "Here is R " << std::endl;
    for (int i = 0; i != R.size(); ++i) {
            std::cout << R[i] << " ";
    }
    //std::cout << std::endl;*/

    // REORDER CMAT
    Eigen::MatrixXi conmat2 = Eigen::MatrixXi::Zero(m_dim, m_dim);

    for (int j = 0; j != m_dim; ++j) {
      for (int i = 0; i != m_dim; ++i) { conmat2(i, j) = m_conmat(R[i], R[j]); }
    }

    m_conmat = std::move(conmat2);

    // std::cout << "REORDERED CON MAT" << std::endl;
    // std::cout << m_conmat << std::endl;

    m_index = R;
  }

  std::vector<int> reordered_idx()
  {
    return m_index;
  }

  template <typename T>
  void reorder(T& p_in)
  {
    T p_out = p_in;
    for (size_t i = 0; i != m_points.size(); ++i) {
      p_out[i] = p_in[m_index[i]];
    }
    p_in = p_out;
  }

};  // end class

}  // namespace math

}  // namespace megalochem

#endif
