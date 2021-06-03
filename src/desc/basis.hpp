#ifndef DESC_BASIS_H
#define DESC_BASIS_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "desc/atom.hpp"

namespace megalochem {

namespace desc {

/// df_Kminus1[k] = (k-1)!!
static constexpr std::array<int64_t, 31> df_Kminus1 = {
    {1LL,
     1LL,
     1LL,
     2LL,
     3LL,
     8LL,
     15LL,
     48LL,
     105LL,
     384LL,
     945LL,
     3840LL,
     10395LL,
     46080LL,
     135135LL,
     645120LL,
     2027025LL,
     10321920LL,
     34459425LL,
     185794560LL,
     654729075LL,
     3715891200LL,
     13749310575LL,
     81749606400LL,
     316234143225LL,
     1961990553600LL,
     7905853580625LL,
     51011754393600LL,
     213458046676875LL,
     1428329123020800LL,
     6190283353629375LL}};

struct Shell {
  std::array<double, 3> O;
  bool pure;
  size_t l;
  std::vector<double> coeff;
  std::vector<double> alpha;

  size_t cartesian_size() const
  {
    return (l + 1) * (l + 2) / 2;
  }

  size_t size() const
  {
    return pure ? (2 * l + 1) : cartesian_size();
  }

  size_t ncontr() const
  {
    return coeff.size();
  }
  size_t nprim() const
  {
    return alpha.size();
  }

  Shell unit_shell()
  {
    Shell out;
    out.O = {0, 0, 0};
    out.l = 0;
    coeff = std::vector<double>{1.0};
    alpha = std::vector<double>{0.0};
    return out;
  }

  bool operator==(const Shell& b) const
  {
    return (this->O == b.O) && (this->pure == b.pure) && (this->l == b.l) &&
        (this->coeff == b.coeff) && (this->alpha == b.alpha);
  }
};

inline std::ostream& operator<<(std::ostream& out, const Shell& s)
{
  out << "{\n";
  out << "\t"
      << "O: [" << s.O[0] << ", " << s.O[1] << ", " << s.O[2] << "]\n";
  out << "\t"
      << "l: " << s.l << "\n";
  out << "\t"
      << "shell: {\n";
  for (size_t ii = 0; ii != s.alpha.size(); ++ii) {
    out << "\t"
        << "\t" << s.alpha[ii] << "\t" << s.coeff[ii] << '\n';
  }
  out << "\t"
      << "}\n";
  out << "}";
  return out;
}

using vshell = std::vector<Shell>;
using vvshell = std::vector<vshell>;

inline size_t nbf(vshell bas)
{
  size_t n = 0;
  for (auto& s : bas) { n += s.size(); }
  return n;
}

inline size_t max_nprim(vshell bas)
{
  size_t n = 0;
  for (auto& s : bas) { n = std::max(n, s.nprim()); }
  return n;
}

inline size_t max_l(vshell bas)
{
  size_t n = 0;
  for (auto& s : bas) { n = std::max(n, s.l); }
  return n;
}

inline int atom_of(Shell& s, std::vector<Atom>& atoms)
{
  for (int i = 0; i != (int)atoms.size(); ++i) {
    double dist = sqrt(
        pow(s.O[0] - atoms[i].x, 2.0) + pow(s.O[1] - atoms[i].y, 2.0) +
        pow(s.O[2] - atoms[i].z, 2.0));
    if (dist < std::numeric_limits<double>::epsilon()) {
      return i;
    }
  }
  return -1;
}

inline static const int DEFAULT_NSPLIT = 8;
inline static const std::string DEFAULT_SPLIT_METHOD = "multi_shell_strict_sp";

struct cluster {
 vshell shells;
 std::array<double,3> O;
 bool diffuse;
};

inline int atom_of(cluster& c, std::vector<Atom>& atoms)
{
  for (int i = 0; i != (int)atoms.size(); ++i) {
    double dist = sqrt(
        pow(c.O[0] - atoms[i].x, 2.0) + pow(c.O[1] - atoms[i].y, 2.0) +
        pow(c.O[2] - atoms[i].z, 2.0));
    if (dist < std::numeric_limits<double>::epsilon()) {
      return i;
    }
  }
  return -1;
}

class cluster_basis {
 private:
 
  std::vector<cluster> m_clusters;
  
 public:
  struct global {
    static inline double cutoff = 1e-8;
    static inline double step = 0.5;
    static inline int maxiter = 5000;
  };

  cluster_basis()
  {
  }

  cluster_basis(
      std::string basname,
      std::vector<desc::Atom>& atoms,
      std::optional<std::string> method = std::nullopt,
      std::optional<int> nsplit = std::nullopt,
      std::optional<bool> augmented = std::nullopt);

  cluster_basis(
      vshell basis,
      std::optional<std::string> method = std::nullopt,
      std::optional<int> nsplit = std::nullopt,
      std::optional<vshell> augbasis = std::nullopt);

  cluster_basis(const cluster_basis& cbasis) :
      m_clusters(cbasis.m_clusters)
  {
  }

  cluster_basis& operator=(const cluster_basis& cbasis)
  {
    if (this != &cbasis) {
      m_clusters = cbasis.m_clusters;
    }

    return *this;
  }

  std::vector<cluster>::iterator begin()
  {
    return m_clusters.begin();
  }

  std::vector<cluster>::iterator end()
  {
    return m_clusters.end();
  }
  
  void add(const cluster& c) {
	  m_clusters.push_back(c);
  }

  cluster& operator[](int i)
  {
    return m_clusters[i];
  }

  cluster& at(int i)
  {
    return m_clusters[i];
  }
  
  std::vector<int> shell_offsets() const;

  const cluster& operator[](int i) const
  {
    return m_clusters[i];
  }

  std::vector<int> cluster_sizes() const;

  std::vector<double> min_alpha() const;

  std::vector<double> radii(
      double cutoff = 1e-8, double step = 0.2, int maxiter = 1000) const;

  std::vector<bool> diffuse() const;
  
  std::vector<std::string> shell_types() const;

  int max_nprim() const;

  int nbf() const;

  int max_l() const;

  size_t size() const
  {
    return m_clusters.size();
  }

  std::vector<int> block_to_atom(std::vector<desc::Atom> atoms) const;

  std::vector<int> nshells() const;

  int nshells_tot() const;

  void print_info() const;

};

using shared_cluster_basis = std::shared_ptr<cluster_basis>;

}  // namespace desc

}  // namespace megalochem

#endif
