#ifndef DESC_BASIS_H
#define DESC_BASIS_H

#include "desc/atom.hpp"
#include "io/data_handler.hpp"
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <cmath>

namespace megalochem {

namespace desc {
	
/// df_Kminus1[k] = (k-1)!!
static constexpr std::array<int64_t,31> df_Kminus1 = {{1LL, 1LL, 1LL, 2LL, 3LL, 8LL, 
	15LL, 48LL, 105LL, 384LL, 945LL, 3840LL, 10395LL, 46080LL, 135135LL,
	645120LL, 2027025LL, 10321920LL, 34459425LL, 185794560LL, 654729075LL,
	3715891200LL, 13749310575LL, 81749606400LL, 316234143225LL, 1961990553600LL,
	7905853580625LL, 51011754393600LL, 213458046676875LL, 1428329123020800LL,
	6190283353629375LL}};
	
struct Shell {
	
	std::array<double,3> O;
	bool pure;
	size_t l;
	std::vector<double> coeff;
	std::vector<double> alpha;
	
	size_t cartesian_size() const {
		return (l + 1) * (l + 2) / 2;
	}
	
	size_t size() const {
		return pure ? (2 * l + 1) : cartesian_size();
    }
    
    size_t ncontr() const { return coeff.size(); }
    size_t nprim() const { return alpha.size(); }
    
    Shell unit_shell() {
		Shell out;
		out.O = {0,0,0};
		out.l = 0;
		coeff = std::vector<double>{1.0};
		alpha = std::vector<double>{0.0};
		return out;
	}
	
	bool operator ==(const Shell &b) const {
		return (this->O == b.O) && (this->pure == b.pure)
			&& (this->l == b.l) && (this->coeff == b.coeff)
			&& (this->alpha == b.alpha);
	}
		

};

inline std::ostream& operator<<(std::ostream& out, const Shell& s) {
	out << "{\n";
	out << "\t" << "O: [" << s.O[0] << ", " << s.O[1] << ", " << s.O[2] << "]\n";
	out << "\t" << "l: " << s.l << "\n";
	out << "\t" << "shell: {\n";
	for (int i = 0; i != s.alpha.size(); ++i) {
		out << "\t" << "\t" << s.alpha[i] << "\t" << s.coeff[i] << '\n';
	} 
	out << "\t" << "}\n";
	out << "}";
	return out;
}

using vshell = std::vector<Shell>;
using vvshell = std::vector<vshell>;

inline size_t nbf(vshell bas) {
	size_t n = 0;
	for (auto& s : bas) {
		n += s.size();
	}
	return n;
}

inline size_t max_nprim(vshell bas) {
	size_t n = 0;
	for (auto& s : bas) {
		n = std::max(n, s.nprim());
	}
	return n;
}

inline size_t max_l(vshell bas) {
	size_t n = 0;
	for (auto& s : bas) {
		n = std::max(n, s.l);
	}
	return n;
}

inline int atom_of(Shell& s, std::vector<Atom>& atoms) {
	for (int i = 0; i != atoms.size(); ++i) {
		double dist = sqrt(pow(s.O[0] - atoms[i].x,2.0)
			+ pow(s.O[1] - atoms[i].y,2.0)
			+ pow(s.O[2] - atoms[i].z,2.0));
		if (dist < std::numeric_limits<double>::epsilon()) {
			return i;
		}
	}
	return -1;
}

inline static const int DEFAULT_NSPLIT = 8;
inline static const std::string DEFAULT_SPLIT_METHOD = "multi_shell_strict_sp";

class cluster_basis {
private:

	std::vector<vshell> m_clusters;
	std::vector<int> m_cluster_sizes;
	std::vector<bool> m_cluster_diff; // false: not diffuse | true : diffuse
	std::vector<std::string> m_cluster_types; // s, p, d, ... sp, spdf, ...
	std::vector<int> m_shell_offsets;
	int m_nsplit;
	std::string m_split_method;
	
public:

	struct global {
		static inline double cutoff = 1e-8;
		static inline double step = 0.5;
		static inline int maxiter = 5000;
	};

	cluster_basis() {}
	
	cluster_basis(std::string basname, std::vector<desc::Atom>& atoms, 
		std::optional<std::string> method = std::nullopt, 
		std::optional<int> nsplit = std::nullopt, 
		std::optional<bool> augmented = std::nullopt);
	
	cluster_basis(vshell basis, 
		std::optional<std::string> method = std::nullopt, 
		std::optional<int> nsplit = std::nullopt, 
		std::optional<vshell> augbasis = std::nullopt);
	
	cluster_basis(const cluster_basis& cbasis) : 
		m_clusters(cbasis.m_clusters),
		m_cluster_sizes(cbasis.m_cluster_sizes),
		m_shell_offsets(cbasis.m_shell_offsets),
		m_nsplit(cbasis.m_nsplit),
		m_split_method(cbasis.m_split_method),
		m_cluster_diff(cbasis.m_cluster_diff),
		m_cluster_types(cbasis.m_cluster_types) {}
		
	cluster_basis& operator =(const cluster_basis& cbasis) {
		if (this != &cbasis) {
			m_clusters = cbasis.m_clusters;
			m_cluster_sizes = cbasis.m_cluster_sizes;
			m_shell_offsets = cbasis.m_shell_offsets;
			m_nsplit = cbasis.m_nsplit;
			m_split_method = cbasis.m_split_method;
			m_cluster_diff = cbasis.m_cluster_diff;
			m_cluster_types = cbasis.m_cluster_types;
		}
		
		return *this;
	}
	
	vvshell::iterator begin() {
		return m_clusters.begin();
	}
	
	vvshell::iterator end() {
		return m_clusters.end();
	}
	
	vshell& operator[](int i) {
		return m_clusters[i];
	}
	
	vshell& at(int i) {
		return m_clusters[i];
	}
	
	const vshell& operator[](int i) const {
		return m_clusters[i];
	}
	
	std::vector<int> cluster_sizes() const;
	
	int shell_offset(int i) const {
		return m_shell_offsets[i];
	}
	
	std::vector<int> block_to_atom(std::vector<desc::Atom> atoms) const;
	
	int nsplit() { return m_nsplit; }
	std::string split_method() { return m_split_method; }
	
	std::vector<double> min_alpha() const;
	
	std::vector<double> radii(double cutoff = 1e-8, double step = 0.2, 
		int maxiter = 1000) const;
	
	std::vector<bool> diffuse() const;
	
	std::vector<std::string> types() const;
	
	size_t max_nprim() const;

	size_t nbf() const;

	size_t max_l() const;
	
	size_t size() const {
		return m_clusters.size();
	}
	
	std::vector<int> nshells() {
		std::vector<int> out;
		for (auto& cluster : m_clusters) {
			out.push_back(cluster.size());
		}
		return out;
	} 
	
	size_t nshells_tot() {
		size_t n = 0;
		for (auto& c : m_clusters) {
			n += c.size();
		}
		return n;
	}
	
	void print_info() const;
	
	std::shared_ptr<cluster_basis> fragment(std::vector<int> block_list);

};

using shared_cluster_basis = std::shared_ptr<cluster_basis>;

}

} // end mega

#endif
