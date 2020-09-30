#ifndef DESC_BASIS_H
#define DESC_BASIS_H

#include "desc/atom.h"
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <cmath>

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
	
	void normalize() {
		
        const auto sqrt_Pi_cubed = double{5.56832799683170784528481798212};
        const auto np = nprim();
        
        assert(l <= 15); // due to df_Kminus1[] a 64-bit integer type; kinda ridiculous restriction anyway
		
		for (int p = 0; p != np; ++p) {
		
			assert(alpha[p] >= 0);
			
			if (alpha[p] != 0) {
			  const auto two_alpha = 2 * alpha[p];
			  const auto two_alpha_to_am32 = std::pow(two_alpha,l+1) * std::sqrt(two_alpha);
			  const auto normalization_factor = std::sqrt(std::pow(2,l) * two_alpha_to_am32/(sqrt_Pi_cubed * df_Kminus1[2*l] ));

			  coeff[p] *= normalization_factor;
			}
			
		}

		// need to force normalization to unity?
		if (true) {
			// compute the self-overlap of the , scale coefficients by its inverse square root
			double norm{0};
			for(int p = 0; p != np; ++p) {
			  for(int q = 0; q<=p; ++q) {
				auto gamma = alpha[p] + alpha[q];
				norm += (p==q ? 1 : 2) * df_Kminus1[2*l] * sqrt_Pi_cubed * coeff[p] * coeff[q] /
						(std::pow(2,l) * std::pow(gamma,l+1) * std::sqrt(gamma));
			  }
			}
			auto normalization_factor = 1 / sqrt(norm);
			for(int p = 0; p != np; ++p) {
			  coeff[p] *= normalization_factor;
			}
		}

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

class cluster_basis {
private:

	std::vector<vshell> m_clusters;
	std::vector<int> m_cluster_sizes;
	std::vector<int> m_shell_offsets;
	int m_nsplit;
	std::string m_split_method;
	
public:

	cluster_basis() {}
	
	cluster_basis(std::string basname, std::vector<desc::Atom>& atoms, std::string method, int nsplit);
	
	cluster_basis(vshell basis, std::string method, int nsplit);
	
	cluster_basis(const cluster_basis& cbasis) : 
		m_clusters(cbasis.m_clusters),
		m_cluster_sizes(cbasis.m_cluster_sizes),
		m_shell_offsets(cbasis.m_shell_offsets),
		m_nsplit(cbasis.m_nsplit),
		m_split_method(cbasis.m_split_method) {}
		
	cluster_basis& operator =(const cluster_basis& cbasis) {
		if (this != &cbasis) {
			m_clusters = cbasis.m_clusters;
			m_cluster_sizes = cbasis.m_cluster_sizes;
			m_shell_offsets = cbasis.m_shell_offsets;
			m_nsplit = cbasis.m_nsplit;
			m_split_method = cbasis.m_split_method;
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
	
	size_t max_nprim() const;

	size_t nbf() const;

	size_t max_l() const;
	
	size_t size() const {
		return m_clusters.size();
	}
	
	std::vector<int> cluster_sizes() const;
	
	int shell_offset(int i) const {
		return m_shell_offsets[i];
	}
	
	std::vector<int> block_to_atom(std::vector<desc::Atom>& atoms) const;
	
	int nsplit() { return m_nsplit; }
	std::string split_method() { return m_split_method; }

};

using shared_cluster_basis = std::shared_ptr<cluster_basis>;

}

#endif
