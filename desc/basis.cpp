#include "desc/basis.h"
#include <libint2/basis.h>

namespace desc {

cluster_basis::cluster_basis(vshell& basis, std::string method) : m_basis(basis) {
	
	std::vector<size_t> shell2bf = libint2::BasisSet::compute_shell2bf(basis);
	
	int nbas = libint2::nbf(basis);
	int nsplit = 1; // = (method == "atomic") ? 1 : INT_MAX;
		
	int nfunc(0);
		
	vshell cluster_part;
	int n = 1;
	libint2::Shell prev_shell;
		
	auto coord = [&](libint2::Shell& s1, libint2::Shell& s2) {	
		return (s1.O[0] == s2.O[0]) && (s1.O[1] == s2.O[1]) && (s1.O[2] == s2.O[2]);
	};
	
	auto angmom = [](libint2::Shell& s1, libint2::Shell& s2) {
		return (s1.contr.size() == s2.contr.size()) && (s1.contr[0].l == s2.contr[0].l);
	};
		
	for (int i = 0; i != basis.size(); ++i) {
			
		//std::cout << "Shell Nr. " << i << std::endl;
			
		if (i == 0) {
				
			cluster_part.push_back(basis[i]);
			prev_shell = basis[i];
				
		} else {
			
			bool push;
			
			if (method == "shell") {
				push = (coord(basis[i], prev_shell) && angmom(basis[i], prev_shell));
			} else {
				push = coord(basis[i], prev_shell);
			}
			
			if (push) {
				
				//std::cout << "  Same as previous" << std::endl;
				cluster_part.push_back(basis[i]);
	
			} else {
				
				if ( n < nsplit ) {
				
					//std::cout << "...still okay" << std::endl;
					cluster_part.push_back(basis[i]);
					n += 1;

				} else {
					
					//std::cout << "...pushing." << std::endl;
					n = 1;
					m_clusters.push_back(cluster_part);
					cluster_part.clear();
					cluster_part.push_back(basis[i]);
					
				}
			}		
		}	
		
		if (i == basis.size() - 1) {
			m_clusters.push_back(cluster_part);
		}
			prev_shell = basis[i];
		
	}
	
	for (auto c : m_clusters) {
		m_cluster_sizes.push_back(libint2::nbf(c));
	}		
	
}

size_t cluster_basis::max_nprim() const {
	
	size_t n = 0;
	for (int i = 0; i != m_clusters.size(); ++i) {
		n = std::max(libint2::max_nprim(m_clusters[i]), n);
	}
	return n;	
}

size_t cluster_basis::nbf() const {
	
	size_t nbas = 0;
	for (int i = 0; i != m_clusters.size(); ++i) {
		nbas += libint2::nbf(m_clusters[i]);
	}
	return nbas;
}

int cluster_basis::max_l() const {
	
	int l = 0;
		for (int i = 0; i != m_clusters.size(); ++i) {
		l = std::max(libint2::max_l(m_clusters[i]), l);
	}
	return l;	
}

std::vector<int> cluster_basis::cluster_sizes() const {
	return m_cluster_sizes;
}

}
