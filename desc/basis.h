#ifndef DESC_BASIS_H
#define DESC_BASIS_H

#include <libint2/shell.h>

using vshell = std::vector<libint2::Shell>;

namespace desc {
	
class cluster_basis {
private:

	vshell m_basis;
	std::vector<vshell> m_clusters;
	std::vector<int> m_cluster_sizes;
	
public:

	cluster_basis() {}
	
	cluster_basis(vshell& basis, std::string method);
	
	cluster_basis(const cluster_basis& cbasis) : 
		m_basis(cbasis.m_basis),
		m_clusters(cbasis.m_clusters),
		m_cluster_sizes(cbasis.m_cluster_sizes) {}
		
	cluster_basis& operator =(const cluster_basis& cbasis) {
		if (this != &cbasis) {
			m_basis = cbasis.m_basis;
			m_clusters = cbasis.m_clusters;
			m_cluster_sizes = cbasis.m_cluster_sizes;
		}
		
		return *this;
	}
	
	vshell libint_basis() const {
		return m_basis;
	}
	
	vshell& operator[](int i) {
		return m_clusters[i];
	}
	
	const vshell& operator[](int i) const {
		return m_clusters[i];
	}
	
	size_t max_nprim() const;

	size_t nbf() const;

	int max_l() const;
	
	size_t size() const {
		return m_clusters.size();
	}
	
	std::vector<int> cluster_sizes() const;

};

}

#endif
