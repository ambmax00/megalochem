#ifndef DESC_BASIS_H
#define DESC_BASIS_H

#include <libint2/shell.h>

using vshell = std::vector<libint2::Shell>;
using vvshell = std::vector<vshell>;

namespace desc {
	
class cluster_basis {
private:

	vshell m_basis;
	std::vector<vshell> m_clusters;
	std::vector<int> m_cluster_sizes;
	std::vector<int> m_shell_offsets;
	
public:

	inline static int shell_split = 10;

	cluster_basis() {}
	
	cluster_basis(vshell& basis, std::string method);
	
	cluster_basis(const cluster_basis& cbasis) : 
		m_basis(cbasis.m_basis),
		m_clusters(cbasis.m_clusters),
		m_cluster_sizes(cbasis.m_cluster_sizes),
		m_shell_offsets(cbasis.m_shell_offsets) {}
		
	cluster_basis& operator =(const cluster_basis& cbasis) {
		if (this != &cbasis) {
			m_basis = cbasis.m_basis;
			m_clusters = cbasis.m_clusters;
			m_cluster_sizes = cbasis.m_cluster_sizes;
			m_shell_offsets = cbasis.m_shell_offsets;
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
	
	int shell_offset(int i) const {
		return m_shell_offsets[i];
	}

};

}

#endif
