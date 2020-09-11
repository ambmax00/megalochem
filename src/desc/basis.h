#ifndef DESC_BASIS_H
#define DESC_BASIS_H

#include "desc/atom.h"

#include <libint2/shell.h>
#include <libint2/atom.h>

#include <memory>

using vshell = std::vector<libint2::Shell>;
using vvshell = std::vector<vshell>;

namespace desc {
	
class cluster_basis {
private:

	vshell m_basis;
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
		m_basis(cbasis.m_basis),
		m_clusters(cbasis.m_clusters),
		m_cluster_sizes(cbasis.m_cluster_sizes),
		m_shell_offsets(cbasis.m_shell_offsets),
		m_nsplit(cbasis.m_nsplit),
		m_split_method(cbasis.m_split_method) {}
		
	cluster_basis& operator =(const cluster_basis& cbasis) {
		if (this != &cbasis) {
			m_basis = cbasis.m_basis;
			m_clusters = cbasis.m_clusters;
			m_cluster_sizes = cbasis.m_cluster_sizes;
			m_shell_offsets = cbasis.m_shell_offsets;
			m_nsplit = cbasis.m_nsplit;
			m_split_method = cbasis.m_split_method;
		}
		
		return *this;
	}
	
	vshell libint_basis() const {
		return m_basis;
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

	int max_l() const;
	
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
