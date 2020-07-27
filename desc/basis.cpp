#include "desc/basis.h"
#include <libint2/basis.h>
#include <list>

namespace desc {
	
static 
	
vvshell split_atomic(vshell& basis) {
	
	vvshell c_out(0);
	vshell cluster(0);
	std::list<libint2::Shell> slist(basis.begin(),basis.end());
	
	while (slist.size()) {
		
		auto fshell = slist.begin();
		cluster.push_back(*fshell);
		
		slist.pop_front();
		auto it = slist.begin();
		
		while (it != slist.end()) {
			if (it->O == fshell->O) {
				cluster.push_back(*it);
				it = slist.erase(it);
			} else {
				++it;
			}
		}
				
		c_out.push_back(cluster);
		cluster.clear();
		
	}
	
	return c_out;
	
}

vvshell split_shell(vshell& basis) {
	
	vvshell c_out(0);
	vshell cluster(0);
	std::list<libint2::Shell> slist(basis.begin(),basis.end());
	
	while (slist.size()) {
		
		auto fshell = slist.begin();
		cluster.push_back(*fshell);
		
		slist.pop_front();
		auto it = slist.begin();
		
		while (it != slist.end()) {
			if (it->O == fshell->O && it->contr[0].l == fshell->contr[0].l) {
				cluster.push_back(*it);
				it = slist.erase(it);
			} else {
				++it;
			}
		}
				
		c_out.push_back(cluster);
		cluster.clear();
		
	}
	
	return c_out;
	
}	

vvshell split_multi_shell(vshell& basis, int nsplit, bool strict, bool sp) {
	
	vvshell c_out(0);
	vshell cluster(0);
	std::list slist(basis.begin(),basis.end());
	
	while (slist.size()) {
		
		auto fshell = slist.begin();
		cluster.push_back(*fshell);
		
		slist.pop_front();
		auto it = slist.begin();
		
		while (it != slist.end()) {
			
			int nbf_cluster = libint2::nbf(cluster);
			int nbf_shell = it->size();
			
			bool dont_split = true;
			
			if (strict) {
				
				int l1 = it->contr[0].l;
				int l2 = fshell->contr[0].l;
				
				//std::cout << "strict: " << l1 << " " << l2 << std::endl;
				
				if (sp) {
					dont_split = (((l1 == 0) || (l1 == 1)) && ((l2 == 0) || (l2 == 1))
						|| (l1 == l2));
				} else {
					dont_split = (l1 == l2);
				}
				
			}
			
			//std::cout << ((dont_split) ? "TRUE" : "FALSE") << std::endl;
			
			if (it->O == fshell->O 
				&& (nbf_cluster + nbf_shell <= nsplit || nbf_cluster == 0)
				&& (dont_split))
			{
				cluster.push_back(*it);
				it = slist.erase(it);
			} else {
				++it;
			}
			
		}
		
		c_out.push_back(cluster);
		cluster.clear();
		
	}
	
	return c_out;
	
} 
	
	

cluster_basis::cluster_basis(vshell& basis, std::string method) : m_basis(basis) {
	
	int nbas = libint2::nbf(basis);
	int vsize = basis.size();
	
	//std::cout << "NBAS: " << nbas << std::endl;
	//std::cout << "VSIZE: " << vsize << std::endl;
	
	if (method == "atomic") {
		
		m_clusters = split_atomic(basis);
		
	} else if (method == "shell") {
		
		m_clusters = split_shell(basis);
	
	} else if (method == "multi_shell") {
		
		m_clusters = split_multi_shell(basis,shell_split,false,false);
		
	} else if (method == "multi_shell_strict") {
		
		m_clusters = split_multi_shell(basis,shell_split,true,false);

	} else if (method == "multi_shell_strict_sp") {
		
		m_clusters = split_multi_shell(basis,shell_split,true,true);
	
	} else {
		
		throw std::runtime_error("Unknown splitting method.\n");
		
	}
	
	/*for (auto c : m_clusters) {
		std::cout << c.size() << " ";
	} std::cout << std::endl;
	
	for (auto c : m_clusters) {
		std::cout << libint2::nbf(c) << " ";
	} std::cout << std::endl;
	
	exit(0);*/
	
	for (auto c : m_clusters) {
		m_cluster_sizes.push_back(libint2::nbf(c));
	}
	
	int off = 0;
	m_shell_offsets.resize(m_clusters.size());
	
	for (int i = 0; i != m_shell_offsets.size(); ++i) {
		m_shell_offsets[i] = off;
		off += m_clusters[i].size();
	}
	
	/*std::cout << "OFFS: " << std::endl;
	for (auto s : m_shell_offsets) {
		std::cout << s << " ";
	} std::cout << std::endl;*/
	
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
