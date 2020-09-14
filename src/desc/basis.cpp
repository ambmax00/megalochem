#include "desc/basis.h"
#include <libint2/basis.h>
#include <list>

namespace desc { 
	
vvshell split_atomic(vshell& basis) {
	
	vvshell c_out(0);
	vshell cluster(0);
	std::list<libint2::Shell> slist(basis.begin(),basis.end());
	
	while (slist.size()) {
		
		auto fshell = *(slist.begin());
		cluster.push_back(fshell);
		
		slist.pop_front();
		auto it = slist.begin();
		
		while (it != slist.end()) {
			if (it->O == fshell.O) {
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
		
		auto fshell = *(slist.begin());
		cluster.push_back(fshell);
		
		slist.pop_front();
		auto it = slist.begin();
		
		while (it != slist.end()) {
			if (it->O == fshell.O && it->contr[0].l == fshell.contr[0].l) {
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
		
		auto fshell = *(slist.begin());
		cluster.push_back(fshell);
		
		slist.pop_front();
		auto it = slist.begin();
		
		while (it != slist.end()) {
			
			int nbf_cluster = libint2::nbf(cluster);
			int nbf_shell = it->size();
			
			bool dont_split = true;
			
			if (strict) {
				
				int l1 = it->contr[0].l;
				int l2 = fshell.contr[0].l;
				
				//std::cout << "strict: " << l1 << " " << l2 << std::endl;
				
				if (sp) {
					dont_split = (((l1 == 0) || (l1 == 1)) && ((l2 == 0) || (l2 == 1))
						|| (l1 == l2));
				} else {
					dont_split = (l1 == l2);
				}
				
			}
			
			//std::cout << ((dont_split) ? "TRUE" : "FALSE") << std::endl;
			
			if (it->O == fshell.O 
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
	
	

cluster_basis::cluster_basis(std::string basname, std::vector<desc::Atom>& atoms_in,
	std::string method, int nsplit) {
	
	std::vector<libint2::Atom> atoms;
	
	for (auto a_in : atoms_in) {
		libint2::Atom a_out;
		a_out.x = a_in.x;
		a_out.y = a_in.y;
		a_out.z = a_in.z;
		a_out.atomic_number = a_in.atomic_number;
		atoms.push_back(a_out);
	}
	
	libint2::BasisSet full_basis(basname, atoms);
	vshell basis = std::move(full_basis);
	
	cluster_basis c(basis, method, nsplit);
	*this = c;
	
}

cluster_basis::cluster_basis(vshell basis, std::string method, int nsplit)
	: m_nsplit(nsplit), m_split_method(method) {
	
	int nbas = libint2::nbf(basis);
	int vsize = basis.size();
	m_basis = basis;
	
	if (method == "atomic") {
		
		m_clusters = split_atomic(basis);
		
	} else if (method == "shell") {
		
		m_clusters = split_shell(basis);
	
	} else if (method == "multi_shell") {
		
		m_clusters = split_multi_shell(basis,m_nsplit,false,false);
		
	} else if (method == "multi_shell_strict") {
		
		m_clusters = split_multi_shell(basis,m_nsplit,true,false);

	} else if (method == "multi_shell_strict_sp") {
		
		m_clusters = split_multi_shell(basis,m_nsplit,true,true);
	
	} else {
		
		throw std::runtime_error("Unknown splitting method: " + method);
		
	}

	
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

std::vector<int> cluster_basis::block_to_atom(std::vector<desc::Atom>& atoms) const {
	
	std::vector<int> blk_to_atom(m_cluster_sizes.size());
	
	auto get_centre = [&atoms](libint2::Shell& s) {
		for (int i = 0; i != atoms.size(); ++i) {
			auto& a = atoms[i];
			double d = sqrt(pow(s.O[0] - a.x, 2)
				+ pow(s.O[1] - a.y,2)
				+ pow(s.O[2] - a.z,2));
			if (d < 1e-12) return i;
		}
		return -1;
	};
	
	for (int iv = 0; iv != m_clusters.size(); ++iv) {
		auto s = m_clusters[iv][0];
		blk_to_atom[iv] = get_centre(s);
	}
	
	return blk_to_atom;
	
}

} // end namespace
