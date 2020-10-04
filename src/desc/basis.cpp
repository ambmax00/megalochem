#include "desc/basis.h"
#include "utils/json.hpp"
#include <list>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#cmakedefine BASIS_ROOT "@BASIS_ROOT@"

namespace desc { 
	
vvshell split_atomic(vshell& basis) {
	
	vvshell c_out(0);
	vshell cluster(0);
	std::list<Shell> slist(basis.begin(),basis.end());
	
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
	std::list<Shell> slist(basis.begin(),basis.end());
	
	while (slist.size()) {
		
		auto fshell = *(slist.begin());
		cluster.push_back(fshell);
		
		slist.pop_front();
		auto it = slist.begin();
		
		while (it != slist.end()) {
			if (it->O == fshell.O && it->l == fshell.l) {
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
			
			auto nbf_cluster = nbf(cluster);
			auto nbf_shell = it->size();
			
			bool dont_split = true;
			
			if (strict) {
				
				auto l1 = it->l;
				auto l2 = fshell.l;
				
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
	
	// check if file exists
	std::string basis_root_dir(BASIS_ROOT);
	std::string filename_1 = basis_root_dir + "/" + basname + ".json";
	std::string filename_2 = basname + ".json";
	std::string filename; // the one we will take
	
	if (std::filesystem::exists(filename_2)) {
		filename = filename_2;
	} else if (std::filesystem::exists(filename_1)) {
		filename = filename_1;
	} else {
		throw std::runtime_error("Could not find basis " + basname 
			+ " in either basis root or work directory.");
	}
	
	nlohmann::json basis_data;
	std::ifstream file(filename);
	
	file >> basis_data;
	auto& elements = basis_data["elements"];
	
	vshell basis;
	
	auto convert = [](std::vector<std::string>& str_vec) {
		std::vector<double> out;
		for (auto s : str_vec) {
			out.push_back(std::stod(s));
		}
		return out;
	};
	
	for (auto& atom : atoms_in) {
		
		int Z = atom.atomic_number;
		auto& Z_basis = elements[std::to_string(Z)]["electron_shells"];
				
		for (auto& Z_shell : Z_basis) {
			
			std::array<double,3> pos = {atom.x, atom.y, atom.z};
			
			std::vector<std::string> alpha_str = Z_shell["exponents"];
			std::vector<double> alpha = convert(alpha_str);
			
			std::vector<int> angmoms = Z_shell["angular_momentum"];
			
			auto& coeff_arrays = Z_shell["coefficients"];
			
			if (angmoms.size() == 1) {
				int l = angmoms[0];
				for (auto& coeffs : coeff_arrays) {
					
					std::vector<std::string> coeff_str = coeffs;
					auto coeffs_full = convert(coeff_str);
					
					std::vector<double> shell_alpha;
					std::vector<double> shell_coeff;
					
					for (int i = 0; i != coeffs_full.size(); ++i) {
						if (fabs(coeffs_full[i]) > std::numeric_limits<double>::epsilon()) {
							shell_alpha.push_back(alpha[i]);
							shell_coeff.push_back(coeffs_full[i]);
						}
					}
					
					Shell s;
					s.pure = true;
					s.l = l;
					s.O = pos;
					s.alpha = shell_alpha;
					s.coeff = shell_coeff;
					
					//std::cout << s << std::endl;
					
					basis.push_back(s);
					
				}
			
			} else { 
			
				for (int i = 0; i != angmoms.size(); ++i) {
					
					std::vector<std::string> coeffs_str = coeff_arrays[i];						
					
					Shell s;
					s.pure = true;
					s.l = angmoms[i];
					s.O = pos;
					s.alpha = alpha;
					s.coeff = convert(coeffs_str);
					
					basis.push_back(s);
					
					//std::cout << s << std::endl;
					
				}
			}
		}
	}
	
	*this = cluster_basis(basis, method, nsplit);
	
}

cluster_basis::cluster_basis(vshell basis, std::string method, int nsplit)
	: m_nsplit(nsplit), m_split_method(method) {
	
	auto nbas = desc::nbf(basis);
	auto vsize = basis.size();
	
	//std::cout << "NBAS: " << nbas << std::endl;
	//std::cout << "VSIZE: " << vsize << std::endl;
	
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
		m_cluster_sizes.push_back(desc::nbf(c));
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
		n = std::max(desc::max_nprim(m_clusters[i]), n);
	}
	return n;	
}

size_t cluster_basis::nbf() const {
	
	size_t nbas = 0;
	for (int i = 0; i != m_clusters.size(); ++i) {
		nbas += desc::nbf(m_clusters[i]);
	}
	return nbas;
}

size_t cluster_basis::max_l() const {
	
	size_t l = 0;
		for (int i = 0; i != m_clusters.size(); ++i) {
		l = std::max(desc::max_l(m_clusters[i]), l);
	}
	return l;	
}

std::vector<int> cluster_basis::cluster_sizes() const {
	return m_cluster_sizes;
}

std::vector<int> cluster_basis::block_to_atom(std::vector<desc::Atom>& atoms) const {
	
	std::vector<int> blk_to_atom(m_cluster_sizes.size());
	
	auto get_centre = [&atoms](Shell& s) {
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
