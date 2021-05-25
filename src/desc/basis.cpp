#include "desc/basis.hpp"
#include "utils/json.hpp"
#include <list>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#cmakedefine BASIS_ROOT "@BASIS_ROOT@"

namespace megalochem {

namespace desc {

const int MAX_L = 6;

const std::vector<std::string> ANGMOMS = {
	"s", "p", "d", "f", "g", "h", "i"
}; 
	
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
					bool is_s_or_p = ((l1 == 0) || (l1 == 1)) && 
						((l2 == 0) || (l2 == 1));
					dont_split = is_s_or_p || (l1 == l2);
					
				} else {
					dont_split = (l1 == l2);
					
				}
				
			}
			
			//std::cout << ((dont_split) ? "TRUE" : "FALSE") << std::endl;
			
			if (it->O == fshell.O 
				&& (int(nbf_cluster + nbf_shell) <= nsplit || nbf_cluster == 0)
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

vshell make_basis(std::string basname, std::vector<desc::Atom>& atoms_in) {
	
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
					
					for (size_t i = 0; i != coeffs_full.size(); ++i) {
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
			
				for (size_t i = 0; i != angmoms.size(); ++i) {
					
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
	
	//std::cout << "BASIS" << std::endl;
	//for (auto s : basis) {
	//	std::cout << s << std::endl;
	//}
	
	return basis;
		
}
	
double exp_radius(int l, double alpha, double threshold, double prefactor,
	double step, int maxiter) {
	
	// g(r) = prefactor * r^l * exp(-alpha*r**2) - threshold = 0
	
	double radius = step;
	double g = 0.0;
	
	for (int i = 0; i != maxiter; ++i) {
		g = fabs(prefactor) * exp(-alpha * radius * radius) * pow(radius,l);
		if (g < fabs(threshold)) break;
		radius = radius + step;
	}
		
	return radius;
	
}

double cluster_min_alpha(const vshell& shells) {
	double min = std::numeric_limits<double>::max();
	for (auto& s : shells) {
		for (auto& e : s.alpha) {
			min = std::min(min,e);
		}
	}
	return min;
}

cluster_basis::cluster_basis(std::string basname, std::vector<desc::Atom>& atoms_in,
	std::optional<std::string> method, 
	std::optional<int> nsplit, 
	std::optional<bool> augmented) {
	
	auto basis = make_basis(basname, atoms_in);
	
	std::optional<vshell> augbasis;
	
	if (augmented && *augmented) {
		std::optional<vshell> augbasis = std::make_optional<vshell>(
			make_basis("aug-" + basname, atoms_in));
	}
	
	*this = cluster_basis(basis, method, nsplit, augbasis);
	
}

vshell extract(vshell& basis, vshell& augbasis) {
	
	vshell aug_shells;
	
	for (auto& s : basis) {
		if (std::find(augbasis.begin(), augbasis.end(), s) == augbasis.end())
		{
			throw std::runtime_error("Basis set is not a subset of augmented basis set!");
		}
	}
	
	for (auto& s : augbasis) {
		auto iter = std::find(basis.begin(), basis.end(), s);
		if (iter == basis.end()) {
			//std::cout << s << std::endl;
			aug_shells.push_back(s);
		}
	}
	
	//std::cout << "AUGSHELLS: " << aug_shells.size() << std::endl;
	
	augbasis.resize(0);
	return aug_shells;	
	
}			

cluster_basis::cluster_basis(vshell basis, 
	std::optional<std::string> opt_method, 
	std::optional<int> opt_nsplit,
	std::optional<vshell> augbasis) : 
	m_nsplit(opt_nsplit ? *opt_nsplit : DEFAULT_NSPLIT), 
	m_split_method(opt_method ? *opt_method : DEFAULT_SPLIT_METHOD) {
		
	//std::cout << "NBAS: " << nbas << std::endl;
	//std::cout << "VSIZE: " << vsize << std::endl;
	
	auto get_cluster = [&](vshell t_basis) {
	
		if (m_split_method == "atomic") {
			
			return split_atomic(t_basis);
			
		} else if (m_split_method == "shell") {
			
			return split_shell(t_basis);
		
		} else if (m_split_method == "multi_shell") {
			
			return split_multi_shell(t_basis,m_nsplit,false,false);
			
		} else if (m_split_method == "multi_shell_strict") {
			
			return split_multi_shell(t_basis,m_nsplit,true,false);

		} else if (m_split_method == "multi_shell_strict_sp") {
			
			return split_multi_shell(t_basis,m_nsplit,true,true);
		
		} else {
			
			throw std::runtime_error("Unknown splitting method: " + m_split_method);
			
		}
		
	};
	
	auto clusters = get_cluster(basis);
	if (augbasis) {
		
		//std::cout << "PREPPING" << std::endl;
		auto aug_extract = extract(basis, *augbasis);
		
		auto aug_clusters = get_cluster(aug_extract);
		decltype(clusters) new_clusters;
		
		for (size_t i = 0; i < clusters.size(); ++i) {
			
			auto& this_shell = clusters[i].back();
			auto& next_shell = (i == clusters.size() - 1) ? 
				clusters[0].front() : clusters[i+1].front();
			
			new_clusters.push_back(clusters[i]);
			m_cluster_diff.push_back(false);
			
			auto this_pos = this_shell.O;
			auto next_pos = next_shell.O;
			
			if (this_pos == next_pos) continue;
			
			// search for all shells with same pos in augmented set
			for (auto& c : aug_clusters) {
				auto& s = c.front();
				if (s.O == this_pos) {
					new_clusters.push_back(c);
					m_cluster_diff.push_back(true);
				}
			}
			
		}
		
		m_clusters = new_clusters;
		
	} else {
		
		m_clusters = clusters;
		m_cluster_diff = std::vector<bool>(m_clusters.size(),false);
		
	}
					
	for (auto c : m_clusters) {
		m_cluster_sizes.push_back(desc::nbf(c));
	}
	
	int off = 0;
	m_shell_offsets.resize(m_clusters.size());
	
	for (size_t i = 0; i != m_shell_offsets.size(); ++i) {
		m_shell_offsets[i] = off;
		off += m_clusters[i].size();
	}
	
	// shelltypes
	for (auto& cluster : m_clusters) {
		std::vector<bool> stypes(MAX_L + 1);
		for (auto& shell : cluster) {
			stypes[shell.l] = true;
		}
		std::string id = "";
		for (int i = 0; i != MAX_L+1; ++i) {
			if (stypes[i]) id += ANGMOMS[i];
		}
		m_cluster_types.push_back(id);
	}
	
}

size_t cluster_basis::max_nprim() const {
	
	size_t n = 0;
	for (size_t i = 0; i != m_clusters.size(); ++i) {
		n = std::max(desc::max_nprim(m_clusters[i]), n);
	}
	return n;	
}

size_t cluster_basis::nbf() const {
	
	size_t nbas = 0;
	for (size_t i = 0; i != m_clusters.size(); ++i) {
		nbas += desc::nbf(m_clusters[i]);
	}
	return nbas;
}

size_t cluster_basis::max_l() const {
	
	size_t l = 0;
		for (size_t i = 0; i != m_clusters.size(); ++i) {
		l = std::max(desc::max_l(m_clusters[i]), l);
	}
	return l;	
}

std::vector<int> cluster_basis::cluster_sizes() const {
	return m_cluster_sizes;
}

std::vector<int> cluster_basis::block_to_atom(std::vector<desc::Atom> atoms) const {
	
	std::vector<int> blk_to_atom(m_cluster_sizes.size());
	
	auto get_centre = [&atoms](Shell& s) {
		for (size_t i = 0; i != atoms.size(); ++i) {
			auto& a = atoms[i];
			double d = sqrt(pow(s.O[0] - a.x, 2)
				+ pow(s.O[1] - a.y,2)
				+ pow(s.O[2] - a.z,2));
			if (d < 1e-12) return int(i);
		}
		return -1;
	};
	
	for (size_t iv = 0; iv != m_clusters.size(); ++iv) {
		auto s = m_clusters[iv][0];
		blk_to_atom[iv] = get_centre(s);
	}
	
	return blk_to_atom;
	
}

std::vector<double> cluster_basis::min_alpha() const {
	std::vector<double> out;
	for (auto& cluster : m_clusters) {
		out.push_back(cluster_min_alpha(cluster));
	}
	return out;
}
	
std::vector<double> cluster_basis::radii(double cutoff, double step, int maxiter) const
{
	// radii
	std::vector<double> cluster_radii;
	
	for (auto& cluster : m_clusters) {
		double max_radius = 0.0;
		for (auto& shell : cluster) {
			for (size_t i = 0; i != shell.nprim(); ++i) {
				max_radius = std::max(max_radius,
					exp_radius(shell.l, shell.alpha[i], 
					cutoff, shell.coeff[i], step, maxiter));
			}
		}
		cluster_radii.push_back(max_radius);
	}
	return cluster_radii;
	
}

std::vector<bool> cluster_basis::diffuse() const {
	return m_cluster_diff;
}
	
std::vector<std::string> cluster_basis::types() const {
	return m_cluster_types;
}

void cluster_basis::print_info() const {
	
	auto cluster_radii = radii();
	
	for (size_t icluster = 0; icluster != m_clusters.size(); ++icluster) {
		std::cout << "Cluster: " << icluster << '\n'
				<< "{" << '\n'
				<< '\t' << "size: " << m_cluster_sizes[icluster] << '\n'
				<< '\t' << "radius: " << cluster_radii[icluster] << '\n'
				<< '\t' << "diffuse: " << m_cluster_diff[icluster] << '\n'
				<< '\t' << "type: " << m_cluster_types[icluster] << '\n'
				<< "}" << std::endl;
	}
	
}

std::shared_ptr<cluster_basis> cluster_basis::fragment(std::vector<int> block_list) {
	
	auto basis_frag = std::make_shared<cluster_basis>();
	
	for (auto iblk : block_list) {
		basis_frag->m_clusters.push_back(this->m_clusters[iblk]);
		basis_frag->m_cluster_sizes.push_back(this->m_cluster_sizes[iblk]);
		basis_frag->m_cluster_diff.push_back(this->m_cluster_diff[iblk]);
		basis_frag->m_cluster_types.push_back(this->m_cluster_types[iblk]);
		basis_frag->m_shell_offsets.push_back(this->m_shell_offsets[iblk]);
	}
	
	basis_frag->m_nsplit = this->m_nsplit;
	basis_frag->m_split_method = this->m_split_method;

	return basis_frag;
	
}	

} // end namespace

} // end mega
