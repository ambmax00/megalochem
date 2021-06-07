#include "desc/basis.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include "utils/json.hpp"
#include "utils/ele_to_int.hpp"

#cmakedefine BASIS_ROOT "@BASIS_ROOT@"

namespace megalochem {

namespace desc {

const int MAX_L = 6;

const std::vector<std::string> ANGMOMS = {"s", "p", "d", "f", "g", "h", "i"};

double exp_radius(
    int l,
    double alpha,
    double threshold,
    double prefactor,
    double step,
    int maxiter)
{
  // g(r) = prefactor * r^l * exp(-alpha*r**2) - threshold = 0

  double radius = step;
  double g = 0.0;

  for (int i = 0; i != maxiter; ++i) {
    g = fabs(prefactor) * exp(-alpha * radius * radius) * pow(radius, l);
    if (g < fabs(threshold))
      break;
    radius = radius + step;
  }

  return radius;
}

double cluster_min_alpha(const vshell& shells)
{
  double min = std::numeric_limits<double>::max();
  for (auto& s : shells) {
    for (auto& e : s.alpha) { min = std::min(min, e); }
  }
  return min;
}

std::vector<cluster> cluster_atomic(vshell& basis)
{
  std::vector<cluster> c_out;
  cluster cltr;
  std::list<Shell> slist(basis.begin(), basis.end());

  while (slist.size()) {
    auto fshell = *(slist.begin());
    cltr.shells.push_back(fshell);

    slist.pop_front();
    auto it = slist.begin();

    while (it != slist.end()) {
      if (it->O == fshell.O) {
        cltr.shells.push_back(*it);
        it = slist.erase(it);
      }
      else {
        ++it;
      }
    }

    cltr.O = cltr.shells[0].O;

    c_out.push_back(cltr);
    cltr.shells.clear();
  }

  return c_out;
}

std::vector<cluster> cluster_shell(vshell& basis)
{
  std::vector<cluster> c_out;
  cluster cltr;
  std::list<Shell> slist(basis.begin(), basis.end());

  while (slist.size()) {
    auto fshell = *(slist.begin());
    cltr.shells.push_back(fshell);

    slist.pop_front();
    auto it = slist.begin();

    while (it != slist.end()) {
      if (it->O == fshell.O && it->l == fshell.l) {
        cltr.shells.push_back(*it);
        it = slist.erase(it);
      }
      else {
        ++it;
      }
    }

    cltr.O = cltr.shells[0].O;

    c_out.push_back(cltr);
    cltr.shells.clear();
  }

  return c_out;
}

std::vector<cluster> cluster_multi_shell(
    vshell& basis, int nsplit, bool strict, bool sp)
{
  std::vector<cluster> c_out;
  cluster cltr;
  std::list slist(basis.begin(), basis.end());

  while (slist.size()) {
    auto fshell = *(slist.begin());
    cltr.shells.push_back(fshell);

    slist.pop_front();
    auto it = slist.begin();

    while (it != slist.end()) {
      auto nbf_cluster = nbf(cltr.shells);
      auto nbf_shell = it->size();

      bool dont_split = true;

      if (strict) {
        auto l1 = it->l;
        auto l2 = fshell.l;

        // std::cout << "strict: " << l1 << " " << l2 << std::endl;

        if (sp) {
          bool is_s_or_p = ((l1 == 0) || (l1 == 1)) && ((l2 == 0) || (l2 == 1));
          dont_split = is_s_or_p || (l1 == l2);
        }
        else {
          dont_split = (l1 == l2);
        }
      }

      // std::cout << ((dont_split) ? "TRUE" : "FALSE") << std::endl;

      if (it->O == fshell.O &&
          (int(nbf_cluster + nbf_shell) <= nsplit || nbf_cluster == 0) &&
          (dont_split)) {
        cltr.shells.push_back(*it);
        it = slist.erase(it);
      }
      else {
        ++it;
      }
    }

    cltr.O = cltr.shells[0].O;

    c_out.push_back(cltr);
    cltr.shells.clear();
  }

  return c_out;
}

std::map<std::string,vshell> read_atom_basis(
   std::string basname,
   std::vector<std::string> symbols)
{
   // check if file exists
  std::string basis_root_dir(BASIS_ROOT);
  std::string filename_1 = basis_root_dir + "/" + basname + ".json";
  std::string filename_2 = basname + ".json";
  std::string filename;  // the one we will take

  if (std::filesystem::exists(filename_2)) {
    filename = filename_2;
  }
  else if (std::filesystem::exists(filename_1)) {
    filename = filename_1;
  }
  else {
    throw std::runtime_error(
        "Could not find basis " + basname +
        " in either basis root or work directory.");
  }

  nlohmann::json basis_data;
  std::ifstream file(filename);

  file >> basis_data;
  auto& elements = basis_data["elements"];

  std::map<std::string, vshell> symbol_basis;

  auto convert = [](std::vector<std::string>& str_vec) {
    std::vector<double> out;
    for (auto s : str_vec) { out.push_back(std::stod(s)); }
    return out;
  };
  
  for (auto& s : symbols) {
	  	  
	vshell atom_basis;  
	 
	int Z = util::ele_to_int[s];
    auto& Z_basis = elements[std::to_string(Z)]["electron_shells"];

    for (auto& Z_shell : Z_basis) {
		
      std::array<double, 3> pos = {0.0, 0.0, 0.0};

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

          // std::cout << s << std::endl;

          atom_basis.push_back(s);
        }
      }
      else {
        for (size_t i = 0; i != angmoms.size(); ++i) {
          std::vector<std::string> coeffs_str = coeff_arrays[i];

          Shell s;
          s.pure = true;
          s.l = angmoms[i];
          s.O = pos;
          s.alpha = alpha;
          s.coeff = convert(coeffs_str);

          atom_basis.push_back(s);
        }
      }
    }
    
    //for (auto s : atom_basis) {
	//	std::cout << s << std::endl;
	//}
    
    symbol_basis[s] = atom_basis;
  }

  return symbol_basis;
}
  

vshell read_basis(std::string basname, std::vector<desc::Atom>& atoms_in)
{
  // check if file exists
  std::string basis_root_dir(BASIS_ROOT);
  std::string filename_1 = basis_root_dir + "/" + basname + ".json";
  std::string filename_2 = basname + ".json";
  std::string filename;  // the one we will take

  if (std::filesystem::exists(filename_2)) {
    filename = filename_2;
  }
  else if (std::filesystem::exists(filename_1)) {
    filename = filename_1;
  }
  else {
    throw std::runtime_error(
        "Could not find basis " + basname +
        " in either basis root or work directory.");
  }

  nlohmann::json basis_data;
  std::ifstream file(filename);

  file >> basis_data;
  auto& elements = basis_data["elements"];

  vshell basis;

  auto convert = [](std::vector<std::string>& str_vec) {
    std::vector<double> out;
    for (auto s : str_vec) { out.push_back(std::stod(s)); }
    return out;
  };

  for (auto& atom : atoms_in) {
    int Z = atom.atomic_number;
    auto& Z_basis = elements[std::to_string(Z)]["electron_shells"];

    for (auto& Z_shell : Z_basis) {
      std::array<double, 3> pos = {atom.x, atom.y, atom.z};

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

          // std::cout << s << std::endl;

          basis.push_back(s);
        }
      }
      else {
        for (size_t i = 0; i != angmoms.size(); ++i) {
          std::vector<std::string> coeffs_str = coeff_arrays[i];

          Shell s;
          s.pure = true;
          s.l = angmoms[i];
          s.O = pos;
          s.alpha = alpha;
          s.coeff = convert(coeffs_str);

          basis.push_back(s);
        }
      }
    }
  }

  return basis;
}

std::vector<std::string> get_unique(std::vector<desc::Atom>& atoms) {
	std::vector<std::string> out;
	for (auto a : atoms) {
		auto a_name = util::int_to_ele[a.atomic_number];
		auto it = std::find(out.begin(), out.end(), a_name);
		if (it == out.end()) {
			out.push_back(a_name);
		}
	}
	return out;
}

vshell extract_augbasis(vshell& basis, vshell& augbasis)
{
  vshell aug_shells;

  for (auto& s : basis) {
    if (std::find(augbasis.begin(), augbasis.end(), s) == augbasis.end()) {
      throw std::runtime_error(
          "Basis set is not a subset of augmented basis set!");
    }
  }

  for (auto& s : augbasis) {
    auto iter = std::find(basis.begin(), basis.end(), s);
    if (iter == basis.end()) {
      // std::cout << s << std::endl;
      aug_shells.push_back(s);
    }
  }

  augbasis.resize(0);
  return aug_shells;
}

std::vector<cluster> get_cluster(
   vshell t_basis,
   std::optional<std::string> opt_method = std::nullopt,
   std::optional<int> opt_nsplit = std::nullopt) 
{
    std::vector<cluster> out;

    auto method = (opt_method) ? *opt_method : DEFAULT_SPLIT_METHOD;
    auto nsplit = (opt_nsplit) ? *opt_nsplit : DEFAULT_NSPLIT;

    if (method == "atomic") {
      out = cluster_atomic(t_basis);
    }
    else if (method == "shell") {
      out = cluster_shell(t_basis);
    }
    else if (method == "multi_shell") {
      out = cluster_multi_shell(t_basis, nsplit, false, false);
    }
    else if (method == "multi_shell_strict") {
      out = cluster_multi_shell(t_basis, nsplit, true, false);
    }
    else if (method == "multi_shell_strict_sp") {
      out = cluster_multi_shell(t_basis, nsplit, true, true);
    }
    else {
      throw std::runtime_error("Unknown splitting method: " + method);
    }
    
    return out;
}

cluster_basis::cluster_basis(
      std::vector<desc::Atom>& atoms,
      std::vector<std::string> symbols,
      std::vector<std::string> basis_names,
      std::optional<std::vector<bool>> augmentations,
      std::optional<std::string> method,
      std::optional<int> nsplit)
{
	
	// first, check if vectors have appropriate sizes
	bool do_throw = false;
	if (!(augmentations) && !(symbols.size() == basis_names.size())) {
		//std::cout <<  symbols.size() << " " << basis_names.size() << std::endl;
		do_throw = true;
	} else if ((augmentations) && (!(augmentations->size() == basis_names.size()) || 
		!(symbols.size() == basis_names.size())) 
	) {
		//std::cout << augmentations->size() << " " << symbols.size() << " " << basis_names.size() << std::endl;
		do_throw = true;
	}
	
	if (do_throw) {
		throw std::runtime_error("Basis set constructor: wrong vector sizes");
	}
	
	// then check wether all atoms are in symbols
	auto unique_names = get_unique(atoms);
	
	for (auto uname : unique_names) {
		auto it = std::find(symbols.begin(), symbols.end(), uname);
		if (it == symbols.end()) {
			throw std::runtime_error("Could not find " + uname + 
				" in basis set symbols");
		}
	}
	
	// get unique basis sets
	std::map<std::string,std::vector<std::string>> unique_basis;
	std::map<std::string,std::vector<std::string>> unique_aug_basis;
	// {basis_set_name, {vector of symbols which use that basis set}}
	
	auto add_basis = [&](auto elename, auto& basname, auto& basis) {
		if (basis.find(basname) == basis.end()) {
			basis[basname] = {};
		}
		basis[basname].push_back(elename);
	};
	
	for (int iele = 0; iele != (int)symbols.size(); ++iele) {
		
		auto iname = basis_names[iele];
		add_basis(symbols[iele], iname, unique_basis);
		
		if (augmentations->at(iele)) {
			add_basis(symbols[iele], iname, unique_aug_basis);
		}
				
	}
	
	/*for (auto& [basis_name,basis_symbols] : unique_basis) {
		std::cout << basis_name << std::endl;
		for (auto e : basis_symbols) {
			std::cout << e << " ";
		} std::cout << std::endl;
	}*/
	
	// now read each individual basis set
	std::map<std::string,vshell> unique_shells;
	std::map<std::string,vshell> unique_aug_shells;
	
	auto add_shells = [&](auto& ubasis, auto& ushells, bool diffuse) {
		for (auto& [basis_name, basis_symbols] : ubasis) {
		   std::string suffix = (diffuse) ? "aug-" : "";
		   auto vec_shells = read_atom_basis(suffix + basis_name, basis_symbols);
		   for (auto& [ele_name, shells] : vec_shells) {
			   ushells[ele_name] = shells;
		   }
		}
	};
		
	add_shells(unique_basis, unique_shells, false);
	if (augmentations) add_shells(unique_aug_basis, unique_aug_shells, true);
	
	/*std::cout << "BASIS" << std::endl;
	for (auto& [elename, shells] : unique_shells) {
		std::cout << elename << " : " << std::endl;
		for (auto& c : shells) {
			std::cout << c << std::endl;
		}
		std::cout << std::endl;
	}
	
	std::cout << "AUGBASIS" << std::endl;
	for (auto& [elename, shells] : unique_aug_shells) {
		std::cout << elename << " : " << std::endl;
		for (auto& c : shells) {
			std::cout << c << std::endl;
		}
		std::cout << std::endl;
	}*/
	
	// form clusters
	std::map<std::string,std::vector<cluster>> unique_clusters;
	
	for (auto& [ele_name, shells] : unique_shells) {
		auto cltr = get_cluster(shells, method, nsplit);
		for (auto& c : cltr) {
			c.diffuse = false;
		}
		
		if (unique_aug_shells.find(ele_name) != unique_aug_shells.end()) {
			auto diff_shells = extract_augbasis(shells, unique_aug_shells[ele_name]);
			auto aug_cltr = get_cluster(diff_shells, method, nsplit);
			for (auto& c : aug_cltr) {
				c.diffuse = true;
			}
			cltr.insert(cltr.end(), aug_cltr.begin(), aug_cltr.end());
		}
		
		unique_clusters[ele_name] = cltr;
	}
	
	/*std::cout << "CLUSTERS" << std::endl;
	for (auto [elename, clstrs] : unique_clusters) {
		std::cout << elename << " : " << std::endl;
		for (auto c : clstrs) {
			std::cout << "c" << std::endl;
			for (auto s : c.shells) {
				std::cout << s << '\n';
			}
		}
		std::cout << std::endl;
	}*/
	
	// assign cluster vector to each atom
	
	for (auto a : atoms) {
		std::array<double,3> pos = {a.x, a.y, a.z};
		auto str = util::int_to_ele[a.atomic_number];
		auto cltr = unique_clusters[str];
		for (auto& c : cltr) {
			c.O = pos;
			for (auto& s : c.shells) {
				s.O = pos;
			}
		}
		m_clusters.insert(m_clusters.end(), cltr.begin(), cltr.end());
	}
	
	/*std::cout << "END" << std::endl;
	for (auto& cltr : m_clusters) {
		std::cout << "=== cluster ===" << std::endl;
		for (auto& s : cltr.shells) {
			std::cout << s << std::endl;
		}
	}*/
		
}


cluster_basis::cluster_basis(
    std::string basname,
    std::vector<desc::Atom>& atoms_in,
    std::optional<std::string> method,
    std::optional<int> nsplit,
    std::optional<bool> augmented)
{
  auto basis = read_basis(basname, atoms_in);

  std::optional<vshell> augbasis = std::nullopt;

  if (augmented && *augmented) {
    augbasis =
        std::make_optional<vshell>(read_basis("aug-" + basname, atoms_in));
  }

  *this = cluster_basis(basis, method, nsplit, augbasis);
}

cluster_basis::cluster_basis(
    vshell basis,
    std::optional<std::string> opt_method,
    std::optional<int> opt_nsplit,
    std::optional<vshell> augbasis)
{
  auto get_cluster = [&](vshell t_basis) {
    std::vector<cluster> out;

    auto method = (opt_method) ? *opt_method : DEFAULT_SPLIT_METHOD;
    auto nsplit = (opt_nsplit) ? *opt_nsplit : DEFAULT_NSPLIT;

    if (method == "atomic") {
      out = cluster_atomic(t_basis);
    }
    else if (method == "shell") {
      out = cluster_shell(t_basis);
    }
    else if (method == "multi_shell") {
      out = cluster_multi_shell(t_basis, nsplit, false, false);
    }
    else if (method == "multi_shell_strict") {
      out = cluster_multi_shell(t_basis, nsplit, true, false);
    }
    else if (method == "multi_shell_strict_sp") {
      out = cluster_multi_shell(t_basis, nsplit, true, true);
    }
    else {
      throw std::runtime_error("Unknown splitting method: " + method);
    }
    return out;
  };

  auto clusters = get_cluster(basis);

  if (augbasis) {
    auto aug_extract = extract_augbasis(basis, *augbasis);

    auto aug_clusters = get_cluster(aug_extract);
    std::vector<cluster> new_clusters;

    for (size_t i = 0; i < clusters.size(); ++i) {
      auto& this_cluster = clusters[i];
      auto& next_cluster =
          (i == clusters.size() - 1) ? clusters[0] : clusters[i + 1];

      new_clusters.push_back(clusters[i]);
      new_clusters.back().diffuse = false;

      auto this_pos = this_cluster.O;
      auto next_pos = next_cluster.O;

      if (this_pos == next_pos)
        continue;

      // search for all shells with same pos in augmented set
      for (auto& c : aug_clusters) {
        if (c.O == this_pos) {
          new_clusters.push_back(c);
          new_clusters.back().diffuse = true;
        }
      }
    }

    m_clusters = new_clusters;
  }
  else {
	for (auto& c : clusters) {
	  c.diffuse = false;
	}
    m_clusters = clusters;
  }
}

int cluster_basis::max_nprim() const
{
  int n = 0;
  for (auto& c : m_clusters) {
    n = std::max((int)desc::max_nprim(c.shells), n);
  }
  return n;
}

int cluster_basis::nbf() const
{
  int nbas = 0;
  for (auto& c : m_clusters) { nbas += (int)desc::nbf(c.shells); }
  return nbas;
}

int cluster_basis::max_l() const
{
  int l = 0;
  for (auto& c : m_clusters) { l = std::max((int)desc::max_l(c.shells), l); }
  return l;
}

std::vector<int> cluster_basis::cluster_sizes() const
{
  std::vector<int> out;
  for (auto& c : m_clusters) { out.push_back(desc::nbf(c.shells)); }
  return out;
}

std::vector<int> cluster_basis::block_to_atom(
    std::vector<desc::Atom> atoms) const
{
  std::vector<int> blk_to_atom(m_clusters.size());

  auto get_centre = [&atoms](const cluster& c) {
    for (size_t i = 0; i != atoms.size(); ++i) {
      auto& a = atoms[i];
      double d = sqrt(
          pow(c.O[0] - a.x, 2) + pow(c.O[1] - a.y, 2) + pow(c.O[2] - a.z, 2));
      if (d < 1e-12)
        return int(i);
    }
    return -1;
  };

  for (size_t ii = 0; ii != m_clusters.size(); ++ii) {
    blk_to_atom[ii] = get_centre(m_clusters[ii]);
  }

  return blk_to_atom;
}

std::vector<double> cluster_basis::min_alpha() const
{
  std::vector<double> out;
  for (auto& cltr : m_clusters) {
    out.push_back(cluster_min_alpha(cltr.shells));
  }
  return out;
}

std::vector<double> cluster_basis::radii(
    double cutoff, double step, int maxiter) const
{
  // radii
  std::vector<double> cluster_radii;

  for (auto& cltr : m_clusters) {
    double max_radius = 0.0;
    for (auto& shell : cltr.shells) {
      for (size_t i = 0; i != shell.nprim(); ++i) {
        max_radius = std::max(
            max_radius,
            exp_radius(
                shell.l, shell.alpha[i], cutoff, shell.coeff[i], step,
                maxiter));
      }
    }
    cluster_radii.push_back(max_radius);
  }
  return cluster_radii;
}

std::vector<bool> cluster_basis::diffuse() const
{
  std::vector<bool> out;
  for (auto& c : m_clusters) { out.push_back(c.diffuse); }
  return out;
}

std::vector<int> cluster_basis::shell_offsets() const
{
  std::vector<int> out;
  int off = 0;

  for (auto& c : m_clusters) {
    out.push_back(off);
    off += (int)c.shells.size();
  }

  return out;
}

std::vector<int> cluster_basis::nshells() const
{
  std::vector<int> out;
  for (auto& c : m_clusters) { out.push_back(c.shells.size()); }
  return out;
}

int cluster_basis::nshells_tot() const
{
  int n = 0;
  for (auto& c : m_clusters) { n += c.shells.size(); }
  return n;
}

std::vector<std::string> cluster_basis::shell_types() const
{
  std::vector<std::string> ctypes;
  for (auto& cltr : m_clusters) {
    std::vector<bool> stypes(MAX_L + 1);
    for (auto& shell : cltr.shells) { stypes[shell.l] = true; }
    std::string id = "";
    for (int i = 0; i != MAX_L + 1; ++i) {
      if (stypes[i])
        id += ANGMOMS[i];
    }
    ctypes.push_back(id);
  }
  return ctypes;
}

}  // namespace desc

}  // namespace megalochem
