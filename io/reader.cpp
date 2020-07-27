#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <utility>
#include <cstdlib>

#include "io/io.h"
#include "io/reader.h"
#include "io/valid_keys.h"

#include "utils/ele_to_int.h"
#include "utils/constants.h"
#include "utils/json.hpp"

#include "desc/molecule.h"

#include "math/other/rcm.h"

#include <libint2/basis.h>
#include <dbcsr_common.hpp>
#include "ints/aofactory.h"

#include <sys/types.h>
#include <sys/stat.h>

namespace filio {

void validate(const json& j, const json& compare) {
	
	for (auto it = j.begin(); it != j.end(); ++it) {
		if (compare.find(it.key()) == compare.end()){
			throw std::runtime_error("Invalid keyword: "+it.key());
		}
		if (it->is_structured() && !it->is_array() && it.key() != "gen_basis") {
			validate(*it, compare[it.key()]);
		}
	}
}

std::vector<libint2::Atom> get_geometry(const json& j, std::string filename, util::mpi_log& LOG) {
	
	std::vector<libint2::Atom> out;
	
	if (j.find("file") == j.end()) {
		// reading from input file
		
		LOG.os<>("Reading XYZ info from file.\n");
		
		auto geometry = j["geometry"];
		auto symbols = j["symbols"];
		
		if ((geometry.size() % 3 != 0) || (geometry.size() / 3 != symbols.size())) {
			throw std::runtime_error("Missing coordinates.");
		}
		
		for (int i = 0; i != geometry.size()/3; ++i) {
			libint2::Atom a;
			a.x = geometry.at(3*i);
			a.y = geometry.at(3*i+1);
			a.z = geometry.at(3*i+2);
			a.atomic_number = util::ele_to_int[symbols.at(i)];
			
			out.push_back(a);
			
		}
		
	} else {
		// read from xyz file
		
		std::string xyzfilename = j["file"];
		
		LOG.os<>("Reading XYZ info from file ", xyzfilename, '\n');
		
		std::ifstream in;
		in.open(xyzfilename);
		
		if (!in) {
			throw std::runtime_error("XYZ data file not found.");
		}
		
		std::string line;
		
		std::string ele_name;
		int nline = 0;
		double x, y, z;
		
		while(std::getline(in,line)) {
			
			if (nline > 1) {
				std::stringstream ss(line);
				
				libint2::Atom atom;
				ss >> ele_name;
				ss >> atom.x >> atom.y >> atom.z;
				
				atom.atomic_number = util::ele_to_int[ele_name];
				
				out.push_back(atom);
			}
			
			++nline;
			
		}
		
	}
	
	double factor;
	
	if (j.find("unit") != j.end()) {
		if (j["unit"] == "angstrom") {
			factor = BOHR_RADIUS;
		} else {
			throw std::runtime_error("Unknown length unit.");
		}
	} else {
		factor = BOHR_RADIUS;
	}
	
	for (auto& a : out) {
		a.x /= factor;
		a.y /= factor;
		a.z /= factor;
		
		//std::cout << a.x << " " << a.y << " " << a.z << std::endl;
		
	}
	
	return out;
	
}

std::vector<libint2::Shell> read_basis(const json &jbas, std::vector<libint2::Atom> &atoms, util::mpi_log& LOG) {
	
	//libint2::Shell::do_enforce_unit_normalization(false);
	
	/*
	// check basis
	auto name = j["basis"];
	
	libint2::BasisSet basis(name, atoms);
	
	std::vector<libint2::Shell> out = std::move(basis);
	
	std::cout << "BASIS" << std::endl;
	
	
	// read input*/
	
	//read per atom basis
	//auto& jbas = j["gen_basis"];
	
	std::map<int,std::vector<libint2::Shell>> basis_map;
	
	// basis name?
	auto& jele = jbas["elements"];
	
	for (auto ele = jele.begin(); ele != jele.end(); ++ele) {
      
      auto strv_dv = [](std::vector<std::string>& str_v) {
		  libint2::svector<double> v(str_v.size());
		  for (int i = 0; i != v.size(); ++i) {
			  v[i] = std::stod(str_v[i]);
		  }
		  return v;
	  };
      
      auto& jshells = ele.value()["electron_shells"];
      
      std::vector<libint2::Shell> vecshell;
      
      for (auto& shell : jshells) {
		  
		  std::vector<int> ang_moms = shell["angular_momentum"];
	
		  std::vector<std::string> str_exp = shell["exponents"];
		  auto exp = strv_dv(str_exp);
		  
		  auto& jcoeffs = shell["coefficients"];
		  
		  libint2::svector<double> max_ln(jcoeffs.size());
		  libint2::svector<libint2::Shell::Contraction> veccons;
		  
		  for (int i = 0; i != jcoeffs.size(); ++i) {
			  
			std::vector<std::string> str_v = jcoeffs[i];
			auto v = strv_dv(str_v);
			
			auto max = std::max_element(v.begin(), v.end());
			max_ln[i] = log(*max);
			
			libint2::Shell::Contraction c;
			c.l = ang_moms[i];
			c.pure = true; 
			c.coeff = v;
			
			veccons.push_back(c);
			
		  }
		  
		  libint2::Shell s;
		  s.alpha = exp;
		  s.contr = veccons;
		  s.max_ln_coeff = max_ln;
		  
		  vecshell.push_back(std::move(s));
		  
	  }
		  
	  int ele_num = std::stoi(ele.key());
	  
	  basis_map[ele_num] = vecshell;
		  
	}
	
	std::vector<libint2::Shell> tot_basis;
	
	for (auto a : atoms) {
		auto& ele_basis = basis_map[a.atomic_number];
		std::array<double,3> new_o = {a.x, a.y, a.z};
		
		for (auto& s : ele_basis) s.move(new_o);
		
		tot_basis.insert(tot_basis.end(), ele_basis.begin(), ele_basis.end());
	}
	
	//for (auto b : tot_basis) {
		//std::cout << b << std::endl;
	//}
	
	return tot_basis;
	
}

void unpack(const json& j_in, desc::options& opt, std::string root, util::mpi_log& LOG) {
	
	auto j = j_in[root];
	
	for (auto it = j.begin(); it != j.end(); ++it) {
		
		if (it->type() == json::value_t::boolean) {
			opt.set<bool>(root + "/" + it.key(), *it);
		} else if (it->type() == json::value_t::number_integer ||
			it->type() == json::value_t::number_unsigned) {
			opt.set<int>(root + "/" + it.key(), *it);
		} else if (it->type() == json::value_t::number_float) {
			opt.set<double>(root + "/" + it.key(), *it);
		} else if (it->type() == json::value_t::string) {
			opt.set<std::string>(root + "/" + it.key(), *it);
		} else {
			throw std::runtime_error("Invalid type for keyword.");
		}
		
		//if (it->type() == json::value_t::array) {
		//	opt.set<std::vector<int>>(root + it.key(), *it);
		//}
		
	}
}	

reader::reader(MPI_Comm comm, std::string filename, int print) : m_comm(comm), LOG(comm, print) {
	
	std::ifstream in;
	in.open(filename + ".json");
	
	if (!in) {
		throw std::runtime_error("Input file not found.");
	}
	
	LOG.os<>("Reading input file...\n\n");
	
	json data;
	
	in >> data;
	
	validate(data, valid_keys);
	
	if (data.find("global") != data.end()) {
		json& jglob = data["global"];
		
		if (jglob.find("block_threshold") != jglob.end()) {
			dbcsr::global::filter_eps = jglob["block_threshold"];
		}
		
		if (jglob.find("integral_precision") != jglob.end()) {
			ints::global::precision = jglob["integral_precision"];
		}
		
		if (jglob.find("integral_omega") != jglob.end()) {
			ints::global::omega = jglob["integral_omega"];
		}
	}
	
	json& jmol = data["molecule"];
	
	optional<int,val> opt_mo_split;
	optional<std::string,val> opt_ao_split_method;
	
	if (jmol.find("mo_split") != jmol.end()) {
		opt_mo_split = jmol["mo_split"];
	}
	
	if (jmol.find("ao_split_method") != jmol.end()) {
		opt_ao_split_method = jmol["ao_split_method"];
	}
	
	LOG.os<>("Processing atomic coordinates...\n");
	auto atoms = get_geometry(jmol,filename,LOG);
	
	bool reorder = (jmol.find("reorder") != jmol.end()) 
		? jmol["reorder"].get<bool>() 
		: true;
	
	if (reorder) {
		LOG.os<>("Reordering atoms...\n");
		
		math::rcm<libint2::Atom> sorter(atoms,5.0,
			[](libint2::Atom a1, libint2::Atom a2) -> double {
				return sqrt(
					pow(a1.x - a2.x,2) +
					pow(a1.y - a2.y,2) +
					pow(a1.z - a2.z,2));
				});
				
		sorter.compute();
		
		LOG.os<>("Reordered Atom Indices:\n");
		for (auto i : sorter.reordered_idx()) {
			LOG.os<>(i, " ");
		} LOG.os<>("\n\n");
	
		sorter.reorder(atoms);
	
	}
	
	if (jmol.find("ao_split") != jmol.end()) {
		desc::cluster_basis::shell_split = jmol["ao_split"];
	}
	
	std::vector<libint2::Shell> basis;
	if (jmol.find("basis") != jmol.end()) {	
		libint2::BasisSet bas(jmol["basis"], atoms);
		basis = std::move(bas);
	} else { 
		basis = read_basis(jmol["gen_basis"],atoms,LOG);
	}
	
	optional<std::vector<libint2::Shell>,val> dfbasis;
	if (jmol.find("dfbasis") != jmol.end()) {
		auto b = libint2::BasisSet(jmol["dfbasis"], atoms);
		dfbasis = optional<std::vector<libint2::Shell>,val>(std::move(b));
	}
	
	//if (dfbasis) std::cout << "DFBASIS IS HERE." << std::endl;
	
	int charge = jmol["charge"];
	int mult = jmol["mult"];
	std::string name = jmol["name"];
	
	LOG.os<>("Molecule: \n");
	int w = 10;
	LOG.setprecision(6);
	LOG.left();
	LOG.setw(w).os<>("Atom Nr.").setw(w).os<>("X").setw(w)
		.os<>("Y").setw(w).os<>("Z").os<>('\n');
	LOG.right();
	LOG.os<>("----------------------------------------------------------\n");
	LOG.left();
	for (auto& a : atoms) {
		LOG.setw(w).os<>(a.atomic_number).setw(w).os<>(a.x).setw(w)
			.os<>(a.y).setw(w).os<>(a.z, '\n');
	}
	LOG.right();
	LOG.os<>("----------------------------------------------------------\n\n");
	
	LOG.reset();
	
	desc::molecule mol = desc::molecule::create().name(name).atoms(atoms).charge(charge)
		.mult(mult).mo_split(opt_mo_split).ao_split_method(opt_ao_split_method)
		.basis(basis);
		
	mol.print_info(m_comm,1);
	LOG.os<>('\n');
	
	desc::options opt;
	
	auto read_section = [&](std::string r)
	{
		if (data.find(r) != data.end()) {
			unpack(data, opt, r, LOG);
			opt.set<bool>("do_"+r, true);
		} else {
			opt.set<bool>("do_"+r, false);
		}
	};
	
	read_section("hf");
	read_section("mp");
	read_section("adc");
	
	//std::cout << opt.get<bool>("hf/diis") << std::endl;
	//std::cout << opt.get<double>("hf/conv") << std::endl;
	
	//if (opt.get<bool>("hf/use_df",false)) {
	//	std::cout << "Using DENSITY FITTING." << std::endl;
	//}
	
	m_mol = mol;
	m_opt = opt;
	
}

}
