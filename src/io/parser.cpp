#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <utility>
#include <cstdlib>

#include "io/io.hpp"
#include "io/parser.hpp"
#include "io/valid_keys.hpp"

#include "utils/ele_to_int.hpp"
#include "utils/constants.hpp"

#include "desc/molecule.hpp"

#include "math/other/rcm.hpp"

#include <dbcsr_common.hpp>
#include "ints/aofactory.hpp"

#include <sys/types.h>
#include <sys/stat.h>

//#include <Python.h>

namespace filio {

void validate(std::string section, const json& j, const json& compare) {
			
	// check if all required keys are present
	std::vector<std::string> reqkeys = compare["_required"];
	
	for (auto key : reqkeys) {
		if (key == "none") break;
		if (j.find(key) == j.end()) {
			throw std::runtime_error("Key " + key + " is required.");
		}
	}
	
	// loop through objects in this section
	for (auto it = j.begin(); it != j.end(); ++it) {
		
		// check if keys valid
		if (compare.find(it.key()) == compare.end()){
			throw std::runtime_error("Section " + section + ", invalid keyword: "+it.key());
		}
		
		// go to nested section, if present
		if (it->is_structured() && !it->is_array() && it.key() != "gen_basis") {
			validate(it.key(), *it, compare[it.key()]);
		}
	}
	
}

std::vector<desc::Atom> get_geometry(const json& j, util::mpi_log& LOG) {
	
	std::vector<desc::Atom> out;
	
	if (j.find("file") == j.end()) {
		// reading from input file
		
		LOG.os<>("Reading XYZ info from file.\n");
		
		auto geometry = j["geometry"];
		auto symbols = j["symbols"];
		
		if ((geometry.size() % 3 != 0) || (geometry.size() / 3 != symbols.size())) {
			throw std::runtime_error("Missing coordinates.");
		}
		
		for (int i = 0; i != geometry.size()/3; ++i) {
			desc::Atom a;
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
				
				desc::Atom atom;
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

void unpack(const json& j_in, desc::options& opt, std::string root, std::string totalpath, util::mpi_log& LOG) {
	
	auto j = j_in[root];
	
	for (auto it = j.begin(); it != j.end(); ++it) {
		
		if (it->type() == json::value_t::boolean) {
			opt.set<bool>(totalpath + "/" + it.key(), *it);
		} else if (it->type() == json::value_t::number_integer ||
			it->type() == json::value_t::number_unsigned) {
			opt.set<int>(totalpath + "/" + it.key(), *it);
		} else if (it->type() == json::value_t::number_float) {
			opt.set<double>(totalpath + "/" + it.key(), *it);
		} else if (it->type() == json::value_t::string) {
			opt.set<std::string>(totalpath + "/" + it.key(), *it);
		} else if (it->type() == json::value_t::object) {
			std::string newroot = it.key();
			std::string newpath = totalpath + "/" + it.key();
			unpack(j, opt, newroot, newpath, LOG);
		} else {
			std::string msg = "Invalid type for keyword in " + root + "/";
			throw std::runtime_error(msg);
		}
		
		//if (it->type() == json::value_t::array) {
		//	opt.set<std::vector<int>>(root + it.key(), *it);
		//}
		
	}
}

template <typename T>
T assign(json& j, std::string name, T default_val) {
	auto it = j.find(name);
	return (it != j.end()) ? (T)it.value() : default_val;
}

desc::shared_molecule parse_molecule(json& jdata, MPI_Comm comm, int nprint) {
	
	util::mpi_log LOG(comm,nprint);
	
	validate("all", jdata, valid_keys);
	
	json& jmol = jdata["molecule"];
	
	int mo_split;
	std::string ao_split_method;
	
	mo_split = assign<int>(jmol, "mo_split", 10);
	ao_split_method = assign<std::string>(jmol, "ao_split_method", "atomic");
	
	LOG.os<>("Processing atomic coordinates...\n");
	auto atoms = get_geometry(jmol,LOG);
	
	bool reorder = assign<bool>(jmol, "reorder", false);
	
	if (reorder) {
		LOG.os<>("Reordering atoms...\n");
		
		math::rcm<desc::Atom> sorter(atoms,5.0,
			[](desc::Atom a1, desc::Atom a2) -> double {
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
	
	int ao_split = assign<int>(jmol, "ao_split", 10);
	
	std::string basname = jmol["basis"];
	std::optional<std::string> augbasname = std::nullopt;
	
	bool augmented = assign<bool>(jmol, "augmentation", false);
	
	desc::shared_cluster_basis cbas = 
		std::make_shared<desc::cluster_basis>(
			basname, atoms, ao_split_method, ao_split, augmented);
			
	desc::shared_cluster_basis cbas2 = nullptr;
	
	if (jmol.find("basis2") != jmol.end()) {
		std::string bas2name = jmol["basis2"];
		cbas2 = std::make_shared<desc::cluster_basis>(
			bas2name, atoms, ao_split_method, ao_split, false);
	}
	
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
	
	auto mol = desc::molecule::create()
		.comm(comm)
		.name(name)
		.atoms(atoms)
		.cluster_basis(cbas)
		.charge(charge)
		.mult(mult)
		.mo_split(mo_split)
		.build();
		
	if (cbas2) mol->set_cluster_basis2(cbas2);
		
	//mol->print_info(1);
	//LOG.os<>('\n');
	
	return mol;
	
}

desc::options parse_options(json& jdata, MPI_Comm comm, int nprint) {
		
	util::mpi_log LOG(comm,nprint);	

	validate("all", jdata, valid_keys);
	
	if (jdata.find("global") != jdata.end()) {
		json& jglob = jdata["global"];
		
		dbcsr::global::filter_eps = assign<double>(jglob, "block_threshold", 1e-9);
		ints::global::precision = assign<double>(jglob, "integral_precision", 1e-9);
		ints::global::omega = assign<double>(jglob, "integral_omega", 0.1);
		ints::global::qr_theta = assign<double>(jglob, "qr_theta", 1e-5);
		ints::global::qr_rho = assign<double>(jglob, "qr_rho", 40);

	}
	
	desc::options opt;
	
	auto read_section = [&](std::string r)
	{
		if (jdata.find(r) != jdata.end()) {
			unpack(jdata, opt, r, r, LOG);
			opt.set<bool>("do_"+r, true);
		} else {
			opt.set<bool>("do_"+r, false);
		}
	};
	
	read_section("hf");
	read_section("mp");
	read_section("adc");
	
	return opt;
	
}

}
