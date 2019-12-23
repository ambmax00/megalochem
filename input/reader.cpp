#include "input/reader.h"
#include "utils/json.hpp"
#include "input/valid_keys.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <utility>
#include "utils/ele_to_int.h"
#include "utils/constants.h"
#include "desc/molecule.h"
#include <libint2/basis.h>

void validate(const json& j, const json& compare) {
	
	for (auto it = j.begin(); it != j.end(); ++it) {
		std::cout << it.key() << std::endl;
		if (compare.find(it.key()) == compare.end()){
			throw std::runtime_error("Invalid keyword: "+it.key());
		}
		if (it->is_structured() && !it->is_array()) {
			validate(*it, compare[it.key()]);
		}
	}
}

std::vector<libint2::Atom> get_geometry(const json& j) {
	
	std::vector<libint2::Atom> out;
	
	if (j.find("file") == j.end()) {
		// reading from input file
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
		std::ifstream in;
		in.open(j["file"]);
		
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
				std::cout << line << std::endl;
				
				libint2::Atom atom;
				atom.x = x;
				atom.y = y;
				atom.z = z;
				atom.atomic_number = util::ele_to_int[ele_name];
				
				out.push_back(atom);
			}
			
			++nline;
			
		}
		
	}
	
	if (j.find("unit") != j.end()) {
		if (j["unit"] == "angstrom") {
			for (auto a : out) {
				a.x /= BOHR_RADIUS;
				a.y /= BOHR_RADIUS;
				a.z /= BOHR_RADIUS;
			}
		}
	}
			
	
	return out;
	
}

std::vector<libint2::Shell> get_basis(const json &j, std::vector<libint2::Atom> &atoms) {
	
	// only ba name for now
	auto name = j["basis"];
	
	libint2::BasisSet basis(name, atoms);
	
	std::vector<libint2::Shell> out = std::move(basis);
	
	return out;
	
}

optional<std::vector<libint2::Shell>,val> 
get_dfbasis(const json &j, std::vector<libint2::Atom> &atoms) {
	
	// only ba name for now
	if (j.find("dfbasis") != j.end()) {
		
		auto name = j["dfbasis"];
		
		libint2::BasisSet basis(name, atoms);
		
		optional<std::vector<libint2::Shell>,val> out(basis);
		
		return out;
		
	}	
	
	optional<std::vector<libint2::Shell>,val> out;
	return out;
	
}

void unpack(const json& j_in, desc::options& opt, std::string root) {
	
	auto j = j_in[root];
	
	for (auto it = j.begin(); it != j.end(); ++it) {
		
		if (it->type() == json::value_t::boolean) {
			opt.set<bool>(root + "/" + it.key(), *it);
		}
		
		if (it->type() == json::value_t::number_integer) {
			opt.set<int>(root + "/" + it.key(), *it);
		}
		
		if (it->type() == json::value_t::number_float) {
			opt.set<double>(root + "/" + it.key(), *it);
		}
		
		if (it->type() == json::value_t::string) {
			opt.set<std::string>(root + "/" + it.key(), *it);
		}
		
		//if (it->type() == json::value_t::array) {
		//	opt.set<std::vector<int>>(root + it.key(), *it);
		//}
		
	}
}	

reader::reader(std::string filename) {
	
	std::ifstream in;
	in.open(filename);
	
	if (!in) {
		throw std::runtime_error("Input file not found.");
	}
	
	json data;
	
	in >> data;
	
	validate(data, valid_keys);
	
	auto atoms = get_geometry(data["molecule"]);
	
	auto basis = get_basis(data["molecule"],atoms);
	
	auto dfbasis = get_dfbasis(data["molecule"],atoms);
	
	int charge = data["molecule"]["charge"];
	int mult = data["molecule"]["mult"];
	
	desc::molecule mol({.atoms = atoms, .charge = charge,
		.mult = mult, .split = 20, .basis = basis, .dfbasis = dfbasis});
		
	mol.print_info(1);
	
	desc::options opt;
	
	unpack(data, opt, "hf");
	
	std::cout << opt.get<bool>("hf/diis") << std::endl;
	std::cout << opt.get<double>("hf/conv") << std::endl;
	
}
