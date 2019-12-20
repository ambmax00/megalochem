#include "input/reader.h"
#include "utils/json.hpp"
#include "input/valid_keys.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <utility>
#include <libint2/atom.h>
#include <libint2/shell.h>
#include <libint2/basis.h>
#include "utils/ele_to_int.h"
#include "utils/constants.h"

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
	
}
