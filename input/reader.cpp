#include "input/reader.h"
#include "utils/json.hpp"
#include "input/valid_keys.h"
#include <fstream>
#include <iostream>

void validate(const json& j, const json& compare) {
	
	for (auto it = j.begin(); it != j.end(); ++it) {
		std::cout << it.key() << std::endl;
		if (compare.find(it.key()) == compare.end()){
			throw std::runtime_error("Invalid keyword: "+it.key());
		}
		if (it->is_structured()) {
			validate(*it, compare[it.key()]);
		}
	}
}

//options make_molecule

reader::reader(std::string filename) {
	
	std::ifstream in;
	in.open(filename);
	
	if (!in) {
		throw std::runtime_error("Input file not found.");
	}
	
	json data;
	
	in >> data;
	
	validate(data, valid_keys);
	
}
