#ifndef MEGALOCHEM_DRIVER_HPP
#define MEGALOCHEM_DRIVER_HPP

#include "megalochem.hpp"
#include "io/data_handler.hpp"
#include "utils/json.hpp"
#include <deque>
#include <map>
#include <string>
#include <any>

namespace megalochem {

enum class megatype {
	globals,
	atoms,
	molecule,
	basis,
	hfwfn,
	mpwfn,
	adcwfn
};

inline megatype str_to_type(std::string str) {
	if (str == "globals") {
		return megatype::globals;
	} else if (str == "atoms") {
		return megatype::atoms; 
	} else if (str == "molecule") {
		return megatype::molecule; 
	} else if (str == "basis") {
		return megatype::basis; 
	} else if (str == "hfwfn") {
		return megatype::hfwfn; 
	} else if (str == "mpwfn") {
		return megatype::mpwfn;
	} else if (str == "adcwfn") {
		return megatype::atoms;
	} else {
		throw std::runtime_error("Unknown type");
	}
};  	

struct job {
	megatype m_type;
	nlohmann::json m_data;
};
	
class driver {
private:

	world m_world;
	filio::data_io m_fh;
	
	util::mpi_log LOG;
	
	std::map<std::string,std::any> m_stack; // variables 
	std::deque<job> m_jobs; // job queue
	
public:

	driver(world w, filio::data_io fh) : m_world(w), m_fh(fh), 
		LOG(w.comm(), 0) {}
	
	void parse_file(std::string filename);
	
	void parse_json(nlohmann::json& data);
	
	void parse_json_section(nlohmann::json& data);
	
	void parse_globals(nlohmann::json& jdata);
	
	void parse_basis(nlohmann::json& jdata);
	
	void parse_atoms(nlohmann::json& jdata);
	
	void parse_molecule(nlohmann::json& jdata);
	
	void parse_hfwfn(nlohmann::json& jdata);
	
	void parse_mpwfn(nlohmann::json& jdata);
	
	void parse_adcwfn(nlohmann::json& jdata);
	
	void run();
	
	template <typename T>
	T& get(std::string key) {
		
		if (m_stack.find(key) == m_stack.end()) {
			throw std::runtime_error("Could not find " + key + " in stack!");
		}
		
		return std::any_cast<T&>(m_stack[key]);
		
	}
		

	~driver() {}
	
};
	
}

#endif
