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

#include <dbcsr_common.hpp>
#include "ints/aofactory.h"

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

std::vector<desc::Atom> get_geometry(const json& j, std::string filename, util::mpi_log& LOG) {
	
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

template <typename T>
T assign(json& j, std::string name, T default_val) {
	auto it = j.find(name);
	return (it != j.end()) ? (T)it.value() : default_val;
}
	  	

reader::reader(MPI_Comm comm, std::string filename, int print) : m_comm(comm), LOG(comm, print) {
	
	std::ifstream in;
	in.open(filename + ".json");
	
	if (!in) {
		throw std::runtime_error("Input file not found.");
	}
	
	LOG.os<>("Reading input file...\n\n");

/*
	std::string c_filename = filename + ".json";
	std::string qcschema_validate = 
R"(
import json
import qcschema

def qcschema_validate(filename):
	json_file = open(filename)
	data  = json.load(json_file)
	try:
		qcschema.validate(data, 'input')
		return str("success")
	except Exception as inst:
		return str(inst)
)"; 

    //std::cout << qcschema_validate << std::endl;
    
    Py_Initialize();
   
	PyObject *main_module, *global_dict, *expression, 
		*result, *temp_bytes, *p_filename, *p_args;
	char* cstr;
   
    PyRun_SimpleString(qcschema_validate.c_str());
    main_module = PyImport_AddModule("__main__");
    global_dict = PyModule_GetDict(main_module);
    expression = PyDict_GetItemString(global_dict, "qcschema_validate");
    p_filename = PyUnicode_FromString(c_filename.c_str());
    p_args = PyTuple_New(1);
    PyTuple_SetItem(p_args, 0, p_filename);
    
    result = PyObject_CallObject(expression, p_args);
    
    if (!result) {
		PyErr_Print();
		throw std::runtime_error("Something went wrong calling Python.");
	}
    
	temp_bytes = PyUnicode_AsEncodedString(result, "UTF-8", "strict");
	cstr = PyBytes_AS_STRING(temp_bytes);
	cstr = strdup(cstr);
    
    if (cstr != "success") {
		LOG.os<>(cstr);
		throw std::runtime_error("Error while validating QCSchema");
	}
*/

	json data;
	
	in >> data;
	
	validate("all", data, valid_keys);
	
	if (data.find("global") != data.end()) {
		json& jglob = data["global"];
		
		dbcsr::global::filter_eps = assign<double>(jglob, "block_threshold", 1e-9);
		ints::global::precision = assign<double>(jglob, "integral_precision", 1e-9);
		ints::global::omega = assign<double>(jglob, "integral_omega", 0.1);

	}
	
	json& jmol = data["molecule"];
	
	int mo_split;
	std::string ao_split_method;
	
	mo_split = assign<int>(jmol, "mo_split", 10);
	ao_split_method = assign<std::string>(jmol, "ao_split_method", "atomic");
	
	LOG.os<>("Processing atomic coordinates...\n");
	auto atoms = get_geometry(jmol,filename,LOG);
	
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
	
	desc::shared_cluster_basis cbas = 
		std::make_shared<desc::cluster_basis>(
			basname, atoms, ao_split_method, ao_split);
	
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
	
	m_mol = desc::create_molecule()
		.comm(m_comm)
		.name(name)
		.atoms(atoms)
		.basis(cbas)
		.charge(charge)
		.mult(mult)
		.mo_split(mo_split)
		.get();
		
	m_mol->print_info(1);
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
	
	m_opt = opt;
	
}

}
