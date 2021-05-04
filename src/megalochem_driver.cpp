#include "megalochem_driver.hpp"
#include "math/other/rcm.hpp"
#include "hf/hfmod.hpp"
#include "mp/mpmod.hpp"
#include "adc/adcmod.hpp"
#include "utils/ele_to_int.hpp"
#include "utils/constants.hpp"
#include "utils/ppdirs.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <filesystem>

#define SINGLE_REFLECTION_DETAIL(ctype, name) \
	.name(json_optional<util::base_type< UNPAREN ctype >::type>(job.jdata, STR(name)))

#define SINGLE_REFLECTION(param) \
	SINGLE_REFLECTION_DETAIL(GET_1 param, GET_2 param)
	
#define JSON_REFLECTION(list) \
	ITERATE_LIST(SINGLE_REFLECTION, (), (), list)

namespace megalochem {

static const nlohmann::json valid_basis = 
{
	{"tag", "string"},
	{"type", "string"},
	{"name", "string"},
	{"atoms", "name"},
	{"augmentation", false}, 
	{"ao_split_method", "atomic"},
	{"ao_split", 10u},
	{"_required", {"tag", "type", "name", "atoms"}}
};

static const nlohmann::json valid_atoms = 
{
	{"tag", "string"},
	{"type", "string"},
	{"unit", "string"},
	{"geometry", {0.0,0.0,0.0}}, // mol xyz
	{"symbols", {"H", "He", "C"}}, // mol elements
	{"file", "filename"},
	{"reorder", true},
	{"_required", {"tag", "type"}}
};
	
static const nlohmann::json valid_molecule = 
{
	{"tag", "string"},
	{"type", "string"},
	{"atoms", "string"},
	{"basis", "string"}, // basisset name
	{"basis2", "string"},
	{"mult", 0u}, //multiplicity
	{"charge", 0}, // total charge
	{"mo_split", 10u},
	{"_required", {"tag", "type", "atoms", "basis", "mult", "charge"}}
};

static const nlohmann::json valid_hfwfn = 
{
	{"tag", "string"},
	{"type", "string"},
	{"molecule", "molname"},
	{"guess", "core"}, // HF guess
	{"scf_thresh", 1e-9}, // convergence criteria
	{"unrestricted", true}, // unrestricted HF calc
	{"diis", true}, // use diis or not
	{"diis_max_vecs", 8u}, // maximum number of diis vectors in subsdpace
	{"diis_min_vecs", 2u}, // minimum number of diis vectors in subspace
	{"diis_start", 0u}, // at what iteration to start diis
	{"diis_beta", true}, // whether to use separate coeficients for beta
	{"build_J", "exact"}, // how Coulomb matrix is constructed
	{"build_K", "exact"}, // how Exchange matrix is constructed
	{"eris", "direct"}, // how eris are held in memory (core/disk/direct)
	{"intermeds", "core"}, // how intermediates are held in memory (core/disk)
	{"df_metric", "coulomb"}, // which metric to use for batchdf
	{"print", 0u}, // print level (0, 1 or 2 at the moment, -1 for silent output)
	{"nbatches_x", 4u},
	{"nbatches_b", 4u},
	{"occ_nbatches", 2u},
	{"read", false}, // skip hartree fock and read from files
	{"max_iter", 10u},
	{"SAD_guess", "core"},
	{"SAD_diis", true},
	{"SAD_spin_average", true},
	{"df_basis", "string"},
	{"df_basis2", "string"},
	{"_required", {"tag", "type", "molecule"}}
};
	
static const nlohmann::json valid_mpwfn = 
{
	{"tag", "string"},
	{"type", "string"},
	{"hfwfn", "string"},
	{"print", 0u},
	{"df_metric", "string"},
	{"nlap", 5u}, // number of laplace points
	{"nbatches_b", 3u},
	{"nbatches_x", 3u},
	{"df_basis", "basis"},
	{"c_os", 1.3},
	{"eris", "core"},
	{"intermeds", "core"},
	{"build_Z", "LLMPFULL"},
	{"_required", {"tag", "type", "hfwfn", "df_basis"}}
};
	
	
static const nlohmann::json valid_adcwfn = 
{
	{"tag", "string"},
	{"type", "string"},
	{"hfwfn", "string"},
	{"method", "sos-cd-ri-adc2"},
	{"print", 1u},
	{"nbatches_b", 3u},
	{"nbatches_x", 3u},
	{"df_basis", "basis"},
	{"nroots", 1u},
	{"block", true},
	{"balanced", true},
	{"nguesses", 1},
	{"do_adc1", true},
	{"do_adc2", true},
	{"df_metric", "string"},
	{"dav_conv", 1e-5},
	{"build_J", "dfao"},
	{"build_K", "dfao"},
	{"build_Z", "llmp_full"},
	{"eris", "core"},
	{"imeds", "core"},
	{"dav_max_iter", 100},
	{"diis_max_iter", 100},
	{"c_os", 1.3},
	{"c_os_coupling", 1.15},
	{"nlap", 5u},
	{"_required", {"tag", "type", "hfwfn", "nroots", "df_basis"}}
};

template <typename T>
std::optional<T> json_optional(nlohmann::json& j, std::string key) {
	std::optional<T> out = std::nullopt;
	if (j.find(key) != j.end()) {
		out = std::make_optional<T>(j[key]);
	}
	return out;
};

void validate(std::string section, const nlohmann::json& j_in, const nlohmann::json& j_ref) {
			
	// check if all required keys are present
	std::vector<std::string> reqkeys = j_ref["_required"];
	
	for (auto key : reqkeys) {
		if (key == "none") break;
		if (j_in.find(key) == j_in.end()) {
			throw std::runtime_error("Key " + key + " is required.");
		}
	}
	
	// loop through objects in this section
	for (auto it = j_in.begin(); it != j_in.end(); ++it) {
		
		// check if keys valid
		auto ref_it = j_ref.find(it.key());
		
		if (ref_it == j_ref.end()){
			throw std::runtime_error("Section " + section + ", invalid keyword: "+it.key());
		}
		
		// check type
		//if (ref_it->type() != it->type()) {
		//	throw std::runtime_error("Section " + section + ", bad type: "+it.key());
		//}
		
		// go to nested section, if present
		if (it->is_structured() && !it->is_array() && it.key() != "gen_basis") {
			validate(it.key(), *it, j_ref[it.key()]);
		}
	}
	
}

void driver::parse_file(std::string filename) {
	
	if (!std::filesystem::exists(filename)) {
		throw std::runtime_error("Could not find file " + filename);
	}
	
	std::ifstream ifile(filename);
	nlohmann::json jdata;
	
	ifile >> jdata;
	
	parse_json(jdata);
	
}

void driver::parse_json(nlohmann::json& jdata) {
	
	auto megalochem_data = jdata["megalochem"];
	
	for (auto& it : megalochem_data.items()) {
		parse_json_section(it.value());
	}
	
}

void driver::parse_json_section(nlohmann::json& jdata) {
	
	std::string strtype = jdata["type"];
	
	megatype mtype = str_to_type(strtype);
	
	switch (mtype) {
		case megatype::basis: {
			parse_basis(jdata);
			break;
		}
		case megatype::atoms: {
			parse_atoms(jdata);
			break;
		}
		case megatype::molecule: {
			parse_molecule(jdata);
			break;
		}
		case megatype::hfwfn: {
			parse_hfwfn(jdata);
			break;
		}
		case megatype::adcwfn: {
			parse_adcwfn(jdata);
			break;
		}
		case megatype::mpwfn: {
			parse_mpwfn(jdata);
			break;
		}
		default: {
			throw std::runtime_error("Unknown type in json string.");
		}
	}
		
}

void driver::parse_atoms(nlohmann::json& jdata) {
	
	validate("atoms", jdata, valid_atoms);
	
	std::vector<desc::Atom> atoms;
	
	if (jdata.find("file") == jdata.end()) {
		// reading from input file
		
		LOG.os<>("Reading XYZ info from file.\n");
		
		auto geometry = jdata["geometry"];
		auto symbols = jdata["symbols"];
		
		if ((geometry.size() % 3 != 0) || (geometry.size() / 3 != symbols.size())) {
			throw std::runtime_error("Missing coordinates.");
		}
		
		for (int i = 0; i != geometry.size()/3; ++i) {
			desc::Atom a;
			a.x = geometry.at(3*i);
			a.y = geometry.at(3*i+1);
			a.z = geometry.at(3*i+2);
			a.atomic_number = util::ele_to_int[symbols.at(i)];
			
			atoms.push_back(a);
			
		}
		
	} else {
		// read from xyz file
		
		std::string xyzfilename = jdata["file"];
		
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
				
				atoms.push_back(atom);
			}
			
			++nline;
			
		}
		
	}
	
	double factor;
	
	if (jdata.find("unit") != jdata.end()) {
		if (jdata["unit"] == "angstrom") {
			factor = BOHR_RADIUS;
		} else {
			throw std::runtime_error("Unknown length unit.");
		}
	} else {
		factor = BOHR_RADIUS;
	}
	
	for (auto& a : atoms) {
		a.x /= factor;
		a.y /= factor;
		a.z /= factor;
		
		//std::cout << a.x << " " << a.y << " " << a.z << std::endl;
		
	}
	
	bool reorder = (jdata.find("reorder") != jdata.end()) ? 
		(bool)jdata["reorder"] : true;
			
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
	
	m_stack[jdata["tag"]] = std::any(atoms);
	
}

void driver::parse_basis(nlohmann::json& jdata) {
	
	validate("basis", jdata, valid_basis);
	
	auto augmentation = json_optional<bool>(jdata, "augmentation");
	auto ao_split_method = json_optional<std::string>(jdata, "ao_split_method");
	auto ao_split = json_optional<int>(jdata, "ao_split");
	
	auto& atoms = get<std::vector<desc::Atom>>(jdata["atoms"]);
	
	auto cbas = std::make_shared<desc::cluster_basis>(jdata["name"], atoms, 
		ao_split_method, ao_split, augmentation);
	
	m_stack[jdata["tag"]] = std::any(cbas);
	
}

void driver::parse_molecule(nlohmann::json& jdata) {
	
	validate("molecule", jdata, valid_molecule);
	
	// get atoms
	
	auto atoms = get<std::vector<desc::Atom>>(jdata["atoms"]);
	
	auto cbas1 = get<desc::shared_cluster_basis>(jdata["basis"]);
	
	decltype(cbas1) cbas2 = nullptr;
	
	auto mo_split = json_optional<int>(jdata, "mo_split");
	
	auto mol = desc::molecule::create()
		.comm(m_world.comm())
		.name(jdata["tag"])
		.atoms(atoms)
		.cluster_basis(cbas1)
		.charge(jdata["charge"])
		.mult(jdata["mult"])
		.mo_split(mo_split)
		.build();
		
	if (jdata.find("basis2") != jdata.end()) {
		cbas2 = get<desc::shared_cluster_basis>(jdata["basis2"]);
		mol->set_cluster_basis2(cbas2);
	}
	
	mol->print_info();
	
	m_stack[jdata["tag"]] = std::any(mol);

}
	
void driver::parse_hfwfn(nlohmann::json& jdata) {
	
	validate("hfwfn", jdata, valid_hfwfn);
	
	auto read = json_optional<bool>(jdata, "read");
	auto mol = get<desc::shared_molecule>(jdata["molecule"]);
	
	if (read && *read) {
		auto myhfwfn = hf::read_hfwfn("hf_wfn", mol, m_world, *m_fh.input_fh);
		m_stack[jdata["tag"]] = std::any(myhfwfn);
	} else {
		megajob j = {megatype::hfwfn, jdata};
		m_jobs.push_back(std::move(j));
	}

}
	
void driver::parse_mpwfn(nlohmann::json& jdata) {
	
	validate("mpwfn", jdata, valid_mpwfn);
	
	megajob j = {megatype::mpwfn, jdata};
	m_jobs.push_back(std::move(j));
	
}

void driver::parse_adcwfn(nlohmann::json& jdata) {
	
	validate("adcwfn", jdata, valid_adcwfn);
	
	megajob j = {megatype::adcwfn, jdata};
	m_jobs.push_back(std::move(j));
	
}

void driver::run() {
	
	for (auto& j : m_jobs) {
		
		switch (j.mtype) {
			case megatype::hfwfn: {
				run_hfmod(j);
				break;
			}
			case megatype::mpwfn: {
				run_mpmod(j);
				break;
			}
			case megatype::adcwfn: {
				run_adcmod(j);
				break;
			}
			default: 
				throw std::runtime_error("Unknown driver method.");
		}
		
	}
	
}

void driver::run_hfmod(megajob& job) {
	
	auto mol = get<desc::shared_molecule>(job.jdata["molecule"]);
	
	std::optional<desc::shared_cluster_basis> dfbas, dfbas2;
	
	if (job.jdata.find("df_basis") != job.jdata.end()) {
		dfbas = get<desc::shared_cluster_basis>(job.jdata["df_basis"]);
	}
	
	std::cout << job.jdata["build_J"] << std::endl;
	
	if (job.jdata.find("df_basis2") != job.jdata.end()) {
		dfbas2 = get<desc::shared_cluster_basis>(job.jdata["df_basis2"]);
	}
	
	auto myhfmod = hf::hfmod::create()
		.set_world(m_world)
		.set_molecule(mol)
		.df_basis(dfbas)
		.df_basis2(dfbas2)
		JSON_REFLECTION(HFMOD_LIST_OPT)
		.build();
		
	myhfmod->compute();
	
	auto myhfwfn = myhfmod->wfn();
	
	m_stack[job.jdata["tag"]] = std::any(myhfwfn);
	
}

void driver::run_mpmod(megajob& job) {
	
	auto hfwfn = get<hf::shared_hf_wfn>(job.jdata["hfwfn"]);
	
	std::optional<desc::shared_cluster_basis> dfbas;
	
	if (job.jdata.find("df_basis") != job.jdata.end()) {
		dfbas = get<desc::shared_cluster_basis>(job.jdata["df_basis"]);
	}
	
	auto mympmod = mp::mpmod::create()
		.set_world(m_world)
		.set_hf_wfn(hfwfn)
		.df_basis(dfbas)
		JSON_REFLECTION(MPMOD_OPTLIST)
		.build();
		
	mympmod->compute();	
	
}

void driver::run_adcmod(megajob& job) {
	
	auto hfwfn = get<hf::shared_hf_wfn>(job.jdata["hfwfn"]);
	
	std::optional<desc::shared_cluster_basis> dfbas;
	
	if (job.jdata.find("df_basis") != job.jdata.end()) {
		dfbas = get<desc::shared_cluster_basis>(job.jdata["df_basis"]);
	}
	
	auto myadcmod = adc::adcmod::create()
		.set_world(m_world)
		.set_hfwfn(hfwfn)
		.df_basis(dfbas)
		JSON_REFLECTION(ADCMOD_OPTLIST)
		.build();
		
	myadcmod->compute();
		
}

} // namespace megalochem
