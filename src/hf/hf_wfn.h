#ifndef HF_HF_WFN_H
#define HF_HF_WFN_H

#include "desc/molecule.h"
#include "io/io.h"
#include "utils/ppdirs.h"

#include <dbcsr_matrix.hpp>
#include <dbcsr_matrix_ops.hpp>
#include <mpi.h>

#include <string>
#include <fstream>
#include <filesystem>

namespace hf {
	
class hfmod;

class hf_wfn {
private:

	desc::smolecule m_mol;
	
	dbcsr::smatrix<double> m_c_bo_A;
	dbcsr::smatrix<double> m_c_bv_A;
	
	dbcsr::smatrix<double> m_c_bo_B;
	dbcsr::smatrix<double> m_c_bv_B;
	
	svector<double> m_eps_occ_A;
	svector<double> m_eps_occ_B;
	svector<double> m_eps_vir_A;
	svector<double> m_eps_vir_B;
	
	double m_scf_energy;
	double m_nuc_energy;
	double m_wfn_energy;
		
public:

	hf_wfn() {};
	hf_wfn(const hf_wfn& wfn_in) = default;
	
	hf_wfn(desc::smolecule mol, dbcsr::smatrix<double> c_bo_A,
		dbcsr::smatrix<double> c_bo_B, dbcsr::smatrix<double> c_bv_A,
		dbcsr::smatrix<double> c_bv_B, svector<double> eps_occ_A,
		svector<double> eps_occ_B, svector<double> eps_vir_A,
		svector<double> eps_vir_B, double scf_energy, double nuc_energy, 
		double wfn_energy) 
	: m_mol(mol), m_c_bo_A(c_bo_A), m_c_bo_B(c_bo_B), m_c_bv_A(c_bv_A), m_c_bv_B(c_bv_B),
		m_eps_occ_A(eps_occ_A), m_eps_occ_B(eps_occ_B), m_eps_vir_A(eps_vir_A),
		m_eps_vir_B(eps_vir_B), m_scf_energy(scf_energy), m_nuc_energy(nuc_energy),
		m_wfn_energy(wfn_energy)
	{}
	
	desc::smolecule mol() { return m_mol; }
	
	//dbcsr::smatrix<double> s_bb() { return m_s_bb; }
	
	dbcsr::smatrix<double> c_bo_A() { return m_c_bo_A; }
	dbcsr::smatrix<double> c_bv_A() { return m_c_bv_A; }
	
	dbcsr::smatrix<double> c_bo_B() { return m_c_bo_B; }
	dbcsr::smatrix<double> c_bv_B() { return m_c_bv_B; }
	
	svector<double> eps_occ_A() { return m_eps_occ_A; }
	svector<double> eps_vir_A() { return m_eps_vir_A; }
	svector<double> eps_occ_B() { return m_eps_occ_B; }
	svector<double> eps_vir_B() { return m_eps_vir_B; }
	
	double scf_energy() { return m_scf_energy; }
	double nuc_energy() { return m_nuc_energy; }
	double wfn_energy() { return m_wfn_energy; }
	
	void write_to_file(std::string name) {
		
		std::string molname = name;
		std::string prefix = molname + "_data/";
		std::string suffix = ".dat";
		
		auto filename = [&](std::string root) {
			return prefix + root + suffix;
		};
		
		//write_matrix(m_s_bb, mname);
		
		filio::write_matrix(m_c_bo_A, filename("c_bo_A"));
		filio::write_matrix(m_c_bv_A, filename("c_bv_A"));
		
		if (m_c_bo_B) {
	
			filio::write_matrix(m_c_bo_B, filename("c_bo_B"));
			filio::write_matrix(m_c_bv_B, filename("c_bv_B"));
			
		}
		
		MPI_Comm comm = m_c_bo_A->get_world().comm();
		
		filio::write_vector(m_eps_occ_A, filename("eps_occ_A"), comm);
		filio::write_vector(m_eps_vir_A, filename("eps_vir_A"), comm);
		
		if (m_eps_occ_B) {
			filio::write_vector(m_eps_occ_B, filename("eps_occ_B"), comm);
			filio::write_vector(m_eps_vir_B, filename("eps_vir_B"), comm);
		}
		
	}
	
	void read_from_file(std::string name, desc::smolecule& mol, dbcsr::world& w) {
		
		m_mol = mol;
		
		std::string molname = name;
		std::string prefix = molname + "_data/" ;
		std::string suffix = ".dat";
		
		auto filename = [&](std::string root) {
			return prefix + root + suffix;
		};
		
		auto b = m_mol->dims().b();
		auto oA = m_mol->dims().oa();
		auto oB = m_mol->dims().ob();
		auto vA = m_mol->dims().va();
		auto vB = m_mol->dims().vb();

		//m_s_bb = read_matrix(m_mol->name(), "s_bb", w, b, b, dbcsr_type_symmetric);
		
		m_c_bo_A = filio::read_matrix(filename("c_bo_A"), "c_bo_A", w, b, oA, dbcsr::type::no_symmetry);
		m_c_bv_A = filio::read_matrix(filename("c_bv_A"), "c_bv_A", w, b, vA, dbcsr::type::no_symmetry);
		
		bool read_beta = (m_mol->nele_alpha() == m_mol->nele_beta()) ? false : true;
		
		if (read_beta) {
				
			m_c_bo_B = filio::read_matrix(filename("c_bo_B"), "c_bo_B", w, b, oB, dbcsr::type::no_symmetry);
			m_c_bv_B = filio::read_matrix(filename("c_bv_B"), "c_bv_B", w, b, vB, dbcsr::type::no_symmetry);
			
		}
		
		filio::read_vector(m_eps_occ_A, filename("eps_occ_A"));
		filio::read_vector(m_eps_vir_A, filename("eps_vir_A"));
		
		if (read_beta) {
		
			filio::read_vector(m_eps_occ_B, filename("eps_occ_B"));
			filio::read_vector(m_eps_vir_B, filename("eps_vir_B"));
			
		}
		
	}	
	
	void write_results(std::string filename) {
		
		nlohmann::json data;
		nlohmann::json hfdata;
		std::ofstream out;
		std::ifstream in;
		
		int rank = m_c_bo_A->get_world().rank();
					
		if (std::filesystem::exists(filename) && rank == 0) {
					
			std::filesystem::remove(filename);
			
		}
		
		out.open(filename);
			
		hfdata["scf_energy"] = m_scf_energy;
		hfdata["nuc_energy"] = m_nuc_energy;
		hfdata["wfn_energy"] = m_wfn_energy;
		
		hfdata["nele_alpha"] = m_mol->nele_alpha();
		hfdata["nele_beta"] = m_mol->nele_beta();
		hfdata["nbf"] = m_mol->c_basis()->nbf();
		hfdata["nocc_alpha"] = m_mol->nocc_alpha();
		hfdata["nvir_alpha"] = m_mol->nvir_alpha();
		hfdata["nocc_beta"] = m_mol->nocc_beta();
		hfdata["nvir_beta"] = m_mol->nvir_beta();
		
		data["hf"] = std::move(hfdata);
		
		if (rank == 0) {
			out << data.dump(4);
		}
		
		out.close();
		
	}
	
	void read_results(std::string filename) {
		
		nlohmann::json data;
		std::ifstream in;
				
		if (std::filesystem::exists(filename)) {
		
			in.open(filename);
		
			in >> data;
			
			in.close();
			
		} else {
			
			throw std::runtime_error("File " + filename + " does not exist");
			
		}
		
		auto& hfdata = data["hf"];
			
		m_scf_energy = hfdata["scf_energy"];
		m_nuc_energy = hfdata["nuc_energy"];
		m_wfn_energy = hfdata["nuc_energy"];
		
	}
	
	dbcsr::smatrix<double> form_density(dbcsr::smatrix<double>& c) {
		
		auto b = c->row_blk_sizes(); 
		auto m = c->col_blk_sizes(); 
		auto w = c->get_world(); 
		
		auto p = dbcsr::create<double>() 
			.name("density") 
			.set_world(w) 
			.row_blk_sizes(b) 
			.col_blk_sizes(m) 
			.matrix_type(dbcsr::type::symmetric) 
			.get(); 
			
		dbcsr::multiply('N', 'T', *c, *c, *p).perform(); 
		
		return p;
		
	}
	
};

using shared_hf_wfn = std::shared_ptr<hf_wfn>;

class create_hf_wfn_base {
private:

	make_param(create_hf_wfn_base, mol, desc::smolecule, required, val);

	make_param(create_hf_wfn_base, c_bo_A, dbcsr::shared_matrix<double>, required, val);
	make_param(create_hf_wfn_base, c_bv_A, dbcsr::shared_matrix<double>, required, val);
	make_param(create_hf_wfn_base, c_bo_B, dbcsr::shared_matrix<double>, required, val);
	make_param(create_hf_wfn_base, c_bv_B, dbcsr::shared_matrix<double>, required, val);
	
	make_param(create_hf_wfn_base, scf_energy, double, required, val);
	make_param(create_hf_wfn_base, nuc_energy, double, required, val);
	make_param(create_hf_wfn_base, wfn_energy, double, required, val);
	
	make_param(create_hf_wfn_base, eps_occ_A, svector<double>, required, val);
	make_param(create_hf_wfn_base, eps_occ_B, svector<double>, required, val);
	make_param(create_hf_wfn_base, eps_vir_A, svector<double>, required, val);
	make_param(create_hf_wfn_base, eps_vir_B, svector<double>, required, val);
	
public:

	create_hf_wfn_base() {}

	shared_hf_wfn get() {
		return std::make_shared<hf_wfn>(*c_mol, *c_c_bo_A, *c_c_bo_B,
			*c_c_bv_A, *c_c_bv_B, *c_eps_occ_A, *c_eps_occ_B, *c_eps_vir_A,
			*c_eps_vir_B, *c_scf_energy, *c_nuc_energy, *c_wfn_energy);
	} 
	
	~create_hf_wfn_base() {}
	
};

inline create_hf_wfn_base create_hf_wfn() {
	return create_hf_wfn_base();
}

}
	
#endif
	
	
