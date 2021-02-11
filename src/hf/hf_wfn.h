#ifndef HF_HF_WFN_H
#define HF_HF_WFN_H

#include "desc/molecule.h"
#include "io/data_handler.h"
#include "io/io.h"

#include <dbcsr_matrix.hpp>
#include <dbcsr_conversions.hpp>
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

	hf_wfn(
		desc::smolecule mol,
		dbcsr::smatrix<double> c_bo_A,
		dbcsr::smatrix<double> c_bo_B,
		dbcsr::smatrix<double> c_bv_A,
		dbcsr::smatrix<double> c_bv_B,
		svector<double> eps_occ_A,
		svector<double> eps_occ_B,
		svector<double> eps_vir_A,
		svector<double> eps_vir_B,
		double scf_energy,
		double nuc_energy,
		double wfn_energy
	) : 
		m_mol(mol), m_c_bo_A(c_bo_A), m_c_bv_A(c_bv_A), m_c_bo_B(c_bo_B), 
		m_c_bv_B(c_bv_B), m_eps_occ_A(eps_occ_A), 
		m_eps_occ_B(eps_occ_B), m_eps_vir_A(eps_vir_A), 
		m_eps_vir_B(eps_vir_B), m_scf_energy(scf_energy),
		m_nuc_energy(nuc_energy), m_wfn_energy(wfn_energy)
	{};
	hf_wfn(const hf_wfn& wfn_in) = default;
	
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
		
	
};

using shared_hf_wfn = std::shared_ptr<hf_wfn>;

inline void write_hfwfn(std::string name, hf_wfn& hfwfn, filio::data_handler& dh) {
	
	dh.open(filio::access_mode::rdwr);
	dh.create_group(name);
		
	// write matrices
	auto write_mat = [&](auto mat, std::string name) {
		if (mat) {
			hsize_t nrows = mat->nfullrows_total();
			hsize_t ncols = mat->nfullcols_total();
			auto eigen = dbcsr::matrix_to_eigen(*mat);
			dh.write<double>(name, eigen.data(), {nrows,ncols}, 0);
		}
	};
	
	auto write_vec = [&](auto vec, std::string name) {
		if (vec) {
			hsize_t ntot = vec->size();
			dh.write<double>(name, vec->data(), {ntot}, 0);
		}
	}; 
	
	write_mat(hfwfn.c_bo_A(), name + "/c_bo_A");
	write_mat(hfwfn.c_bo_B(), name + "/c_bo_B");
	write_mat(hfwfn.c_bv_A(), name + "/c_bv_A");
	write_mat(hfwfn.c_bv_B(), name + "/c_bv_B");
	write_vec(hfwfn.eps_occ_A(), name + "/eps_occ_A");
	write_vec(hfwfn.eps_vir_A(), name + "/eps_vir_A");
	write_vec(hfwfn.eps_occ_B(), name + "/eps_occ_B");
	write_vec(hfwfn.eps_vir_B(), name + "/eps_vir_B");
	
	dh.write<double>(name + "/scf_energy", hfwfn.scf_energy());
	dh.write<double>(name + "/nuc_energy", hfwfn.nuc_energy());
	dh.write<double>(name + "/wfn_energy", hfwfn.wfn_energy());
	
	dh.close();

}

inline shared_hf_wfn read_hfwfn(std::string name, desc::smolecule mol, 
	dbcsr::world w, filio::data_handler& dh) 
{
	
	dh.open(filio::access_mode::rdonly);
	
	// read matrix
	auto read_mat = [&](std::string name, std::vector<int> r, 
		std::vector<int> c, bool throw_if_not_present) 
	{
		
		int nrows = std::accumulate(r.begin(), r.end(), 0);
		int ncols = std::accumulate(c.begin(), c.end(), 0);
		
		if (throw_if_not_present && !dh.exists(name)) {
			throw std::runtime_error("Read hfwfn: could not find " + name + ".");
		} else if (!dh.exists(name)) {
			dbcsr::shared_matrix<double> out = nullptr;
			return out;
		}
		
		auto darray = dh.read<double>(name);
		
		if (nrows != darray.dims[0] || ncols != darray.dims[1]) {
			throw std::runtime_error("Read hfwfn: incompatible matrix dimensions.");
		}
		
		Eigen::Map<Eigen::MatrixXd> emap(darray.data, nrows, ncols);
		
		auto mat = dbcsr::eigen_to_matrix(emap, w, name, r, c, 
			dbcsr::type::no_symmetry);
		
		return mat;
		
	};
	
	auto read_vec = [&](std::string name) {
		
		auto darray = dh.read<double>(name);
		return std::make_shared<std::vector<double>(darray.data);
		
	};
	
	auto b = mol->dims().b();
	auto oa = mol->dims().oa();
	auto ob = mol->dims().ob();
	auto va = mol->dims().va();
	auto vb = mol->dims().vb();
	
	auto c_bo_A = read_mat(name + "/c_bo_A", b, oa, true);
	auto c_bo_B = read_mat(name + "/c_bo_B", b, ob, false);
	auto c_bv_A = read_mat(name + "/c_bv_A", b, va, true);
	auto c_bv_B = read_mat(name + "/c_bv_B", b, vb, false);
	
	auto eps_occ_A = read_vec("/eps_occ_A");
	auto eps_occ_B = read_vec("/eps_occ_B");
	auto eps_vir_A = read_vec("/eps_vir_A");
	auto eps_vir_B = read_vec("/eps_vir_B");
	
	double scf_energy = dh.read_single<double>("/scf_energy");
	double nuc_energy = dh.read_single<double>("/nuc_energy");
	double wfn_energy = dh.read_single<double>("/wfn_energy");
	
	auto hfwfn = std::make_shared<hf_wfn>(
		mol, c_bo_A, c_bo_B, c_bv_A, c_bv_B, eps_occ_A, eps_occ_B,
		eps_vir_A, eps_vir_B, scf_energy, nuc_energy, wfn_energy
	);
		
	return hfwfn;
	
}		

}
	
#endif
	
	
