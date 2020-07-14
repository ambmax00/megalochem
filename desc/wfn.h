#ifndef DESC_WFN_H
#define DESC_WFN_H

#include "desc/molecule.h"
#include "io/io.h"
#include <dbcsr_matrix.hpp>
#include <mpi.h>

#include <string>
#include <fstream>

namespace hf {
	class hfmod;
}

namespace desc {
	
class hf_wfn {
private:

	smolecule m_mol;

	//dbcsr::smatrix<double> m_s_bb;
	
	dbcsr::smatrix<double> m_f_bb_A;
	dbcsr::smatrix<double> m_c_bm_A;
	dbcsr::smatrix<double> m_c_bo_A;
	dbcsr::smatrix<double> m_c_bv_A;
	dbcsr::smatrix<double> m_po_bb_A;
	dbcsr::smatrix<double> m_pv_bb_A;
	
	dbcsr::smatrix<double> m_f_bb_B;
	dbcsr::smatrix<double> m_c_bm_B;
	dbcsr::smatrix<double> m_c_bo_B;
	dbcsr::smatrix<double> m_c_bv_B;
	dbcsr::smatrix<double> m_po_bb_B;
	dbcsr::smatrix<double> m_pv_bb_B;
	
	svector<double> m_eps_occ_A;
	svector<double> m_eps_occ_B;
	svector<double> m_eps_vir_A;
	svector<double> m_eps_vir_B;
	
	double m_scf_energy;
	double m_nuc_energy;
	double m_wfn_energy;
	
	friend class hf::hfmod;
	
public:

	hf_wfn() {};
	
	desc::smolecule mol() { return m_mol; }
	
	//dbcsr::smatrix<double> s_bb() { return m_s_bb; }
	
	dbcsr::smatrix<double> f_bb_A() { return m_f_bb_A; }
	dbcsr::smatrix<double> c_bo_A() { return m_c_bo_A; }
	dbcsr::smatrix<double> c_bv_A() { return m_c_bv_A; }
	dbcsr::smatrix<double> po_bb_A() { return m_po_bb_A; }
	dbcsr::smatrix<double> pv_bb_A() { return m_pv_bb_A; }
	
	dbcsr::smatrix<double> f_bb_B() { return m_f_bb_B; }
	dbcsr::smatrix<double> c_bo_B() { return m_c_bo_B; }
	dbcsr::smatrix<double> c_bv_B() { return m_c_bv_B; }
	dbcsr::smatrix<double> po_bb_B() { return m_po_bb_B; }
	dbcsr::smatrix<double> pv_bb_B() { return m_pv_bb_B; }
	
	svector<double> eps_occ_A() { return m_eps_occ_A; }
	svector<double> eps_vir_A() { return m_eps_vir_A; }
	svector<double> eps_occ_B() { return m_eps_occ_B; }
	svector<double> eps_vir_B() { return m_eps_vir_B; }
	
	double scf_energy() { return m_scf_energy; }
	double nuc_energy() { return m_nuc_energy; }
	double wfn_energy() { return m_wfn_energy; }
	
	void write_to_file() {
		
		std::string molname = m_mol->name();
		std::string prefix = "data/" + molname + "_";
		std::string suffix = ".dat";
		
		auto filename = [&](std::string root) {
			return prefix + root + suffix;
		};
		
		//write_matrix(m_s_bb, mname);
		
		filio::write_matrix(m_f_bb_A, filename("f_bb_A"));
		filio::write_matrix(m_po_bb_A, filename("po_bb_A"));
		filio::write_matrix(m_pv_bb_A, filename("pv_bb_A"));
		filio::write_matrix(m_c_bo_A, filename("c_bo_A"));
		filio::write_matrix(m_c_bv_A, filename("c_bv_A"));
		
		if (m_f_bb_B) {
	
			filio::write_matrix(m_f_bb_B, filename("f_bb_B"));
			filio::write_matrix(m_po_bb_B, filename("po_bb_B"));
			filio::write_matrix(m_pv_bb_B, filename("pv_bb_B"));
			filio::write_matrix(m_c_bo_B, filename("c_bo_B"));
			filio::write_matrix(m_c_bv_B, filename("c_bv_B"));
			
		}
		
		MPI_Comm comm = m_f_bb_A->get_world().comm();
		
		filio::write_vector(m_eps_occ_A, filename("eps_occ_A"), comm);
		filio::write_vector(m_eps_vir_A, filename("eps_vir_A"), comm);
		
		if (m_eps_occ_B) {
			filio::write_vector(m_eps_occ_B, filename("eps_occ_B"), comm);
			filio::write_vector(m_eps_vir_B, filename("eps_vir_B"), comm);
		}
		
	}
	
	void read_from_file(smolecule& mol, dbcsr::world& w) {
		
		m_mol = mol;
		
		std::string molname = m_mol->name();
		std::string prefix = "data/" + molname + "_";
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
		
		m_f_bb_A = filio::read_matrix(filename("f_bb_A"), "f_bb_A", w, b, b, dbcsr_type_symmetric);
		m_po_bb_A = filio::read_matrix(filename("po_bb_A"), "po_bb_A", w, b, b, dbcsr_type_symmetric);
		m_pv_bb_A = filio::read_matrix(filename("pv_bb_A"), "pv_bb_A", w, b, b, dbcsr_type_symmetric);
		m_c_bo_A = filio::read_matrix(filename("c_bo_A"), "c_bo_A", w, b, oA, dbcsr_type_no_symmetry);
		m_c_bv_A = filio::read_matrix(filename("c_bv_A"), "c_bv_A", w, b, vA, dbcsr_type_no_symmetry);
		
		bool read_beta = (m_mol->nele_alpha() == m_mol->nele_beta()) ? false : true;
		
		if (read_beta) {
				
			m_f_bb_B = filio::read_matrix(filename("f_bb_B"), "f_bb_B", w, b, b, dbcsr_type_symmetric);
			m_po_bb_B = filio::read_matrix(filename("po_bb_B"), "po_bb_B", w, b, b, dbcsr_type_symmetric);
			m_pv_bb_B = filio::read_matrix(filename("pv_bb_B"), "pv_bb_B", w, b, b, dbcsr_type_symmetric);
			m_c_bo_B = filio::read_matrix(filename("c_bo_B"), "c_bo_B", w, b, oB, dbcsr_type_no_symmetry);
			m_c_bv_B = filio::read_matrix(filename("c_bv_B"), "c_bv_B", w, b, vB, dbcsr_type_no_symmetry);
			
		}
		
		filio::read_vector(m_eps_occ_A, filename("eps_occ_A"));
		filio::read_vector(m_eps_vir_A, filename("eps_vir_A"));
		
		if (read_beta) {
		
			filio::read_vector(m_eps_occ_B, filename("eps_occ_B"));
			filio::read_vector(m_eps_vir_B, filename("eps_vir_B"));
			
		}
		
	}
};

using shf_wfn = std::shared_ptr<hf_wfn>;

}
	
#endif
	
	
