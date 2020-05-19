#ifndef DESC_WFN_H
#define DESC_WFN_H

#include "desc/molecule.h"
#include "desc/io.h"
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

	dbcsr::smatrix<double> m_s_bb;
	
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
	
	dbcsr::smatrix<double> s_bb() { return m_s_bb; }
	
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
		
		auto mname = m_mol->name();
		
		write_matrix(m_s_bb, mname);
		
		write_matrix(m_f_bb_A, mname);
		write_matrix(m_po_bb_A, mname);
		write_matrix(m_pv_bb_A, mname);
		write_matrix(m_c_bo_A, mname);
		write_matrix(m_c_bv_A, mname);
		
		if (m_f_bb_B) {
	
			write_matrix(m_f_bb_B, mname);
			write_matrix(m_po_bb_B, mname);
			write_matrix(m_pv_bb_B, mname);
			write_matrix(m_c_bo_B, mname);
			write_matrix(m_c_bv_B, mname);
			
		}
		
		MPI_Comm comm = m_s_bb->get_world().comm();
		
		write_vector(m_eps_occ_A, m_mol->name(), "eps_occ_A", comm);
		write_vector(m_eps_vir_A, m_mol->name(), "eps_vir_A", comm);
		
		if (m_eps_occ_B) {
			write_vector(m_eps_occ_B, m_mol->name(), "eps_occ_B", comm);
			write_vector(m_eps_vir_B, m_mol->name(), "eps_vir_B", comm);
		}
		
	}
	
	void read_from_file(smolecule& mol, dbcsr::world& w) {
		
		m_mol = mol;
		
		auto b = m_mol->dims().b();
		auto oA = m_mol->dims().oa();
		auto oB = m_mol->dims().ob();
		auto vA = m_mol->dims().va();
		auto vB = m_mol->dims().vb();

		m_s_bb = read_matrix(m_mol->name(), "s_bb", w, b, b, dbcsr_type_symmetric);
		
		m_f_bb_A = read_matrix(m_mol->name(), "f_bb_A", w, b, b, dbcsr_type_symmetric);
		m_po_bb_A = read_matrix(m_mol->name(), "p_bb_A", w, b, b, dbcsr_type_symmetric);
		m_pv_bb_A = read_matrix(m_mol->name(), "pv_bb_A", w, b, b, dbcsr_type_symmetric);
		m_c_bo_A = read_matrix(m_mol->name(), "c_bo_A", w, b, oA, dbcsr_type_no_symmetry);
		m_c_bv_A = read_matrix(m_mol->name(), "c_bv_A", w, b, vA, dbcsr_type_no_symmetry);
		
		bool read_beta = (m_mol->nele_alpha() == m_mol->nele_beta()) ? false : true;
		
		if (read_beta) {
				
			m_f_bb_B = read_matrix(m_mol->name(), "f_bb_B", w, b, b, dbcsr_type_symmetric);
			m_po_bb_B = read_matrix(m_mol->name(), "p_bb_B", w, b, b, dbcsr_type_symmetric);
			m_pv_bb_B = read_matrix(m_mol->name(), "pv_bb_B", w, b, b, dbcsr_type_symmetric);
			m_c_bo_B = read_matrix(m_mol->name(), "c_bo_B", w, b, oB, dbcsr_type_no_symmetry);
			m_c_bv_B = read_matrix(m_mol->name(), "c_bv_B", w, b, vB, dbcsr_type_no_symmetry);
			
		}
		
		read_vector(m_eps_occ_A, m_mol->name(), "eps_occ_A");
		read_vector(m_eps_vir_A, m_mol->name(), "eps_vir_A");
		
		if (read_beta) {
		
			read_vector(m_eps_occ_B, m_mol->name(), "eps_occ_B");
			read_vector(m_eps_vir_B, m_mol->name(), "eps_vir_B");
			
		}
		
	}
};

using shf_wfn = std::shared_ptr<hf_wfn>;

}
	
#endif
	
	
