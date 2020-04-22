#ifndef DESC_WFN_H
#define DESC_WFN_H

#include "desc/molecule.h"
#include "desc/io.h"
#include "tensor/dbcsr_conversions.h"

#include <string>
#include <fstream>

namespace hf {
	class hfmod;
}

namespace desc {
	
class hf_wfn {
private:

	smolecule m_mol;

	dbcsr::stensor<2> m_s_bb;
	
	dbcsr::stensor<2> m_f_bb_A;
	dbcsr::stensor<2> m_c_bm_A;
	dbcsr::stensor<2> m_c_bo_A;
	dbcsr::stensor<2> m_c_bv_A;
	dbcsr::stensor<2> m_po_bb_A;
	dbcsr::stensor<2> m_pv_bb_A;
	
	dbcsr::stensor<2> m_f_bb_B;
	dbcsr::stensor<2> m_c_bm_B;
	dbcsr::stensor<2> m_c_bo_B;
	dbcsr::stensor<2> m_c_bv_B;
	dbcsr::stensor<2> m_po_bb_B;
	dbcsr::stensor<2> m_pv_bb_B;
	
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
	
	dbcsr::stensor<2> s_bb() { return m_s_bb; }
	
	dbcsr::stensor<2> f_bb_A() { return m_f_bb_A; }
	dbcsr::stensor<2> c_bo_A() { return m_c_bo_A; }
	dbcsr::stensor<2> c_bv_A() { return m_c_bv_A; }
	dbcsr::stensor<2> po_bb_A() { return m_po_bb_A; }
	dbcsr::stensor<2> pv_bb_A() { return m_pv_bb_A; }
	
	dbcsr::stensor<2> f_bb_B() { return m_f_bb_B; }
	dbcsr::stensor<2> c_bo_B() { return m_c_bo_B; }
	dbcsr::stensor<2> c_bv_B() { return m_c_bv_B; }
	dbcsr::stensor<2> po_bb_B() { return m_po_bb_B; }
	dbcsr::stensor<2> pv_bb_B() { return m_pv_bb_B; }
	
	svector<double> eps_occ_A() { return m_eps_occ_A; }
	svector<double> eps_vir_A() { return m_eps_vir_A; }
	svector<double> eps_occ_B() { return m_eps_occ_B; }
	svector<double> eps_vir_B() { return m_eps_vir_B; }
	
	double scf_energy() { return m_scf_energy; }
	double nuc_energy() { return m_nuc_energy; }
	double wfn_energy() { return m_wfn_energy; }
	
	void write_to_file() {
		
		write_2dtensor(m_s_bb, m_mol->name());
		write_2dtensor(m_f_bb_A, m_mol->name());
		write_2dtensor(m_c_bo_A, m_mol->name());
		write_2dtensor(m_c_bv_A, m_mol->name());
		write_2dtensor(m_po_bb_A, m_mol->name());
		write_2dtensor(m_pv_bb_A, m_mol->name());
		
		if (m_f_bb_B) {
		
			write_2dtensor(m_f_bb_B, m_mol->name());
			write_2dtensor(m_c_bo_B, m_mol->name());
			write_2dtensor(m_c_bv_B, m_mol->name());
			write_2dtensor(m_po_bb_B, m_mol->name());
			write_2dtensor(m_pv_bb_B, m_mol->name());
			
		}
		
		auto comm = m_s_bb->comm();
		
		write_vector(m_eps_occ_A, m_mol->name(), "eps_occ_A", comm);
		write_vector(m_eps_vir_A, m_mol->name(), "eps_vir_A", comm);
		
		if (m_eps_occ_B) {
			write_vector(m_eps_occ_B, m_mol->name(), "eps_occ_B", comm);
			write_vector(m_eps_vir_B, m_mol->name(), "eps_vir_B", comm);
		}
		
	}
	
	void read_from_file(smolecule& mol, MPI_Comm comm) {
		
		std::cout << "READING" << std::endl;
		
		m_mol = mol;
		
		arrvec<int,2> bb = {m_mol->dims().b(), m_mol->dims().b()};
		arrvec<int,2> boA = {m_mol->dims().b(), m_mol->dims().oa()};
		arrvec<int,2> bvA = {m_mol->dims().b(), m_mol->dims().va()};
		arrvec<int,2> boB = {m_mol->dims().b(), m_mol->dims().ob()};
		arrvec<int,2> bvB = {m_mol->dims().b(), m_mol->dims().vb()};
		
		bool read_beta = (m_mol->nele_beta() != 0 && m_mol->nele_beta() != m_mol->nele_alpha());
		
		read_2dtensor(m_s_bb, m_mol->name(), "s_bb", comm, bb);
		read_2dtensor(m_f_bb_A, m_mol->name(), "f_bb_A", comm, bb);
		read_2dtensor(m_c_bo_A, m_mol->name(), "c_bo_A", comm, boA);
		read_2dtensor(m_c_bv_A, m_mol->name(), "c_bv_A", comm, bvA);
		
		read_2dtensor(m_po_bb_A, m_mol->name(), "p_bb_A", comm, bb);
		read_2dtensor(m_pv_bb_A, m_mol->name(), "pv_bb_A", comm, bb);
		
		if (read_beta) {
		
			read_2dtensor(m_f_bb_B, m_mol->name(), "f_bb_B", comm, bb);
			read_2dtensor(m_c_bo_B, m_mol->name(), "c_bo_B", comm, boB);
			read_2dtensor(m_c_bv_B, m_mol->name(), "c_bv_B", comm, bvB);
			read_2dtensor(m_po_bb_B, m_mol->name(), "p_bb_B", comm, bb);
			read_2dtensor(m_pv_bb_B, m_mol->name(), "pv_bb_B", comm, bb);
			
		}
		
		read_vector(m_eps_occ_A, m_mol->name(), "eps_occ_A");
		read_vector(m_eps_vir_A, m_mol->name(), "eps_vir_A");
		
		if (read_beta) {
		
			read_vector(m_eps_occ_B, m_mol->name(), "eps_occ_B");
			read_vector(m_eps_vir_B, m_mol->name(), "eps_vir_B");
			
		}
		
		std::cout << "Done Reading" << std::endl;

		
	}
};

using shf_wfn = std::shared_ptr<hf_wfn>;

}
	
#endif
	
	
