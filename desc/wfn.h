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
		
		m_s_bb->write(m_s_bb->name() + "_" + m_mol->name() + ".dat");
		m_f_bb_A->write(m_f_bb_A->name() + "_" + m_mol->name() + ".dat");
		m_c_bo_A->write(m_c_bo_A->name() + "_" + m_mol->name() + ".dat");
		m_c_bv_A->write(m_c_bv_A->name() + "_" + m_mol->name() + ".dat");
		m_po_bb_A->write(m_po_bb_A->name() + "_" + m_mol->name() + ".dat");
		m_pv_bb_A->write(m_pv_bb_A->name() + "_" + m_mol->name() + ".dat");
		
		m_s_bb->write(m_s_bb->name() + "_" + m_mol->name() + ".dat");
		m_f_bb_A->write(m_f_bb_A->name() + "_" + m_mol->name() + ".dat");
		m_c_bo_A->write(m_c_bo_A->name() + "_" + m_mol->name() + ".dat");
		m_c_bv_A->write(m_c_bv_A->name() + "_" + m_mol->name() + ".dat");
		m_po_bb_A->write(m_po_bb_A->name() + "_" + m_mol->name() + ".dat");
		m_pv_bb_A->write(m_pv_bb_A->name() + "_" + m_mol->name() + ".dat");
		
		if (m_f_bb_B) {
	
			m_f_bb_B->write(m_f_bb_B->name() + "_" + m_mol->name() + ".dat");
			m_c_bo_B->write(m_c_bo_B->name() + "_" + m_mol->name() + ".dat");
			m_c_bv_B->write(m_c_bv_B->name() + "_" + m_mol->name() + ".dat");
			m_po_bb_B->write(m_po_bb_B->name() + "_" + m_mol->name() + ".dat");
			m_pv_bb_B->write(m_pv_bb_B->name() + "_" + m_mol->name() + ".dat");
			
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
		
		std::cout << "READING" << std::endl;
		
		auto b = m_mol->dims().b();
		auto oA = m_mol->dims().oa();
		auto oB = m_mol->dims().ob();
		auto vA = m_mol->dims().va();
		auto vB = m_mol->dims().vb();
		
		vec<int> vec_b_row = dbcsr::default_dist(b.size(), w.dims()[0], b);
		vec<int> vec_b_col = dbcsr::default_dist(b.size(), w.dims()[1], b);
		vec<int> vec_oA = dbcsr::default_dist(oA.size(), w.dims()[1], oA);
		vec<int> vec_oB = dbcsr::default_dist(oB.size(), w.dims()[1], oB);
		vec<int> vec_vA = dbcsr::default_dist(vA.size(), w.dims()[1], vA);
		vec<int> vec_vB = dbcsr::default_dist(vB.size(), w.dims()[1], vB);
		
		dbcsr::dist dist_bb = dbcsr::dist::create().set_world(w).row_dist(vec_b_row).col_dist(vec_b_col);
		dbcsr::dist dist_boA = dbcsr::dist::create().set_world(w).row_dist(vec_b_row).col_dist(vec_oA);
		dbcsr::dist dist_bvA = dbcsr::dist::create().set_world(w).row_dist(vec_b_row).col_dist(vec_vA);
		dbcsr::dist dist_boB = dbcsr::dist::create().set_world(w).row_dist(vec_b_row).col_dist(vec_oB);
		dbcsr::dist dist_bvB = dbcsr::dist::create().set_world(w).row_dist(vec_b_row).col_dist(vec_vB);
		
		bool read_beta = (m_mol->nele_beta() != 0 && m_mol->nele_beta() != m_mol->nele_alpha());
		
		dbcsr::mat_d S = dbcsr::mat_d::read().filepath("s_bb_" + m_mol->name() + ".dat")
			.distribution(dist_bb).set_world(w);
			
		m_s_bb = S.get_smatrix();
			
		dbcsr::mat_d FA = dbcsr::mat_d::read().filepath("f_bb_A_" + m_mol->name() + ".dat")
			.distribution(dist_bb).set_world(w);
		
		dbcsr::mat_d COA = dbcsr::mat_d::read().filepath("c_bo_A_" + m_mol->name() + ".dat")
			.distribution(dist_boA).set_world(w);
			
		dbcsr::mat_d CVA = dbcsr::mat_d::read().filepath("c_bv_A_" + m_mol->name() + ".dat")
			.distribution(dist_bvA).set_world(w);
		
		dbcsr::mat_d POA = dbcsr::mat_d::read().filepath("p_bb_A_" + m_mol->name() + ".dat")
			.distribution(dist_bb).set_world(w);
			
		dbcsr::mat_d PVA = dbcsr::mat_d::read().filepath("pv_bb_A_" + m_mol->name() + ".dat")
			.distribution(dist_bb).set_world(w);
			
		m_f_bb_A = FA.get_smatrix();
		m_c_bo_A = COA.get_smatrix();
		m_c_bv_A = CVA.get_smatrix();
		m_po_bb_A = POA.get_smatrix();
		m_pv_bb_A = PVA.get_smatrix();
			
		if (read_beta) {
				
			dbcsr::mat_d FB = dbcsr::mat_d::read().filepath("f_bb_B_" + m_mol->name() + ".dat")
				.distribution(dist_bb).set_world(w);
			
			dbcsr::mat_d COB = dbcsr::mat_d::read().filepath("c_bo_B_" + m_mol->name() + ".dat")
				.distribution(dist_boB).set_world(w);
				
			dbcsr::mat_d CVB = dbcsr::mat_d::read().filepath("c_bv_B_" + m_mol->name() + ".dat")
				.distribution(dist_bvB).set_world(w);
			
			dbcsr::mat_d POB = dbcsr::mat_d::read().filepath("p_bb_B_" + m_mol->name() + ".dat")
				.distribution(dist_bb).set_world(w);
				
			dbcsr::mat_d PVB = dbcsr::mat_d::read().filepath("pv_bb_B_" + m_mol->name() + ".dat")
				.distribution(dist_bb).set_world(w);
				
			m_f_bb_B = FB.get_smatrix();
			m_c_bo_B = COB.get_smatrix();
			m_c_bv_B = CVB.get_smatrix();
			m_po_bb_B = POB.get_smatrix();
			m_pv_bb_B = PVB.get_smatrix();
			
		}
		
		read_vector(m_eps_occ_A, m_mol->name(), "eps_occ_A");
		read_vector(m_eps_vir_A, m_mol->name(), "eps_vir_A");
		
		if (read_beta) {
		
			read_vector(m_eps_occ_B, m_mol->name(), "eps_occ_B");
			read_vector(m_eps_vir_B, m_mol->name(), "eps_vir_B");
			
		}
		
		std::cout << "Done Reading" << std::endl;
		dbcsr::print(*m_c_bo_A);
		
	}
};

using shf_wfn = std::shared_ptr<hf_wfn>;

}
	
#endif
	
	
