#ifndef HF_MOD_H
#define HF_MOD_H

#include "desc/molecule.h"
#include "desc/options.h"
#include "hf/hf_wfn.h"
#include <dbcsr_conversions.hpp>
#include "utils/mpi_time.h"

#include <mpi.h>
#include <memory>
#include <iostream>

using mat_d = dbcsr::mat_d;
using smat_d = dbcsr::smat_d;

namespace hf {
	
class hfmod {
private:
	
	// descriptors
	desc::smolecule m_mol;
	desc::options m_opt;
	dbcsr::world m_world;
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	// options
	bool m_restricted;
	bool m_nobetaorb;
	bool m_diis;
	bool m_diis_beta;
	std::string m_guess;
	int m_max_iter;
	double m_scf_threshold;
	bool m_locc, m_lvir;
	
	// results
	double m_nuc_energy;
	double m_scf_energy;
	
	// other
	int m_SAD_rank;
	
	dbcsr::shared_matrix<double> m_s_bb, //overlap
		  m_v_bb, // nuclear reulsion
		  m_t_bb, // kinetic
		  m_core_bb, // core hamiltonian
		  m_x_bb, // orthogonalizing matrix
		  m_f_bb_A, // alpha fock mat
		  m_p_bb_A, // alpha/beta density matrix
		  m_pv_bb_A,
		  m_c_bm_A; // alpha/beta coefficient matrix	  
				  
	dbcsr::shared_matrix<double> m_f_bb_B, m_p_bb_B, m_pv_bb_B, m_c_bm_B;
	
	svector<double> m_eps_A, m_eps_B;
	
	void compute_nucrep();
	void one_electron();
	
	void compute_guess();
	dbcsr::shared_matrix<double> compute_errmat(dbcsr::shared_matrix<double>& F, 
		dbcsr::shared_matrix<double>& P, dbcsr::shared_matrix<double>& S, std::string x);
	void diag_fock();
	
	void compute_scf_energy();
	
	void compute_virtual_density();

public:

	hfmod(dbcsr::world wrd, desc::smolecule mol, desc::options opt);
	
	hfmod() = delete;
	hfmod(hfmod& hfmod_in) = delete;
	
	~hfmod();
	
	void compute();	
	
	hf::shared_hf_wfn wfn() { 
		
		// separate occupied and virtual coefficient matrix
		auto separate = [&](dbcsr::shared_matrix<double>& in, 
			dbcsr::shared_matrix<double>& out_o, 
			dbcsr::shared_matrix<double>& out_v, std::string x) {
				
				vec<int> o, v, b;
				int noblks, nvblks;
				int nocc, nvir;
				
				if (x == "A") { 
					o = m_mol->dims().oa();
					v = m_mol->dims().va(); 
					nocc = m_mol->nocc_alpha();
					nvir = m_mol->nvir_alpha();
				} else {
					o = m_mol->dims().ob(); 
					v = m_mol->dims().vb();
					nocc = m_mol->nocc_beta();
					nvir = m_mol->nvir_beta();
				}
				
				b = m_mol->dims().b();
				int nbas = m_mol->c_basis()->nbf();
				
				auto eigen_cbm = dbcsr::matrix_to_eigen(*in);
				
				Eigen::MatrixXd eigen_cbo = eigen_cbm.block(0,0,nbas,nocc);
				Eigen::MatrixXd eigen_cbv = eigen_cbm.block(0,nocc,nbas,nvir);
				
				//std::cout << eigen_cbo << std::endl;
				//std::cout << eigen_cbv << std::endl;
				
				if (nocc != 0)
					out_o = dbcsr::eigen_to_matrix(eigen_cbo, m_world, "c_bo_"+x, b, o, dbcsr::type::no_symmetry);
				if (nvir != 0) 
					out_v = dbcsr::eigen_to_matrix(eigen_cbv, m_world, "c_bv_"+x, b, v, dbcsr::type::no_symmetry);
				
		};
		
		std::shared_ptr<std::vector<double>> epsoA, epsoB, epsvA, epsvB;
		
		epsoA = std::make_shared<std::vector<double>>(m_eps_A->begin(), 
			m_eps_A->begin() + m_mol->nocc_alpha());
		if (m_eps_B) 
			epsoB = std::make_shared<std::vector<double>>(m_eps_B->begin(), 
				m_eps_B->begin() + m_mol->nocc_beta());
			
		epsvA = std::make_shared<std::vector<double>>(m_eps_A->begin() 
			+ m_mol->nocc_alpha(), m_eps_A->end());
		if (m_eps_B) 
			epsvB = std::make_shared<std::vector<double>>(m_eps_B->begin() 
				+ m_mol->nocc_beta(), m_eps_B->end());
			
		smat_d cboA, cboB, cbvA, cbvB;
		separate(m_c_bm_A, cboA, cbvA, "A");
				
		if (m_c_bm_B) separate(m_c_bm_B, cboB, cbvB, "B");
		
		auto out = std::make_shared<hf_wfn>(m_mol, cboA, cboB, cbvA, cbvB,
			epsoA, epsoB, epsvA, epsvB, m_scf_energy, m_nuc_energy,
			m_nuc_energy + m_scf_energy);
		
		return out; 
	
	}
	
};

} // end namespace hf

#endif
