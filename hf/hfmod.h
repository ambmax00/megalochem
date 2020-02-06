#ifndef HF_MOD_H
#define HF_MOD_H

#include "desc/molecule.h"
#include "desc/options.h"
#include "math/tensor/dbcsr.hpp"
#include "utils/mpi_time.h"

#include <mpi.h>
#include <memory>
#include <iostream>

namespace hf {
	
class hfmod {
	
	template <int N>
	using tensor_ptr = std::shared_ptr<dbcsr::tensor<N,double>>;
	
private:
	
	// descriptors
	desc::molecule& m_mol;
	desc::options& m_opt;
	MPI_Comm m_comm;
	util::mpi_log LOG;
	
	// options
	bool m_restricted;
	bool m_nobeta;
	bool m_diis;
	std::string m_guess;
	int m_max_iter;
	double m_scf_threshold;
	
	// results
	double m_nuc_energy;
	double m_scf_energy;
	
	dbcsr::stensor<2> m_s_bb, //overlap
				  m_v_bb, // nuclear reulsion
				  m_t_bb, // kinetic
				  m_core_bb, // core hamiltonian
				  m_x_bb, // orthogonalizing matrix
				  m_f_bb_A, // alpha fock mat
				  m_p_bb_A, // alpha/beta density matrix
				  m_c_bm_A; // alpha/beta coefficient matrix	  
				  
	dbcsr::stensor<2> m_f_bb_B, m_p_bb_B, m_c_bm_B;
	
	void compute_nucrep();
	void one_electron();
	
	void compute_guess();
	dbcsr::tensor<2> compute_errmat(dbcsr::tensor<2>& F, dbcsr::tensor<2>& P, 
		dbcsr::tensor<2>& S, std::string x);
	void diag_fock();
	
	void compute_scf_energy();

public:

	hfmod(desc::molecule& mol, desc::options& opt, MPI_Comm comm);
	
	hfmod() = delete;
	hfmod(hfmod& hfmod_in) = delete;
	
	~hfmod() {
		
		/*
		m_s_bb->destroy();
		m_v_bb.destroy();
		m_t_bb.destroy();
		m_core_bb.destroy();
		m_x_bb.destroy();
		m_f_bb_A.destroy();
		m_p_bb_A.destroy();
		m_c_bm_A.destroy();
		
		if (m_f_bb_B) m_f_bb_B->destroy();
		if (m_p_bb_B) m_p_bb_B->destroy();
		if (m_c_bm_B) m_c_bm_B->destroy();
		*/
	}
	
	void compute();	
	
};

} // end namespace hf

#endif
