#ifndef HF_MOD_H
#define HF_MOD_H

#include "desc/molecule.h"
#include "desc/options.h"
#include "math/tensor/dbcsr.hpp"
#include "utils/mpi_time.h"

#include <mpi.h>
#include <memory>
#include <iostream>

template <typename T>
using svector = std::shared_ptr<std::vector<T>>;

namespace hf {
	
class hfmod {
	
	template <int N>
	using tensor_ptr = std::shared_ptr<dbcsr::tensor<N,double>>;
	
private:
	
	// descriptors
	desc::smolecule m_mol;
	desc::options m_opt;
	MPI_Comm m_comm;
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	// options
	bool m_restricted;
	bool m_nobeta;
	bool m_diis;
	bool m_diis_beta;
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
	
	svector<double> m_eps_A, m_eps_B;
	
	void compute_nucrep();
	void one_electron();
	
	void compute_guess();
	dbcsr::tensor<2> compute_errmat(dbcsr::tensor<2>& F, dbcsr::tensor<2>& P, 
		dbcsr::tensor<2>& S, std::string x);
	void diag_fock();
	
	void compute_scf_energy();

public:

	hfmod(desc::smolecule mol, desc::options opt, MPI_Comm comm);
	
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
	
	desc::smolecule mol() {
		return m_mol;
	}
	
	svector<double> eps_A() {
		return m_eps_A;
	}
	
	svector<double> eps_B() {
		return m_eps_B;
	}
	
	dbcsr::stensor<2> c_bm_A() {
		return m_c_bm_A;
	}
	
	dbcsr::stensor<2> c_bm_B() {
		return m_c_bm_B;
	}
	
	dbcsr::stensor<2> p_bb_A() {
		return m_p_bb_A;
	}
	
	dbcsr::stensor<2> p_bb_B() {
		return m_p_bb_B;
	}
	
	dbcsr::stensor<2> f_bb_A() {
		return m_f_bb_A;
	}
	
	dbcsr::stensor<2> f_bb_B() {
		return m_f_bb_B;
	}
	
	dbcsr::stensor<2> s_bb() {
		return m_s_bb;
	}
	
	double tot_energy() {
		return m_scf_energy + m_nuc_energy;
	}
	
};

} // end namespace hf

#endif
