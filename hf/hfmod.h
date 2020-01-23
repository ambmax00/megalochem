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
	std::string m_guess;
	int m_max_iter;
	double m_scf_threshold;
	
	// results
	double m_nuc_energy;
	double m_scf_energy;
	
	dbcsr::tensor<2> m_s_bb, //overlap
				  m_v_bb, // nuclear reulsion
				  m_t_bb, // kinetic
				  m_core_bb, // core hamiltonian
				  m_x_bb, // orthogonalizing matrix
				  m_f_bb_A, m_f_bb_B, // alpha/beta fock mat
				  m_p_bb_A, m_p_bb_B, // alpha/beta density matrix
				  m_c_bm_A, m_c_bm_B; // alpha/beta coefficient matrix	  
	
	void compute_nucrep();
	void one_electron();
	
	void compute_guess();
	void diag_fock();
	
	void calc_scf_energy();

public:

	hfmod(desc::molecule& mol, desc::options& opt, MPI_Comm comm);
	
	hfmod() = delete;
	hfmod(hfmod& hfmod_in) = delete;
	
	~hfmod() {}
	
	void compute();	
	
};

} // end namespace hf

#endif
