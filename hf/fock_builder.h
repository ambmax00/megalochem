#ifndef HF_FOCK_BUILDER_H
#define HF_FOCK_BUILDER_H

#include "math/tensor/dbcsr.hpp"
#include "desc/options.h"
#include "desc/molecule.h"
#include "utils/params.hpp"
#include "utils/mpi_time.h"
#include <mpi.h>

namespace hf {
	
class fockbuilder {
public:

	struct compute_param {
		dbcsr::stensor<2> core;
		dbcsr::stensor<2> c_A;
		dbcsr::stensor<2> p_A;
		dbcsr::stensor<2> c_B;
		dbcsr::stensor<2> p_B;
		optional<bool,val> SAD_iter;
	};
	
private:

	desc::smolecule m_mol;
	desc::options m_opt;
	MPI_Comm m_comm;
	util::mpi_log LOG;
	util::mpi_time& TIME;
	
	//options
	bool m_use_df;
	bool m_restricted;
	
	// pure 3c2e ints
	dbcsr::stensor<3> m_3c2e_ints;
	dbcsr::stensor<2> m_inv_xx; // <- inverse metric
	dbcsr::stensor<2> m_sqrtinv_xx; // <- square root inverse metric
	
	// 2 electron ints
	dbcsr::stensor<4> m_2e_ints;
	
	dbcsr::stensor<2> m_j_bb;
	dbcsr::stensor<2> m_k_bb_A;
	dbcsr::stensor<2> m_f_bb_A;
	
	dbcsr::stensor<2> m_k_bb_B;
	dbcsr::stensor<2> m_f_bb_B;
	
	void build_j(compute_param&& p);
	void build_k(compute_param&& p);
	
public:

	fockbuilder(desc::smolecule mol, desc::options opt, MPI_Comm comm, 
		int print, util::mpi_time& TIME_IN);
	
	~fockbuilder() {}
			
	void compute(compute_param&& p);
	
	dbcsr::stensor<2> fock_alpha() {
		return m_f_bb_A;
	}
	
	dbcsr::stensor<2> fock_beta() {
		return m_f_bb_B;
	}
	
};
	
} // end namespace

#endif
