#ifndef HF_FOCK_BUILDER_H
#define HF_FOCK_BUILDER_H

#include "math/tensor/dbcsr.hpp"
#include "desc/options.h"
#include "desc/molecule.h"
#include "utils/params.hpp"
#include "utils/mpi_log.h"
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

	desc::molecule& m_mol;
	desc::options& m_opt;
	MPI_Comm m_comm;
	util::mpi_log LOG;
	
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

	fockbuilder(desc::molecule& mol, desc::options& opt, MPI_Comm comm, int print = 0);
	
	~fockbuilder() {
		if (m_3c2e_ints) m_3c2e_ints->destroy();
		if (m_inv_xx) m_inv_xx->destroy();
		if (m_sqrtinv_xx) m_sqrtinv_xx->destroy();
		if (m_2e_ints) m_2e_ints->destroy();
		m_j_bb->destroy();
		m_k_bb_A->destroy();
		m_f_bb_A->destroy();
		if (m_k_bb_B) m_k_bb_B->destroy();
		if (m_f_bb_B) m_f_bb_B->destroy();
	}
			
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
