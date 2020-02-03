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
		required<dbcsr::tensor<2>,ref> core;
		required<dbcsr::tensor<2>,ref> c_A;
		required<dbcsr::tensor<2>,ref> p_A;
		optional<dbcsr::tensor<2>,ref> c_B;
		optional<dbcsr::tensor<2>,ref> p_B;
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
	optional<dbcsr::tensor<3>,val> m_3c2e_ints;
	optional<dbcsr::tensor<2>,val> m_inv_xx; // <- inverse metric
	optional<dbcsr::tensor<2>,val> m_sqrtinv_xx; // <- square root inverse metric
	
	// 2 electron ints
	optional<dbcsr::tensor<4>,val> m_2e_ints;
	
	dbcsr::tensor<2> m_j_bb;
	dbcsr::tensor<2> m_k_bb_A;
	dbcsr::tensor<2> m_f_bb_A;
	
	optional<dbcsr::tensor<2>,val> m_k_bb_B;
	optional<dbcsr::tensor<2>,val> m_f_bb_B;
	
	void build_j(compute_param&& p);
	void build_k(compute_param&& p);
	
public:

	fockbuilder(desc::molecule& mol, desc::options& opt, MPI_Comm comm, int print = 0);
			
	void compute(compute_param&& p);
	
	dbcsr::tensor<2>& fock_alpha() {
		return m_f_bb_A;
	}
	
};
	
} // end namespace

#endif
