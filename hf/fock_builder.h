#ifndef HF_FOCK_BUILDER_H
#define HF_FOCK_BUILDER_H

#include "tensor/dbcsr.hpp"
#include "desc/options.h"
#include "desc/molecule.h"
#include "utils/params.hpp"
#include "utils/mpi_time.h"
#include <mpi.h>

namespace hf {
	
class fockbuilder {
private:

	desc::smolecule m_mol;
	desc::options m_opt;
	MPI_Comm m_comm;
	util::mpi_log LOG;
	util::mpi_time& TIME;

	dbcsr::stensor<2> m_core;
	dbcsr::stensor<2> m_c_A;
	dbcsr::stensor<2> m_p_A;
	dbcsr::stensor<2> m_c_B;
	dbcsr::stensor<2> m_p_B;
	bool m_SAD_iter = false;
		
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
	
	void build_j();
	void build_k();
	
public:

	inline fockbuilder& core(dbcsr::stensor<2>& i_core) { m_core = i_core; return *this; }
	inline fockbuilder& c_A(dbcsr::stensor<2>& i_c_A) { m_c_A = i_c_A; return *this; }
	inline fockbuilder& c_B(dbcsr::stensor<2>& i_c_B) { m_c_B = i_c_B; return *this; }
	inline fockbuilder& p_A(dbcsr::stensor<2>& i_p_A) { m_p_A = i_p_A; return *this; }
	inline fockbuilder& p_B(dbcsr::stensor<2>& i_p_B) { m_p_B = i_p_B; return *this; }
	inline fockbuilder& SAD(bool val) { m_SAD_iter = val; return *this; }
	
	fockbuilder(desc::smolecule mol, desc::options opt, MPI_Comm comm, 
		int print, util::mpi_time& TIME_IN);
	
	~fockbuilder() {}
			
	void compute();
	
	dbcsr::stensor<2> fock_alpha() {
		return m_f_bb_A;
	}
	
	dbcsr::stensor<2> fock_beta() {
		return m_f_bb_B;
	}
	
};
	
} // end namespace

#endif
