#ifndef HF_FOCK_BUILDER_H
#define HF_FOCK_BUILDER_H

#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor_ops.hpp>
#include "desc/options.h"
#include "desc/molecule.h"
#include "utils/params.hpp"
#include "utils/mpi_time.h"
#include <mpi.h>

namespace hf {
	
using mat = dbcsr::mat_d;
using smat = dbcsr::smat_d; 

template <int N>
using tensor = dbcsr::tensor<N,double>;

template <int N>
using stensor = dbcsr::stensor<N,double>;
	
class fockbuilder {
private:

	desc::smolecule m_mol;
	desc::options m_opt;
	dbcsr::world m_world;
	util::mpi_log LOG;
	util::mpi_time& TIME;

	//options
	bool m_use_df;
	bool m_restricted;
	bool m_nobetaorb;
	
	// pure 3c2e ints
	stensor<3> m_3c2e_ints;
	stensor<2> m_inv_xx; // <- inverse metric
	stensor<2> m_sqrtinv_xx; // <- square root inverse metric
	
	// 2 electron ints
	stensor<4> m_2e_ints;
	
	stensor<2> m_j_bb;
	stensor<2> m_k_bb_A;
	smat m_f_bb_A;
	
	stensor<2> m_k_bb_B;
	smat m_f_bb_B;
	
	void build_j(stensor<2>& p_A, stensor<2>& p_B);
	void build_k(stensor<2>& p_A, stensor<2>& p_B, stensor<2>& c_A, 
		stensor<2>& c_B, bool SAD);
	
public:
	
	fockbuilder(desc::smolecule mol, desc::options opt, dbcsr::world& w, 
		int print, util::mpi_time& TIME_IN);
	
	~fockbuilder() {}
			
	void compute(smat& core, smat& p_A, smat& c_A, smat& p_B, smat& c_B, bool SAD);
	
	smat fock_alpha() {
		return m_f_bb_A;
	}
	
	smat fock_beta() {
		return m_f_bb_B;
	}
	
};
	
} // end namespace

#endif
