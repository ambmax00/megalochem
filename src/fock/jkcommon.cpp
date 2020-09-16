#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include "math/linalg/LLT.h"
#include <dbcsr_tensor_ops.hpp>

namespace fock {

void JK::init() {
	
	// set up K's
	auto b = m_mol->dims().b();
	
	m_mat_A = dbcsr::create<double>()
		.name("mat_bb_A")
		.set_world(m_world)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type((m_sym) ? dbcsr::type::symmetric : dbcsr::type::no_symmetry)
		.get();
	
	auto p_B = m_reg.get_matrix<double>("p_bb_B");
	
	if (p_B) {
		m_mat_B = dbcsr::create<double>()
			.name("mat_bb_B")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(b)
			.matrix_type((m_sym) ? dbcsr::type::symmetric : dbcsr::type::no_symmetry)
			.get();
	}
	
}
	
} // end namespace
