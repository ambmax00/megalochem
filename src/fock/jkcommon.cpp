#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include "math/linalg/LLT.h"
#include <dbcsr_tensor_ops.hpp>

namespace fock {
	
JK_common::JK_common(dbcsr::world& w, desc::options opt, std::string name) :
	m_world(w), m_opt(opt),
	LOG(m_world.comm(),m_opt.get<int>("print", FOCK_PRINT_LEVEL)),
	TIME(m_world.comm(), name, LOG.global_plev()) {}
	
void J::init() {
	
	// set up J
	auto b = m_mol->dims().b();
	
	m_J = dbcsr::create<double>()
		.name("J_bb")
		.set_world(m_world)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type((m_sym) ? dbcsr::type::symmetric : dbcsr::type::no_symmetry)
		.get();
	
}

void K::init() {
	
	// set up K's
	auto b = m_mol->dims().b();
	
	m_K_A = dbcsr::create<double>()
		.name("K_bb_A")
		.set_world(m_world)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type((m_sym) ? dbcsr::type::symmetric : dbcsr::type::no_symmetry)
		.get();
		
	if (m_p_B) {
		m_K_B = dbcsr::create<double>()
			.name("K_bb_B")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(b)
			.matrix_type((m_sym) ? dbcsr::type::symmetric : dbcsr::type::no_symmetry)
			.get();
	}
	
}
	
} // end namespace
