#include "fock/jkbuilder.hpp"
#include "fock/fock_defaults.hpp"
#include "math/linalg/LLT.hpp"
#include <dbcsr_tensor_ops.hpp>

namespace fock {
	
JK_common::JK_common(dbcsr::cart w, desc::shared_molecule mol, int print, std::string name) :
	m_cart(w), m_mol(mol),
	LOG(m_cart.comm(),print),
	TIME(m_cart.comm(), name, print) {}
	
void J::init_base() {
	
	// set up J
	auto b = m_mol->dims().b();
	
	m_J = dbcsr::matrix<>::create()
		.name("J_bb")
		.set_cart(m_cart)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type((m_sym) ? dbcsr::type::symmetric : dbcsr::type::no_symmetry)
		.build();
	
}

void K::init_base() {
	
	// set up K's
	auto b = m_mol->dims().b();
	
	m_K_A = dbcsr::matrix<>::create()
		.name("K_bb_A")
		.set_cart(m_cart)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type((m_sym) ? dbcsr::type::symmetric : dbcsr::type::no_symmetry)
		.build();
		
	if (m_p_B) {
		m_K_B = dbcsr::matrix<>::create()
			.name("K_bb_B")
			.set_cart(m_cart)
			.row_blk_sizes(b)
			.col_blk_sizes(b)
			.matrix_type((m_sym) ? dbcsr::type::symmetric : dbcsr::type::no_symmetry)
			.build();
	}
	
}
	
} // end namespace
