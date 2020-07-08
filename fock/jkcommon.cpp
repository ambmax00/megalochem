#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include "math/linalg/LLT.h"
#include <dbcsr_tensor_ops.hpp>

namespace fock {
	
JK_common::JK_common(dbcsr::world& w, desc::options opt) :
	m_world(w), m_opt(opt),
	LOG(m_world.comm(),m_opt.get<int>("print", FOCK_PRINT_LEVEL)),
	TIME(m_world.comm(), "JK Builder", LOG.global_plev()) {}
	
void J::init() {
	
	// set up J
	auto b = m_p_A->row_blk_sizes();
	
	m_J = std::make_shared<dbcsr::mat_d>(
		dbcsr::mat_d::create().name("J_bb").set_world(m_world)
		.row_blk_sizes(b).col_blk_sizes(b).type(dbcsr_type_symmetric));
	
}

void K::init() {
	
	// set up K's
	auto b = m_p_A->row_blk_sizes();
	
	m_K_A = std::make_shared<dbcsr::mat_d>(
		dbcsr::mat_d::create().name("K_bb_A").set_world(m_world)
		.row_blk_sizes(b).col_blk_sizes(b).type(dbcsr_type_symmetric));
		
	if (m_p_B) m_K_B = std::make_shared<dbcsr::mat_d>(
		dbcsr::mat_d::create().name("K_bb_B").set_world(m_world)
		.row_blk_sizes(b).col_blk_sizes(b).type(dbcsr_type_symmetric));
	
}
	
} // end namespace
