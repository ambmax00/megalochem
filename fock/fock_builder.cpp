#include "fock/fockmod.h"
#include "fock/fock_defaults.h"

namespace fock {
	
fockmod::fockmod (dbcsr::world iworld, desc::smolecule imol, desc::options iopt) :
	m_world(iworld),
	m_mol(imol),
	m_opt(iopt),
	LOG(m_world.comm(),m_opt.get<int>("print", FOCK_PRINT_LEVEL)),
	TIME(m_world.comm(), "Fock Builder", LOG.global_plev())
{
	// set up tensors
	auto b = m_mol->dims().b();
	dbcsr::mat_d fA = dbcsr::matrix<>::create().set_world(m_world).name("f_bb_A")
		.row_blk_sizes(b).col_blk_sizes(b).type(dbcsr_type_symmetric);
	
	m_f_bb_A = fA.get_smatrix();
	
	if (m_mol->nele_alpha() != m_mol->nele_beta() && m_mol->nele_beta() != 0) {
		dbcsr::mat_d fB = dbcsr::matrix<>::create_template(*m_f_bb_A).name("f_bb_B");
		m_f_bb_B = fB.get_smatrix();
	}
	
}

void fockmod::init() {
	
	std::string j_method = m_opt.get<std::string>("build_J", FOCK_BUILD_J);
	std::string k_method = m_opt.get<std::string>("build_K", FOCK_BUILD_K);
	
	// set J
	if (j_method == "exact") {
		
		J* builder = new EXACT_J(m_world, m_opt);
		m_J_builder.reset(builder);
		
	}
	
	// set K
	if (k_method == "exact") {
		
		K* builder = new EXACT_K(m_world, m_opt);
		m_K_builder.reset(builder);
		
	}
	
	LOG.os<>("Setting up JK builder.\n");
	LOG.os<>("J method: ", j_method, '\n');
	LOG.os<>("K method: ", k_method, '\n');
	
	std::shared_ptr<ints::aofactory> aofac =
		std::make_shared<ints::aofactory>(*m_mol,m_world);
	
	m_J_builder->set_density_alpha(m_p_A);
	m_J_builder->set_density_beta(m_p_B);
	m_J_builder->set_coeff_alpha(m_c_A);
	m_J_builder->set_coeff_beta(m_c_B);
	m_J_builder->set_factory(aofac);
	
	m_K_builder->set_density_alpha(m_p_A);
	m_K_builder->set_density_beta(m_p_B);
	m_K_builder->set_coeff_alpha(m_c_A);
	m_K_builder->set_coeff_beta(m_c_B);
	m_K_builder->set_factory(aofac);
	
	m_J_builder->init();
	m_K_builder->init();
	
	LOG.os<>("Finished setting up JK builder \n \n");
	
}

void fockmod::compute() {
	
	LOG.os<1>("Computing coulomb matrix.\n");
	m_J_builder->compute_J();
	LOG.os<1>("Computing exchange matrix.\n");
	m_K_builder->compute_K();
	
	auto Jtensor = m_J_builder->get_J();
	auto KtensorA = m_K_builder->get_K_A();
	auto KtensorB = m_K_builder->get_K_B();
	
	m_f_bb_A->add(0.0,1.0,*m_core);
	dbcsr::copy_tensor_to_matrix(*Jtensor,*m_f_bb_A,true);
	dbcsr::copy_tensor_to_matrix(*KtensorA,*m_f_bb_A,true);
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_f_bb_A);
	}
	
	if (m_f_bb_B) {
		m_f_bb_B->add(0.0,1.0,*m_core);
		dbcsr::copy_tensor_to_matrix(*Jtensor,*m_f_bb_B,true);
		dbcsr::copy_tensor_to_matrix(*KtensorB,*m_f_bb_B,true);
		
		if (LOG.global_plev() >= 2) {
			dbcsr::print(*m_f_bb_B);
		}
	}
		
}

} // end namespace
