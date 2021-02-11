#include "fock/fockmod.hpp"
#include "ints/screening.hpp"
#include "utils/registry.hpp"
#include "ints/aoloader.hpp"
#include "math/linalg/LLT.hpp"
#include "math/solvers/hermitian_eigen_solver.hpp"
#include "fock/fock_defaults.hpp"
#include <dbcsr_btensor.hpp>

namespace fock {
	
fockmod::fockmod (dbcsr::world iworld, desc::shared_molecule imol, desc::options iopt) :
	m_world(iworld),
	m_mol(imol),
	m_opt(iopt),
	LOG(m_world.comm(),m_opt.get<int>("print", FOCK_PRINT_LEVEL)),
	TIME(m_world.comm(), "Fock Builder", LOG.global_plev()),
	m_ao(iworld, imol, iopt)
{
	// set up tensors
	auto b = m_mol->dims().b();
	m_f_bb_A = dbcsr::create<double>()
		.set_world(m_world)
		.name("f_bb_A")
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::symmetric)
		.get();
		
	if (m_mol->nele_alpha() != m_mol->nele_beta() && m_mol->nele_beta() != 0) {
		m_f_bb_B = dbcsr::create_template<double>(*m_f_bb_A)
			.name("f_bb_B").get();
	}
	
}

void fockmod::init() {
	
	std::string j_method_str = m_opt.get<std::string>("build_J", FOCK_BUILD_J);
	std::string k_method_str = m_opt.get<std::string>("build_K", FOCK_BUILD_K);
	std::string metric_str = m_opt.get<std::string>("df_metric", FOCK_METRIC);
	std::string eris_mem = m_opt.get<std::string>("eris", FOCK_ERIS);
		
	jmethod jmet = str_to_jmethod(j_method_str);
	kmethod kmet = str_to_kmethod(k_method_str);	
	ints::metric metr = ints::str_to_metric(metric_str);
		
	// jload
	
	load_jints(jmet, metr, m_ao);
	
	// kload 
	
	load_kints(kmet, metr, m_ao);
		
	m_ao.compute();
		
	LOG.os<>("Setting up JK builder.\n");
	LOG.os<>("J method: ", j_method_str, '\n');
	LOG.os<>("K method: ", k_method_str, '\n');
	
	int nprint = LOG.global_plev();
	
	int noccbatches = m_opt.get<int>("occ_nbatches", FOCK_NOCCBATCHES);
	
	std::shared_ptr<J> jbuilder = create_j()
		.world(m_world)
		.mol(m_mol)
		.metric(metr)
		.method(jmet)
		.aoloader(m_ao)
		.print(nprint)
		.get();
	
	std::shared_ptr<K> kbuilder = create_k()
		.world(m_world)
		.mol(m_mol)
		.metric(metr)
		.method(kmet)
		.aoloader(m_ao)
		.print(nprint)
		.occ_nbatches(noccbatches)
		.get();
	
	m_J_builder = jbuilder;
	m_K_builder = kbuilder;
	
	m_J_builder->set_sym(true);
	m_K_builder->set_sym(true);
	
	LOG.os<>("Initializing J...\n");
	m_J_builder->init();
	LOG.os<>("Initializing K... \n");
	m_K_builder->init();
	
	LOG.os<>("Finished setting up JK builder \n \n");
	
}

void fockmod::compute(bool SAD_iter, int rank) {
	
	TIME.start();
	
	m_J_builder->set_SAD(SAD_iter,rank);
	m_K_builder->set_SAD(SAD_iter,rank);
	
	auto& t_j = TIME.sub("J builder");
	auto& t_k = TIME.sub("K builder");
	
	m_J_builder->set_coeff_alpha(m_c_A);
	m_J_builder->set_coeff_beta(m_c_B);
	m_J_builder->set_density_alpha(m_p_A);
	m_J_builder->set_density_beta(m_p_B);
	
	m_K_builder->set_coeff_alpha(m_c_A);
	m_K_builder->set_coeff_beta(m_c_B);
	m_K_builder->set_density_alpha(m_p_A);
	m_K_builder->set_density_beta(m_p_B);
	
	LOG.os<1>("Computing coulomb matrix.\n");
	t_j.start();
	m_J_builder->compute_J();
	t_j.finish();
		
	LOG.os<1>("Computing exchange matrix.\n");
	t_k.start();
	m_K_builder->compute_K();
	t_k.finish();
	
	auto Jmat = m_J_builder->get_J();
	
	auto KmatA = m_K_builder->get_K_A();
	auto KmatB = m_K_builder->get_K_B();
	
	m_f_bb_A->add(0.0,1.0,*m_core);
	m_f_bb_A->add(1.0,1.0,*Jmat);
	m_f_bb_A->add(1.0,1.0,*KmatA);
	//dbcsr::copy_tensor_to_matrix(*Jtensor,*m_f_bb_A,true);
	//dbcsr::copy_tensor_to_matrix(*KtensorA,*m_f_bb_A,true);
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_f_bb_A);
	}
	
	if (m_f_bb_B) {
		m_f_bb_B->add(0.0,1.0,*m_core);
		m_f_bb_B->add(1.0,1.0,*Jmat);
		m_f_bb_B->add(1.0,1.0,*KmatB);
		//dbcsr::copy_tensor_to_matrix(*Jtensor,*m_f_bb_B,true);
		//dbcsr::copy_tensor_to_matrix(*KtensorB,*m_f_bb_B,true);
		
		if (LOG.global_plev() >= 2) {
			dbcsr::print(*m_f_bb_B);
		}
	}
	
	TIME.finish();
		
}

} // end namespace
