#include "fock/fockmod.h"
#include "ints/screening.h"
#include "utils/registry.h"
#include "ints/aoloader.h"
#include "math/linalg/LLT.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include "fock/fock_defaults.h"
#include <dbcsr_btensor.hpp>

namespace fock {
	
fockmod::fockmod (dbcsr::world iworld, desc::smolecule imol, desc::options iopt) :
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
		m_f_bb_B = dbcsr::create_template<double>(m_f_bb_A)
			.name("f_bb_B").get();
	}
	
}

void fockmod::init() {
	
	std::string j_method = m_opt.get<std::string>("build_J", FOCK_BUILD_J);
	std::string k_method = m_opt.get<std::string>("build_K", FOCK_BUILD_K);
	std::string metric = m_opt.get<std::string>("df_metric", FOCK_METRIC);
	std::string eris_mem = m_opt.get<std::string>("eris", FOCK_ERIS);
			
	// set J
	if (j_method == "exact") {
		
		J* builder = new EXACT_J(m_world, m_opt);
		m_J_builder.reset(builder);
		
		m_ao.request(ints::key::coul_bbbb);
		
	} else if (j_method == "batchdf") {
		
		J* builder = new BATCHED_DF_J(m_world,m_opt);
		m_J_builder.reset(builder);
		
		m_ao.request(ints::key::scr_xbb);
		
		if (metric == "coulomb") {
			m_ao.request(ints::key::coul_xx);
			m_ao.request(ints::key::coul_xx_inv);
			m_ao.request(ints::key::coul_xbb);
		} else if (metric == "erfc_coulomb") {
			m_ao.request(ints::key::erfc_xx);
			m_ao.request(ints::key::erfc_xx_inv);
			m_ao.request(ints::key::erfc_xbb);
		}
	
	} else if (j_method == "batchqr") {
		
		J* builder = new BATCHED_DF_J(m_world,m_opt);
		m_J_builder.reset(builder);
		
		m_ao.request(ints::key::scr_xbb);
		
		m_ao.request(ints::key::coul_xx);
		m_ao.request(ints::key::ovlp_xx);
		m_ao.request(ints::key::ovlp_xx_inv);
		m_ao.request(ints::key::dfit_qr_xbb);
	
	} else {
		
		throw std::runtime_error("Unknown J method: " + j_method);
		
	}
		
	
	// set K
	if (k_method == "exact") {
		
		K* builder = new EXACT_K(m_world, m_opt);
		m_K_builder.reset(builder);
		
		m_ao.request(ints::key::coul_bbbb);
		
	} else if (k_method == "batchdfao") {
		
		K* builder = new BATCHED_DFAO_K(m_world,m_opt);
		m_K_builder.reset(builder);
		
		m_ao.request(ints::key::scr_xbb);
		
		if (metric == "coulomb") {
			m_ao.request(ints::key::coul_xx);
			m_ao.request(ints::key::coul_xx_inv);
			m_ao.request(ints::key::coul_xbb);
			m_ao.request(ints::key::dfit_coul_xbb);
		} else if (metric == "erfc_coulomb") {
			m_ao.request(ints::key::erfc_xx);
			m_ao.request(ints::key::erfc_xx_inv);
			m_ao.request(ints::key::erfc_xbb);
			m_ao.request(ints::key::dfit_erfc_xbb);
		}
		
	} else if (k_method == "batchdfmo") {
		
		K* builder = new BATCHED_DFMO_K(m_world,m_opt);
		m_K_builder.reset(builder);
		
		m_ao.request(ints::key::scr_xbb);
		
		if (metric == "coulomb") {
			m_ao.request(ints::key::coul_xx);
			m_ao.request(ints::key::coul_xx_invsqrt);
			m_ao.request(ints::key::coul_xbb);
		} else if (metric == "erfc_coulomb") {
			throw std::runtime_error("DFMO with erfc disabled.");
		}
		
	} else if (k_method == "batchpari") {
		
		K* builder = new BATCHED_PARI_K(m_world,m_opt);
		m_K_builder.reset(builder);
		
		m_ao.request(ints::key::scr_xbb);
		
		m_ao.request(ints::key::coul_xx);
		m_ao.request(ints::key::coul_xx_inv);
		m_ao.request(ints::key::coul_xbb);
		m_ao.request(ints::key::dfit_pari_xbb);
		
	} else if (k_method == "batchqr") {
	
		K* builder = new BATCHED_DFMEM_K(m_world,m_opt);
		m_K_builder.reset(builder);
		
		m_ao.request(ints::key::scr_xbb);
		
		m_ao.request(ints::key::coul_xx);
		m_ao.request(ints::key::ovlp_xx);
		m_ao.request(ints::key::ovlp_xx_inv);
		m_ao.request(ints::key::dfit_qr_xbb);
	
	} else {
		
		throw std::runtime_error("Unknown K method: " + k_method);
		
	}
	
	m_ao.compute();
	auto aoreg = m_ao.get_registry();
	
	LOG.os<>("Setting up JK builder.\n");
	LOG.os<>("J method: ", j_method, '\n');
	LOG.os<>("K method: ", k_method, '\n');
	
	m_J_builder->set_density_alpha(m_p_A);
	m_J_builder->set_density_beta(m_p_B);
	m_J_builder->set_coeff_alpha(m_c_A);
	m_J_builder->set_coeff_beta(m_c_B);
	m_J_builder->set_mol(m_mol);
	
	m_K_builder->set_density_alpha(m_p_A);
	m_K_builder->set_density_beta(m_p_B);
	m_K_builder->set_coeff_alpha(m_c_A);
	m_K_builder->set_coeff_beta(m_c_B);
	m_K_builder->set_mol(m_mol);
	
	// registries
	
	util::key_registry<Jkey> jreg;
	util::key_registry<Kkey> kreg;
	
	if (j_method == "exact") {
		jreg.add(aoreg, ints::key::coul_bbbb, Jkey::eri_bbbb);
	} else if (j_method == "batchdf") {
		if (metric == "coulomb") {
			jreg.add(aoreg, ints::key::coul_xbb, Jkey::eri_xbb);
			jreg.add(aoreg, ints::key::coul_xx_inv, Jkey::v_inv_xx);
		} else {
			jreg.add(aoreg, ints::key::erfc_xbb, Jkey::eri_xbb);
			jreg.add(aoreg, ints::key::erfc_xx_inv, Jkey::v_inv_xx);
		}
	} else if (j_method == "batchqr") {
		jreg.add(aoreg, ints::key::dfit_qr_xbb, Jkey::eri_xbb);
		jreg.add(aoreg, ints::key::coul_xx, Jkey::v_inv_xx);
	}
	
	// set K
	if (k_method == "exact") {
		
		kreg.add(aoreg, ints::key::coul_bbbb, Kkey::eri_bbbb);
		
	} else if (k_method == "batchdfao") {
		
		if (metric == "coulomb") {
			kreg.add(aoreg, ints::key::coul_xbb, Kkey::eri_xbb);
			kreg.add(aoreg, ints::key::dfit_coul_xbb, Kkey::dfit_xbb);
		} else if (metric == "erfc_coulomb") {
			kreg.add(aoreg, ints::key::erfc_xbb, Kkey::eri_xbb);
			kreg.add(aoreg, ints::key::dfit_erfc_xbb, Kkey::dfit_xbb);
		}
		
	} else if (k_method == "batchdfmo") {
		
		kreg.add(aoreg, ints::key::coul_xbb, Kkey::eri_xbb);
		kreg.add(aoreg, ints::key::coul_xx_invsqrt, Kkey::v_inv_xx_sqrt);
		
	} else if (k_method == "batchpari") {
		
		kreg.add(aoreg, ints::key::coul_xx, Kkey::v_xx);
		kreg.add(aoreg, ints::key::coul_xbb, Kkey::eri_xbb);
		kreg.add(aoreg, ints::key::dfit_pari_xbb, Kkey::dfit_pari_xbb);
		
	} else if (k_method == "batchqr") {
	
		kreg.add(aoreg, ints::key::coul_xx, Kkey::v_xx);
		kreg.add(aoreg, ints::key::dfit_qr_xbb, Kkey::dfit_xbb);
	
	}
	
	m_J_builder->set_reg(jreg);
	m_K_builder->set_reg(kreg);
	
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
