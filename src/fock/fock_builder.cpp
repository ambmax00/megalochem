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
	
	std::string j_method_str = m_opt.get<std::string>("build_J", FOCK_BUILD_J);
	std::string k_method_str = m_opt.get<std::string>("build_K", FOCK_BUILD_K);
	std::string metric_str = m_opt.get<std::string>("df_metric", FOCK_METRIC);
	std::string eris_mem = m_opt.get<std::string>("eris", FOCK_ERIS);
		
	jmethod jmet = str_to_jmethod(j_method_str);
	kmethod kmet = str_to_kmethod(k_method_str);	
	ints::metric metr = ints::str_to_metric(metric_str);
		
	// set J
	if (jmet == jmethod::exact) {
		
		m_ao.request(ints::key::coul_bbbb, true);
		
	} else if (jmet == jmethod::dfao) {
		
		m_ao.request(ints::key::scr_xbb,true);
		
		if (metr == ints::metric::coulomb) {
			m_ao.request(ints::key::coul_xx, false);
			m_ao.request(ints::key::coul_xx_inv, true);
			m_ao.request(ints::key::coul_xbb, true);
		} else if (metr == ints::metric::erfc_coulomb) {
			m_ao.request(ints::key::erfc_xx, false);
			m_ao.request(ints::key::erfc_xx_inv, true);
			m_ao.request(ints::key::erfc_xbb, true);
		} else if (metr == ints::metric::qr_fit) {
			m_ao.request(ints::key::coul_xx, true);
			m_ao.request(ints::key::ovlp_xx, false);
			m_ao.request(ints::key::ovlp_xx_inv, false);
			m_ao.request(ints::key::qr_xbb, true);
		}
		
	} else {
		
		throw std::runtime_error("Unknown J method: " + j_method_str);
		
	}
		
	
	// set K
	if (kmet == kmethod::exact) {
		
		m_ao.request(ints::key::coul_bbbb,true);
		
	} else if (kmet == kmethod::dfao) {
		
		m_ao.request(ints::key::scr_xbb,true);
		
		if (metr == ints::metric::coulomb) {
			m_ao.request(ints::key::coul_xx,false);
			m_ao.request(ints::key::coul_xx_inv,false);
			m_ao.request(ints::key::coul_xbb,true);
			m_ao.request(ints::key::dfit_coul_xbb,true);
		} else if (metr == ints::metric::erfc_coulomb) {
			m_ao.request(ints::key::erfc_xx,false);
			m_ao.request(ints::key::erfc_xx_inv,false);
			m_ao.request(ints::key::erfc_xbb,true);
			m_ao.request(ints::key::dfit_erfc_xbb,true);
		} else if (metr == ints::metric::qr_fit) {
			m_ao.request(ints::key::coul_xx, true);
			m_ao.request(ints::key::ovlp_xx, false);
			m_ao.request(ints::key::ovlp_xx_inv, false);
			m_ao.request(ints::key::qr_xbb, true);
			m_ao.request(ints::key::dfit_qr_xbb, true);
		}
		
	} else if (kmet == kmethod::dfmo) {
		
		m_ao.request(ints::key::scr_xbb,true);
		
		if (metr == ints::metric::coulomb) {
			m_ao.request(ints::key::coul_xx,false);
			m_ao.request(ints::key::coul_xx_invsqrt,true);
			m_ao.request(ints::key::coul_xbb,true);
		} else {
			throw std::runtime_error("DFMO with non coulomb metric disabled.");
		}
		
	} else if (kmet == kmethod::dfmem) {
		
		m_ao.request(ints::key::scr_xbb,true);
		
		if (metr == ints::metric::coulomb) {
			m_ao.request(ints::key::coul_xx,false);
			m_ao.request(ints::key::coul_xx_inv,false);
			m_ao.request(ints::key::coul_xbb,true);
			m_ao.request(ints::key::dfit_coul_xbb,true);
		} else if (metr == ints::metric::erfc_coulomb) {
			m_ao.request(ints::key::erfc_xx,false);
			m_ao.request(ints::key::erfc_xx_inv,false);
			m_ao.request(ints::key::erfc_xbb,true);
			m_ao.request(ints::key::dfit_erfc_xbb,true);
		} else if (metr == ints::metric::qr_fit) {
			m_ao.request(ints::key::coul_xx, false);
			m_ao.request(ints::key::ovlp_xx, false);
			m_ao.request(ints::key::ovlp_xx_inv, false);
			m_ao.request(ints::key::qr_xbb, false);
			m_ao.request(ints::key::dfit_qr_xbb, true);
		}
		
	} else {
		
		throw std::runtime_error("Unknown K method: " + k_method_str);
		
	}
	
	
	m_ao.request(ints::key::dfit_pari_xbb,false);
	
	m_ao.compute();
	auto aoreg = m_ao.get_registry();
	
	exit(0);
	
	LOG.os<>("Setting up JK builder.\n");
	LOG.os<>("J method: ", j_method_str, '\n');
	LOG.os<>("K method: ", k_method_str, '\n');
	
	std::shared_ptr<J> jbuilder;
	std::shared_ptr<K> kbuilder;
	
	int nprint = LOG.global_plev();
	
	if (jmet == jmethod::exact) {
		
		auto eris = aoreg.get<dbcsr::sbtensor<4,double>>(ints::key::coul_bbbb);
		
		jbuilder = create_EXACT_J(m_world, m_mol, nprint)
			.eri4c2e_batched(eris)
			.get();
		
	} else if (jmet == jmethod::dfao) {
		
		dbcsr::sbtensor<3,double> eris;
		dbcsr::shared_matrix<double> v_inv;
		
		if (metr == ints::metric::coulomb) {
			
			eris = aoreg.get<decltype(eris)>(ints::key::coul_xbb);
			v_inv = aoreg.get<decltype(v_inv)>(ints::key::coul_xx_inv);
			
		} else if (metr == ints::metric::erfc_coulomb) {
			
			eris = aoreg.get<decltype(eris)>(ints::key::erfc_xbb);
			v_inv = aoreg.get<decltype(v_inv)>(ints::key::erfc_xx_inv);
			
		} else if (metr == ints::metric::qr_fit) {
			
			eris = aoreg.get<decltype(eris)>(ints::key::qr_xbb);
			v_inv = aoreg.get<decltype(v_inv)>(ints::key::coul_xx);
			
		}
		
		jbuilder = create_BATCHED_DF_J(m_world, m_mol, nprint)
			.eri3c2e_batched(eris)
			.v_inv(v_inv)
			.get();
		
	}
	
	// set K
	if (kmet == kmethod::exact) {
		
		auto eris = aoreg.get<dbcsr::sbtensor<4,double>>(ints::key::coul_bbbb);
		
		kbuilder = create_EXACT_K(m_world, m_mol, nprint)
			.eri4c2e_batched(eris)
			.get();
		
	} else if (kmet == kmethod::dfao) {
		
		dbcsr::sbtensor<3,double> eris;
		dbcsr::sbtensor<3,double> cfit;
		
		switch (metr) {
			case ints::metric::coulomb:
				eris = aoreg.get<decltype(eris)>(ints::key::coul_xbb);
				cfit = aoreg.get<decltype(eris)>(ints::key::dfit_coul_xbb);
				break;
			case ints::metric::erfc_coulomb:
				eris = aoreg.get<decltype(eris)>(ints::key::erfc_xbb);
				cfit = aoreg.get<decltype(eris)>(ints::key::dfit_erfc_xbb);
				break;
			case ints::metric::qr_fit:
				eris = aoreg.get<decltype(eris)>(ints::key::qr_xbb);
				cfit = aoreg.get<decltype(eris)>(ints::key::dfit_qr_xbb);
				break;
		}
		
		kbuilder = create_BATCHED_DFAO_K(m_world, m_mol, nprint)
			.eri3c2e_batched(eris)
			.fitting_batched(cfit)
			.get();
		
	} else if (kmet == kmethod::dfmo) {
		
		auto eris = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::coul_xbb);
		auto invsqrt = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::coul_xx_invsqrt);
		int nbatches = m_opt.get<int>("occ_nbatches");
		
		kbuilder = create_BATCHED_DFMO_K(m_world, m_mol, nprint)
			.eri3c2e_batched(eris)
			.v_invsqrt(invsqrt)
			.occ_nbatches(nbatches)
			.get();
		
	} else if (kmet == kmethod::dfmem) {
		
		dbcsr::sbtensor<3,double> eris;
		dbcsr::shared_matrix<double> v_xx;
		
		switch (metr) {
			case ints::metric::coulomb:
				eris = aoreg.get<decltype(eris)>(ints::key::coul_xbb);
				v_xx = aoreg.get<decltype(v_xx)>(ints::key::coul_xx_inv);
				break;
			case ints::metric::erfc_coulomb:
				eris = aoreg.get<decltype(eris)>(ints::key::erfc_xbb);
				v_xx = aoreg.get<decltype(v_xx)>(ints::key::erfc_xx_inv);
				break;
			case ints::metric::qr_fit:
				eris = aoreg.get<decltype(eris)>(ints::key::qr_xbb);
				v_xx = aoreg.get<decltype(v_xx)>(ints::key::coul_xx);
				break;
		}
		
		kbuilder = create_BATCHED_DFMEM_K(m_world, m_mol, nprint)
			.eri3c2e_batched(eris)
			.v_xx(v_xx)
			.get();
			
	}
	
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
