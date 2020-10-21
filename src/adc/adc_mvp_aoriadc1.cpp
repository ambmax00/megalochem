#include "adc/adc_mvp.h"

namespace adc {

void MVP_ao_ri_adc1::init() {
	
	LOG.os<>("Initializing AO-ADC(1)\n");
	
	m_c_bo = m_reg.get_matrix<double>("c_bo");
	m_c_bv = m_reg.get_matrix<double>("c_bv");
	
	auto kmethod = m_opt.get<std::string>("k_method", "batchdfao");
	
	LOG.os<>("Setting up j builder for AO-ADC1.\n");
	
	fock::J* jbuilder = new fock::BATCHED_DF_J(m_world,m_opt);
	m_jbuilder.reset(jbuilder);
	
	m_jbuilder->set_sym(false);
	m_jbuilder->set_reg(m_reg);
	m_jbuilder->set_mol(m_mol);
	
	m_jbuilder->init();
	
	LOG.os<>("Setting up k builder for AO-ADC1.\n");
	
	if (kmethod == "batchdfao") {
		fock::K* kbuilder = new fock::BATCHED_DFAO_K(m_world,m_opt);
		m_kbuilder.reset(kbuilder);
	} else if (kmethod == "batchpari") {
		fock::K* kbuilder = new fock::BATCHED_PARI_K(m_world,m_opt);
		m_kbuilder.reset(kbuilder);
	}
	
	m_kbuilder->set_sym(false);
	m_kbuilder->set_reg(m_reg);
	m_kbuilder->set_mol(m_mol);
	
	m_kbuilder->init();
	
	LOG.os<>("Done with setting up.\n");
	
}

smat MVP_ao_ri_adc1::compute(smat u_ia, double omega) {
	
	LOG.os<>("Computing ADC0.\n");
	// compute ADC0 part in MO basis
	smat sig_0 = compute_sigma_0(u_ia);
	
	std::cout << "SIG0" << std::endl;
	dbcsr::print(*sig_0);
	
	
	// transform u to ao coordinated
	smat u_ao = u_transform(u_ia, 'N', m_c_bo, 'T', m_c_bv);
	
	LOG.os<>("U transformed: \n");
	dbcsr::print(*u_ao);
	
	u_ao->filter(dbcsr::global::filter_eps);
	
	m_jbuilder->set_density_alpha(u_ao);
	m_kbuilder->set_density_alpha(u_ao);
	
	m_jbuilder->compute_J();
	m_kbuilder->compute_K();
	
	auto jmat = m_jbuilder->get_J();
	auto kmat = m_kbuilder->get_K_A();
	
	auto j = u_transform(jmat, 'T', m_c_bo, 'N', m_c_bv);
	auto k = u_transform(kmat, 'T', m_c_bo, 'N', m_c_bv);
	
	// recycle u_ao
	u_ao->add(0.0, 1.0, *jmat);
	u_ao->add(1.0, 1.0, *kmat);
	
	//LOG.os<>("Sigma adc1 ao:\n");
	//dbcsr::print(*u_ao);
	
	// transform back
	smat sig_1 = u_transform(u_ao, 'T', m_c_bo, 'N', m_c_bv);
	
	LOG.os<>("Sigma adc1 mo:\n");
	dbcsr::print(*sig_1);
	
	sig_0->add(1.0, 1.0, *sig_1);
	
	LOG.os<>("Sigma adc1 tot:\n");
	dbcsr::print(*sig_0);
	
	return sig_0;
	
}
	
} // end namespace
