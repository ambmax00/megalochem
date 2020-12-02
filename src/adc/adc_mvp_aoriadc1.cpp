#include "adc/adc_mvp.h"
#include "ints/aoloader.h"

namespace adc {

void MVP_AOADC1::init() {
	
	LOG.os<>("Initializing AO-ADC(1)\n");
	
	LOG.os<>("Setting up j/k builders for AO-ADC1.\n");
	
	int nprint = LOG.global_plev();
	
	switch (m_jmethod) {
		case fock::jmethod::dfao: 
		{
			m_jbuilder = fock::create_DF_J(m_world, m_mol, nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.v_inv(m_v_xx)
				.get();
			break;
		}
		default:
		{
			throw std::runtime_error("Invalid jmethod for AO-ADC(1)");
		}
	}
	
	switch (m_kmethod) {
		case fock::kmethod::dfao:
		{
			m_kbuilder = fock::create_DFAO_K(m_world, m_mol, nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.fitting_batched(m_fitting_batched)
				.get();
			break;
		}
		case fock::kmethod::dfmem:
		{
			m_kbuilder = fock::create_DFMEM_K(m_world, m_mol, nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.v_xx(m_v_xx)
				.get();
			break;
		}
		default:
		{
			throw std::runtime_error("Invalid kmethod for AO-ADC(1)");
		}
	}
	
	m_jbuilder->set_sym(false);
	m_jbuilder->init();
		
	m_kbuilder->set_sym(false);	
	m_kbuilder->init();
	
	LOG.os<>("Done with setting up.\n");
	
}

smat MVP_AOADC1::compute(smat u_ia, double omega) {
	
	TIME.start();
	
	LOG.os<1>("Computing ADC0.\n");
	// compute ADC0 part in MO basis
	smat sig_0 = compute_sigma_0(u_ia, *m_eps_occ, *m_eps_vir);
		
	// transform u to ao coordinated
	
	LOG.os<1>("Computing ADC1.\n");
	smat u_ao = u_transform(u_ia, 'N', m_c_bo, 'T', m_c_bv);
	
	//LOG.os<>("U transformed: \n");
	//dbcsr::print(*u_ao);
	
	u_ao->filter(dbcsr::global::filter_eps);
	
	m_jbuilder->set_density_alpha(u_ao);
	m_kbuilder->set_density_alpha(u_ao);
	
	m_jbuilder->compute_J();
	m_kbuilder->compute_K();
	
	auto jmat = m_jbuilder->get_J();
	auto kmat = m_kbuilder->get_K_A();
	
	// recycle u_ao
	u_ao->add(0.0, 1.0, *jmat);
	u_ao->add(1.0, 1.0, *kmat);
	
	//LOG.os<>("Sigma adc1 ao:\n");
	//dbcsr::print(*u_ao);
	
	// transform back
	smat sig_1 = u_transform(u_ao, 'T', m_c_bo, 'N', m_c_bv);
	
	//LOG.os<>("Sigma adc1 mo:\n");
	//dbcsr::print(*sig_1);
	
	sig_0->add(1.0, 1.0, *sig_1);
	
	//LOG.os<>("Sigma adc1 tot:\n");
	//dbcsr::print(*sig_0);
	
	TIME.finish();
	
	return sig_0;
	
}
	
} // end namespace
