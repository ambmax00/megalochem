#include "adc/adcmod.h"
#include "adc/adc_mvp.h"
#include "math/solvers/davidson.h"

#include <dbcsr_conversions.hpp>

namespace adc {

void adcmod::compute() {
	
		// BEFORE: init tensors base (ints, metrics, etc..., put into m_reg)
		
		init_ao_tensors();
		
		// AFTER: init tensors (2) (mo-ints, diags, amplitudes ...) 
		
		init_mo_tensors();
				
		// SECOND: Generate guesses
		
		compute_diag();
		
		LOG.os<>("--- Starting Computation ---\n\n");
				
		int nocc = m_hfwfn->mol()->nocc_alpha();
		int nvir = m_hfwfn->mol()->nvir_alpha();
		auto epso = m_hfwfn->eps_occ_A();
		auto epsv = m_hfwfn->eps_vir_A();
		
		LOG.os<>("Computing guess vectors...\n");
		// now order it : there is probably a better way to do it
		auto eigen_ia = dbcsr::matrix_to_eigen(m_d_ov);
		
		std::cout << eigen_ia << std::endl;
		
		std::vector<int> index(eigen_ia.size(), 0);
		for (int i = 0; i!= index.size(); ++i) {
			index[i] = i;
		}
		
		std::sort(index.begin(), index.end(), 
			[&](const int& a, const int& b) {
				return (eigen_ia.data()[a] < eigen_ia.data()[b]);
		});
		
		std::vector<dbcsr::shared_matrix<double>> dav_guess(m_nroots);
			
		// generate the guesses
		
		auto o = m_hfwfn->mol()->dims().oa();
		auto v = m_hfwfn->mol()->dims().va();
		
		for (int i = 0; i != m_nroots; ++i) {
			
			LOG.os<>("Guess ", i, '\n');
			
			Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(nocc,nvir);
			mat.data()[index[i]] = 1.0;
			
			std::string name = "guess_" + std::to_string(i);
			
			auto guessmat = dbcsr::eigen_to_matrix(mat, m_world, name,
				o, v, dbcsr::type::no_symmetry);
			
			dav_guess[i] = guessmat;
			
			dbcsr::print(*guessmat);
			
		}
		
		MVP* mvfacptr = new MVP_ao_ri_adc1(m_world, m_hfwfn->mol(), 
			m_opt, m_reg, epso, epsv);
		std::shared_ptr<MVP> mvfac(mvfacptr);
		
		mvfac->init();
		auto out = mvfac->compute(dav_guess[0], 0.0); 
		
		math::davidson<MVP> dav;
		
		dav.set_factory(mvfac);
		dav.set_diag(m_d_ov);
		dav.pseudo(false);
		dav.conv(1e-6);
		dav.maxiter(100);	
		
		int nroots = m_opt.get<int>("nroots", ADC_NROOTS);
		
		dav.compute(dav_guess, nroots);
		
		/*
		ri_adc1_u1 ri_adc1(m_mo.eps_o, m_mo.eps_v, m_mo.b_xoo, m_mo.b_xov, m_mo.b_xvv); 
		
		math::davidson<ri_adc1_u1> dav = math::davidson<ri_adc1_u1>::create()
			.factory(ri_adc1).diag(m_mo.d_ov);
		
		LOG.os<>("Running ADC(1) davidson...\n\n");
		dav.compute(dav_guess, m_nroots);
		
		auto rvs = dav.ritz_vectors();
		
		auto rn = rvs[m_nroots - 1];
		double omega = dav.eigval();

		// compute ADC2
		if (!m_use_sos) {
			
			ri_adc2_diis_u1 ri_adc2(m_mo.eps_o, m_mo.eps_v, m_mo.b_xoo, m_mo.b_xov, m_mo.b_xvv, m_mo.t_ovov);
		
			math::modified_davidson<ri_adc2_diis_u1> dav
				= math::modified_davidson<ri_adc2_diis_u1>::create()
				.factory(ri_adc2).diag(m_mo.d_ov);
			
			LOG.os<>("Running RI-ADC(2)...\n\n");
			dav.compute(rvs, m_nroots, omega);
			
		} else {
			
			std::cout << "SOS" << std::endl;
			lp_ri_adc2_diis_u1 ri_adc2(m_mo.eps_o, m_mo.eps_v, m_mo.b_xoo, m_mo.b_xov, m_mo.b_xvv);
		
			//math::modified_davidson<lp_ri_adc2_diis_u1> dav 
			//	= math::modified_davidson<lp_ri_adc2_diis_u1>::create()
			//	.factory(ri_adc2).diag(m_mo.d_ov);
			
			LOG.os<>("Running SOS-RI-ADC(2)...\n\n");
			//dav.compute(rvs, m_nroots, omega);
			
		}*/
		
}

}
