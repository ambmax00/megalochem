#include "adc/adcmod.h"
#include "ints/aoloader.h"
#include "ints/screening.h"
#include "math/linalg/LLT.h"
#include "locorb/locorb.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include <Eigen/Eigenvalues>

namespace adc {
		
adcmod::adcmod(dbcsr::world w, hf::shared_hf_wfn hfref, desc::options& opt) :
	m_hfwfn(hfref), 
	m_opt(opt), 
	m_world(w),
	m_nroots(m_opt.get<int>("nroots", ADC_NROOTS)),
	m_diag_order(m_opt.get<int>("diag_order", m_order)),
	m_c_os(m_opt.get<double>("c_os", ADC_C_OS)),
	m_c_osc(m_opt.get<double>("c_os_coupling", ADC_C_OS_COUPLING)),
	LOG(w.comm(), m_opt.get<int>("print", ADC_PRINT_LEVEL)),
	TIME(w.comm(), "ADC Module", LOG.global_plev())
{
	
	LOG.banner("ADC MODULE",50,'*');
	
	auto mstr = m_opt.get<std::string>("method", ADC_METHOD);
	auto jstr = m_opt.get<std::string>("build_J", ADC_BUILD_J);
	auto kstr = m_opt.get<std::string>("build_K", ADC_BUILD_K);
	
	m_mvpmethod = str_to_mvpmethod(mstr);
	m_jmethod = fock::str_to_jmethod(jstr);
	m_kmethod = fock::str_to_kmethod(kstr);
	
	if (m_mvpmethod == mvpmethod::invalid) {
		throw std::runtime_error("Invalid method for adc.");
	}
		
	std::string dfbasname = m_opt.get<std::string>("dfbasis");
	int nsplit = m_hfwfn->mol()->c_basis()->nsplit();
	std::string splitmethod = m_hfwfn->mol()->c_basis()->split_method();
	auto atoms = m_hfwfn->mol()->atoms();
	
	bool augmented = m_opt.get<bool>("df_augmentation", false);
	auto dfbasis = std::make_shared<desc::cluster_basis>(
		dfbasname, atoms, splitmethod, nsplit, augmented);
	m_hfwfn->mol()->set_cluster_dfbasis(dfbasis);
	
	init();
	
	LOG.os<>("--- Ready for launching computation. --- \n\n");
	
}

void adcmod::init() {
	
	// setup pgrids
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	int nbf = m_hfwfn->mol()->c_basis()->nbf();
	int nocc = m_hfwfn->mol()->nocc_alpha();
	int nvir = m_hfwfn->mol()->nvir_alpha();
	int xnbf = m_hfwfn->mol()->c_dfbasis()->nbf();
	
	std::array<int,2> bosizes = {nbf,nocc};
	std::array<int,2> bvsizes = {nbf,nvir};
	
	m_spgrid2_bo = dbcsr::create_pgrid<2>(m_world.comm())
		.tensor_dims(bosizes).get();
		
	m_spgrid2_bv = dbcsr::create_pgrid<2>(m_world.comm())
		.tensor_dims(bvsizes).get();

	std::array<int,3> xbbsizes = {xnbf,nbf,nbf};
	
	m_spgrid3_xbb = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(xbbsizes).get();
	
}

void adcmod::init_ao_tensors() {
	
	LOG.os<>("Setting up AO integral tensors.\n");
	
	// setting up what we need
	bool compute_3c2e(false), 
		compute_metric_inv(false), 
		compute_metric_invsqrt(false),
		compute_s_bb(true);
	
	std::string metric_str = m_opt.get<std::string>("df_metric", ADC_METRIC);
	auto met = ints::str_to_metric(metric_str);
	m_metric = met;
	
	ints::aoloader m_aoloader(m_world, m_hfwfn->mol(), m_opt);
	
	m_aoloader.request(ints::key::ovlp_bb,true);
	
	if (met == ints::metric::coulomb) {
		
		m_aoloader.request(ints::key::coul_xx, true);
		m_aoloader.request(ints::key::coul_xx_inv, true);
		m_aoloader.request(ints::key::scr_xbb, true);
		m_aoloader.request(ints::key::coul_xbb, true);
		
		if (m_kmethod == fock::kmethod::dfao) {
			m_aoloader.request(ints::key::dfit_coul_xbb, true);
		}
		
	} else if (met == ints::metric::erfc_coulomb) {
		
		m_aoloader.request(ints::key::coul_xx, true);
		m_aoloader.request(ints::key::erfc_xx, true);
		m_aoloader.request(ints::key::erfc_xx_inv, true);
		m_aoloader.request(ints::key::scr_xbb, true);
		m_aoloader.request(ints::key::erfc_xbb, true);
		
		if (m_kmethod == fock::kmethod::dfao) {
			m_aoloader.request(ints::key::dfit_erfc_xbb, true);
		}
		
	} else if (met == ints::metric::qr_fit) {
		
		m_aoloader.request(ints::key::coul_xx, true);
		m_aoloader.request(ints::key::ovlp_xx, false);
		m_aoloader.request(ints::key::ovlp_xx_inv, false);
		m_aoloader.request(ints::key::scr_xbb, true);
		m_aoloader.request(ints::key::qr_xbb, true);
		
		if (m_kmethod == fock::kmethod::dfao) {
			m_aoloader.request(ints::key::dfit_qr_xbb, true);
		}
		
	}
	
	m_aoloader.compute();
	auto aoreg = m_aoloader.get_registry();
	
	dbcsr::sbtensor<3,double> eri3c2e_batched;
	dbcsr::shared_matrix<double> metric_matrix;
	dbcsr::sbtensor<3,double> fitting_batched;
	
	if (met == ints::metric::coulomb) {
		
		eri3c2e_batched = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::coul_xbb);
		metric_matrix = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::coul_xx_inv);
		if (m_kmethod == fock::kmethod::dfao) {
			fitting_batched = aoreg.get<dbcsr::sbtensor<3,double>>(
				ints::key::dfit_coul_xbb);
		}
		
	} else if (met == ints::metric::erfc_coulomb) {
		
		eri3c2e_batched = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::erfc_xbb);
		metric_matrix = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::erfc_xx_inv);
		if (m_kmethod == fock::kmethod::dfao) {
			fitting_batched = aoreg.get<dbcsr::sbtensor<3,double>>(
				ints::key::dfit_erfc_xbb);
		}
		
	} else if (met == ints::metric::qr_fit) {
		
		eri3c2e_batched = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::qr_xbb);
		metric_matrix = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::coul_xx);
		if (m_kmethod == fock::kmethod::dfao) {
			fitting_batched = aoreg.get<dbcsr::sbtensor<3,double>>(
				ints::key::dfit_qr_xbb);
		}
		
	}
	
	LOG.os<>("Finished setting up AO integral tensors.\n");
	
	m_adc1_mvp = create_MVPAOADC1(
			m_world, m_hfwfn->mol(), LOG.global_plev())
			.eri3c2e_batched(eri3c2e_batched)
			.fitting_batched(fitting_batched)
			.v_xx(metric_matrix)
			.c_bo(m_hfwfn->c_bo_A())
			.c_bv(m_hfwfn->c_bv_A())
			.eps_occ(m_hfwfn->eps_occ_A())
			.eps_vir(m_hfwfn->eps_vir_A())
			.kmethod(m_kmethod)
			.jmethod(m_jmethod)
			.get();
			
	m_adc1_mvp->init();
	
	m_s_bb = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::ovlp_bb);
	
}

void adcmod::init_mo_tensors() {
	
	LOG.os<>("Setting up MO tensors.\n");
	
		
}


} // end namespace
