#include "adc/adcmod.hpp"
#include "ints/aoloader.hpp"
#include "ints/screening.hpp"
#include "math/linalg/LLT.hpp"
#include "locorb/locorb.hpp"
#include "math/solvers/hermitian_eigen_solver.hpp"
#include <Eigen/Eigenvalues>

#include <type_traits>

namespace adc {
		
adcmod::adcmod(dbcsr::cart w, hf::shared_hf_wfn hfref, desc::options& opt) :
	m_hfwfn(hfref), 
	m_opt(opt), 
	m_cart(w),
	LOG(w.comm(), m_opt.get<int>("print", ADC_PRINT_LEVEL)),
	TIME(w.comm(), "ADC Module", LOG.global_plev())
{
	
	LOG.banner("ADC MODULE",50,'*');
	
	std::string dfbasname = m_opt.get<std::string>("dfbasis");
	int nsplit = m_hfwfn->mol()->c_basis()->nsplit();
	std::string splitmethod = m_hfwfn->mol()->c_basis()->split_method();
	auto atoms = m_hfwfn->mol()->atoms();
	
	bool augmented = m_opt.get<bool>("df_augmentation", false);
	auto dfbasis = std::make_shared<desc::cluster_basis>(
		dfbasname, atoms, splitmethod, nsplit, augmented);
	m_hfwfn->mol()->set_cluster_dfbasis(dfbasis);
	
	std::optional<int> nbatches_b_opt = m_opt.present("nbatches_b") ? 
		std::make_optional<int>(m_opt.get<int>("nbatches_b")) : 
		std::nullopt;
		
	std::optional<int> nbatches_x_opt = m_opt.present("nbatches_x") ? 
		std::make_optional<int>(m_opt.get<int>("nbatches_x")) : 
		std::nullopt;
		
	std::optional<dbcsr::btype> btype_e = m_opt.present("eris") ?
		std::make_optional<dbcsr::btype>(dbcsr::get_btype(m_opt.get<std::string>("eris"))) :
		std::nullopt;
		
	std::optional<dbcsr::btype> btype_i = m_opt.present("intermeds") ?
		std::make_optional<dbcsr::btype>(dbcsr::get_btype(m_opt.get<std::string>("intermeds"))) :
		std::nullopt;
	
	m_aoloader = ints::aoloader::create()
		.set_cart(m_cart)
		.set_molecule(m_hfwfn->mol())
		.print(LOG.global_plev())
		.nbatches_b(nbatches_b_opt)
		.nbatches_x(nbatches_x_opt)
		.btype_eris(btype_e)
		.btype_intermeds(btype_i)
		.build();
	
	init_ao_tensors();
	
	LOG.os<>("--- Ready for launching computation. --- \n\n");
	
}

void adcmod::init_ao_tensors() {
	
	LOG.os<>("Setting up AO integral tensors.\n");
		
	// LOAD ALL ADC1 INTEGRALS
	
	bool do_adc1 = m_opt.get<bool>("do_adc1",true);
	bool do_adc2 = m_opt.get<bool>("do_adc2",ADC_DO_ADC2);
	
	if (do_adc1) {
	
		auto adc1_opt = m_opt.subtext("adc1");
			
		auto jstr_adc1 = adc1_opt.get<std::string>("jmethod", ADC_ADC1_JMETHOD);
		auto kstr_adc1 = adc1_opt.get<std::string>("kmethod", ADC_ADC1_KMETHOD);
		auto mstr_adc1 = adc1_opt.get<std::string>("df_metric", ADC_ADC1_DF_METRIC);
		
		std::cout << "K: " << mstr_adc1 << std::endl;
		
		auto jmet_adc1 = fock::str_to_jmethod(jstr_adc1);
		auto kmet_adc1 = fock::str_to_kmethod(kstr_adc1);
		auto metr_adc1 = ints::str_to_metric(mstr_adc1);
		
		fock::load_jints(jmet_adc1, metr_adc1, *m_aoloader);
		fock::load_kints(kmet_adc1, metr_adc1, *m_aoloader);
		
	}
	
	if (do_adc2) {
		
		auto adc2_opt = m_opt.subtext("adc2");
		bool local = adc2_opt.get<bool>("local", ADC_ADC2_LOCAL);
		
		// if not local, we are postponing integral evaluation for later
		if (!local) {
			auto jstr_adc2 = adc2_opt.get<std::string>("jmethod", ADC_ADC2_JMETHOD);
			auto kstr_adc2 = adc2_opt.get<std::string>("kmethod", ADC_ADC2_KMETHOD);
			auto zstr_adc2 = adc2_opt.get<std::string>("zmethod", ADC_ADC2_ZMETHOD);
			auto mstr_adc2 = adc2_opt.get<std::string>("df_metric", ADC_ADC2_DF_METRIC);
		
			auto jmet_adc2 = fock::str_to_jmethod(jstr_adc2);
			auto kmet_adc2 = fock::str_to_kmethod(kstr_adc2);
			auto zmet_adc2 = mp::str_to_zmethod(zstr_adc2);
			
			auto metr_adc2 = ints::str_to_metric(mstr_adc2);
		
			fock::load_jints(jmet_adc2, metr_adc2, *m_aoloader);
			fock::load_kints(kmet_adc2, metr_adc2, *m_aoloader);
		
		} else {
			
			m_aoloader->request(ints::key::coul_xx, true);
		
		}	
		
		// overlap always needed
		m_aoloader->request(ints::key::ovlp_bb, true);
		
	}
	
	m_aoloader->request(ints::key::ovlp_bb, true);
	int nprint = LOG.global_plev();

	m_aoloader->compute();
	
}

std::shared_ptr<MVP> adcmod::create_adc1() {
	
	dbcsr::shared_matrix<double> v_xx;
	dbcsr::sbtensor<3,double> eri3c2e, fitting;
	
	auto jmeth = fock::str_to_jmethod(m_opt.get<std::string>("adc1/jmethod", ADC_ADC1_JMETHOD));
	auto kmeth = fock::str_to_kmethod(m_opt.get<std::string>("adc1/kmethod", ADC_ADC1_KMETHOD));
	auto metr = ints::str_to_metric(m_opt.get<std::string>("adc1/df_metric", ADC_ADC1_DF_METRIC));
		
	auto aoreg = m_aoloader->get_registry();
	
	auto get = [&aoreg](auto& tensor, ints::key aokey) {
		if (aoreg.present(aokey)) {
			tensor = aoreg.get<typename std::remove_reference<decltype(tensor)>::type>(aokey);
		} else {
			std::cout << "NOT PRESENT :" << static_cast<int>(aokey) << std::endl;
			tensor = nullptr;
		}
	};
	
	switch (metr) {
		case ints::metric::coulomb:
		{
			get(eri3c2e, ints::key::coul_xbb);
			get(fitting, ints::key::dfit_coul_xbb);
			get(v_xx, ints::key::coul_xx_inv);
			break;
		}
		case ints::metric::erfc_coulomb:
		{
			get(eri3c2e, ints::key::erfc_xbb);
			get(fitting, ints::key::dfit_erfc_xbb);
			get(v_xx, ints::key::erfc_xx_inv);
			break;
		}
		case ints::metric::pari:
		{
			get(eri3c2e, ints::key::pari_xbb);
			if (!eri3c2e) std::cout << "NULL" << std::endl;
			get(fitting, ints::key::dfit_pari_xbb);
			get(v_xx, ints::key::coul_xx);
			break;
		}
		case ints::metric::qr_fit:
		{
			get(eri3c2e, ints::key::qr_xbb);
			get(fitting, ints::key::dfit_qr_xbb);
			get(v_xx, ints::key::coul_xx);
			break;
		}
	}
	
	auto ptr = MVP_AORIADC1::create()
		.set_cart(m_cart)
		.set_molecule( m_hfwfn->mol())
		.print(LOG.global_plev())
		.c_bo(m_hfwfn->c_bo_A())
		.c_bv(m_hfwfn->c_bv_A())
		.eps_occ(*m_hfwfn->eps_occ_A())
		.eps_vir(*m_hfwfn->eps_vir_A())
		.eri3c2e_batched(eri3c2e)
		.fitting_batched(fitting)
		.metric_inv(v_xx)
		.jmethod(jmeth)
		.kmethod(kmeth)
		.build();
		
	ptr->init();
	
	return ptr;
	
}

std::shared_ptr<MVP> adcmod::create_adc2(std::optional<canon_lmo> clmo) {
	
	desc::shared_molecule mol;
	std::shared_ptr<std::vector<double>> eps_o, eps_v;
	dbcsr::shared_matrix<double> v_xx, s_bb, c_bo, c_bv;
	dbcsr::sbtensor<3,double> eri3c2e, fitting;
	
	auto jmeth = fock::str_to_jmethod(m_opt.get<std::string>("adc2/jmethod", ADC_ADC2_JMETHOD));
	auto kmeth = fock::str_to_kmethod(m_opt.get<std::string>("adc2/kmethod", ADC_ADC2_KMETHOD));
	auto zmeth = mp::str_to_zmethod(m_opt.get<std::string>("adc2/zmethod", ADC_ADC2_ZMETHOD));
	auto mytype = dbcsr::get_btype(m_opt.get<std::string>("adc2/intermeds", ADC_ADC2_INTERMEDS));
	double c_os = m_opt.get<double>("adc2/c_os", ADC_ADC2_C_OS);
	double c_os_coupling = m_opt.get<double>("adc2/c_os_coupling", ADC_ADC2_C_OS_COUPLING);
	int nlap = m_opt.get<int>("adc2/nlap", ADC_ADC2_NLAP);
	
	auto metr = ints::str_to_metric(m_opt.get<std::string>("adc2/df_metric", ADC_ADC2_DF_METRIC));
	auto aoreg = m_aoloader->get_registry();
	
	auto get = [&aoreg](auto& tensor, ints::key aokey) {
		if (aoreg.present(aokey)) {
			tensor = aoreg.get<typename std::remove_reference<decltype(tensor)>::type>(aokey);
		} else {
			tensor = nullptr;
		}
	};
	
	switch (metr) {
		case ints::metric::coulomb:
		{
			get(eri3c2e, ints::key::coul_xbb);
			get(fitting, ints::key::dfit_coul_xbb);
			get(v_xx, ints::key::coul_xx_inv);
			break;
		}
		case ints::metric::erfc_coulomb:
		{
			get(eri3c2e, ints::key::erfc_xbb);
			get(fitting, ints::key::dfit_erfc_xbb);
			get(v_xx, ints::key::erfc_xx_inv);
			break;
		}
		case ints::metric::pari:
		{
			get(eri3c2e, ints::key::pari_xbb);
			get(fitting, ints::key::dfit_pari_xbb);
			get(v_xx, ints::key::coul_xx);
			break;
		}
		case ints::metric::qr_fit:
		{
			get(eri3c2e, ints::key::qr_xbb);
			get(fitting, ints::key::dfit_qr_xbb);
			get(v_xx, ints::key::coul_xx);
			break;
		}
	}
	
	s_bb = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::ovlp_bb);
	
	if (clmo) {
		
		int natoms = m_hfwfn->mol()->atoms().size();
	
		std::vector<int> iat(natoms, 0);
		std::iota(iat.begin(), iat.end(), 0);
	
		int noa = clmo->c_br->nfullcols_total();
		int nva = clmo->c_bs->nfullcols_total();
	
		mol = m_hfwfn->mol()->fragment(noa, noa, nva, nva, iat);
		
		c_bo = clmo->c_br;
		c_bv = clmo->c_bs;
		
		eps_o = std::make_shared<std::vector<double>>(clmo->eps_r);
		eps_v = std::make_shared<std::vector<double>>(clmo->eps_s);
		
	} else {
		
		mol = m_hfwfn->mol();
		c_bo = m_hfwfn->c_bo_A();
		c_bv = m_hfwfn->c_bv_A();
		eps_o = m_hfwfn->eps_occ_A();
		eps_v = m_hfwfn->eps_vir_A();
		
	}

	auto ptr = MVP_AORISOSADC2::create()
		.set_cart(m_cart)
		.set_molecule(mol)
		.print(LOG.global_plev())
		.c_bo(c_bo)
		.c_bv(c_bv)
		.s_bb(s_bb)
		.eps_occ(*eps_o)
		.eps_vir(*eps_v)
		.eri3c2e_batched(eri3c2e)
		.fitting_batched(fitting)
		.metric_inv(v_xx)
		.jmethod(jmeth)
		.kmethod(kmeth)
		.zmethod(zmeth)
		.btype(mytype)
		.nlap(nlap)
		.c_os(c_os)
		.c_os_coupling(c_os_coupling)
		.build();
				
	ptr->init();
	
	return ptr;
	
}

} // end namespace
