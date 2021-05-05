#include "adc/adcmod.hpp"
#include "ints/aoloader.hpp"
#include "ints/screening.hpp"
#include "math/linalg/LLT.hpp"
#include "locorb/locorb.hpp"
#include "math/solvers/hermitian_eigen_solver.hpp"
#include <Eigen/Eigenvalues>

#include <type_traits>

namespace megalochem {

namespace adc {
		
void adcmod::init()
{
	
	LOG.banner("ADC MODULE",50,'*');
	
	m_wfn->mol->set_cluster_dfbasis(m_df_basis);
		
	dbcsr::btype btype_e = dbcsr::get_btype(m_eris);
		
	dbcsr::btype btype_i = dbcsr::get_btype(m_imeds);
	
	m_aoloader = ints::aoloader::create()
		.set_world(m_world)
		.set_molecule(m_wfn->mol)
		.print(LOG.global_plev())
		.nbatches_b(m_nbatches_b)
		.nbatches_x(m_nbatches_x)
		.btype_eris(btype_e)
		.btype_intermeds(btype_i)
		.build();
	
	m_adcmethod = str_to_adcmethod(m_method);
	
	init_ao_tensors();
	
	LOG.os<>("--- Ready for launching computation. --- \n\n");
	
}

void adcmod::init_ao_tensors() {
	
	LOG.os<>("Setting up AO integral tensors.\n");
	
	switch (m_adcmethod) {
		case adcmethod::ri_ao_adc1: 
		{
			auto jmet_adc1 = fock::str_to_jmethod(m_build_J);
			auto kmet_adc1 = fock::str_to_kmethod(m_build_K);
			auto metr_adc1 = ints::str_to_metric(m_df_metric);
			
			fock::load_jints(jmet_adc1, metr_adc1, *m_aoloader);
			fock::load_kints(kmet_adc1, metr_adc1, *m_aoloader);
		
			break;
		}
		case adcmethod::sos_cd_ri_adc2:
		{
			auto jmet_adc2 = fock::str_to_jmethod(m_build_J);
			auto kmet_adc2 = fock::str_to_kmethod(m_build_K);
			auto zmet_adc2 = mp::str_to_zmethod(m_build_Z);
			auto metr_adc2 = ints::str_to_metric(m_df_metric);
		
			fock::load_jints(jmet_adc2, metr_adc2, *m_aoloader);
			fock::load_kints(kmet_adc2, metr_adc2, *m_aoloader);
			
			break;
		}
	}
	
	m_aoloader->request(ints::key::ovlp_bb, true);

	m_aoloader->compute();
	
}

std::shared_ptr<MVP> adcmod::create_adc1() {
	
	dbcsr::shared_matrix<double> v_xx;
	dbcsr::sbtensor<3,double> eri3c2e, fitting;
	
	auto jmeth = fock::str_to_jmethod(m_build_J);
	auto kmeth = fock::str_to_kmethod(m_build_K);
	auto metr = ints::str_to_metric(m_df_metric);
		
	auto aoreg = m_aoloader->get_registry();
	
	auto get = [&aoreg](auto& tensor, ints::key aokey) {
		if (aoreg.present(aokey)) {
			tensor = aoreg.get<typename std::remove_reference<decltype(tensor)>::type>(aokey);
		} else {
			//std::cout << "NOT PRESENT :" << static_cast<int>(aokey) << std::endl;
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
		.set_world(m_world)
		.set_molecule(m_wfn->mol)
		.print(LOG.global_plev())
		.c_bo(m_wfn->hf_wfn->c_bo_A())
		.c_bv(m_wfn->hf_wfn->c_bv_A())
		.eps_occ(*m_wfn->hf_wfn->eps_occ_A())
		.eps_vir(*m_wfn->hf_wfn->eps_vir_A())
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
	
	auto jmeth = fock::str_to_jmethod(m_build_J);
	auto kmeth = fock::str_to_kmethod(m_build_K);
	auto zmeth = mp::str_to_zmethod(m_build_Z);
	
	auto itype = dbcsr::get_btype(m_imeds);
	
	auto metr = ints::str_to_metric(m_df_metric);
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
	
	/*if (clmo) {
		
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
		
	} else {*/
		
	mol = m_wfn->mol;
	c_bo = m_wfn->hf_wfn->c_bo_A();
	c_bv = m_wfn->hf_wfn->c_bv_A();
	eps_o = m_wfn->hf_wfn->eps_occ_A();
	eps_v = m_wfn->hf_wfn->eps_vir_A();

	auto ptr = MVP_AORISOSADC2::create()
		.set_world(m_world)
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
		.btype(itype)
		.nlap(m_nlap)
		.c_os(m_c_os)
		.c_os_coupling(m_c_os_coupling)
		.build();
				
	ptr->init();
	
	return ptr;
	
}

} // end namespace

} // end namespace mega
