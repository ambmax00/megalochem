#ifndef ADC_ADCMOD_H 
#define ADC_ADCMOD_H

#ifndef TEST_MACRO
#include "megalochem.hpp"
#include "desc/wfn.hpp"
#include "utils/mpi_time.hpp"
#include "adc/adc_mvp.hpp"
#include "ints/registry.hpp"
#include "ints/fitting.hpp"
#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>
#include <mpi.h>
#include <tuple>
#endif

#include "utils/ppdirs.hpp"

namespace megalochem {

namespace adc {
	
enum class adcmethod {
	ri_ao_adc1,
	sos_cd_ri_adc2
};

inline adcmethod str_to_adcmethod(std::string str) {
	if (str == "ri_ao_adc1") {
		return adcmethod::ri_ao_adc1;
	} else if (str == "sos_cd_ri_adc2") {
		return adcmethod::sos_cd_ri_adc2;
	} else {
		throw std::runtime_error("Unknwon adc method");
	}
}
	
#define ADCMOD_LIST (\
	((world), set_world),\
	((desc::shared_wavefunction), set_wfn),\
	((util::optional<desc::shared_cluster_basis>), df_basis))
	
#define ADCMOD_OPTLIST (\
	((util::optional<int>), print, 0),\
	((util::optional<int>), nbatches_b, 5),\
	((util::optional<int>), nbatches_x, 5),\
	((util::optional<int>), nroots, 1),\
	((util::optional<int>), nguesses, 1),\
	((util::optional<bool>), block, false),\
	((util::optional<bool>), balanced, false),\
	((util::optional<std::string>), method, "ri_ao_adc1"),\
	((util::optional<std::string>), df_metric, "coulomb"),\
	((util::optional<double>), conv, 1e-5),\
	((util::optional<int>), dav_max_iter, 40),\
	((util::optional<int>), diis_max_iter, 40),\
	((util::optional<std::string>), eris, "core"),\
	((util::optional<std::string>), imeds, "core"),\
	((util::optional<std::string>), build_K, "dfao"),\
	((util::optional<std::string>), build_J, "dfao"),\
	((util::optional<std::string>), build_Z, "llmp_full"),\
	((util::optional<double>), c_os, 1.3),\
	((util::optional<double>), c_os_coupling, 1.17),\
	((util::optional<int>), nlap, 5),\
	((util::optional<bool>), local, false),\
	((util::optional<double>), cutoff, 1.0),\
	((util::optional<std::string>), guess, "hf"))
	
struct eigenpair {
	std::vector<double> eigvals;
	std::vector<dbcsr::shared_matrix<double>> eigvecs;
};

class adcmod {
private:

	struct canon_lmo {
		dbcsr::shared_matrix<double> c_br, c_bs, u_or, u_vs;
		std::vector<double> eps_r, eps_s;
	};

	desc::shared_wavefunction m_wfn;
	megalochem::world m_world;
	desc::shared_cluster_basis m_df_basis;
	
	dbcsr::cart m_cart;
	
	MAKE_MEMBER_VARS(ADCMOD_OPTLIST)
	
	adcmethod m_adcmethod;
	
	util::mpi_time TIME;
	util::mpi_log LOG;
	
	std::shared_ptr<ints::aoloader> m_aoloader;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<2> m_spgrid2_bo;
	dbcsr::shared_pgrid<2> m_spgrid2_bv;
	dbcsr::shared_pgrid<3> m_spgrid3_xbb;

	dbcsr::shared_matrix<double> m_d_ov;
		
	void init_ao_tensors();
	
	std::shared_ptr<MVP> create_adc1();
	std::shared_ptr<MVP> create_adc2(std::optional<canon_lmo> clmo = 
		std::nullopt);
	
	void compute_diag();
	
	eigenpair guess();
		
	eigenpair run_adc1(eigenpair& dav);
		
	eigenpair run_adc2(eigenpair& dav);
	
	dbcsr::shared_matrix<double> compute_diag_0();
	
	//dbcsr::shared_matrix<double> compute_diag_1();
	
	std::vector<bool> get_significant_blocks(
		dbcsr::shared_matrix<double> u_ia, 
		double theta);
		
	/*canon_lmo get_canon_nto(dbcsr::shared_matrix<double> u_ia, dbcsr::shared_matrix<double> c_bo,
		dbcsr::shared_matrix<double> c_bv, std::vector<double> eps_o, std::vector<double> eps_v,
		double theta);
		
	canon_lmo get_canon_pao(dbcsr::shared_matrix<double> u_ia, dbcsr::shared_matrix<double> c_bo,
		dbcsr::shared_matrix<double> c_bv, std::vector<double> eps_o, std::vector<double> eps_v,
		double theta);*/
	
	void init();
	
	std::tuple<dbcsr::shared_matrix<double>, dbcsr::sbtensor<3,double>>
	test_fitting(std::vector<bool> atom_idx);
	
	dbcsr::sbtensor<3,double> m_fit;
	
public:	

	MAKE_PARAM_STRUCT(create, CONCAT(ADCMOD_LIST, ADCMOD_OPTLIST), ())
	
	MAKE_BUILDER_CLASS(adcmod, create, CONCAT(ADCMOD_LIST, ADCMOD_OPTLIST), ())
	
	adcmod(create_pack&& p) :
		m_world(p.p_set_world),
		m_wfn(p.p_set_wfn),
		m_cart(m_world.dbcsr_grid()),
		m_df_basis(p.p_df_basis ? *p.p_df_basis : nullptr),
		MAKE_INIT_LIST_OPT(ADCMOD_OPTLIST),
		LOG(m_world.comm(), m_print),
		TIME(m_world.comm(), "adcmod", m_print)
	{
		init();
	}
	
	~adcmod() {}
	
	desc::shared_wavefunction compute();
	
};

} // end namespace

} // end namespace mega

#endif
	
