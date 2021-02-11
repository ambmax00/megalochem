#ifndef ADC_ADCMOD_H 
#define ADC_ADCMOD_H

#include "desc/options.hpp"
#include "hf/hf_wfn.hpp"
#include "utils/mpi_time.hpp"
#include "adc/adc_defaults.hpp"
#include "adc/adc_mvp.hpp"
#include "utils/registry.hpp"
#include "ints/fitting.hpp"
#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>

#include <mpi.h>

namespace adc {

class adcmod {
private:

	struct canon_lmo {
		dbcsr::shared_matrix<double> c_br, c_bs, u_or, u_vs;
		std::vector<double> eps_r, eps_s;
	};

	hf::shared_hf_wfn m_hfwfn;
	desc::options m_opt;
	dbcsr::world m_world;
	
	util::mpi_time TIME;
	util::mpi_log LOG;
	
	ints::aoloader m_ao;
	
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
	dbcsr::shared_matrix<double> compute_diag_0();
	dbcsr::shared_matrix<double> compute_diag_1();
		
	std::tuple<
		std::vector<int>, 
		std::vector<int>> 
	get_significant_blocks(dbcsr::shared_matrix<double> u_bb, 
		double theta, dbcsr::shared_matrix<double> metric_bb, double gamma);
		
	canon_lmo get_canon_nto(dbcsr::shared_matrix<double> u_ia, dbcsr::shared_matrix<double> c_bo,
		dbcsr::shared_matrix<double> c_bv, std::vector<double> eps_o, std::vector<double> eps_v,
		double theta);
		
	canon_lmo get_canon_pao(dbcsr::shared_matrix<double> u_ia, dbcsr::shared_matrix<double> c_bo,
		dbcsr::shared_matrix<double> c_bv, std::vector<double> eps_o, std::vector<double> eps_v,
		double theta);
	
public:	

	adcmod(dbcsr::world w, hf::shared_hf_wfn hfref, desc::options& opt);
	~adcmod() {}
	
	void compute();
	
};

} // end namespace

#endif
	
