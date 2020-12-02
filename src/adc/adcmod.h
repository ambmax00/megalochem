#ifndef ADC_ADCMOD_H 
#define ADC_ADCMOD_H

#include "desc/options.h"
#include "hf/hf_wfn.h"
#include "utils/mpi_time.h"
#include "adc/adc_defaults.h"
#include "adc/adc_mvp.h"
#include "utils/registry.h"
#include "ints/fitting.h"
#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>

#include <mpi.h>

namespace adc {

	
class adcmod {
private:

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
	
	void compute_diag();
	dbcsr::shared_matrix<double> compute_diag_0();
	dbcsr::shared_matrix<double> compute_diag_1();
		
	std::vector<int> get_significant_blocks(dbcsr::shared_matrix<double> u_bb, 
		double theta, dbcsr::shared_matrix<double> metric_bb, double gamma);
	
public:	

	adcmod(dbcsr::world w, hf::shared_hf_wfn hfref, desc::options& opt);
	~adcmod() {}
	
	void compute();
	
};

} // end namespace

#endif
	
