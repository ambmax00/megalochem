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
	
	int m_order;
	int m_diag_order;
	
	int m_nroots; 
	
	mvpmethod m_mvpmethod;
	ints::metric m_metric;
	fock::kmethod m_kmethod;
	fock::jmethod m_jmethod;
	
	std::shared_ptr<MVP> m_adc1_mvp;
	
	double m_c_os;
	double m_c_osc;
		
	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<2> m_spgrid2_bo;
	dbcsr::shared_pgrid<2> m_spgrid2_bv;
	dbcsr::shared_pgrid<3> m_spgrid3_xbb;

	dbcsr::shared_matrix<double> m_d_ov;
		
	void analyze_sparsity(dbcsr::shared_matrix<double> u_ia, 
		dbcsr::shared_matrix<double> c_loc_o, dbcsr::shared_matrix<double> u_loc_o,
		dbcsr::shared_matrix<double> c_loc_v, dbcsr::shared_matrix<double> u_loc_v); 
	
	void init();
	void init_ao_tensors();
	void init_mo_tensors();
	
	void compute_diag();
	dbcsr::shared_matrix<double> compute_diag_0();
	dbcsr::shared_matrix<double> compute_diag_1();
	
public:	

	adcmod(dbcsr::world w, hf::shared_hf_wfn hfref, desc::options& opt);
	~adcmod() {}
	
	void compute();
	
};

} // end namespace

#endif
	
