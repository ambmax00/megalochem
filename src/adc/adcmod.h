#ifndef ADC_ADCMOD_H 
#define ADC_ADCMOD_H

#include "desc/options.h"
#include "hf/hf_wfn.h"
#include "utils/mpi_time.h"
#include "adc/adc_defaults.h"
#include "utils/registry.h"
#include "ints/fitting.h"
#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>

#include <mpi.h>

namespace adc {
	
enum class method {
	invalid,
	//ri_adc_1,
	//ri_adc_2,
	//sos_ri_adc_2,
	ao_adc_1,
	ao_adc_2
};

static const std::map<std::string,method> method_map = 
{
	//{"ri_adc_1", method::ri_adc_1},
	//{"ri_adc_2", method::ri_adc_2},
	//{"sos_ri_adc_2", method::sos_ri_adc_2},
	{"ao_adc_1", method::ao_adc_1},
	{"ao_adc_2", method::ao_adc_2}
};
	
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
	
	method m_method;
	
	std::string m_jmethod, m_kmethod, m_zmethod;
	
	double m_c_os;
	double m_c_osc;
	
	util::registry m_reg;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<2> m_spgrid2_bo;
	dbcsr::shared_pgrid<2> m_spgrid2_bv;
	dbcsr::shared_pgrid<3> m_spgrid3_xbb;

	dbcsr::shared_matrix<double> m_d_ov;
	
	std::shared_ptr<ints::dfitting> m_dfit;
	
	void analyze_sparsity(dbcsr::shared_matrix<double> u_ia, 
		dbcsr::shared_matrix<double> c_loc_o, dbcsr::shared_matrix<double> u_loc_o,
		dbcsr::shared_matrix<double> c_loc_v, dbcsr::shared_matrix<double> u_loc_v); 
	
	void init();
	void init_ao_tensors();
	void init_mo_tensors();
	void compute_diag();
	
public:	

	adcmod(hf::shared_hf_wfn hfref, desc::options& opt, dbcsr::world& w);
	~adcmod() {}
	
	void compute();
	
};

} // end namespace

#endif
	
