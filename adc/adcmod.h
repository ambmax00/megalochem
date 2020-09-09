#ifndef ADC_ADCMOD_H 
#define ADC_ADCMOD_H

#include "desc/options.h"
#include "desc/wfn.h"
#include "utils/mpi_time.h"
#include "adc/adc_defaults.h"
#include "utils/registry.h"
#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>

#include <mpi.h>

namespace adc {
	
enum class method {
	invalid,
	ri_adc_1,
	ri_adc_2,
	sos_ri_adc_2,
	ao_ri_adc_1,
	ao_ri_adc_2
};

static const std::map<std::string,method> method_map = 
{
	{"ri_adc_1", method::ri_adc_1},
	{"ri_adc_2", method::ri_adc_2},
	{"sos_ri_adc_2", method::sos_ri_adc_2},
	{"ao_ri_adc_1", method::ao_ri_adc_1},
	{"ao_ri_adc_2", method::ao_ri_adc_2}
};
	
class adcmod {
private:

	desc::shf_wfn m_hfwfn;
	desc::options m_opt;
	dbcsr::world m_world;
	
	util::mpi_time TIME;
	util::mpi_log LOG;
	
	int m_order;
	int m_diag_order;
	
	int m_nroots; 
	
	method m_method;
	
	double m_c_os;
	double m_c_osc;
	
	util::registry m_reg;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<2> m_spgrid2_bo;
	dbcsr::shared_pgrid<2> m_spgrid2_bv;
	dbcsr::shared_pgrid<3> m_spgrid3_xbb;
	dbcsr::shared_pgrid<3> m_spgrid3_xoo;
	dbcsr::shared_pgrid<3> m_spgrid3_xvv;
	dbcsr::shared_pgrid<3> m_spgrid3_xov;
	
	dbcsr::shared_matrix<double> m_d_ov;
	
	void init();
	void init_ao_tensors();
	void init_mo_tensors();
	void compute_diag();
	
	//void mo_load();
	
	//void mo_compute_diag_0();
	//void mo_compute_diag_1();
	//void mo_compute_diag();
	
	//void mo_amplitudes();
	
	//void antisym(dbcsr::tensor<4>& t, vec<int>& o, vec<int>& v);
	
	//void scale(dbcsr::tensor<4>& t, vec<double>& eo, vec<double>& ev);
	
public:	

	adcmod(desc::shf_wfn hfref, desc::options& opt, dbcsr::world& w);
	~adcmod() {}
	
	void compute();
	
};

} // end namespace

#endif
	
