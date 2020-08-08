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
	
class adcmod {
private:

	desc::shf_wfn m_hfwfn;
	desc::options m_opt;
	MPI_Comm m_comm;
	
	util::mpi_time TIME;
	util::mpi_log LOG;
	
	int m_order;
	bool m_use_ao;
	bool m_use_sos;
	bool m_use_lp;
	int m_diag_order;
	
	int m_nroots; 
	
	double m_c_os;
	double m_c_osc;
	
	util::registry m_reg;
	
	void init();
	
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
	
