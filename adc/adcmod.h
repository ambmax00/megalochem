#ifndef ADC_ADCMOD_H 
#define ADC_ADCMOD_H

#include "desc/options.h"
#include "desc/wfn.h"
#include "tensor/dbcsr.hpp"
#include "utils/mpi_time.h"
#include "adc/adc_defaults.h"

#include <mpi.h>

namespace adc {

struct dims {
	
	vec<int> b, o, v, x;
	
};
	
struct moctx {
	
	svector<double> eps_o, eps_v;
	dbcsr::stensor<3> b_xoo, b_xov, b_xvv;
	dbcsr::stensor<4> t_ovov;
	dbcsr::stensor<2> d_ov; // matrix diagonal
	
};
	
class adcmod {
private:

	desc::shf_wfn m_hfwfn;
	desc::options m_opt;
	MPI_Comm m_comm;
	
	util::mpi_time TIME;
	util::mpi_log LOG;
	
	// have two structs: mo_ctx, ao_ctx
	dims m_dims;
	moctx m_mo;
	
	int m_method; // 0: MO-RI-ADC, 1: AO-RI-ADC
	
	int m_nroots; 
	
	double m_c_os;
	double m_c_osc;
	
	void mo_load();
	
	void mo_compute_diag();
	
	void mo_amplitudes();
	
	void antisym(dbcsr::tensor<4>& t, vec<int>& o, vec<int>& v);
	
	void scale(dbcsr::tensor<4>& t, vec<double>& eo, vec<double>& ev);
	
public:	

	adcmod(desc::shf_wfn hfref, desc::options& opt, MPI_Comm comm);
	~adcmod() {}
	
	void compute();
	
};

} // end namespace

#endif
	
