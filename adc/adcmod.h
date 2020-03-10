#ifndef ADC_ADCMOD_H 
#define ADC_ADCMOD_H

#include "desc/options.h"
#include "desc/wfn.h"
#include "math/tensor/dbcsr.hpp"
#include "utils/mpi_time.h"
#include "adc/adc_defaults.h"

#include <mpi.h>

namespace adc {
	
class adcmod {
private:

	desc::shf_wfn m_hfwfn;
	desc::options m_opt;
	MPI_Comm m_comm;
	
	util::mpi_time TIME;
	util::mpi_log LOG;
	
	int m_nroots;
	
public:	

	adcmod(desc::shf_wfn hfref, desc::options opt, MPI_Comm comm);
	~adcmod() {}
	
	void compute();
	
};

} // end namespace

#endif
	
