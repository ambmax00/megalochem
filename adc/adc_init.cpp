#include "adc/adcmod.h"

namespace adc {

adcmod::adcmod(desc::shf_wfn hfref, desc::options opt, MPI_Comm comm) :
	m_hfwfn(hfref), 
	m_opt(opt), 
	m_comm(comm),
	m_nroots(ADC_NROOTS),
	LOG(comm, ADC_PRINT_LEVEL),
	TIME(comm, "ADC Module", LOG.global_plev())
{
		
	// something will happen here I guess	
		
		
}

} // end namespace
