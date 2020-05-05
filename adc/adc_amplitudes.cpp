#include "adc/adcmod.h"
#include "adc/adc_ops.h"

namespace adc {
			
void adcmod::mo_amplitudes() {
	
	// form 2e electron integrals
	
	if (m_use_lp) {
		LOG.os<>("Generating amplitudes on the fly.\n\n");
		return;
	}
	
	LOG.os<>("Computing MO amplitudes...\n");
	
	dbcsr::pgrid<4> grid4(m_comm);
	arrvec<int,4> blksizes = {m_dims.o, m_dims.v, m_dims.o, m_dims.v};
	
	dbcsr::tensor<4> mo_ints = dbcsr::tensor<4>::create().name("t_ovov").ngrid(grid4)
		.map1({0,1}).map2({2,3}).blk_sizes(blksizes);
		
	dbcsr::contract(*m_mo.b_xov, *m_mo.b_xov, mo_ints).perform("Xia, Xjb -> iajb");
	
	adc::scale<double>(mo_ints,*m_mo.eps_o, *m_mo.eps_v);

	if (!m_use_sos) {
		LOG.os<>("-- Antisymmetrizing...\n");
		adc::antisym<double>(mo_ints,2.0);
	} else {
		LOG.os<>("-- Scaling amplitudes: ", m_c_os, '\n');
		mo_ints.scale(m_c_os);
	}
	
	if (LOG.global_plev() >= 2)
		dbcsr::print(mo_ints);
	
	m_mo.t_ovov = mo_ints.get_stensor();
	
}

} // end namespace
