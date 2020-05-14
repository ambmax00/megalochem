#include "adc/adcmod.h"
#include "adc/adc_ops.h"

namespace adc {
			
void adcmod::mo_amplitudes() {
	
	// form 2e electron integrals
	
	dbcsr::pgrid<4> grid4(m_comm);
	arrvec<int,4> blksizes = {m_dims.o, m_dims.v, m_dims.o, m_dims.v};
	
	dbcsr::tensor<4> mo_ints = dbcsr::tensor<4>::create().name("t_ovov").ngrid(grid4)
		.map1({0,1}).map2({2,3}).blk_sizes(blksizes);
		
	//dbcsr::einsum<3,3,4>({.x = "Xia, Xjb -> iajb", .t1 = *m_mo.b_xov, .t2 = *m_mo.b_xov, .t3 = mo_ints});
	dbcsr::contract(*m_mo.b_xov, *m_mo.b_xov, mo_ints).perform("Xia, Xjb -> iajb");
	
	dbcsr::print(mo_ints);
	
	adc::scale<double>(mo_ints,*m_mo.eps_o, *m_mo.eps_v);
	
	dbcsr::print(mo_ints);
	
	if (m_method == 0) {
		adc::antisym<double>(mo_ints,2.0);
	} else {
		mo_ints.scale(m_c_os);
	}
	dbcsr::print(mo_ints);
	
	m_mo.t_ovov = mo_ints.get_stensor();
	
}

} // end namespace
