#include "adc/adc_mvp.h"

namespace adc {

void MVP_ri_adc1::init() {
	
	// init external tensors
	
	m_d_xoo = m_reg.get_btensor<3,double>("d_xoo")->get_stensor();
	m_d_xov = m_reg.get_btensor<3,double>("d_xov")->get_stensor();
	m_d_xvv = m_reg.get_btensor<3,double>("d_xvv")->get_stensor();
	
	auto blksizes = m_d_xov->blk_sizes();
	
	m_x = blksizes[0];
	m_o = blksizes[1];
	m_v = blksizes[2];
	
	// init pgrids

	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	m_spgrid3 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	// init internal tensors
	
	arrvec<int,2> xd = {m_x, vec<int>{1}};
	arrvec<
	
	m_c_x = dbcsr::tensor_create().name("c_x")
		.pgrid(m_spgrid2).map1({0}).map2({1})
		.blk_sizes(xd).get();
		
	m_c_xov = dbcsr::tensor_create_template(*m_d_xov)
		.name("c_xov").get();
	
}

smat_d MVP_ri_adc1::compute(smat_d u_ia, double omega) {
	
	
	
	
	// c_X = b_Xia * u_ia
	
	return u_ia;
	
}

}
