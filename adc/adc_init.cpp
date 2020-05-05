#include "adc/adcmod.h"
#include "ints/registry.h"
#include "ints/aofactory.h"
#include "ints/gentran.h"

namespace adc {
	
void adcmod::mo_load() {
	
	LOG.os<>("Constructing MO integrals...\n");
	
	dbcsr::pgrid<2> grid2(m_comm);
	
	auto epso = m_hfwfn->eps_occ_A();
	auto epsv = m_hfwfn->eps_vir_A();
	
	m_mo.eps_o = epso;
	m_mo.eps_v = epsv;

	ints::aofactory aofac(*m_hfwfn->mol(), m_comm);
	
	LOG.os<>("-- Loading 3c2e AO integrals\n"); 
	auto aoints = aofac.op("coulomb").dim("xbb").map1({0}).map2({1,2}).compute<3>(); 
	
	LOG.os<>("-- Loading df overlap integrals\n");
	auto metric = aofac.op("coulomb").dim("xx").map1({0}).map2({1}).compute<2>();
	
	int nocc = m_hfwfn->mol()->nocc_alpha();
	int nvir = m_hfwfn->mol()->nvir_alpha();
	
	auto c_bo = m_hfwfn->c_bo_A();
	auto c_bv = m_hfwfn->c_bv_A();
	
	vec<int> d = {1};
	
	LOG.os<>("-- Inverting metric matrix and contracting...\n");
	auto PQSQRT = aofac.invert(metric,2);
	
	dbcsr::pgrid<3> grid3(m_comm);
	
	// contract X
	arrvec<int,3> xbb = {m_dims.x, m_dims.b, m_dims.b};
	
	dbcsr::stensor<3> d_xbb = dbcsr::make_stensor<3>(
		dbcsr::tensor<3>::create().name("i_xbb(xx)^-1/2").ngrid(grid3) 
			.map1({0}).map2({1,2}).blk_sizes(xbb)); 
			
	dbcsr::contract(*aoints, *PQSQRT, *d_xbb).move(true).perform("XMN, XY -> YMN");
	
	aoints.reset();
	
	dbcsr::print(*c_bo);
	dbcsr::print(*c_bv);
	
	LOG.os<>("-- AO -> MO transformation...\n");
	
	m_mo.b_xoo = ints::transform3(d_xbb, c_bo, c_bo, "i_xoo(xx)^-1/2");
	m_mo.b_xov = ints::transform3(d_xbb, c_bo, c_bv, "i_xov(xx)^-1/2");
	m_mo.b_xvv = ints::transform3(d_xbb, c_bv, c_bv, "i_xvv(xx)^-1/2");
	
	d_xbb->destroy();

	mo_amplitudes();
	
	LOG.os<>("Finished computing MO quantities.\n\n");
		
}
		
adcmod::adcmod(desc::shf_wfn hfref, desc::options& opt, MPI_Comm comm) :
	m_hfwfn(hfref), 
	m_opt(opt), 
	m_comm(comm),
	m_nroots(m_opt.get<int>("nroots", ADC_NROOTS)),
	m_order(m_opt.get<int>("order", ADC_ORDER)),
	m_use_ao(m_opt.get<bool>("use_ao", ADC_USE_AO)),
	m_use_sos(m_opt.get<bool>("use_sos", ADC_USE_SOS)),
	m_use_lp(m_opt.get<bool>("use_lp", ADC_USE_LP)),
	m_diag_order(m_opt.get<int>("diag_order", m_order)),
	m_c_os(m_opt.get<double>("c_os", ADC_C_OS)),
	m_c_osc(m_opt.get<double>("c_os_coupling", ADC_C_OS_COUPLING)),
	LOG(comm, m_opt.get<int>("print", ADC_PRINT_LEVEL)),
	TIME(comm, "ADC Module", LOG.global_plev())
{
	
	LOG.banner("ADC MODULE",50,'*');	
	
	m_dims.o = m_hfwfn->mol()->dims().oa();
	m_dims.v = m_hfwfn->mol()->dims().va();
	m_dims.b = m_hfwfn->mol()->dims().b();
	m_dims.x = m_hfwfn->mol()->dims().x();
	
	if (!m_use_ao) {
		mo_load();
	} else {
		//ao_load();
	}
	
	LOG.os<>("--- Ready for launching computation. --- \n\n");
	
}

} // end namespace
