#include "adc/adcmod.h"
#include "ints/registry.h"
#include "ints/aofactory.h"
#include "ints/gentran.h"

namespace adc {
	
void adcmod::mo_load() {
	
	dbcsr::pgrid<2> grid2({.comm = m_comm});
	
	auto epso = m_hfwfn->eps_occ_A();
	auto epsv = m_hfwfn->eps_vir_A();
	
	std::cout << "Energy: " << std::endl;
	for (auto e : *epsv) {
		std::cout << e << std::endl;
	}
	
	m_mo.eps_o = epso;
	m_mo.eps_v = epsv;
	
	std::cout << "H1" << std::endl;
	
	
	
	ints::aofactory aofac(*m_hfwfn->mol(), m_comm);
		
	ints::registry INT_REGISTRY;
	
	auto aoints = aofac.compute<3>({.op = "coulomb", .bas = "xbb", .map1 = {0}, .map2 = {1,2}});
	auto metric = aofac.compute<2>({.op = "coulomb", .bas = "xx", .map1 = {0}, .map2 = {1}});
	
	int nocc = m_hfwfn->mol()->nocc_alpha();
	int nvir = m_hfwfn->mol()->nvir_alpha();
	
	auto c_bo = m_hfwfn->c_bo_A();
	auto c_bv = m_hfwfn->c_bv_A();
	
	vec<int> d = {1};
	
	auto PQSQRT = aofac.invert(metric,2);
	
	//dbcsr::print(*aoints);
	//dbcsr::print(*PQSQRT);
	
	// contract X
	dbcsr::pgrid<3> grid3({.comm = m_comm});
	
	dbcsr::stensor<3> d_xbb = dbcsr::make_stensor<3>({.name = "i_xbb(xx)^-1/2", .pgridN = grid3, 
			.map1 = {0}, .map2 = {1,2}, .blk_sizes = {m_dims.x, m_dims.b, m_dims.b}});
			
	dbcsr::einsum<3,2,3>({.x = "XMN, XY -> YMN", .t1 = *aoints, .t2 = *PQSQRT, .t3 = *d_xbb, .move = true});
	
	aoints.reset();
	
	dbcsr::print(*c_bo);
	dbcsr::print(*c_bv);
	
	m_mo.b_xoo = ints::transform3({.t_in = *d_xbb, .c_1 = *c_bo, .c_2 = *c_bo, .name = "i_xoo(xx)^-1/2"});
	//dbcsr::print(*d_xoo);
	m_mo.b_xov = ints::transform3({.t_in = *d_xbb, .c_1 = *c_bo, .c_2 = *c_bv, .name = "i_xov(xx)^-1/2"});
	//dbcsr::print(*d_xov);
	m_mo.b_xvv = ints::transform3({.t_in = *d_xbb, .c_1 = *c_bv, .c_2 = *c_bv, .name = "i_xvv(xx)^-1/2"});
	//dbcsr::print(*d_xvv);
	
	d_xbb->destroy();
		
}

adcmod::adcmod(desc::shf_wfn hfref, desc::options& opt, MPI_Comm comm) :
	m_hfwfn(hfref), 
	m_opt(opt), 
	m_comm(comm),
	m_nroots(m_opt.get<int>("nroots", ADC_NROOTS)),
	m_c_os(m_opt.get<double>("c_os", ADC_C_OS)),
	m_c_osc(m_opt.get<double>("c_os_coupling", ADC_C_OS_COUPLING)),
	LOG(comm, ADC_PRINT_LEVEL),
	TIME(comm, "ADC Module", LOG.global_plev())
{
		
	// prepare integrals
	auto method = m_opt.get<std::string>("method", ADC_METHOD);
	
	m_dims.o = m_hfwfn->mol()->dims().oa();
	m_dims.v = m_hfwfn->mol()->dims().va();
	m_dims.b = m_hfwfn->mol()->dims().b();
	m_dims.x = m_hfwfn->mol()->dims().x();
	
	if (method == "MO") { 
		m_method = 0; 
		mo_load();
	} else if (method == "SOS-MO") {
		m_method = 1;
		mo_load();
	} else if (method == "AO") { 
		m_method = 10; 
		//load_ao();
	} else { 
		throw std::runtime_error("Unknown Method.");
	}
	
}

} // end namespace
