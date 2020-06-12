#include "mp/mpmod.h"
#include "mp/mp_defaults.h"
#include "math/laplace/laplace.h"

namespace mp {
	
mpmod::mpmod(desc::shf_wfn& wfn_in, desc::options& opt_in, dbcsr::world& w_in) :
	m_hfwfn(wfn_in),
	m_opt(opt_in),
	m_world(w_in),
	LOG(m_world.comm(),m_opt.get<int>("print", MP_PRINT_LEVEL)),
	TIME(m_world.comm(), "Moller Plesset", LOG.global_plev())
{
	
	
}

void mpmod::compute() {
	
	// get energies
	auto eps_o = m_hfwfn->eps_occ_A();
	auto eps_v = m_hfwfn->eps_vir_A();
	
	int nlap = m_opt.get<int>("nlap",MP_NLAP);
	
	// laplace
	double emin = eps_o->front();
	double ehomo = eps_o->back();
	double elumo = eps_v->front();
	double emax = eps_v->back();
	
	math::laplace lp(nlap, emin, ehomo, elumo, emax);
	
	lp.compute();
	
	// integrals
	//ints::aofactory fac(m_hfwfn->mol(), m_world);
	
	//auto B_mn = fac.ao_3c2e();
	//auto M_xy = 

}

} // end namespace 
