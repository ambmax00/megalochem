#ifndef ADC_MVP_H
#define ADC_MVP_H

#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_tensor_ops.hpp>
#include "utils/registry.h"
#include "desc/options.h"
#include "fock/jkbuilder.h"
#include "adc/adc_defaults.h"

namespace adc {
	
using smat = dbcsr::shared_matrix<double>;
using stensor2 = dbcsr::shared_tensor<2,double>;
using stensor3 = dbcsr::shared_tensor<3,double>;
using sbtensor3 = dbcsr::sbtensor<3,double>;

class MVP {
protected:

	dbcsr::world m_world;
	desc::smolecule m_mol;
	desc::options m_opt;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	util::registry m_reg;
	
	svector<double> m_epso;
	svector<double> m_epsv;
		
	vec<int> m_o, m_v, m_b, m_x;
	
	smat compute_sigma_0(smat& u_ia);
	
	smat u_transform(smat& u_ao, char to, smat& c_bo, char tv, smat& c_bv);

public:

	MVP(dbcsr::world& w, desc::smolecule smol, 
		desc::options opt, util::registry& reg,
		svector<double> epso, svector<double> epsv) :
		m_world(w), m_mol(smol), m_opt(opt), 
		LOG(w.comm(), m_opt.get<int>("print", ADC_PRINT_LEVEL)),
		TIME(w.comm(), "MVP"),
		m_reg(reg), m_epso(epso), m_epsv(epsv) {}
		
	virtual smat compute(smat u_ia, double omega = 0.0) = 0;
	
	virtual void init() = 0;
	
	virtual ~MVP() {}
	
};

class MVP_ao_ri_adc1 : public MVP {
private:
	
	std::shared_ptr<fock::J> m_jbuilder;
	std::shared_ptr<fock::K> m_kbuilder;
	
	smat m_c_bo;
	smat m_c_bv;

public:

	MVP_ao_ri_adc1(dbcsr::world& w, desc::smolecule smol,
		desc::options opt, util::registry& reg,
		svector<double> epso, svector<double> epsv) :
		MVP(w,smol,opt,reg,epso,epsv) {}
		
	void init() override;
	
	smat compute(smat u_ia, double omega) override;
	
	~MVP_ao_ri_adc1() override {}
	
};
	

} // end namespace

#endif

