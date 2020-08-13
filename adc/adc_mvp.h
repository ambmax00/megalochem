#ifndef ADC_MVP_H
#define ADC_MVP_H

#include <dbcsr_matrix.hpp>
#include "utils/registry.h"
#include "desc/options.h"

namespace adc {
	
using smat = dbcsr::shared_matrix<double>;
using stensor2 = dbcsr::shared_tensor<2,double>;
using stensor3 = dbcsr::shared_tensor<3,double>;
using sbtensor3 = dbcsr::sbtensor<3,double>
	
class MVP {
private:

	dbcsr::world m_world;
	
	desc::options m_opt;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	util::registry m_reg;
	
	svector<double> m_epso;
	svector<double> m_epsv;
		
	vec<int> m_o, m_v, m_b, m_x;

public:

	MVP(dbcsr::world w, desc::options opt, util::registry reg,
		svector<double> epso, svector<double> epsv) :
		m_world(w), m_opt(opt), LOG(w.comm(), m_opt.get<int>("print")),
		m_reg(reg), m_epso(epso), m_epsv(epsv) {}
		
	virtual smat compute(smat u_ia, double omega);
	
	virtual void init();
	
	virtual ~MVP();
	
};

class MVP_ri_adc1 : MVP {
private:

	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<3> m_spgrid3;
	
	stensor m_d_xoo, m_d_xov, m_d_xvv;
	
	stensor2 m_c_xd;
	stensor3 m_c_xov;

public:

	MVP_ri_adc1(dbcsr::world w, desc::options opt, util::registry reg,
		svector<double> epso, svector<double> epsv) :
		MVP(w,opt,reg,epso,epsv) {}
		
	void init() override;
	
	smat compute(smat u_ia, double omega) override;
	
	~MVP_ri_adc1() {}
	
};
	

} // end namespace

