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

class MVP_ao_ri_adc2 : public MVP {
private:

	smat m_c_bo;
	smat m_c_bv;
	smat m_s_xx_inv;
	
	smat m_i_oo;
	smat m_i_vv;
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	
	std::vector<double> m_omega;
	std::vector<double> m_alpha;
	int m_nlap;
	
	std::vector<smat> m_pseudo_occs;
	std::vector<smat> m_pseudo_virs;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
	std::shared_ptr<fock::J> m_jbuilder;
	std::shared_ptr<fock::K> m_kbuilder;
	
	double m_c_os;
	double m_c_osc;
	
	void compute_intermeds();
	std::pair<smat,smat> compute_jk(smat& u_ao);
	smat compute_sigma_1(smat& jmat, smat& kmat);
	smat compute_sigma_2a(smat& u_ia);
	smat compute_sigma_2b(smat& u_ia);
	smat compute_sigma_2c(smat& jmat, smat& kmat);
	//void compute_sigma_2d();
	//void compute_sigma_2e();
	
public:

	MVP_ao_ri_adc2(dbcsr::world& w, desc::smolecule smol,
		desc::options opt, util::registry& reg,
		svector<double> epso, svector<double> epsv) :
		MVP(w,smol,opt,reg,epso,epsv) {}
		
	void init() override;
	
	smat compute(smat u_ia, double omega) override;
	
	~MVP_ao_ri_adc2() override {}
	
};
	

} // end namespace

#endif

