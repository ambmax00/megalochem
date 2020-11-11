#ifndef ADC_MVP_H
#define ADC_MVP_H

#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_tensor_ops.hpp>
#include "utils/ppdirs.h"
#include "desc/options.h"
#include "fock/jkbuilder.h"
#include "mp/z_builder.h"
#include "ints/fitting.h"
#include "adc/adc_defaults.h"

namespace adc {
	
enum class mvpmethod {
	invalid,
	ao_adc_1,
	ao_adc_2
};

inline mvpmethod str_to_mvpmethod(std::string str) {
	if (str == "ao_adc_1") {
		return mvpmethod::ao_adc_1;
	} else if (str == "ao_adc_2") {
		return mvpmethod::ao_adc_2;
	} else {
		return mvpmethod::invalid;
	}
}
	
using smat = dbcsr::shared_matrix<double>;
using stensor2 = dbcsr::shared_tensor<2,double>;
using stensor3 = dbcsr::shared_tensor<3,double>;
using sbtensor3 = dbcsr::sbtensor<3,double>;

class MVP {
protected:

	dbcsr::world m_world;
	desc::smolecule m_mol;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	smat compute_sigma_0(smat& u_ia, vec<double> epso, vec<double> epsv);
	
	smat u_transform(smat& u_ao, char to, smat& c_bo, char tv, smat& c_bv);

public:

	MVP(dbcsr::world w, desc::smolecule smol, int nprint, std::string name);
		
	virtual smat compute(smat u_ia, double omega = 0.0) = 0;
	
	virtual void init() = 0;
	
	virtual ~MVP() {}
	
	virtual void print_info() = 0;
	
};

class create_MVPAOADC1_base;

class MVPAOADC1 : public MVP {
private:
	
	std::shared_ptr<fock::J> m_jbuilder;
	std::shared_ptr<fock::K> m_kbuilder;
	
	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::sbtensor<3,double> m_fitting_batched;
	dbcsr::shared_matrix<double> m_v_xx;
	
	svector<double> m_eps_occ;
	svector<double> m_eps_vir;

	smat m_c_bo;
	smat m_c_bv;
	
	fock::kmethod m_kmethod;
	fock::jmethod m_jmethod;

	friend class create_MVPAOADC1_base;

public:

	MVPAOADC1(dbcsr::world w, desc::smolecule smol, int nprint) : 
		MVP(w,smol,nprint,"MVPAOADC1") {}
		
	void init() override;
	
	smat compute(smat u_ia, double omega) override;
	
	void print_info() override {
		LOG.os<>("Timings for AO-ADC(1): \n");
		TIME.print_info();
		m_jbuilder->print_info();
		m_kbuilder->print_info();
	}
	
	~MVPAOADC1() override {}
	
};

MAKE_STRUCT(
	MVPAOADC1, MVP,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(fitting_batched, (dbcsr::sbtensor<3,double>), optional, val, nullptr),
		(v_xx, (dbcsr::shared_matrix<double>), optional, val, nullptr),
		(c_bo, (dbcsr::shared_matrix<double>), required, val),
		(c_bv, (dbcsr::shared_matrix<double>), required, val),
		(eps_occ, (std::shared_ptr<std::vector<double>>), required, val),
		(eps_vir, (std::shared_ptr<std::vector<double>>), required, val),
		(kmethod, (fock::kmethod), optional, val, fock::kmethod::dfao),
		(jmethod, (fock::jmethod), optional, val, fock::jmethod::dfao)
	)
)

/*
class MVP_ao_ri_adc2 : public MVP {
private:

	std::array<int,3> m_nbatches;
	dbcsr::btype m_bmethod;

	smat m_po_bb;
	smat m_pv_bb;
	smat m_s_xx_inv;
	
	smat m_i_oo;
	smat m_i_vv;
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	
	std::vector<double> m_weigths, m_weights_dd;
	std::vector<double> m_xpoints, m_xpoints_dd;
	int m_nlap;
	
	std::vector<smat> m_pseudo_occs;
	std::vector<smat> m_pseudo_virs;
	std::vector<smat> m_pseudo_occs_dd;
	std::vector<smat> m_pseudo_virs_dd;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
	std::shared_ptr<fock::J> m_jbuilder;
	std::shared_ptr<fock::K> m_kbuilder;
	std::shared_ptr<mp::Z> m_zbuilder;
	
	double m_c_os;
	double m_c_osc;
	
	void compute_intermeds();
	std::pair<smat,smat> compute_jk(smat& u_ao);
	smat compute_sigma_1(smat& jmat, smat& kmat);
	smat compute_sigma_2a(smat& u_ia);
	smat compute_sigma_2b(smat& u_ia);
	smat compute_sigma_2c(smat& jmat, smat& kmat);
	smat compute_sigma_2d(smat& u_ia);
	smat compute_sigma_2e(smat& u_ia, double omega);
	
	dbcsr::sbtensor<3,double> compute_J(smat& u_ia);
	std::pair<smat,smat> compute_sigma_2e_ilap(dbcsr::sbtensor<3,double>& J_xbb_batched, 
		smat& FA, smat& FB, smat& pseudo_o, smat& pseudo_v
#if 0
	, double omega, int ilap
#endif	
	);
	
#if 0
	std::vector<smat> m_FS;
	stensor3 m_I;
#endif

	arrvec<int,2> m_bb;
	arrvec<int,3> m_xbb;
	
	std::shared_ptr<ints::dfitting> m_dfit;
	std::shared_ptr<Eigen::MatrixXi> m_shellpair_info;
	
public:

	MVP_ao_ri_adc2(dbcsr::world& w, desc::smolecule smol,
		desc::options opt, util::registry& reg,
		svector<double> epso, svector<double> epsv) :
		MVP(w,smol,opt,reg,epso,epsv) {}
		
	void init() override;
	
	smat compute(smat u_ia, double omega) override;
	
	~MVP_ao_ri_adc2() override {}

};*/
	

} // end namespace

#endif

