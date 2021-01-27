#ifndef ADC_MVP_H
#define ADC_MVP_H

#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_tensor_ops.hpp>
#include "utils/ppdirs.h"
#include "desc/options.h"
#include "fock/jkbuilder.h"
#include "mp/z_builder.h"
#include "adc/adc_defaults.h"

namespace adc {
	
enum class mvpmethod {
	ao_adc_1,
	ao_adc_2
};

inline mvpmethod str_to_mvpmethod(std::string str) {
	if (str == "ao_adc_1") {
		return mvpmethod::ao_adc_1;
	} else if (str == "ao_adc_2") {
		return mvpmethod::ao_adc_2;
	} else {
		throw std::runtime_error("Invalid MVP method");
	}
}
	
using smat = dbcsr::shared_matrix<double>;
using stensor2 = dbcsr::shared_tensor<2,double>;
using stensor3 = dbcsr::shared_tensor<3,double>;
using sbtensor3 = dbcsr::sbtensor<3,double>;

smat u_transform(smat& u_ao, char to, smat& c_bo, char tv, smat& c_bv);

class MVP {
protected:

	dbcsr::world m_world;
	desc::smolecule m_mol;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	smat compute_sigma_0(smat& u_ia, vec<double> epso, vec<double> epsv);

public:

	MVP(dbcsr::world w, desc::smolecule smol, int nprint, std::string name);
		
	virtual smat compute(smat u_ia, double omega = 0.0) = 0;
	
	virtual void init() = 0;
	
	virtual ~MVP() {}
	
	virtual void print_info() = 0;
	
};

class create_MVP_AOADC1_base;

class MVP_AOADC1 : public MVP {
private:
	
	fock::jmethod m_jmethod;
	fock::kmethod m_kmethod;
	
	sbtensor3 m_eri3c2e_batched;
	sbtensor3 m_fitting_batched;
	smat m_v_xx;
	
	std::shared_ptr<fock::J> m_jbuilder;
	std::shared_ptr<fock::K> m_kbuilder;
	
	std::vector<double> m_eps_occ;
	std::vector<double> m_eps_vir;

	smat m_c_bo;
	smat m_c_bv;

	friend class create_MVP_AOADC1_base;

public:

	MVP_AOADC1(dbcsr::world w, desc::smolecule smol, int nprint) : 
		MVP(w,smol,nprint,"MVP_AOADC1") {}
		
	void init() override;
	
	smat compute(smat u_ia, double omega) override;
	
	void print_info() override {
		LOG.os<>("Timings for AO-ADC(1): \n");
		TIME.print_info();
		m_jbuilder->print_info();
		m_kbuilder->print_info();
	}
	
	~MVP_AOADC1() override {}
	
};

MAKE_STRUCT(
	MVP_AOADC1, MVP,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(c_bo, (dbcsr::shared_matrix<double>), required, val),
		(c_bv, (dbcsr::shared_matrix<double>), required, val),
		(eps_occ, (std::vector<double>), required, val),
		(eps_vir, (std::vector<double>), required, val),
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(fitting_batched, (dbcsr::sbtensor<3,double>), optional, val, nullptr),
		(v_xx, (dbcsr::shared_matrix<double>), required, val),
		(kmethod, (fock::kmethod), required, val),
		(jmethod, (fock::jmethod), required, val)
	)
)

class create_MVP_AOADC2_base;

class MVP_AOADC2 : public MVP {
private:
	
	// input:
	svector<double> m_eps_occ;
	svector<double> m_eps_vir;

	smat m_s_bb;
	smat m_c_bo;
	smat m_c_bv;
	smat m_v_xx;
	
	sbtensor3 m_eri3c2e_batched;
	sbtensor3 m_fitting_batched;
	
	fock::jmethod m_jmethod;
	fock::kmethod m_kmethod;
	mp::zmethod m_zmethod;
	
	int m_nlap;
	double m_c_os;
	double m_c_os_coupling;
	
	dbcsr::btype m_btype;
	
	// created with init()
	std::vector<double> m_weights, m_xpoints, 
		m_weights_dd, m_xpoints_dd;
	std::shared_ptr<fock::J> m_jbuilder;
	std::shared_ptr<fock::K> m_kbuilder;
	std::shared_ptr<mp::Z> m_zbuilder;
	std::vector<smat> m_pseudo_occs, m_pseudo_virs;
	dbcsr::shared_pgrid<2> m_spgrid2;
	std::shared_ptr<Eigen::MatrixXi> m_shellpairs;
	dbcsr::shared_matrix<double> m_ssqrt_bb, m_sinvqrt_bb;
	
	// private functions
	smat get_scaled_coeff(char dim, double wght, double xpt, 
		double wfactor, double xfactor);
	smat get_ortho_cholesky(char dim, double wght, double xpt, 
		double wfactor, double xfactor);
	smat get_density(smat coeff);
	
	// adc 1
	std::pair<smat,smat> compute_jk(smat& u_ao);
	smat compute_sigma_1(smat& jmat, smat& kmat);
	
	// adc 2
	void compute_intermeds();
	
	smat compute_sigma_2a(smat& u_ia);
	
	smat compute_sigma_2b(smat& u_ia);
	
	smat compute_sigma_2c(smat& jmat, smat& kmat);
	
	smat compute_sigma_2d(smat& u_ia);
		
	smat compute_sigma_2e(smat& u_ao, double omega);
	
	std::tuple<dbcsr::sbtensor<3,double>,dbcsr::sbtensor<3,double>>
		compute_laplace_batchtensors(smat& u_ia, smat& L_bo, smat& pv_bb);
	
	std::tuple<dbcsr::shared_tensor<2,double>,dbcsr::shared_tensor<2,double>>
		compute_F(dbcsr::sbtensor<3,double> eri_xob_batched,
		dbcsr::sbtensor<3,double> J_xob_batched,
		dbcsr::shared_matrix<double> L_bo);
		
	dbcsr::sbtensor<3,double> compute_I(dbcsr::sbtensor<3,double>& eri,
		dbcsr::sbtensor<3,double>& J, dbcsr::shared_tensor<2,double>& F_A,
		dbcsr::shared_tensor<2,double>& F_B);
		
	std::tuple<smat,smat> compute_sigma_2e_ilap(
		dbcsr::sbtensor<3,double>& I_xob_batched, 
		smat& L_bo, double omega);
	
	// intermediates
	smat m_i_oo;
	smat m_i_vv;
	
	friend class create_MVP_AOADC2_base;

public:

	MVP_AOADC2(dbcsr::world w, desc::smolecule smol, int nprint) : 
		MVP(w,smol,nprint,"MVP_AOADC2") {}
		
	void init() override;
	
	smat compute(smat u_ia, double omega) override;
	
	void print_info() override {}
	
	~MVP_AOADC2() override {}
	
};

MAKE_STRUCT(
	MVP_AOADC2, MVP,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(c_bo, (dbcsr::shared_matrix<double>), required, val),
		(c_bv, (dbcsr::shared_matrix<double>), required, val),
		(s_bb, (dbcsr::shared_matrix<double>), required, val),
		(v_xx, (dbcsr::shared_matrix<double>), required, val),
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(fitting_batched, (dbcsr::sbtensor<3,double>), optional, val, nullptr),
		(eps_occ, (std::shared_ptr<std::vector<double>>), required, val),
		(eps_vir, (std::shared_ptr<std::vector<double>>), required, val),
		(kmethod, (fock::kmethod), required, val),
		(jmethod, (fock::jmethod), required, val),
		(zmethod, (mp::zmethod), required, val),
		(btype, (dbcsr::btype), required, val),
		(nlap, (int), optional, val, ADC_ADC2_NLAP),
		(c_os, (double), optional, val, ADC_ADC2_C_OS),
		(c_os_coupling, (double), optional, val, ADC_ADC2_C_OS_COUPLING)
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

