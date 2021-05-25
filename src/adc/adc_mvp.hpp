#ifndef ADC_MVP_H
#define ADC_MVP_H

#ifndef TEST_MACRO
#include "megalochem.hpp"
#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_tensor_ops.hpp>
#include "fock/jkbuilder.hpp"
#include "mp/z_builder.hpp"
#include "math/laplace/laplace_helper.hpp"
#endif

#include "utils/ppdirs.hpp"

namespace megalochem {

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

	megalochem::world m_world;
	dbcsr::cart m_cart;
	desc::shared_molecule m_mol;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	smat compute_sigma_0(smat& u_ia, vec<double> epso, vec<double> epsv);

public:

	MVP(megalochem::world w, desc::shared_molecule smol, int nprint, std::string name);
		
	virtual smat compute(smat u_ia, double omega = 0.0) = 0;
	
	virtual void init() = 0;
	
	virtual ~MVP() {}
	
	virtual void print_info() = 0;
	
};

class MVP_AORIADC1 : public MVP {
private:
	
	smat m_c_bo;
	smat m_c_bv;
	smat m_v_xx;
	
	sbtensor3 m_eri3c2e_batched;
	sbtensor3 m_fitting_batched;
	
	std::vector<double> m_eps_occ;
	std::vector<double> m_eps_vir;
	
	int m_nbatches_occ;
	
	fock::jmethod m_jmethod;
	fock::kmethod m_kmethod;
	
	std::shared_ptr<fock::J> m_jbuilder;
	std::shared_ptr<fock::K> m_kbuilder;

public:

#define AORIADC1_LIST (\
	((megalochem::world), set_world),\
	((desc::shared_molecule), set_molecule),\
	((util::optional<int>), print),\
	((dbcsr::shared_matrix<double>), c_bo),\
	((dbcsr::shared_matrix<double>), c_bv),\
	((dbcsr::shared_matrix<double>), metric_inv),\
	((std::vector<double>), eps_occ),\
	((std::vector<double>), eps_vir),\
	((dbcsr::sbtensor<3,double>),eri3c2e_batched),\
	((dbcsr::sbtensor<3,double>),fitting_batched),\
	((fock::kmethod), kmethod),\
	((fock::jmethod), jmethod),\
	((util::optional<int>), nbatches_occ))

	MAKE_PARAM_STRUCT(create, AORIADC1_LIST, ())
	MAKE_BUILDER_CLASS(MVP_AORIADC1, create, AORIADC1_LIST, ())
	
	MVP_AORIADC1(create_pack&& p) :
		MVP(p.p_set_world, p.p_set_molecule, (p.p_print) ? *p.p_print : 0,
			"MVP_AORIADC1"),
		m_c_bo(p.p_c_bo), 
		m_c_bv(p.p_c_bv), 
		m_v_xx(p.p_metric_inv),
		m_eri3c2e_batched(p.p_eri3c2e_batched),
		m_fitting_batched(p.p_fitting_batched),
		m_eps_occ(p.p_eps_occ), 
		m_eps_vir(p.p_eps_vir), 
		m_nbatches_occ(p.p_nbatches_occ ? *p.p_nbatches_occ : 5),
		m_jmethod(p.p_jmethod),
		m_kmethod(p.p_kmethod)
	{}
		
	void init() override;
	
	smat compute(smat u_ia, double omega) override;
	
	void print_info() override {
		LOG.os<>("Timings for AO-ADC(1): \n");
		TIME.print_info();
		m_jbuilder->print_info();
		m_kbuilder->print_info();
	}
	
	~MVP_AORIADC1() override {}
	
};

inline constexpr bool _use_doubles_ob = false; 

class MVP_AORISOSADC2 : public MVP {
private:
	
	// input:
	smat m_s_bb;
	smat m_c_bo;
	smat m_c_bv;
	smat m_v_xx;
	
	sbtensor3 m_eri3c2e_batched;
	sbtensor3 m_fitting_batched;
	
	std::vector<double> m_eps_occ;
	std::vector<double> m_eps_vir;
	
	fock::jmethod m_jmethod;
	fock::kmethod m_kmethod;
	mp::zmethod m_zmethod;
	
	dbcsr::btype m_btype;
	
	int m_nlap;
	double m_c_os;
	double m_c_os_coupling;
	int m_nbatches_occ;
	
	std::vector<int> m_o, m_v, m_b, m_x;
	
	// created with init()
	
	std::vector<double> m_weights, m_xpoints, 
		m_weights_dd, m_xpoints_dd;
		
	double m_prev_omega = -1;
	
	double m_old_omega = -1.0;
	
	std::shared_ptr<math::laplace_helper> m_laphelper_ss, 
		m_laphelper_dd;
	
	std::shared_ptr<fock::J> m_jbuilder;
	std::shared_ptr<fock::K> m_kbuilder;
	std::shared_ptr<mp::Z> m_zbuilder;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	std::shared_ptr<Eigen::MatrixXi> m_shellpairs;
	dbcsr::shared_matrix<double> m_s_sqrt_bb, m_s_invsqrt_bb;
	
	// adc 1
	std::pair<smat,smat> compute_jk(smat& u_ia, smat& u_ao);
	smat compute_sigma_1(smat& jmat, smat& kmat);
	
	// adc 2
	void compute_intermeds();
	
	smat compute_sigma_2a(smat& u_ia);
	
	smat compute_sigma_2b(smat& u_ia);
	
	smat compute_sigma_2c(smat& jmat, smat& kmat);
	
	smat compute_sigma_2d(smat& u_ia);
		
	smat compute_sigma_2e_OB(smat& u_ao, double omega);
	
	std::tuple<dbcsr::sbtensor<3,double>,dbcsr::sbtensor<3,double>>
		compute_laplace_batchtensors_OB(smat& u_ia, smat& L_bo, smat& pv_bb);
	
	std::tuple<dbcsr::shared_tensor<2,double>,dbcsr::shared_tensor<2,double>>
		compute_F_OB(dbcsr::sbtensor<3,double> eri_xob_batched,
		dbcsr::sbtensor<3,double> J_xob_batched,
		dbcsr::shared_matrix<double> L_bo);
		
	dbcsr::sbtensor<3,double> compute_I_OB(dbcsr::sbtensor<3,double>& eri,
		dbcsr::sbtensor<3,double>& J, dbcsr::shared_tensor<2,double>& F_A,
		dbcsr::shared_tensor<2,double>& F_B);
		
	std::tuple<smat,smat> compute_sigma_2e_ilap_OB(
		dbcsr::sbtensor<3,double>& I_xob_batched, 
		smat& L_bo, double omega);
		
	smat compute_sigma_2e_OV(smat& u_ao, double omega);
	
	std::tuple<dbcsr::sbtensor<3,double>,dbcsr::sbtensor<3,double>>
		compute_laplace_batchtensors_OV(smat& u_ia, smat& L_bo, smat& L_bv);
	
	std::tuple<dbcsr::shared_tensor<2,double>,dbcsr::shared_tensor<2,double>>
		compute_F_OV(dbcsr::sbtensor<3,double> eri_xov_batched,
		dbcsr::sbtensor<3,double> J_xov_batched);
		
	dbcsr::sbtensor<3,double> compute_I_OV(dbcsr::sbtensor<3,double>& eri,
		dbcsr::sbtensor<3,double>& J, dbcsr::shared_tensor<2,double>& F_A,
		dbcsr::shared_tensor<2,double>& F_B);
		
	std::tuple<smat,smat> compute_sigma_2e_ilap_OV(
		dbcsr::sbtensor<3,double>& I_xob_batched, 
		smat& L_bo, smat& L_bv, double omega);
	
	// intermediates
	smat m_i_oo;
	smat m_i_vv;
	
public:

	static inline bool TEST_MVP = false;

#define AORIADC2_LIST (\
	((megalochem::world), set_world),\
	((desc::shared_molecule), set_molecule),\
	((util::optional<int>), print),\
	((dbcsr::shared_matrix<double>), c_bo),\
	((dbcsr::shared_matrix<double>), c_bv),\
	((dbcsr::shared_matrix<double>), s_bb),\
	((dbcsr::shared_matrix<double>), metric_inv),\
	((std::vector<double>), eps_occ),\
	((std::vector<double>), eps_vir),\
	((dbcsr::sbtensor<3,double>),eri3c2e_batched),\
	((dbcsr::sbtensor<3,double>),fitting_batched),\
	((util::optional<int>), nbatches_occ),\
	((fock::kmethod), kmethod),\
	((mp::zmethod), zmethod),\
	((fock::jmethod), jmethod),\
	((dbcsr::btype), btype),\
	((int), nlap),\
	((double), c_os),\
	((double), c_os_coupling))

	MAKE_PARAM_STRUCT(create, AORIADC2_LIST, ())
	MAKE_BUILDER_CLASS(MVP_AORISOSADC2, create, AORIADC2_LIST, ())

	MVP_AORISOSADC2(create_pack&& p) :
		MVP(p.p_set_world, p.p_set_molecule, (p.p_print) ? *p.p_print : 0, 
			"MVP_AORISOSADC2"),
		m_s_bb(p.p_s_bb),
		m_c_bo(p.p_c_bo), 
		m_c_bv(p.p_c_bv), 
		m_v_xx(p.p_metric_inv), 
		m_eri3c2e_batched(p.p_eri3c2e_batched),
		m_fitting_batched(p.p_fitting_batched),
		m_eps_occ(p.p_eps_occ), 
		m_eps_vir(p.p_eps_vir), 
		m_jmethod(p.p_jmethod), 
		m_kmethod(p.p_kmethod), 
		m_zmethod(p.p_zmethod),
		m_btype(p.p_btype), 
		m_nlap((p.p_nlap)),
		m_c_os((p.p_c_os)),
		m_c_os_coupling((p.p_c_os_coupling)),
		m_nbatches_occ((p.p_nbatches_occ) ? *p.p_nbatches_occ : 5),
		m_o(p.p_c_bo->col_blk_sizes()),
		m_v(p.p_c_bv->col_blk_sizes()),
		m_b(p.p_c_bo->row_blk_sizes()),
		m_x(p.p_metric_inv->row_blk_sizes())
	{}
		
	void init() override;
	
	smat compute(smat u_ia, double omega) override;
	
	void print_info() override {}
	
	~MVP_AORISOSADC2() override {}
	
};

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

	MVP_ao_ri_adc2(dbcsr::cart& w, desc::shared_molecule smol,
		desc::options opt, util::registry& reg,
		svector<double> epso, svector<double> epsv) :
		MVP(w,smol,opt,reg,epso,epsv) {}
		
	void init() override;
	
	smat compute(smat u_ia, double omega) override;
	
	~MVP_ao_ri_adc2() override {}

};*/
	

} // end namespace

} // end namespace mega

#endif

