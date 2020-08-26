#include "adc/adc_mvp.h"
#include "math/laplace/laplace.h"
#include "math/linalg/piv_cd.h"
#include "mp/z_builder.h"
#include "adc/adc_defaults.h"

namespace adc {
	
void MVP_ao_ri_adc2::compute_intermeds() {
	
	/* computes
	 *   
	 * I_ab = 0.5 * sum_kcl [ c_os t_kalc (kb|lc) ] a <-> b
	 * I_ij = 0.5 * sum_ckd [ c_os t_ickd (jc|kd) ] i <-> j
	 * 
	 * In AO steps:
	 * 
	 * F(t)_XY = M_XX' Z_X'Y'(t) * M_YY'
	 * 
	 * I_ab :
	 * K kernel with P = Pocc(t) and s_xx_inv = F(t)_XY
	 * I_ab = antisym[ C_Bb sum_t C_Aa exp(t) * K_AB ]
	 * 
	 * I_ij
	 * K kernel with P = Pvir(t) and s_xx_inv = F(t)_XY
	 * I_ij = antisym[ C_Jj sum_t C_Ii exp(t) * K_IJ ]
	 * 
	 */
	 
	LOG.os<>("Computing intermediates.\n");
	 
	LOG.os<>("Setting up ZBUILDER.\n"); 
	
	std::string zmethod = m_opt.get<std::string>("build_Z", ADC_BUILD_Z);
	auto zbuilder = mp::get_Z(zmethod, m_world, m_opt);
	
	zbuilder->set_reg(m_reg);
	zbuilder->init_tensors();
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	auto o = m_mol->dims().oa();
	auto v = m_mol->dims().va();
	
	arrvec<int,2> xx = {x,x};
	
	m_i_oo = dbcsr::create<double>()
		.name("I_ij")
		.set_world(m_world)
		.row_blk_sizes(o)
		.col_blk_sizes(o)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	m_i_vv = dbcsr::create<double>()
		.name("I_ab")
		.set_world(m_world)
		.row_blk_sizes(v)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto i_ob = dbcsr::create<double>()
		.name("I_ij_part")
		.set_world(m_world)
		.row_blk_sizes(o)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto i_vb = dbcsr::create<double>()
		.name("I_ab_part")
		.set_world(m_world)
		.row_blk_sizes(v)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto i_oo_tmp = dbcsr::create_template(m_i_oo)
		.name("i_oo_temp").get();
		
	auto i_vv_tmp = dbcsr::create_template(m_i_vv)
		.name("i_vv_temp").get();
	
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		LOG.os<>("LAPLACE POINT ", ilap, '\n');
		
		auto po = m_pseudo_occs[ilap];
		auto pv = m_pseudo_virs[ilap];
		
		math::pivinc_cd chol(po, LOG.global_plev());
		
		chol.compute();
		
		int rank = chol.rank();
		
		auto u = dbcsr::split_range(rank, m_mol->mo_split());
		
		LOG.os<>("Cholesky decomposition rank: ", rank, '\n');
	
		auto L_bu = chol.L(b, u);
		
		L_bu->filter(dbcsr::global::filter_eps);

		pv->filter(dbcsr::global::filter_eps);
		
		zbuilder->set_occ_coeff(L_bu);
		zbuilder->set_vir_density(pv);
	
		LOG.os<>("Computing Z.\n");
	
		zbuilder->compute();
		auto z_xx_ilap = zbuilder->zmat();
		
		dbcsr::print(*z_xx_ilap);
		
		auto temp = dbcsr::create_template(z_xx_ilap)
			.name("temp_xx").get();
			
		dbcsr::multiply('N', 'N', *m_s_xx_inv, *z_xx_ilap, *temp)
			.filter_eps(dbcsr::global::filter_eps)
			.perform();
			
		dbcsr::multiply('N', 'N', *temp, *m_s_xx_inv, *z_xx_ilap)
			.filter_eps(dbcsr::global::filter_eps)
			.perform();
			
		auto f_xx_ilap = z_xx_ilap;
		f_xx_ilap->setname("f_xx_ilap");
		
		dbcsr::print(*f_xx_ilap);
		
		auto f_xx_01 = dbcsr::tensor_create<2,double>()
			.name("f_xx_ilap_tensor")
			.pgrid(m_spgrid2)
			.map1({0}).map2({1})
			.blk_sizes(xx)
			.get();
			
		dbcsr::copy_matrix_to_tensor(*f_xx_ilap, *f_xx_01);
		//f_xx_ilap->release();
		
		util::registry reg_ilap;
		
		reg_ilap.insert_btensor<3,double>("i_xbb_batched", m_eri_batched);
		reg_ilap.insert_tensor<2,double>("s_xx_inv", f_xx_01);
		
		std::string kmethod = m_opt.get<std::string>("build_K", ADC_BUILD_K);
		
		LOG.os<>("Setting up K_ilap.\n");
		
		auto kbuilder_ilap = fock::get_K(kmethod, m_world, m_opt);
		
		kbuilder_ilap->set_reg(reg_ilap);
		kbuilder_ilap->set_mol(m_mol);
		kbuilder_ilap->set_sym(false);
		
		kbuilder_ilap->init();
		kbuilder_ilap->init_tensors();
		kbuilder_ilap->set_density_alpha(pv);
		
		LOG.os<>("Computing K_ilap.\n");
		
		kbuilder_ilap->compute_K();
		
		auto ko_ilap = kbuilder_ilap->get_K_A();
		
		auto c_bo_eps = dbcsr::create_template(m_c_bo)
			.name("c_bo_eps")
			.get();
			
		auto c_bv_eps = dbcsr::create_template(m_c_bv)
			.name("c_bv_eps")
			.get();
			
		std::vector<double> exp_occ = *m_epso;
		std::vector<double> exp_vir = *m_epsv;
		
		double alpha = m_alpha[ilap];
		double omega = m_omega[ilap];
		
		std::for_each(exp_occ.begin(),exp_occ.end(),
			[alpha,omega](double& eps) {
				eps = exp(0.25 * log(omega) + eps * alpha);
			});
			
		std::for_each(exp_vir.begin(),exp_vir.end(),
			[alpha,omega](double& eps) {
				eps = exp(0.25 * log(omega) - eps * alpha);
			});
			
		c_bo_eps->copy_in(*m_c_bo);
		c_bv_eps->copy_in(*m_c_bv);
		
		c_bo_eps->scale(exp_occ, "right");
		c_bv_eps->scale(exp_vir, "right");
		
		//c_bo_eps->scale(pow(omega,0.25));
		//c_bv_eps->scale(pow(omega,0.25));
		
		LOG.os<>("Forming partly-transformed intermediates.\n");
		
		dbcsr::multiply('T', 'N', *c_bo_eps, *ko_ilap, *i_ob)
			//.filter_eps(dbcsr::global::filter_eps)
			.beta(1.0)
			.perform();
			
		kbuilder_ilap->set_density_alpha(po);
		kbuilder_ilap->compute_K();
		
		auto kv_ilap = kbuilder_ilap->get_K_A();
			
		dbcsr::multiply('T', 'N', *c_bv_eps, *kv_ilap, *i_vb)
			//.filter_eps(dbcsr::global::filter_eps)
			.beta(1.0)
			.perform();
			
		kbuilder_ilap.reset();
			
		//dbcsr::multiply('N', 'N', *i_ob, *m_c_bo, *i_oo_tmp)
			//.perform();
		//dbcsr::multiply('N', 'N', *i_vb, *m_c_bv, *i_vv_tmp)
			//.perform();
			
		//auto i_oo_tr = dbcsr::transpose(i_oo_tmp).get();
		//auto i_vv_tr = dbcsr::transpose(i_vv_tmp).get();
	
		//i_oo_tmp->add(1.0, 1.0, *i_oo_tr);
		//i_vv_tmp->add(1.0, 1.0, *i_vv_tr);
		
		//m_i_oo->add(1.0, 1.0, *i_oo_tmp);
		//m_i_vv->add(1.0, 1.0, *i_vv_tmp);
		
		//i_oo_tmp->clear();
		//i_vv_tmp->clear();
			
	}
	
	LOG.os<>("Forming fully transformed intermediates.\n");
	
	dbcsr::multiply('N', 'N', *i_ob, *m_c_bo, *m_i_oo)
			.perform();
	dbcsr::multiply('N', 'N', *i_vb, *m_c_bv, *m_i_vv)
			.perform();
			
	auto i_oo_tr = dbcsr::transpose(m_i_oo).get();
	auto i_vv_tr = dbcsr::transpose(m_i_vv).get();
	
	m_i_oo->add(1.0, 1.0, *i_oo_tr);
	m_i_vv->add(1.0, 1.0, *i_vv_tr);
	
	m_i_oo->scale(0.5 * m_c_os);
	m_i_vv->scale(0.5 * m_c_os);
	
	dbcsr::print(*m_i_oo);
	dbcsr::print(*m_i_vv);
			
}

void MVP_ao_ri_adc2::init() {
	
	LOG.os<>("Initializing AO-ADC(2)\n");
	
	// laplace
	LOG.os<>("Computing laplace points.\n");
	
	int nlap = m_opt.get<int>("nlap", ADC_NLAP);
	
	double emin = m_epso->front();
	double ehomo = m_epso->back();
	double elumo = m_epsv->front();
	double emax = m_epsv->back();
	
	double ymin = 2*(elumo - ehomo);
	double ymax = 2*(emax - emin);
	
	LOG.os<>("eps_min/eps_homo/eps_lumo/eps_max ", emin, " ", ehomo, " ", elumo, " ", emax, '\n');
	LOG.os<>("ymin/ymax ", ymin, " ", ymax, '\n');
	
	math::laplace lp(m_world.comm(), LOG.global_plev());
	
	lp.compute(nlap, ymin, ymax);
	
	m_nlap = nlap;
	
	m_omega = lp.omega();
	m_alpha = lp.alpha();
	
	m_c_bo = m_reg.get_matrix<double>("c_bo");
	m_c_bv = m_reg.get_matrix<double>("c_bv");
	
	auto po = m_reg.get_matrix<double>("po_bb");
	auto pv = m_reg.get_matrix<double>("pv_bb");
	
	LOG.os<>("Constructing pseudo densities.\n");
	
	// construct pseudo densities
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
	
		std::vector<double> exp_occ = *m_epso;
		std::vector<double> exp_vir = *m_epsv;
		
		double alpha = m_alpha[ilap];
		double omega = m_omega[ilap];
		
		std::for_each(exp_occ.begin(),exp_occ.end(),
			[alpha](double& eps) {
				eps = exp(0.5 * eps * alpha);
			});
			
		std::for_each(exp_vir.begin(),exp_vir.end(),
			[alpha](double& eps) {
				eps = exp(-0.5 * eps * alpha);
			});
			
		auto c_occ_exp = dbcsr::create_template<double>(m_c_bo)
			.name("c_bo_exp").get();
		auto c_vir_exp = dbcsr::create_template<double>(m_c_bv)
			.name("c_bv_exp").get();
			
		auto pseudo_o = dbcsr::create_template<double>(po)
			.name("pseudo occ density").get();
			
		auto pseudo_v = dbcsr::create_template<double>(pv)
			.name("pseudo vir density").get();
			
		c_occ_exp->copy_in(*m_c_bo);
		c_vir_exp->copy_in(*m_c_bv);
		
		c_occ_exp->scale(exp_occ, "right");
		c_vir_exp->scale(exp_vir, "right");
				
		//c_occ_exp->filter();
		//c_vir_exp->filter();
		
		dbcsr::multiply('N', 'T', *c_occ_exp, *c_occ_exp, *pseudo_o)
			.alpha(pow(omega,0.25)).perform();
		dbcsr::multiply('N', 'T', *c_vir_exp, *c_vir_exp, *pseudo_v)
			.alpha(pow(omega,0.25)).perform();
			
		//pseudo_o->filter(dbcsr::global::filter_eps);
		//pseudo_v->filter(dbcsr::global::filter_eps);
			
		m_pseudo_occs.push_back(pseudo_o);
		m_pseudo_virs.push_back(pseudo_v);
		
	}
	
	m_s_xx_inv = m_reg.get_matrix<double>("s_xx_inv_mat");
	
	m_eri_batched = m_reg.get_btensor<3,double>("i_xbb_batched");
	
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	m_c_os = m_opt.get<double>("c_os", ADC_C_OS);
	m_c_osc = m_opt.get<double>("c_osc", ADC_C_OS_COUPLING);
	
	compute_intermeds();
	
	auto build_J = m_opt.get<std::string>("build_J", ADC_BUILD_J);
	auto build_K = m_opt.get<std::string>("build_K", ADC_BUILD_K);
	
	LOG.os<>("Setting up J builder.\n");
	
	m_jbuilder = fock::get_J(build_J, m_world, m_opt);
	
	m_jbuilder->set_sym(false);
	m_jbuilder->set_reg(m_reg);
	m_jbuilder->set_mol(m_mol);
	
	m_jbuilder->init();
	m_jbuilder->init_tensors();
	
	LOG.os<>("Setting up K builder.\n");
	
	m_kbuilder = fock::get_K(build_K, m_world, m_opt);
	
	m_kbuilder->set_sym(false);
	m_kbuilder->set_reg(m_reg);
	m_kbuilder->set_mol(m_mol);
	
	m_kbuilder->init();
	m_kbuilder->init_tensors();
	
	LOG.os<>("Done with setting up.\n");
	
}

std::pair<smat,smat> MVP_ao_ri_adc2::compute_jk(smat& u_ao) {
	
	m_jbuilder->set_density_alpha(u_ao);
	m_kbuilder->set_density_alpha(u_ao);
	
	m_jbuilder->compute_J();
	m_kbuilder->compute_K();
	
	auto jmat = m_jbuilder->get_J();
	auto kmat = m_kbuilder->get_K_A();
	
	std::pair<smat,smat> out = {jmat, kmat};
	
	return out;
	
}

smat MVP_ao_ri_adc2::compute_sigma_1(smat& jmat, smat& kmat) {
	
	auto j = u_transform(jmat, 'T', m_c_bo, 'N', m_c_bv);
	auto k = u_transform(kmat, 'T', m_c_bo, 'N', m_c_bv);
	
	smat sig_ao = dbcsr::create_template<double>(jmat)
		.name("sig_ao").get();
	
	sig_ao->add(0.0, 1.0, *jmat);
	sig_ao->add(1.0, 1.0, *kmat);
	
	//LOG.os<>("Sigma adc1 ao:\n");
	//dbcsr::print(*u_ao);
	
	// transform back
	smat sig_1 = u_transform(sig_ao, 'T', m_c_bo, 'N', m_c_bv);
	
	sig_1->setname("sig_1");
	
	return sig_1;
	
} 

smat MVP_ao_ri_adc2::compute_sigma_2a(smat& u_ia) {
	
	// sig_2a = i_vv_ab * u_ib
	auto sig_2a = dbcsr::create_template<double>(u_ia)
		.name("sig_2a").get();
		
	dbcsr::multiply('N', 'T', *u_ia, *m_i_vv, *sig_2a).perform();
	
	return sig_2a;
	
}

smat MVP_ao_ri_adc2::compute_sigma_2b(smat& u_ia) {
	
	// sig_2b = i_oo_ij * u_ja
	auto sig_2b = dbcsr::create_template<double>(u_ia)
		.name("sig_2b").get();
		
	dbcsr::multiply('N', 'N', *m_i_oo, *u_ia, *sig_2b).perform();
	
	return sig_2b;
	
} 

smat MVP_ao_ri_adc2::compute_sigma_2c(smat& jmat, smat& kmat) {
	
	// sig_2c = -1/2 t_iajb^SOS * I_jb
	// I_ia = [2*(jb|ia) - (ja|ib)] u_jb
	
	// in AO:
	/* sig_2c = -1/2 * c_os * sum_t c_mi * exp(eps_i t) 
	 * 				c_na * exp(-eps_a t) * (mn|X) * d_X(t)
	 * d_X(t) = (nk|X) * I_n'k' * Po_nn'(t) * Pv_kk'(t)
	 * I_mk = jmat + transpose(K)
	 */
	 
	auto I_ao = dbcsr::create_template<double>(jmat).name("I_ao").get();
	auto jmat_t = dbcsr::transpose(jmat).get();
	auto kmat_t = dbcsr::transpose(kmat).get();
	
	auto jmo = u_transform(jmat_t, 'T', m_c_bo, 'N', m_c_bv);
	auto kmo = u_transform(kmat_t, 'T', m_c_bo, 'N', m_c_bv);
	
	dbcsr::print(*jmo);
	dbcsr::print(*kmo);
	
	I_ao->add(0.0, 1.0, *jmat_t);
	I_ao->add(1.0, 1.0, *kmat_t);
	
	return u_transform(I_ao, 'T', m_c_bo, 'N', m_c_bv);
	 
}
	

smat MVP_ao_ri_adc2::compute(smat u_ia, double omega) {
	
	LOG.os<>("Computing ADC0.\n");
	// compute ADC0 part in MO basis
	smat sig_0 = compute_sigma_0(u_ia);
	
	std::cout << "SIG0" << std::endl;
	dbcsr::print(*sig_0);

	sig_0->setname("sigma ADC2 vector");

	// transform u to ao coordinated
	smat u_ao = u_transform(u_ia, 'N', m_c_bo, 'T', m_c_bv);
	
	LOG.os<>("U transformed: \n");
	dbcsr::print(*u_ao);
	
	auto jk = compute_jk(u_ao);
	
	auto sig_1 = compute_sigma_1(jk.first, jk.second);
	
	LOG.os<>("Sigma adc1 mo:\n");
	dbcsr::print(*sig_1);
	
	sig_0->add(1.0, 1.0, *sig_1);
	sig_1->release();
	
	auto sig_2a = compute_sigma_2a(u_ia);
	auto sig_2b = compute_sigma_2b(u_ia);
	
	dbcsr::print(*sig_2a);
	dbcsr::print(*sig_2b);
	
	sig_0->add(1.0, 1.0, *sig_2a);
	sig_0->add(1.0, 1.0, *sig_2b);
	
	sig_2a->release();
	sig_2b->release();
		
	auto sig_2c = compute_sigma_2c(jk.first, jk.second);
	
	std::cout << "INTERMEDIATE:" << std::endl;
	dbcsr::print(*sig_2c);
	
	exit(0);
	
	return sig_0;
	
}
	
} // end namespace
