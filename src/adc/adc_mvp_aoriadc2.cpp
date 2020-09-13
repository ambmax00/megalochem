#include "adc/adc_mvp.h"
#include "math/laplace/laplace.h"
#include "math/linalg/piv_cd.h"
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
	 * I_ab = - antisym[ C_Bb sum_t C_Aa exp(t) * K_AB ]
	 * 
	 * I_ij
	 * K kernel with P = Pvir(t) and s_xx_inv = F(t)_XY
	 * I_ij = - antisym[ C_Jj sum_t C_Ii exp(t) * K_IJ ]
	 * 
	 */
	 
	LOG.os<>("Computing intermediates.\n");
	 
	LOG.os<>("Setting up ZBUILDER.\n"); 
	
	std::string zmethod = m_opt.get<std::string>("build_Z", ADC_BUILD_Z);
	m_zbuilder = mp::get_Z(zmethod, m_world, m_mol, m_opt);
	
	m_zbuilder->set_reg(m_reg);
	m_zbuilder->init_tensors();
	
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
		
		m_zbuilder->set_occ_coeff(L_bu);
		m_zbuilder->set_vir_density(pv);
	
		LOG.os<>("Computing Z.\n");
	
		m_zbuilder->compute();
		auto z_xx_ilap = m_zbuilder->zmat();
		
		//dbcsr::print(*z_xx_ilap);
		
		auto temp = dbcsr::create_template(z_xx_ilap)
			.name("temp_xx")
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
			
		dbcsr::multiply('N', 'N', *m_s_xx_inv, *z_xx_ilap, *temp)
			.filter_eps(dbcsr::global::filter_eps)
			.perform();
			
		dbcsr::multiply('N', 'N', *temp, *m_s_xx_inv, *z_xx_ilap)
			.filter_eps(dbcsr::global::filter_eps)
			.perform();
			
		auto f_xx_ilap = z_xx_ilap;
		f_xx_ilap->setname("f_xx_ilap");
		
#if 0
		auto F = dbcsr::copy<double>(f_xx_ilap).get();
		m_FS.push_back(F);
#endif
		
		//dbcsr::print(*f_xx_ilap);
		
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
		
		double xpt = m_xpoints[ilap];
		double wght = m_weigths[ilap];
		
		std::for_each(exp_occ.begin(),exp_occ.end(),
			[xpt,wght](double& eps) {
				eps = exp(0.25 * log(wght) + eps * xpt);
			});
			
		std::for_each(exp_vir.begin(),exp_vir.end(),
			[xpt,wght](double& eps) {
				eps = exp(0.25 * log(wght) - eps * xpt);
			});
			
		c_bo_eps->copy_in(*m_c_bo);
		c_bv_eps->copy_in(*m_c_bv);
		
		c_bo_eps->scale(exp_occ, "right");
		c_bv_eps->scale(exp_vir, "right");
		
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
		
		f_xx_ilap->clear();
			
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
	
	m_i_oo->scale(-0.5 * m_c_os);
	m_i_vv->scale(-0.5 * m_c_os);
	
	//dbcsr::print(*m_i_oo);
	//dbcsr::print(*m_i_vv);
			
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
	
	m_weigths = lp.omega();
	m_xpoints = lp.alpha();
	
	m_c_bo = m_reg.get_matrix<double>("c_bo");
	m_c_bv = m_reg.get_matrix<double>("c_bv");
	
	auto po = m_reg.get_matrix<double>("po_bb");
	auto pv = m_reg.get_matrix<double>("pv_bb");
	
	m_po_bb = po;
	m_pv_bb = pv;
	
	LOG.os<>("Constructing pseudo densities.\n");
	
	// construct pseudo densities
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
	
		std::vector<double> exp_occ = *m_epso;
		std::vector<double> exp_vir = *m_epsv;
		
		double xpt = m_xpoints[ilap];
		double wght = m_weigths[ilap];
		
		std::for_each(exp_occ.begin(),exp_occ.end(),
			[xpt](double& eps) {
				eps = exp(0.5 * eps * xpt);
			});
			
		std::for_each(exp_vir.begin(),exp_vir.end(),
			[xpt](double& eps) {
				eps = exp(-0.5 * eps * xpt);
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
			.alpha(pow(wght,0.25)).perform();
		dbcsr::multiply('N', 'T', *c_vir_exp, *c_vir_exp, *pseudo_v)
			.alpha(pow(wght,0.25)).perform();
			
		//pseudo_o->filter(dbcsr::global::filter_eps);
		//pseudo_v->filter(dbcsr::global::filter_eps);
			
		m_pseudo_occs.push_back(pseudo_o);
		m_pseudo_virs.push_back(pseudo_v);
		
	}
	
	m_s_bb = m_reg.get_matrix<double>("s_bb");
	
	m_s_xx_inv = m_reg.get_matrix<double>("s_xx_inv_mat");
	
	m_eri_batched = m_reg.get_btensor<3,double>("i_xbb_batched");
	
	m_nbatches = m_eri_batched->batch_dims();
	auto bmethod_str = m_opt.get<std::string>("intermeds", ADC_INTERMEDS);
	
	m_bmethod = dbcsr::get_btype(bmethod_str);
	
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
		.name("sig_ao")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
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
	/* sig_2c = +1/2 * c_os * sum_t c_mi * exp(eps_i t) 
	 * 				c_na * exp(-eps_a t) * (mn|X') * (X'|X)^inv d_X(t)
	 * d_X(t) = (nk|X) * I_n'k' * Po_nn'(t) * Pv_kk'(t)
	 * I_mk = jmat + transpose(K)
	 */
	 
	LOG.os<>("Computing sigma_2_c.\n");
	
	LOG.os<>("-- Computing intermediate...\n");
	
	auto I_ao = dbcsr::create_template<double>(jmat).name("I_ao").get();
	auto jmat_t = dbcsr::transpose(jmat).get();
	auto kmat_t = dbcsr::transpose(kmat).get();
	
	I_ao->add(0.0, 1.0, *jmat_t);
	I_ao->add(1.0, 1.0, *kmat_t);
	
	jmat_t->release();
	kmat_t->release();
	
	auto o = m_mol->dims().oa();
	auto v = m_mol->dims().va();
	
	auto sig_2c = dbcsr::create<double>()
		.set_world(m_world)
		.name("sig_2c")
		.row_blk_sizes(o)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	LOG.os<>("-- Contracting over laplace points...\n");
	
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		auto pseudo_o = m_pseudo_occs[ilap];
		auto pseudo_v = m_pseudo_virs[ilap];
		
		auto I_pseudo = u_transform(I_ao, 'T', pseudo_o, 'N', pseudo_v);
		
		m_jbuilder->set_density_alpha(I_pseudo);
		m_jbuilder->set_sym(false);
		
		m_jbuilder->compute_J();
		
		auto jmat_pseudo = m_jbuilder->get_J();
		
		auto c_bo_copy = dbcsr::copy(m_c_bo).get();
		auto c_bv_copy = dbcsr::copy(m_c_bv).get();
		
		std::vector<double> exp_occ = *m_epso;
		std::vector<double> exp_vir = *m_epsv;
		
		double xpt = m_xpoints[ilap];
		double wght = m_weigths[ilap];
		
		std::for_each(exp_occ.begin(),exp_occ.end(),
			[xpt,wght](double& eps) {
				eps = exp(0.25 * log(wght) + eps * xpt);
			});
			
		std::for_each(exp_vir.begin(),exp_vir.end(),
			[xpt,wght](double& eps) {
				eps = exp(0.25 * log(wght) - eps * xpt);
			});
			
		c_bo_copy->scale(exp_occ, "right");
		c_bv_copy->scale(exp_vir, "right");
		
		auto jmat_trans = u_transform(jmat_pseudo, 
			'T', c_bo_copy, 'N', c_bv_copy);
		
		sig_2c->add(1.0, 1.0, *jmat_trans);
		
	}
				
	sig_2c->scale(-0.25 * m_c_os);
	
	return sig_2c;
}

smat MVP_ao_ri_adc2::compute_sigma_2d(smat& u_ia) {
	
	/* sig_2d = - 0.5 sum_jb [2 * (ia|bj) - (ja|ib)] * I_jb
	 * where I_ia = sum_jb t_iajb^SOS u_jb
	 * 
	 * in AO:
	 * 
	 * sig_2a = + 2 * J(I_nr)_transpose - K(I_nr)_transpose 
	 * I_nr = sum_t c_os Po(t) * J(u_pseudo_ao(t)) * Pv(t)
	 */
	 
	auto I_ao = dbcsr::create_template<double>(m_pseudo_occs[0])
		.name("I_ao")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	 
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		double xpt = m_xpoints[ilap];
		double wght = m_weigths[ilap];
		
		auto c_bo_copy = dbcsr::copy(m_c_bo).get();
		auto c_bv_copy = dbcsr::copy(m_c_bv).get();
		
		std::vector<double> exp_occ = *m_epso;
		std::vector<double> exp_vir = *m_epsv;
		
		std::for_each(exp_occ.begin(),exp_occ.end(),
			[xpt,wght](double& eps) {
				eps = exp(0.25 * log(wght) + eps * xpt);
			});
			
		std::for_each(exp_vir.begin(),exp_vir.end(),
			[xpt,wght](double& eps) {
				eps = exp(0.25 * log(wght) - eps * xpt);
			});
			
		c_bo_copy->scale(exp_occ, "right");
		c_bv_copy->scale(exp_vir, "right");
		
		auto u_pseudo = u_transform(u_ia, 'N', c_bo_copy, 'T', c_bv_copy);
		
		c_bo_copy->release();
		c_bv_copy->release();
		
		m_jbuilder->set_density_alpha(u_pseudo);
		m_jbuilder->set_sym(false);
		m_jbuilder->compute_J();
		
		auto jpseudo1 = m_jbuilder->get_J();
		
		auto po = m_pseudo_occs[ilap];
		auto pv = m_pseudo_virs[ilap];
		auto jpseudo2 = u_transform(jpseudo1, 'N', po, 'T', pv); 
		
		I_ao->add(1.0, 0.5, *jpseudo2);
		
	}
	
	I_ao->scale(m_c_os);
	
	m_jbuilder->set_density_alpha(I_ao);
	m_jbuilder->set_sym(false);
	m_kbuilder->set_density_alpha(I_ao);
	m_kbuilder->set_sym(false);
	
	m_jbuilder->compute_J();
	m_kbuilder->compute_K();
	
	auto jmat = m_jbuilder->get_J();
	auto kmat = m_kbuilder->get_K_A();
	
	auto jmat_trans = dbcsr::transpose(jmat).get();
	auto kmat_trans = dbcsr::transpose(kmat).get();
	
	jmat_trans->add(1.0, 1.0, *kmat_trans);
	
	auto sig_2d = u_transform(jmat_trans, 'T', m_c_bo, 'N', m_c_bv);
	sig_2d->setname("sig_2d");
	
	sig_2d->scale(-0.5);
	
	return sig_2d;
		
}

dbcsr::sbtensor<3,double> MVP_ao_ri_adc2::compute_J(smat& u_ao) {
	
	LOG.os<>("Forming J_xbb\n");
	// form J
	auto eri = m_eri_batched->get_stensor();
	
	auto J_xbb_0_12 = dbcsr::tensor_create_template<3,double>(eri)
		.name("J_xbb_0_12").get();
		
	auto J_xbb_2_01_t = dbcsr::tensor_create_template<3,double>(eri)
		.name("J_xbb_2_01_t")
		.map1({2}).map2({0,1})
		.get();
		
	auto J_xbb_2_01 = dbcsr::tensor_create_template<3,double>(eri)
		.name("J_xbb_2_01")
		.map1({2}).map2({0,1})
		.get();
	
	auto J_xbb_batched = dbcsr::btensor_create<3>(J_xbb_0_12)
		.name(m_mol->name() + "_j_xbb_batched")
		.batch_dims(m_nbatches)
		.btensor_type(m_bmethod)
		.print(LOG.global_plev())
		.get();
			
	// Form half-projected AO u
	// u_v_mu,nu = u_mu,nu' * S_nu',nu
	// u_o_mu.nu = S_mu',mu * u_mu',nu
		
	auto u_hto = dbcsr::create_template<double>(u_ao)
		.name("u_hto").get();
	auto u_htv = dbcsr::create_template<double>(u_ao)
		.name("u_htv").get();
	
	dbcsr::multiply('N', 'N', *m_s_bb, *u_ao, *u_hto).perform();
	dbcsr::multiply('N', 'N', *u_ao, *m_s_bb, *u_htv).perform();

	auto b = m_mol->dims().b();
	arrvec<int,2> bb = {b,b};

	auto u_hto_01 = dbcsr::tensor_create<2,double>()
		.name("u_hto_01")
		.pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bb).get();
		
	auto u_htv_01 = dbcsr::tensor_create_template<2,double>(u_hto_01)
		.name("u_htv_01").get();
	
	dbcsr::copy_matrix_to_tensor(*u_hto, *u_hto_01);
	dbcsr::copy_matrix_to_tensor(*u_htv, *u_htv_01);
	
	u_hto->release();
	u_htv->release();

	m_eri_batched->reorder(vec<int>{2},vec<int>{0,1});

	m_eri_batched->decompress_init({0});
	J_xbb_batched->compress_init({0,2});
	
	int nxbatches = m_eri_batched->nbatches_dim(0);
	int nnbatches = m_eri_batched->nbatches_dim(2);
	
	auto xbounds = m_eri_batched->bounds(0);
	auto bbounds = m_eri_batched->bounds(2);
	auto fullbbounds = m_eri_batched->full_bounds(1);
	
	for (int ix = 0; ix != nxbatches; ++ix) {
		
		m_eri_batched->decompress({ix});
		auto eri_2_01 = m_eri_batched->get_stensor();
		
		for (int inu = 0; inu != nnbatches; ++inu) {
			
			vec<vec<int>> nu_bounds = {
				bbounds[inu]
			};
			
			vec<vec<int>> xmu_bounds = {
				xbounds[ix],
				fullbbounds
			};
			
			vec<vec<int>> xnu_bounds = {
				xbounds[ix],
				bbounds[inu]
			};
			
			dbcsr::contract(*u_hto_01, *eri_2_01, *J_xbb_2_01_t)
				.alpha(-1.0)
				.bounds3(xnu_bounds)
				.print(LOG.global_plev() >= 3)
				.filter(dbcsr::global::filter_eps / nxbatches)
				.perform("mg, Yag -> Yam");
				
			dbcsr::copy(*J_xbb_2_01_t, *J_xbb_2_01)
				.move_data(true)
				.order(vec<int>{0,2,1})
				.perform();
			
			dbcsr::contract(*u_htv_01, *eri_2_01, *J_xbb_2_01)
				.bounds2(nu_bounds)
				.bounds3(xmu_bounds)
				.print(LOG.global_plev() >= 3)
				.filter(dbcsr::global::filter_eps / nxbatches)
				.beta(1.0)
				.perform("ka, Ymk -> Yma");
				
			dbcsr::copy(*J_xbb_2_01, *J_xbb_0_12).move_data(true)
				.perform();
				
			J_xbb_batched->compress({ix,inu}, J_xbb_0_12);
			
		}
	}
	
#if 0

	auto x = m_mol->dims().x();
	auto o = m_mol->dims().oa();
	auto v = m_mol->dims().va();
	
	arrvec<int,3> xob = {x,o,b};
	arrvec<int,3> xov = {x,o,v};
	
	dbcsr::shared_pgrid<3> spgrid3 = dbcsr::create_pgrid<3>(m_world.comm()).get();
	auto J_xob = dbcsr::tensor_create<3,double>()
		.name("J_xob").pgrid(spgrid3)
		.map1({0}).map2({1,2})
		.blk_sizes(xob).get();
	auto J_xov1 = dbcsr::tensor_create<3,double>()
		.name("J_xov1").pgrid(spgrid3)
		.map1({0}).map2({1,2})
		.blk_sizes(xov).get();	
	auto J_xov2 = dbcsr::tensor_create<3,double>()
		.name("J_xov2").pgrid(spgrid3)
		.map1({0}).map2({1,2})
		.blk_sizes(xov).get();	
	
	arrvec<int,2> bo = {b,o};
	arrvec<int,2> bv = {b,v};
	
	auto c_bo = dbcsr::tensor_create<2,double>()
		.name("c").pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bo).get();
		
	auto c_bv = dbcsr::tensor_create<2,double>()
		.name("c").pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bv).get();
		
	dbcsr::copy_matrix_to_tensor(*m_c_bo, *c_bo);
	dbcsr::copy_matrix_to_tensor(*m_c_bv, *c_bv);
	
	auto J_ao = J_xbb_batched->get_stensor();
	dbcsr::contract(*c_bo, *J_ao, *J_xob).perform("mi, Xmn -> Xin");
	dbcsr::contract(*c_bv, *J_xob, *J_xov1).perform("na, Xin -> Xia");
	auto invs = m_reg.get_tensor<2,double>("s_xx_invsqrt");
	dbcsr::contract(*invs, *J_xov1, *J_xov2).perform("XY, Xia -> Yia"); 
	
	J_xov2->filter(dbcsr::global::filter_eps);
	
	dbcsr::print(*invs);
	
	dbcsr::print(*J_ao);
	
	dbcsr::print(*J_xov2);

	exit(0);

#endif

	return J_xbb_batched;
	
}

std::pair<smat,smat> MVP_ao_ri_adc2::compute_sigma_2e_ilap(
	dbcsr::sbtensor<3,double>& J_xbb_batched, 
	smat& FA, smat& FB, smat& pseudo_o, smat& pseudo_v 
#if 0
	, double omega, int ilap
#endif	
	) {
	
	// two different ways: full or memory efficient
	auto imethod = m_opt.get<std::string>("doubles", ADC_DOUBLES);
	
	LOG.os<1>("Computing doubles part of sigma vector.\n");
	
	auto x = m_mol->dims().x();
	arrvec<int,2> xx = {x,x};
	
	auto FA_01 = dbcsr::tensor_create<2,double>()
		.name("FA_01")
		.pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(xx)
		.get();
		
	auto FB_01 = dbcsr::tensor_create_template<2,double>(FA_01)
		.name("FB_01").get();
	
	dbcsr::copy_matrix_to_tensor(*FA, *FA_01);
	dbcsr::copy_matrix_to_tensor(*FB, *FB_01);
	
	FA->clear();
	FB->clear();
	
	auto sigma_ilap_A = dbcsr::create_template<double>(pseudo_o)
		.name("sigma_ilap_A")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto sigma_ilap_B = dbcsr::create_template<double>(pseudo_o)
		.name("sigma_ilap_B")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	if (imethod == "full") {
		
		// Intermediate is held in-core/on-disk
		auto eri = m_eri_batched->get_stensor();
		
		auto I_xbb_batched = dbcsr::btensor_create<3>(eri)
			.name(m_mol->name() + "_i_xbb_batched")
			.batch_dims(m_nbatches)
			.btensor_type(m_bmethod)
			.print(LOG.global_plev())
			.get();
			
		// Form I
		I_xbb_batched->reorder(vec<int>{0},vec<int>{1,2});
		J_xbb_batched->reorder(vec<int>{0},vec<int>{1,2});
		m_eri_batched->reorder(vec<int>{0},vec<int>{1,2});
		
		I_xbb_batched->compress_init({0,2});
		
		auto xbds = m_eri_batched->bounds(0);
		auto bbds = m_eri_batched->bounds(2);
		auto fullxb = m_eri_batched->full_bounds(0);
		auto fullbb = m_eri_batched->full_bounds(2);
		
		auto I_xbb_0_12 = dbcsr::tensor_create_template<3,double>(eri)
			.name("I_xbb_0_12").map1({0}).map2({1,2}).get();
		
		for (int ix = 0; ix != xbds.size(); ++ix) {
			for (int inu = 0; inu != bbds.size(); ++inu) {
				
				//I_xbb_0_12->batched_contract_init();
				
				J_xbb_batched->decompress_init({0,2});
				
				for (int iy = 0; iy != xbds.size(); ++iy) {
					
					J_xbb_batched->decompress({iy,inu});
					auto J_xbb_0_12 = J_xbb_batched->get_stensor();
					
					vec<vec<int>> xbounds = { xbds[ix] };
					vec<vec<int>> ybounds = { xbds[iy] };
					vec<vec<int>> kabounds = { fullbb, bbds[inu] };
					
					dbcsr::contract(*FA_01, *J_xbb_0_12, *I_xbb_0_12)
						.bounds1(ybounds)
						.bounds2(xbounds)
						.bounds3(kabounds)
						.filter(dbcsr::global::filter_eps)
						.beta(1.0)
						.perform("YX, Yka -> Xka");
						
				}
				
				J_xbb_batched->decompress_finalize();
				
				m_eri_batched->decompress_init({0,2});
				
				for (int iy = 0; iy != xbds.size(); ++iy) {
					
					m_eri_batched->decompress({iy,inu});
					auto eri_0_12 = m_eri_batched->get_stensor();
					
					vec<vec<int>> xbounds = { xbds[ix] };
					vec<vec<int>> ybounds = { xbds[iy] };
					vec<vec<int>> kabounds = { fullbb, bbds[inu] };
					
					dbcsr::contract(*FB_01, *eri_0_12, *I_xbb_0_12)
						.bounds1(ybounds)
						.bounds2(xbounds)
						.bounds3(kabounds)
						.filter(dbcsr::global::filter_eps)
						.beta(1.0)
						.perform("YX, Yka -> Xka");
						
				}
				
				m_eri_batched->decompress_finalize();
				
				//I_xbb_0_12->batched_contract_finalize();
				
				I_xbb_batched->compress({ix,inu}, I_xbb_0_12);
				
			}
		}
		
		I_xbb_batched->compress_finalize();

#if 0
		auto Iout = I_xbb_batched->get_stensor();
		double xpt = m_xpoints_dd[ilap];
		Iout->scale(exp(xpt * omega));
		dbcsr::copy(*Iout, *m_I).perform();
		
#endif
		// form sigma
		auto b = m_mol->dims().b();
		arrvec<int,2> bb = {b,b};
		
		auto sigma_ilap_01 = dbcsr::tensor_create<2,double>()
			.name("sigma_01")
			.pgrid(m_spgrid2)
			.map1({0}).map2({1})
			.blk_sizes(bb).get();
			
		auto po_01 = dbcsr::tensor_create_template<2,double>
			(sigma_ilap_01)
			.name("po_01").get();
			
		auto pv_01 = dbcsr::tensor_create_template<2,double>
			(sigma_ilap_01)
			.name("pv_01").get();
			
		dbcsr::copy_matrix_to_tensor(*pseudo_o, *po_01);
		dbcsr::copy_matrix_to_tensor(*pseudo_v, *pv_01);
		
		m_eri_batched->reorder(vec<int>{1},vec<int>{0,2});
		I_xbb_batched->reorder(vec<int>{0,1}, vec<int>{2});
		
		auto erireo = m_eri_batched->get_stensor();
		
		auto cbar_1_02 = dbcsr::tensor_create_template<3,double>(erireo)
			.map1({1}).map2({0,2})
			.name("cbar_1_02").get();
			
		auto cbar_01_2 = dbcsr::tensor_create_template<3,double>(erireo)
			.map1({0,1}).map2({2})
			.name("cbar_01_2")
			.get();
			
		auto I_01_2 = dbcsr::tensor_create_template<3,double>(erireo)
			.map1({0,1}).map2({2})
			.name("I_02_1")
			.get();
		
		I_xbb_batched->decompress_init({0,1});
		m_eri_batched->decompress_init({0});
		
		for (int ix = 0; ix != xbds.size(); ++ix) {
			
			m_eri_batched->decompress({ix});
			auto eri_1_02 = m_eri_batched->get_stensor();
			
			for (int inu = 0; inu != bbds.size(); ++inu) {
				
				vec<vec<int>> xmbounds = {xbds[ix],fullbb};
				vec<vec<int>> nbounds = {bbds[inu]};
				
				// form cbar
				dbcsr::contract(*eri_1_02, *po_01, *cbar_1_02)
					.bounds2(xmbounds)
					.bounds3(nbounds)
					.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps / xbds.size())
					.perform("Xlm, lk -> Xkm");
					
				dbcsr::copy(*cbar_1_02, *cbar_01_2).move_data(true).perform();
				
				I_xbb_batched->decompress({ix,inu});
				auto I_xbb_01_2 = I_xbb_batched->get_stensor();
				
				vec<vec<int>> xnubounds = {xbds[ix],bbds[inu]};
				
				// form sig_A
				dbcsr::contract(*I_xbb_01_2, *cbar_01_2, *sigma_ilap_01)
					.bounds1(xnubounds)
					.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps / xbds.size())
					.beta(1.0)
					.perform("Xka, Xkm -> ma");
				
				cbar_01_2->clear();
								
			}
			
		}
		
		dbcsr::copy_tensor_to_matrix(*sigma_ilap_01, *sigma_ilap_A);
		sigma_ilap_01->clear();
		
		m_eri_batched->decompress_finalize();
		I_xbb_batched->decompress_finalize();
		
		I_xbb_batched->reorder(vec<int>{0,2},vec<int>{1});
		
		m_eri_batched->decompress_init({0});
		I_xbb_batched->decompress_init({0,2});
		
		for (int ix = 0; ix != xbds.size(); ++ix) {
			
			m_eri_batched->decompress({ix});
			auto eri_1_02 = m_eri_batched->get_stensor();
			
			for (int inu = 0; inu != bbds.size(); ++inu) {
				
				vec<vec<int>> xmbounds = {xbds[ix],fullbb};
				vec<vec<int>> nbounds = {bbds[inu]};
				
				// form cbar
				dbcsr::contract(*eri_1_02, *pv_01, *cbar_1_02)
					.bounds2(xmbounds)
					.bounds3(nbounds)
					.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps / xbds.size())
					.perform("Xda, dg -> Xga");
					
				dbcsr::copy(*cbar_1_02, *cbar_01_2).move_data(true).perform();
				
				I_xbb_batched->decompress({ix,inu});
				auto I_xbb_02_1 = I_xbb_batched->get_stensor();
				
				vec<vec<int>> xnubounds = {xbds[ix],bbds[inu]};
				
				// form sig_A
				dbcsr::contract(*I_xbb_02_1, *cbar_01_2, *sigma_ilap_01)
					.bounds1(xnubounds)
					.filter(dbcsr::global::filter_eps / xbds.size())
					.print(LOG.global_plev() >= 3)
					.beta(1.0)
					.perform("Xmg, Xga -> ma");
					
			}
			
		}
		
		m_eri_batched->decompress_finalize();
		I_xbb_batched->decompress_finalize();
		
		dbcsr::copy_tensor_to_matrix(*sigma_ilap_01, *sigma_ilap_B);
		
		I_xbb_batched->reset();		
				
	} else if (imethod == "mem") {
		
		std::cout << "NOTHING YET." << std::endl;		
		
	} else {
		
		throw std::runtime_error("Invalid value for option doubles.");
		
	}
	
	//double xpt = m_xpoints_dd[ilap];
	//sigma_ilap_A->scale(exp(omega * xpt));
	//sigma_ilap_B->scale(exp(omega * xpt));
	
	std::pair<smat,smat> out = {sigma_ilap_A, sigma_ilap_B};
	
	return out;
	
}

smat MVP_ao_ri_adc2::compute_sigma_2e(smat& u_ao, double omega) {
	
	/* IN AO:
	 * sig_e2 = - c_os_c ^2 [ sum_t
	 * 	exp(omega t) C_μi C_αa * exp(-ε_a t) * I_{Xκα}(t) * (X|μκ') Po(t)_κκ'
	 * + exp(omega t) C_μi C_αa * exp(ε_i t) * I_{Xμγ}(t) * (X|γ'α) Pv(t)_γγ']
	 * 
	 * with
	 * 
	 * I_{Xκα}(t) = J_{Yκα} * F_{YX}(t) + (Y|κα) Ftilde_{XY}(t)
	 * 
	 * Ftilde_{X'Y'}(t) = (X'|X) (X|μν) * J_{Xμ'ν'} * Po(t)_{μμ'} * Pv(t)_{νν'} (YY')
	 * 
	 * J_{Χκα} = [ wv_κα (X|μκ) - wo_μγ (X|γα)]
	 * wv_κα = co_κi * S_αα' cv_α'a u_ia
	 * wo_μγ = S_μμ' * co_μ'i * cv_γa * u_ia
	 */
	
	auto o = m_mol->dims().oa();
	auto v = m_mol->dims().va();
#if 0
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
#endif 
	
	auto sigma_2e_A = dbcsr::create<double>()
		.name("sigma_2e_A")
		.set_world(m_world)
		.row_blk_sizes(o)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto sigma_2e_B = dbcsr::create_template<double>(sigma_2e_A)
		.name("sigma_2e_B").get();
	
	// Form J
	auto J_xbb_batched = compute_J(u_ao);
	m_reg.insert_btensor<3,double>("t_xbb_batched", J_xbb_batched);
	
	// Prepare asym zbuilder
	mp::LLMP_ASYM_Z zbuilder_asym(m_world, m_mol, m_opt);
		
	zbuilder_asym.set_reg(m_reg);
	zbuilder_asym.init_tensors();

#if 0
	auto eri = m_eri_batched->get_stensor();
	m_I = dbcsr::tensor_create_template<3,double>(eri).name("I").get();

	arrvec<int,3> xov = {x,o,v};
	arrvec<int,3> xob = {x,o,b};

	auto sp3 = dbcsr::create_pgrid<3>(m_world.comm()).get();

	auto I_mo = dbcsr::tensor_create<3,double>()
		.name("I_mo").pgrid(sp3)
		.map1({0}).map2({1,2})
		.blk_sizes(xov).get();
		
	auto I_HT = dbcsr::tensor_create<3,double>()
		.name("I_HT").pgrid(sp3)
		.map1({0}).map2({1,2})
		.blk_sizes(xob).get();
		
#endif

	// Loop
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		double wght = m_weights_dd[ilap];
		double xpt = m_xpoints_dd[ilap];
		
		// Form pseudo densities
		auto c_bo_eps = dbcsr::copy(m_c_bo).get();
		auto c_bv_eps = dbcsr::copy(m_c_bv).get();
		
		std::vector<double> exp_occ = *m_epso;
		std::vector<double> exp_vir = *m_epsv;
		
		std::for_each(exp_occ.begin(),exp_occ.end(),
			[xpt,wght](double& eps) {
				eps = exp(0.5 * eps * xpt);
			});
			
		std::for_each(exp_vir.begin(),exp_vir.end(),
			[xpt,wght](double& eps) {
				eps = exp(- 0.5 * eps * xpt);
			});
			
		c_bo_eps->scale(exp_occ, "right");
		c_bv_eps->scale(exp_vir, "right");
		
		auto pseudo_o = dbcsr::create_template<double>(m_po_bb)
			.name("pseudo_o").get();
			
		auto pseudo_v = dbcsr::create_template<double>(m_pv_bb)
			.name("pseudo_v").get();
			
		dbcsr::multiply('N', 'T', *c_bo_eps, *c_bo_eps, *pseudo_o)
			.alpha(pow(wght,0.25)).perform();
		dbcsr::multiply('N', 'T', *c_bv_eps, *c_bv_eps, *pseudo_v)
			.alpha(pow(wght,0.25)).perform();
		
		//pseudo_o->scale(exp(0.25 * omega * xpt));
		//pseudo_v->scale(exp(0.25 * omega * xpt)); 
		
		math::pivinc_cd chol(pseudo_o, LOG.global_plev());
		chol.compute();
		
		int rank = chol.rank();
		
		auto u = dbcsr::split_range(rank, m_mol->mo_split());
		auto b = m_mol->dims().b();
		
		auto L_bu = chol.L(b, u);
		
		L_bu->filter(dbcsr::global::filter_eps);
		pseudo_o->filter(dbcsr::global::filter_eps);
		pseudo_v->filter(dbcsr::global::filter_eps);
		
		//LOG.os<1>("DENSITIES: \n");
		//dbcsr::print(*pseudo_o);
		//dbcsr::print(*pseudo_v);
		
		// Form F_A
		m_zbuilder->set_occ_coeff(L_bu);
		m_zbuilder->set_vir_density(pseudo_v);
		
		/*
		dbcsr::multiply('N', 'T', *L_bu, *L_bu, *pseudo_o)
			.beta(-1.0).perform();
		
		std::cout << "PO" << std::endl;
		pseudo_o->filter(1e-12);
		dbcsr::print(*pseudo_o);
		*/
		
		LOG.os<1>("Forming ZA.\n");
		m_zbuilder->compute();
		
		auto Z_xx = m_zbuilder->zmat();
		
		auto temp = dbcsr::create_template(Z_xx)
			.name("temp_xx")
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
			
		dbcsr::multiply('N', 'N', *m_s_xx_inv, *Z_xx, *temp)
			.filter_eps(dbcsr::global::filter_eps)
			.perform();
			
		dbcsr::multiply('N', 'N', *temp, *m_s_xx_inv, *Z_xx)
			.filter_eps(dbcsr::global::filter_eps)
			.perform();
			
		auto F_xx_A = dbcsr::copy(Z_xx).get();
		Z_xx->clear();
		
		F_xx_A->setname("F_xx_A");
		
		Z_xx->clear();
		
		// Form F_B
		zbuilder_asym.set_occ_coeff(L_bu);
		zbuilder_asym.set_vir_density(pseudo_v);
		
		LOG.os<1>("Forming ZB.\n");
		zbuilder_asym.compute();
		
		auto Z_xx_B = zbuilder_asym.zmat();
		auto F_xx_B = u_transform(Z_xx_B, 'N', m_s_xx_inv, 'T', m_s_xx_inv);
		
		F_xx_B->setname("F_xx_B");
		
		if (LOG.global_plev() >= 2) dbcsr::print(*F_xx_A);
		if (LOG.global_plev() >= 2) dbcsr::print(*F_xx_B);
		
		auto sig_pair = compute_sigma_2e_ilap(J_xbb_batched, F_xx_A, F_xx_B, 
			pseudo_o, pseudo_v
#if 0
		, omega, ilap
#endif
		);
		auto sig_ilap_A = sig_pair.first;
		auto sig_ilap_B = sig_pair.second;
		
		c_bo_eps = dbcsr::copy(m_c_bo).get();
		c_bv_eps = dbcsr::copy(m_c_bv).get();
		
		exp_occ = *m_epso;
		exp_vir = *m_epsv;
		
		std::for_each(exp_occ.begin(),exp_occ.end(),
			[xpt,wght](double& eps) {
				eps = exp(0.25 * log(wght) + eps * xpt);
			});
			
		std::for_each(exp_vir.begin(),exp_vir.end(),
			[xpt,wght](double& eps) {
				eps = exp(0.25 * log(wght) - eps * xpt);
			});
			
		c_bo_eps->scale(exp_occ, "right");
		c_bv_eps->scale(exp_vir, "right");
		
		//std::cout << "AOILAP" << std::endl;
		//dbcsr::print(*sig_ilap_A);
		
		auto sig_ilap_A_ia = u_transform(sig_ilap_A, 'T', m_c_bo, 'N', c_bv_eps);
		auto sig_ilap_B_ia = u_transform(sig_ilap_B, 'T', c_bo_eps, 'N', m_c_bv); 
		
		//std::cout << "ILAP" << std::endl;
		//dbcsr::print(*sig_ilap_A_ia);
		
		//std::cout << "SCALE: " << omega << " " << xpt << " " << exp(omega * xpt) << std::endl;
		
		sig_ilap_A_ia->scale(exp(omega * xpt));
		sig_ilap_B_ia->scale(exp(omega * xpt));
		
		sigma_2e_A->add(1.0, 1.0, *sig_ilap_A_ia);
		sigma_2e_B->add(1.0, 1.0, *sig_ilap_B_ia);
		
#if 0

		arrvec<int,2> bo = {b,o};
		arrvec<int,2> bv = {b,v};
		
		auto co = dbcsr::tensor_create<2,double>()
			.name("co")
			.pgrid(m_spgrid2)
			.map1({0}).map2({1})
			.blk_sizes(bo)
			.get();
			
		auto cv = dbcsr::tensor_create<2,double>()
			.name("cv")
			.pgrid(m_spgrid2)
			.map1({0}).map2({1})
			.blk_sizes(bv)
			.get();
			
		dbcsr::copy_matrix_to_tensor(*c_bo_eps, *co);
		dbcsr::copy_matrix_to_tensor(*c_bv_eps, *cv);
			
		dbcsr::contract(*co, *m_I, *I_HT)
			.perform("mi, Xmn -> Xin");
			
		dbcsr::contract(*cv, *I_HT, *I_mo)
			.beta(1.0).perform("na, Xin -> Xia");
		
#endif
		
		
		
	} // end loop over laplace points
	
	m_reg.erase("t_xbb_batched");
	
	sigma_2e_A->scale(-pow(m_c_osc, 2));
	sigma_2e_B->scale(pow(m_c_osc, 2));

#if 0
	I_mo->scale(m_c_osc);
	dbcsr::print(*I_mo);
	
	arrvec<int,3> xoo = {x,o,o};
	
	auto ints_mo = dbcsr::tensor_create<3,double>()
		.name("ints_mo")
		.pgrid(sp3)
		.map1({0}).map2({1,2})
		.blk_sizes(xoo)
		.get();
	
	arrvec<int,2> bo = {b,o};
	arrvec<int,2> bv = {b,v};
	arrvec<int,2> ov = {o,v};
	
	auto co = dbcsr::tensor_create<2,double>()
		.name("co")
		.pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bo)
		.get();
		
	auto cv = dbcsr::tensor_create<2,double>()
		.name("cv")
		.pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bv)
		.get();
		
	auto sig = dbcsr::tensor_create<2,double>()
		.name("sig")
		.pgrid(m_spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(ov)
		.get();
		
	dbcsr::copy_matrix_to_tensor(*m_c_bo, *co);
	
	auto e = m_eri_batched->get_stensor();
	dbcsr::contract(*co, *e, *I_HT)
		.perform("mi, Xmn -> Xin");
		
	dbcsr::contract(*co, *I_HT, *ints_mo)
		.perform("nj, Xin -> Xij");
		
	dbcsr::print(*ints_mo);
		
	dbcsr::contract(*I_mo, *ints_mo, *sig)
		.perform("Xka, Xkm -> ma");
	
	sig->scale(m_c_osc);
	dbcsr::print(*sig);
	
#endif
	
	//dbcsr::print(*sigma_2e_A);
	//dbcsr::print(*sigma_2e_B);
	
	sigma_2e_A->setname("sigma_2e");
	sigma_2e_A->add(1.0, 1.0, *sigma_2e_B);
	
	return sigma_2e_A;
	
}
	
smat MVP_ao_ri_adc2::compute(smat u_ia, double omega) {
	
	LOG.os<>("Computing ADC0.\n");
	// compute ADC0 part in MO basis
	smat sig_0 = compute_sigma_0(u_ia);
	
	//std::cout << "SIG0" << std::endl;
	//dbcsr::print(*sig_0);

	sig_0->setname("sigma ADC2 vector");

	// transform u to ao coordinated
	smat u_ao = u_transform(u_ia, 'N', m_c_bo, 'T', m_c_bv);
	
	//LOG.os<>("U transformed: \n");
	//dbcsr::print(*u_ao);
	
	auto jk = compute_jk(u_ao);
	
	auto sig_1 = compute_sigma_1(jk.first, jk.second);
	
	LOG.os<2>("Sigma adc1 mo:\n");
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*sig_1);
	}
	
	sig_0->add(1.0, 1.0, *sig_1);
	sig_1->release();
	
	auto sig_2a = compute_sigma_2a(u_ia);
	auto sig_2b = compute_sigma_2b(u_ia);
	
	LOG.os<2>("SIG A and SIG B\n");
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*sig_2a);
		dbcsr::print(*sig_2b);
	}
	
	sig_0->add(1.0, 1.0, *sig_2a);
	sig_0->add(1.0, 1.0, *sig_2b);
	
	sig_2a->release();
	sig_2b->release();
		
	auto sig_2c = compute_sigma_2c(jk.first, jk.second);
	
	LOG.os<2>("SIG_2C:\n");
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*sig_2c);
	}
	
	auto sig_2d = compute_sigma_2d(u_ia);
	
	LOG.os<2>("SIG_2D:\n");
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*sig_2d);
	}
	
	sig_0->add(1.0,1.0,*sig_2c);
	sig_0->add(1.0,1.0,*sig_2d);
	
	sig_2d->release();
	sig_2c->release();
		
	// compute new laplace points
	math::laplace lp_dd(m_world.comm(), LOG.global_plev());
	
	double emin = m_epso->front();
	double ehomo = m_epso->back();
	double elumo = m_epsv->front();
	double emax = m_epsv->back();
	
	double ymin = 2*(elumo - ehomo) + omega;
	double ymax = 2*(emax - emin) + omega;
	
	lp_dd.compute(m_nlap, ymin, ymax);
	m_weights_dd = lp_dd.omega();
	m_xpoints_dd = lp_dd.alpha();
	
	auto sig_2e = compute_sigma_2e(u_ao,omega);
	
	sig_0->add(1.0,1.0,*sig_2e);
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*sig_0);
	}
		
	return sig_0;
	
}
	
} // end namespace
