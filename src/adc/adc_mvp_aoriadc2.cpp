#include "adc/adc_mvp.h"
#include "math/laplace/laplace.h"
#include "math/linalg/piv_cd.h"
#include "math/linalg/LLT.h"
#include "adc/adc_defaults.h"
#include "ints/fitting.h"
#include "utils/matrix_plot.h"

//#define _DLOG
//SYM!!

namespace adc {
	
/* =====================================================================
 *                         AUXILIARY FUNCTIONS
 * ====================================================================*/
	
smat MVP_AOADC2::get_scaled_coeff(char dim, double wght, double xpt, 
	double wfactor, double xfactor) {
	
	auto c_bm = (dim == 'O') ? m_c_bo : m_c_bv;
	auto eps_m = (dim == 'O') ? m_eps_occ : m_eps_vir;
	int sign = (dim == 'O') ? 1 : -1;
	
	auto c_bm_scaled = dbcsr::copy(*c_bm).get();
	auto eps_scaled = *eps_m;
	
	std::for_each(eps_scaled.begin(),eps_scaled.end(),
		[xpt,wght,wfactor,xfactor,sign](double& eps) {
			eps = exp(wfactor * log(wght) + sign * xfactor * eps * xpt);
	});
			
	c_bm_scaled->scale(eps_scaled, "right");
	
	return c_bm_scaled;
	
}

smat MVP_AOADC2::get_density(smat coeff) {
	
	auto b = m_mol->dims().b();
	
	auto p_bb = dbcsr::create<double>()
		.set_world(m_world)
		.name("density matrix")
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::symmetric)
		.get();
		
	dbcsr::multiply('N', 'T', *coeff, *coeff, *p_bb).perform();
	
	return p_bb;
	
}

smat MVP_AOADC2::get_ortho_cholesky(char dim, double wght, double xpt, 
	double wfactor, double xfactor) {
		
	auto coeff = get_scaled_coeff(dim, wght, xpt, wfactor, xfactor);
		
	auto coeff_ortho = dbcsr::create_template<double>(*coeff)
		.name("co_ortho")
		.get();
	
	dbcsr::multiply('N', 'N', *m_ssqrt_bb, *coeff, *coeff_ortho)
		.perform();
		
	auto p_ortho = get_density(coeff_ortho);
	
	math::pivinc_cd chol(p_ortho, LOG.global_plev());
	chol.compute();
	
	int rank = chol.rank();
	
	auto b = m_mol->dims().b();
	auto u = dbcsr::split_range(rank, m_mol->mo_split());
	
	LOG.os<1>("Cholesky decomposition rank: ", rank, '\n');

	static int i = 0;
	++i;
	
	std::string filename_ortho = std::string(std::filesystem::current_path())
		+ "/cholortho_" + dim + "_" + std::to_string(i);

	std::string filename = std::string(std::filesystem::current_path())
		+ "/chol_" + dim + "_" + std::to_string(i);

	auto L_bu_ortho = chol.L(b, u);
	
	util::plot(L_bu_ortho, 1e-5, filename_ortho);
	
	auto L_bu = dbcsr::create_template<double>(*L_bu_ortho)
		.name("L_bu")
		.get();
		
	dbcsr::multiply('N', 'N', *m_sinvqrt_bb, *L_bu_ortho, *L_bu)
		.perform();
		
	util::plot(L_bu, 1e-5, filename);
		
	return L_bu;
	
}	

/* =====================================================================
 *                         INITIALIZING FUNCTIONS
 * ====================================================================*/
	
void MVP_AOADC2::init() {
	
	LOG.os<1>("Initializing AO-ADC(2)\n");
	
	auto& t_init = TIME.sub("Initializing ADC(2)");
	
	TIME.start();
	t_init.start();
	
	// laplace
	LOG.os<1>("Computing laplace points.\n");
		
	double emin = m_eps_occ->front();
	double ehomo = m_eps_occ->back();
	double elumo = m_eps_vir->front();
	double emax = m_eps_vir->back();
	
	double ymin = 2*(elumo - ehomo);
	double ymax = 2*(emax - emin);
	
	LOG.os<1>("eps_min/eps_homo/eps_lumo/eps_max ", emin, " ", ehomo, " ", elumo, " ", emax, '\n');
	LOG.os<1>("ymin/ymax ", ymin, " ", ymax, '\n');
	
	math::laplace lp(m_world.comm(), LOG.global_plev());
	
	lp.compute(m_nlap, ymin, ymax);
		
	m_weights = lp.omega();
	m_xpoints = lp.alpha();
	
	m_pseudo_occs.resize(m_nlap);
	m_pseudo_virs.resize(m_nlap);
	
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		double wght = m_weights[ilap];
		double xpt = m_xpoints[ilap];
		
		auto c_bo_ilap = get_scaled_coeff('O', wght, xpt, 0.0, 0.5);
		auto c_bv_ilap = get_scaled_coeff('V', wght, xpt, 0.0, 0.5);
		
		auto Do_pseudo = get_density(c_bo_ilap);
		auto Dv_pseudo = get_density(c_bv_ilap);
		
		double wfac = pow(wght,0.25);
		
		Do_pseudo->scale(wfac);
		Dv_pseudo->scale(wfac);
				
		m_pseudo_occs[ilap] = Do_pseudo;
		m_pseudo_virs[ilap] = Dv_pseudo;
	}
	
	LOG.os<1>("Computing Cholesky decomposition of S\n");
	math::LLT chol(m_s_bb, LOG.global_plev());
	
	chol.compute();
	auto b = m_mol->dims().b();
	
	m_ssqrt_bb = chol.L(b);
	m_sinvqrt_bb = chol.L_inv(b);
	
	LOG.os<1>("Setting up J,K,Z builders.\n");
	
	int nprint = LOG.global_plev();
	
	// J builder
	switch (m_jmethod) {
		case fock::jmethod::dfao:
		{
			m_jbuilder = fock::create_DF_J(m_world, m_mol, nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.v_inv(m_v_xx)
				.get();
			break;
		}
		default:
		{
			throw std::runtime_error("Invalid J method in AO-ADC2.");
		}
	}
	
	// K builder
	switch (m_kmethod) {
		case fock::kmethod::dfao:
		{			
			m_kbuilder = fock::create_DFAO_K(m_world, m_mol, nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.fitting_batched(m_fitting_batched)
				.get();
			break;
		}
		case fock::kmethod::dfmem:
		{
			m_kbuilder = fock::create_DFMEM_K(m_world, m_mol, nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.v_xx(m_v_xx)
				.get();
			break;
		}
		default:
		{
			throw std::runtime_error("Invalid K method in AO-ADC2.");
		}
	}
	
	// Z builder
	m_shellpairs = mp::get_shellpairs(m_eri3c2e_batched);
	
	switch (m_zmethod) {
		case mp::zmethod::llmp_full:
		{
			m_zbuilder = mp::create_LLMP_FULL_Z(m_world, m_mol, nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.intermeds(m_btype)
				.get();
			break;
		}
		case mp::zmethod::llmp_mem:
		{
			m_zbuilder = mp::create_LLMP_MEM_Z(m_world, m_mol, nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.get();
			break;
		}
		default:
		{
			throw std::runtime_error("Invalid K method in AO-ADC2.");
		}
	}
	
	m_jbuilder->set_sym(false);
	m_jbuilder->init();
	
	m_kbuilder->set_sym(false);
	m_kbuilder->init();
	
	m_zbuilder->set_shellpair_info(m_shellpairs);
	m_zbuilder->init();
	
	// Intermeds
	LOG.os<1>("Computing intermediates.\n");
	compute_intermeds();
	
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	t_init.finish();
	TIME.finish();
	
	LOG.os<1>("Done with setting up.\n");
	
}

/* =====================================================================
 *                          MVP FUNCTIONS (ADC1)
 * ====================================================================*/

// Computes the pseudo J and K matrices with the excited state density
std::pair<smat,smat> MVP_AOADC2::compute_jk(smat& u_ao) {
	
	auto& t_jk = TIME.sub("Computing pseudo-JK");
	t_jk.start();
	
	m_jbuilder->set_density_alpha(u_ao);
	m_kbuilder->set_density_alpha(u_ao);
	
	m_jbuilder->compute_J();
	m_kbuilder->compute_K();
	
	auto jmat = m_jbuilder->get_J();
	auto kmat = m_kbuilder->get_K_A();
	
	std::pair<smat,smat> out = {jmat, kmat};
	
	t_jk.finish();
	
	return out;
	
}

// computes the ADC(1) MVP part using the pseudo J,K matrices
smat MVP_AOADC2::compute_sigma_1(smat& jmat, smat& kmat) {
	
	auto j = u_transform(jmat, 'T', m_c_bo, 'N', m_c_bv);
	auto k = u_transform(kmat, 'T', m_c_bo, 'N', m_c_bv);
	
	smat sig_ao = dbcsr::create_template<double>(*jmat)
		.name("sig_ao")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	sig_ao->add(0.0, 1.0, *jmat);
	sig_ao->add(1.0, 1.0, *kmat);
	
	// transform back
	smat sig_1 = u_transform(sig_ao, 'T', m_c_bo, 'N', m_c_bv);
	
	sig_1->setname("sigma_1");
	
	return sig_1;
	
} 

/* =====================================================================
 *                         ADC(2) FUNCTIONS
 * ====================================================================*/
 
/* =====================================================================
 *                         ADC(2) INTERMEDIATES
 * ====================================================================*/
 
void MVP_AOADC2::compute_intermeds() {
	
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
	 
	LOG.os<1>("==== Computing ADC(2) intermediates ====\n");
	
	auto& t_intermeds = TIME.sub("Computing Intermediates");
	
	auto& t_intermeds_1 = t_intermeds.sub("Part1");
	auto& t_intermeds_2 = t_intermeds.sub("Part2");
	
	t_intermeds.start();
	
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
		
	auto i_oo_tmp = dbcsr::create_template(*m_i_oo)
		.name("i_oo_temp").get();
		
	auto i_vv_tmp = dbcsr::create_template(*m_i_vv)
		.name("i_vv_temp").get();
	
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		LOG.os<1>("ADC(2) intermediates, laplace point nr. ", ilap, '\n');
		
		t_intermeds_1.start();
		
		//auto po = m_pseudo_occs[ilap];
		
		auto L_bu = get_ortho_cholesky('O', m_weights[ilap], 
			m_xpoints[ilap], 0.125, 0.5);
			
		auto po = m_pseudo_occs[ilap];
		auto pv = m_pseudo_virs[ilap];
		
		L_bu->filter(dbcsr::global::filter_eps);

		pv->filter(dbcsr::global::filter_eps);
		
		m_zbuilder->set_occ_coeff(L_bu);
		m_zbuilder->set_vir_density(pv);
	
		LOG.os<1>("Computing Z.\n");
	
		m_zbuilder->compute();
		auto z_xx_ilap = m_zbuilder->zmat();
				
		auto f_xx_ilap = u_transform(z_xx_ilap, 'N', m_v_xx, 'N', m_v_xx);
		f_xx_ilap->setname("f_xx_ilap");
		
		t_intermeds_1.finish();
		t_intermeds_2.start();
		
		LOG.os<1>("Setting up K_ilap.\n");
		std::shared_ptr<fock::K> k_inter;
		
		int nprint = LOG.global_plev();
		
		switch (m_kmethod) {
			case fock::kmethod::dfao: 
			{
				ints::dfitting dfit(m_world, m_mol, nprint);
				auto I_fit_xbb = dfit.compute(m_eri3c2e_batched, f_xx_ilap, m_btype);
				k_inter = fock::create_DFAO_K(m_world, m_mol, nprint)
					.eri3c2e_batched(m_eri3c2e_batched)
					.fitting_batched(I_fit_xbb)
					.get();
				break;
			}
			case fock::kmethod::dfmem:
			{
				k_inter = fock::create_DFMEM_K(m_world, m_mol, nprint)
					.eri3c2e_batched(m_eri3c2e_batched)
					.v_xx(f_xx_ilap)
					.get();
				break;
			}
		}
			
		k_inter->set_sym(false);
		k_inter->init();
		k_inter->set_density_alpha(pv);
		
		LOG.os<1>("Computing K_ilap.\n");
		
		k_inter->compute_K();
		auto ko_ilap = k_inter->get_K_A();
		
		double wght = m_weights[ilap];
		double xpt = m_xpoints[ilap];
		
		auto c_bo_scaled = get_scaled_coeff('O', wght, xpt, 0.25, 1.0);
		auto c_bv_scaled = get_scaled_coeff('V', wght, xpt, 0.25, 1.0);
		
		LOG.os<1>("Forming partly-transformed intermediates.\n");
		
		dbcsr::multiply('T', 'N', *c_bo_scaled, *ko_ilap, *i_ob)
			.beta(1.0)
			.perform();
			
		k_inter->set_density_alpha(po);
		k_inter->compute_K();
		
		auto kv_ilap = k_inter->get_K_A();
			
		dbcsr::multiply('T', 'N', *c_bv_scaled, *kv_ilap, *i_vb)
			.beta(1.0)
			.perform();
			
		t_intermeds_2.finish();
								
	}
	
	LOG.os<>("Forming fully transformed intermediates.\n");
	
	dbcsr::multiply('N', 'N', *i_ob, *m_c_bo, *m_i_oo)
			.perform();
	dbcsr::multiply('N', 'N', *i_vb, *m_c_bv, *m_i_vv)
			.perform();
			
	auto i_oo_tr = dbcsr::transpose(*m_i_oo).get();
	auto i_vv_tr = dbcsr::transpose(*m_i_vv).get();
	
	m_i_oo->add(1.0, 1.0, *i_oo_tr);
	m_i_vv->add(1.0, 1.0, *i_vv_tr);
	
	m_i_oo->scale(-0.5 * m_c_os);
	m_i_vv->scale(-0.5 * m_c_os);
	
	t_intermeds.finish();
			
}

/* =====================================================================
 *                         ADC(2) P-H CONTRIBUTIONS
 * ====================================================================*/

smat MVP_AOADC2::compute_sigma_2a(smat& u_ia) {
	
	auto& t_sig = TIME.sub("Computing sigma (A)");
	t_sig.start();
	
	LOG.os<1>("==== Computing ADC(2) SIGMA 2A ====\n");
	
	// sig_2a = i_vv_ab * u_ib
	auto sig_2a = dbcsr::create_template<double>(*u_ia)
		.name("sig_2a")
		.get();
		
	dbcsr::multiply('N', 'T', *u_ia, *m_i_vv, *sig_2a).perform();
	
	t_sig.finish();
	
	return sig_2a;
	
}

smat MVP_AOADC2::compute_sigma_2b(smat& u_ia) {
	
	auto& t_sig = TIME.sub("Computing sigma (B)");
	t_sig.start();
	
	LOG.os<1>("==== Computing ADC(2) SIGMA 2B ====\n");
	
	// sig_2b = i_oo_ij * u_ja
	auto sig_2b = dbcsr::create_template<double>(*u_ia)
		.name("sig_2b")
		.get();
		
	dbcsr::multiply('N', 'N', *m_i_oo, *u_ia, *sig_2b).perform();
	
	t_sig.finish();
	
	return sig_2b;
	
} 

smat MVP_AOADC2::compute_sigma_2c(smat& jmat, smat& kmat) {
	
	// sig_2c = -1/2 t_iajb^SOS * I_jb
	// I_ia = [2*(jb|ia) - (ja|ib)] u_jb
	
	// in AO:
	/* sig_2c = +1/2 * c_os * sum_t c_mi * exp(eps_i t) 
	 * 				c_na * exp(-eps_a t) * (mn|X') * (X'|X)^inv d_X(t)
	 * d_X(t) = (nk|X) * I_n'k' * Po_nn'(t) * Pv_kk'(t)
	 * I_mk = jmat + transpose(K)
	 */
	 
	auto& t_sig = TIME.sub("Computing sigma (C)");
	t_sig.start();
	
	LOG.os<1>("==== Computing ADC(2) SIGMA 2C ====\n");
	 		
	auto I_ao = dbcsr::create_template<double>(*jmat)
		.name("I_ao").get();
	auto jmat_t = dbcsr::transpose(*jmat)
		.get();
	auto kmat_t = dbcsr::transpose(*kmat)
		.get();
	
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
		
	LOG.os<1>("-- Contracting over laplace points...\n");
	
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		auto pseudo_o = m_pseudo_occs[ilap];
		auto pseudo_v = m_pseudo_virs[ilap];
		
		auto I_pseudo = u_transform(I_ao, 'T', pseudo_o, 'N', pseudo_v);
		
		m_jbuilder->set_density_alpha(I_pseudo);		
		m_jbuilder->compute_J();
		
		auto jmat_pseudo = m_jbuilder->get_J();
		
		double wght = m_weights[ilap];
		double xpt = m_xpoints[ilap];
		
		auto c_bo_scaled = get_scaled_coeff('O', wght, xpt, 0.25, 1.0);
		auto c_bv_scaled = get_scaled_coeff('V', wght, xpt, 0.25, 1.0);
		
		auto jmat_trans = u_transform(jmat_pseudo, 
			'T', c_bo_scaled, 'N', c_bv_scaled);
		
		sig_2c->add(1.0, 1.0, *jmat_trans);
		
	}
				
	sig_2c->scale(-0.25 * m_c_os);
	
	t_sig.finish();
	
	return sig_2c;
}

smat MVP_AOADC2::compute_sigma_2d(smat& u_ia) {
	
	/* sig_2d = - 0.5 sum_jb [2 * (ia|bj) - (ja|ib)] * I_jb
	 * where I_ia = sum_jb t_iajb^SOS u_jb
	 * 
	 * in AO:
	 * 
	 * sig_2a = + 2 * J(I_nr)_transpose - K(I_nr)_transpose 
	 * I_nr = sum_t c_os Po(t) * J(u_pseudo_ao(t)) * Pv(t)
	 */
	 
	auto& t_sig = TIME.sub("Computing sigma (D)");
	t_sig.start();
	 
	LOG.os<1>("==== Computing ADC(2) SIGMA 2D ====\n");
	 
	auto I_ao = dbcsr::create_template<double>(*m_pseudo_occs[0])
		.name("I_ao")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	 
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		double wght = m_weights[ilap];
		double xpt = m_xpoints[ilap];
		
		auto c_bo_scaled = get_scaled_coeff('O', wght, xpt, 0.25, 1.0);
		auto c_bv_scaled = get_scaled_coeff('V', wght, xpt, 0.25, 1.0);
		
		auto u_pseudo = u_transform(u_ia, 'N', c_bo_scaled, 
			'T', c_bv_scaled);
		
		c_bo_scaled->release();
		c_bv_scaled->release();
		
		m_jbuilder->set_density_alpha(u_pseudo);
		m_jbuilder->compute_J();
		
		auto jpseudo1 = m_jbuilder->get_J();
		
		auto po = m_pseudo_occs[ilap];
		auto pv = m_pseudo_virs[ilap];
		auto jpseudo2 = u_transform(jpseudo1, 'N', po, 'T', pv); 
		
		I_ao->add(1.0, 0.5, *jpseudo2);
		
	}
	
	I_ao->scale(m_c_os);
	
	m_jbuilder->set_density_alpha(I_ao);
	m_kbuilder->set_density_alpha(I_ao);
	
	m_jbuilder->compute_J();
	m_kbuilder->compute_K();
	
	auto jmat = m_jbuilder->get_J();
	auto kmat = m_kbuilder->get_K_A();
	
	auto jmat_trans = dbcsr::transpose(*jmat).get();
	auto kmat_trans = dbcsr::transpose(*kmat).get();
	
	jmat_trans->add(1.0, 1.0, *kmat_trans);
	
	auto sig_2d = u_transform(jmat_trans, 'T', m_c_bo, 'N', m_c_bv);
	sig_2d->setname("sig_2d");
	
	sig_2d->scale(-0.5);
	
	t_sig.finish();
	
	return sig_2d;
		
}

/* =====================================================================
 *                  ADC(2) 2H-P, H-2P 2P-2H CONTRIBUTIONS
 * ====================================================================*/
 
dbcsr::sbtensor<3,double> MVP_AOADC2::compute_R(smat& u_ao) {
	
	LOG.os<1>("Forming R_xbb\n");
	// form R
	
	auto& time_R = TIME.sub("Computing R");
	time_R.start();
	
	auto spgrid3_xbb = m_eri3c2e_batched->spgrid();
	
	arrvec<int,3> xbb = {m_mol->dims().x(), m_mol->dims().b(), m_mol->dims().b()};
	
	auto R_xbb_0_12 = dbcsr::tensor_create<3,double>()
		.name("R_xbb_0_12")
		.pgrid(spgrid3_xbb)
		.map1({0}).map2({1,2})
		.blk_sizes(xbb)
		.get();
		
	auto R_xbb_2_01_t = dbcsr::tensor_create_template<3,double>(R_xbb_0_12)
		.name("R_xbb_2_01_t")
		.map1({2}).map2({0,1})
		.get();
		
	auto R_xbb_2_01 = dbcsr::tensor_create_template<3,double>(R_xbb_0_12)
		.name("R_xbb_2_01")
		.map1({2}).map2({0,1})
		.get();	
	
	auto R_xbb_batched = dbcsr::btensor_create_template<3>
		(m_eri3c2e_batched)
		.name(m_mol->name() + "_R_xbb_batched")
		.btensor_type(m_btype)
		.print(LOG.global_plev())
		.get();
		
	// Form half-projected AO u
	// u_v_mu,nu = u_mu,nu' * S_nu',nu
	// u_o_mu.nu = S_mu',mu * u_mu',nu
		
	auto u_hto = dbcsr::create_template<double>(*u_ao)
		.name("u_hto").get();
	auto u_htv = dbcsr::create_template<double>(*u_ao)
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
	
	u_hto_01->filter(dbcsr::global::filter_eps);
	u_htv_01->filter(dbcsr::global::filter_eps);
	
	u_hto->release();
	u_htv->release();

	m_eri3c2e_batched->decompress_init({0}, vec<int>{2}, vec<int>{0,1});
	R_xbb_batched->compress_init({0}, vec<int>{0}, vec<int>{1,2});
	
	auto nxbatches = m_eri3c2e_batched->nbatches(0);
	auto nbbatches = m_eri3c2e_batched->nbatches(2);
	auto fullbbounds = m_eri3c2e_batched->full_bounds(1);
	
	for (int ix = 0; ix != nxbatches; ++ix) {
		
		m_eri3c2e_batched->decompress({ix});
		auto eri_2_01 = m_eri3c2e_batched->get_work_tensor();
		
		auto xbounds = m_eri3c2e_batched->bounds(0,ix);
		
		for (int inu = 0; inu != nbbatches; ++inu) {
			
			auto bbounds = m_eri3c2e_batched->bounds(2,inu);
			
			vec<vec<int>> nu_bounds = {
				bbounds
			};
			
			vec<vec<int>> xmu_bounds = {
				xbounds,
				fullbbounds
			};
			
			vec<vec<int>> xnu_bounds = {
				xbounds,
				bbounds
			};
			
			dbcsr::contract(*u_hto_01, *eri_2_01, *R_xbb_2_01_t)
				.alpha(-1.0)
				.bounds3(xnu_bounds)
				.filter(dbcsr::global::filter_eps / nxbatches)
				.perform("mg, Yag -> Yam");
			
			dbcsr::copy(*R_xbb_2_01_t, *R_xbb_2_01)
				.move_data(true)
				.order(vec<int>{0,2,1})
				.perform();
			
			dbcsr::print(*R_xbb_2_01);
			
			dbcsr::contract(*u_htv_01, *eri_2_01, *R_xbb_2_01)
				.bounds2(nu_bounds)
				.bounds3(xmu_bounds)
				.filter(dbcsr::global::filter_eps / nxbatches)
				.beta(1.0)
				.perform("ka, Ymk -> Yma");
								
			dbcsr::copy(*R_xbb_2_01, *R_xbb_0_12)
				.move_data(true)
				.sum(true)
				.perform();
			
		}
		
		R_xbb_0_12->filter(dbcsr::global::filter_eps);
		R_xbb_batched->compress({ix}, R_xbb_0_12);
		
	}
	
	m_eri3c2e_batched->decompress_finalize();
	R_xbb_batched->compress_finalize();
	
	time_R.finish();
	
	LOG.os<1>("Occupation of R: ", R_xbb_batched->occupation() * 100, "%\n");
	
	//exit(0);
	
	return R_xbb_batched;
	
}

std::pair<smat,smat> MVP_AOADC2::compute_sigma_2e_ilap(
	dbcsr::sbtensor<3,double>& R_xbb_batched, 
	smat& FA, smat& FB, smat& pseudo_o, smat& pseudo_v,
	bool mem) {

#ifdef _DLOG
	auto get_sum = [](auto& in) {
		long long int datasize;
		double* ptr = in->data(datasize);
		return std::accumulate(ptr, ptr + datasize, 0.0);
	};

	
	auto RR = R_xbb_batched->get_work_tensor();
	std::cout << "R LONG SUM " << std::setprecision(16) << get_sum(RR) << '\n';
#endif
		
	auto& ltime = TIME.sub("Computing sigma ilap");
	auto& ltime_1 = ltime.sub("Part1");
	auto& ltime_2 = ltime.sub("Part2");
	
	ltime.start();
		
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

#ifdef _DLOG
	std::cout << "FA LONG SUM " << std::setprecision(16) << get_sum(FA_01) << '\n';
	std::cout << "FB LONG SUM " << std::setprecision(16) << get_sum(FB_01) << '\n';
#endif

	FA->clear();
	FB->clear();
	
	auto sigma_ilap_A = dbcsr::create_template<double>(*pseudo_o)
		.name("sigma_ilap_A")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto sigma_ilap_B = dbcsr::create_template<double>(*pseudo_o)
		.name("sigma_ilap_B")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	if (!mem) {
		
		// Intermediate is held in-core/on-disk
		ltime_1.start();
		
		auto spgrid3_xbb = m_eri3c2e_batched->spgrid();
		
		auto I_xbb_batched = dbcsr::btensor_create_template<3,double>(
			m_eri3c2e_batched)
			.name(m_mol->name() + "_i_xbb_batched")
			.btensor_type(m_btype)
			.print(LOG.global_plev())
			.get();
			
		auto I_xbb_0_12 = I_xbb_batched->get_template("I_xbb_0_12", 
			vec<int>{0}, vec<int>{1,2});
			
		// Form I
	
		I_xbb_batched->compress_init({2}, vec<int>{0}, vec<int>{1,2});
		R_xbb_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
		m_eri3c2e_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
		
		int nxbatches = m_eri3c2e_batched->nbatches(0);
		int nbbatches = m_eri3c2e_batched->nbatches(2);
		auto fullbbds = m_eri3c2e_batched->full_bounds(1);
		
		for (int inu = 0; inu != nbbatches; ++inu) {
			
			R_xbb_batched->decompress({inu});
			auto R_xbb_0_12 = R_xbb_batched->get_work_tensor();
			
			m_eri3c2e_batched->decompress({inu});
			auto eri_0_12 = m_eri3c2e_batched->get_work_tensor();
			
			auto bbds = m_eri3c2e_batched->bounds(2,inu);
			
			for (int ix = 0; ix != nxbatches; ++ix) {		
				
				auto xbds = m_eri3c2e_batched->bounds(0,ix);
				
				vec<vec<int>> xbounds = { 
					xbds
				};
				
				vec<vec<int>> kabounds = { 
					fullbbds,
					bbds
				};
				
				dbcsr::contract(*FA_01, *R_xbb_0_12, *I_xbb_0_12)
					.bounds2(xbounds)
					.bounds3(kabounds)
					.filter(dbcsr::global::filter_eps)
					.beta(1.0)
					.perform("XY, Yka -> Xka");
								
			}
			
			for (int ix = 0; ix != nxbatches; ++ix) {		
				
				auto xbds = m_eri3c2e_batched->bounds(0,ix);
				
				vec<vec<int>> xbounds = { 
					xbds
				};
				
				vec<vec<int>> kabounds = { 
					fullbbds,
					bbds
				};
					
				dbcsr::contract(*FB_01, *eri_0_12, *I_xbb_0_12)
					.bounds2(xbounds)
					.bounds3(kabounds)
					.filter(dbcsr::global::filter_eps)
					.beta(1.0)
					.perform("YX, Yka -> Xka");
									
			}
			
			I_xbb_batched->compress({inu}, I_xbb_0_12);
			
		}
		
		m_eri3c2e_batched->decompress_finalize();				
		R_xbb_batched->decompress_finalize();
		I_xbb_batched->compress_finalize();
		
		ltime_1.finish();
		ltime_2.start();
		
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

#ifdef _DLOG
		std::cout << "PO LONG SUM " << std::setprecision(16) << get_sum(po_01) << '\n';
		std::cout << "PV LONG SUM " << std::setprecision(16) << get_sum(pv_01) << '\n';
#endif

		auto cbar_1_02 = dbcsr::tensor_create_template<3,double>(I_xbb_0_12)
			.map1({1}).map2({0,2})
			.name("cbar_1_02").get();
			
		auto cbar_01_2 = dbcsr::tensor_create_template<3,double>(I_xbb_0_12)
			.map1({0,1}).map2({2})
			.name("cbar_01_2")
			.get();
			
		auto I_xbb_01_2 = dbcsr::tensor_create_template<3,double>(I_xbb_0_12)
			.map1({0,1}).map2({2})
			.name("I_xbb_02_1")
			.get();
		
		I_xbb_batched->decompress_init({0},vec<int>{1},vec<int>{0,2});
		m_eri3c2e_batched->decompress_init({0},vec<int>{0,1},vec<int>{2});
		
		for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
			
			m_eri3c2e_batched->decompress({ix});
			auto eri_1_02 = m_eri3c2e_batched->get_work_tensor();
			
			I_xbb_batched->decompress({ix});
			auto I_xbb_01_2 = I_xbb_batched->get_work_tensor();
			
			for (int inu = 0; inu != m_eri3c2e_batched->nbatches(2); ++inu) {
				
				vec<vec<int>> xmbounds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->full_bounds(1)
				};
				vec<vec<int>> nbounds = {
					m_eri3c2e_batched->bounds(2,inu)
				};
				
				// form cbar
				dbcsr::contract(*eri_1_02, *po_01, *cbar_1_02)
					.bounds2(xmbounds)
					.bounds3(nbounds)
					.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps 
						/ m_eri3c2e_batched->nbatches(0))
					.perform("Xlm, lk -> Xkm");
										
				dbcsr::copy(*cbar_1_02, *cbar_01_2).move_data(true).perform();
				
				vec<vec<int>> xnubounds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->bounds(2,inu)
				};
				
				//std::cout << "I LONG SUM " << std::setprecision(16) << get_sum(I_xbb_01_2) << '\n';
				//std::cout << "C LONG SUM " << std::setprecision(16) << get_sum(cbar_01_2) << '\n';
				
				// form sig_A
				dbcsr::contract(*I_xbb_01_2, *cbar_01_2, *sigma_ilap_01)
					.bounds1(xnubounds)
					.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps 
						/ m_eri3c2e_batched->nbatches(0))
					.beta(1.0)
					.perform("Xka, Xkm -> ma");
				
				cbar_01_2->clear();
								
			}
			
		}
		
		dbcsr::copy_tensor_to_matrix(*sigma_ilap_01, *sigma_ilap_A);
		
		sigma_ilap_01->clear();
		
		m_eri3c2e_batched->decompress_finalize();
		I_xbb_batched->decompress_finalize();
		
		LOG.os<1>("Occupation of I_xbb: ", I_xbb_batched->occupation() * 100, "%\n");
				
		m_eri3c2e_batched->decompress_init({0},vec<int>{1},vec<int>{0,2});
		I_xbb_batched->decompress_init({0,2},vec<int>{0,2},vec<int>{1});
		
		for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
			
			m_eri3c2e_batched->decompress({ix});
			auto eri_1_02 = m_eri3c2e_batched->get_work_tensor();
						
			for (int inu = 0; inu != m_eri3c2e_batched->nbatches(2); ++inu) {
				
				vec<vec<int>> xmbounds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->full_bounds(1)
				};
				vec<vec<int>> nbounds = {
					m_eri3c2e_batched->bounds(2,inu)
				};
				
				// form cbar
				dbcsr::contract(*eri_1_02, *pv_01, *cbar_1_02)
					.bounds2(xmbounds)
					.bounds3(nbounds)
					.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps 
						/ m_eri3c2e_batched->nbatches(0))
					.perform("Xda, dg -> Xga");
					
				dbcsr::copy(*cbar_1_02, *cbar_01_2).move_data(true).perform();
				
				I_xbb_batched->decompress({ix,inu});
				auto I_xbb_02_1 = I_xbb_batched->get_work_tensor();
				
				vec<vec<int>> xnubounds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->bounds(2,inu)
				};
				
				
				// form sig_A
				dbcsr::contract(*I_xbb_02_1, *cbar_01_2, *sigma_ilap_01)
					.bounds1(xnubounds)
					.filter(dbcsr::global::filter_eps 
						/ m_eri3c2e_batched->nbatches(0))
					.print(LOG.global_plev() >= 3)
					.beta(1.0)
					.perform("Xmg, Xga -> ma");
					
			}
			
		}
		
		m_eri3c2e_batched->decompress_finalize();
		I_xbb_batched->decompress_finalize();
		
		dbcsr::copy_tensor_to_matrix(*sigma_ilap_01, *sigma_ilap_B);
		
		I_xbb_batched->reset();		
		
		ltime_2.finish();
				
	} else {
		
		throw std::runtime_error("NYI");
		
	}

#ifdef _DLOG
	std::cout << "ILAP A" << std::endl;
	dbcsr::print(*sigma_ilap_A);
#endif

#ifdef _DLOG
	std::cout << "ILAP B" << std::endl;
	dbcsr::print(*sigma_ilap_B);
#endif

	std::pair<smat,smat> out = {sigma_ilap_A, sigma_ilap_B};

	ltime.finish();
	
	return out;
	
}
#ifndef _DOUBLES_REDUCED
smat MVP_AOADC2::compute_sigma_2e(smat& u_ao, double omega) {
	
	/* IN AO:
	 * sig_e2 = - c_os_c ^2 [ sum_t
	 * 	exp(omega t) C_μi C_αa * exp(-ε_a t) * I_{Xκα}(t) * (X|μκ') Po(t)_κκ'
	 * + exp(omega t) C_μi C_αa * exp(ε_i t) * I_{Xμγ}(t) * (X|γ'α) Pv(t)_γγ']
	 * 
	 * with
	 * 
	 * I_{Xκα}(t) = R_{Yκα} * F_{YX}(t) + (Y|κα) Ftilde_{XY}(t)
	 * 
	 * Ftilde_{X'Y'}(t) = (X'|X) (X|μν) * R_{Xμ'ν'} * Po(t)_{μμ'} * Pv(t)_{νν'} (YY')
	 * 
	 * R_{Χκα} = [ wv_κα (X|μκ) - wo_μγ (X|γα)]
	 * wv_κα = co_κi * S_αα' cv_α'a u_ia
	 * wo_μγ = S_μμ' * co_μ'i * cv_γa * u_ia
	 */
	
	LOG.os<1>("==== Computing ADC(2) SIGMA 2E ====\n");
	
	auto& time_2e = TIME.sub("Computing sigma(2e)");
	time_2e.start();
	
	double emin = m_eps_occ->front();
	double ehomo = m_eps_occ->back();
	double elumo = m_eps_vir->front();
	double emax = m_eps_vir->back();
	
	double ymin = 2*(elumo - ehomo) + omega;
	double ymax = 2*(emax - emin) + omega;
	
	LOG.os<1>("eps_min/eps_homo/eps_lumo/eps_max ", emin, " ", ehomo, " ", elumo, " ", emax, '\n');
	LOG.os<1>("ymin/ymax ", ymin, " ", ymax, '\n');
	
	math::laplace lp_dd(m_world.comm(), LOG.global_plev());
	
	lp_dd.compute(m_nlap, ymin, ymax);
		
	auto weights_dd = lp_dd.omega();
	auto xpoints_dd = lp_dd.alpha();
	
	auto o = m_mol->dims().oa();
	auto v = m_mol->dims().va();
	
	auto sigma_2e_A = dbcsr::create<double>()
		.name("sigma_2e_A")
		.set_world(m_world)
		.row_blk_sizes(o)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto sigma_2e_B = dbcsr::create_template<double>(*sigma_2e_A)
		.name("sigma_2e_B").get();
	
	// Form J
	auto R_xbb_batched = compute_R(u_ao);
	
	// Prepare asym zbuilder
	std::shared_ptr<mp::Z> zbuilder_asym = 
		mp::create_LLMP_ASYM_Z(m_world, m_mol, LOG.global_plev())
		.t3c2e_right_batched(R_xbb_batched)
		.t3c2e_left_batched(m_eri3c2e_batched)
		.get();
		
	zbuilder_asym->set_shellpair_info(m_shellpairs);
	zbuilder_asym->init();

	// Loop
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		double wght_dd = weights_dd[ilap];
		double xpt_dd = xpoints_dd[ilap];
		
		auto c_bo_scaled = get_scaled_coeff('O', wght_dd, xpt_dd, 0.0, 0.5);
		auto c_bv_scaled = get_scaled_coeff('V', wght_dd, xpt_dd, 0.0, 0.5);
		auto L_bu = get_ortho_cholesky('O', wght_dd, xpt_dd, 0.125, 0.5);
		
		auto pseudo_o = get_density(c_bo_scaled);
		auto pseudo_v = get_density(c_bv_scaled);
		
		pseudo_o->scale(pow(wght_dd,0.25));
		pseudo_v->scale(pow(wght_dd,0.25));
		
		L_bu->filter(dbcsr::global::filter_eps);
		pseudo_o->filter(dbcsr::global::filter_eps);
		pseudo_v->filter(dbcsr::global::filter_eps);
		
		// Form F_A
		m_zbuilder->set_occ_coeff(L_bu);
		m_zbuilder->set_vir_density(pseudo_v);
		
		LOG.os<1>("Forming ZA.\n");
		m_zbuilder->compute();
		
		auto Z_xx_A = m_zbuilder->zmat();
		
#ifdef _DLOG
		dbcsr::print(*Z_xx_A);
#endif
			
		auto F_xx_A = u_transform(Z_xx_A, 'N', m_v_xx, 'T', m_v_xx);
		F_xx_A->setname("F_xx_A");
		
		Z_xx_A->clear();
		
		// Form F_B
		zbuilder_asym->set_occ_coeff(L_bu);
		zbuilder_asym->set_vir_density(pseudo_v);
		
		LOG.os<1>("Forming ZB.\n");
		zbuilder_asym->compute();
		
		auto Z_xx_B = zbuilder_asym->zmat();
		
#ifdef _DLOG
		dbcsr::print(*Z_xx_B);
#endif
		
		auto F_xx_B = u_transform(Z_xx_B, 'N', m_v_xx, 'T', m_v_xx);
		
		F_xx_B->setname("F_xx_B");
		
		if (LOG.global_plev() > 2) dbcsr::print(*F_xx_A);
		if (LOG.global_plev() > 2) dbcsr::print(*F_xx_B);
		
		auto sig_pair = compute_sigma_2e_ilap(R_xbb_batched, F_xx_A, F_xx_B, 
			pseudo_o, pseudo_v, false);
			
		auto sig_ilap_A = sig_pair.first;
		auto sig_ilap_B = sig_pair.second;
		
		c_bo_scaled = get_scaled_coeff('O', wght_dd, xpt_dd, 0.25, 1.0);
		c_bv_scaled = get_scaled_coeff('V', wght_dd, xpt_dd, 0.25, 1.0);
		
		auto sig_ilap_A_ia = u_transform(sig_ilap_A, 'T', m_c_bo, 'N', c_bv_scaled);
		auto sig_ilap_B_ia = u_transform(sig_ilap_B, 'T', c_bo_scaled, 'N', m_c_bv); 
		
		double xpt = xpoints_dd[ilap];
		
		sig_ilap_A_ia->scale(exp(omega * xpt));
		sig_ilap_B_ia->scale(exp(omega * xpt));
		
#ifdef _DLOG
		dbcsr::print(*sig_ilap_A_ia);
		dbcsr::print(*sig_ilap_B_ia);
#endif
		
		sigma_2e_A->add(1.0, 1.0, *sig_ilap_A_ia);
		sigma_2e_B->add(1.0, 1.0, *sig_ilap_B_ia);
		
	} // end loop over laplace points
		
	sigma_2e_A->scale(-pow(m_c_os_coupling, 2));
	sigma_2e_B->scale(pow(m_c_os_coupling, 2));

#ifdef _DLOG
	std::cout << "SIGMA E A" << std::endl;
	dbcsr::print(*sigma_2e_A);
	
	std::cout << "SIGMA E B" << std::endl;
	dbcsr::print(*sigma_2e_B);
#endif

	sigma_2e_A->setname("sigma_2e");
	sigma_2e_A->add(1.0, 1.0, *sigma_2e_B);
	
	time_2e.finish();
	
	return sigma_2e_A;
	
}

#else

std::tuple<dbcsr::sbtensor<3,double>,dbcsr::sbtensor<3,double>>
	MVP_AOADC2::compute_laplace_batchtensors(smat& u_ao, smat& L_bo, smat& L_bv)
{
	
	LOG.os<1>("Compute J and Teri\n");
	
	auto& time = TIME.sub("Laplace batch tensors");
	auto& time_setup = time.sub("Setup");
	auto& time_contr = time.sub("Contraction");
	
	time.start();
	time_setup.start();
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	auto o_chol = L_bo->col_blk_sizes();
	auto v_chol = L_bv->col_blk_sizes();
	
	arrvec<int,2> bo_chol = {b,o_chol};
	arrvec<int,2> ob_chol = {o_chol,b};
	arrvec<int,2> bv_chol = {b,v_chol};
	arrvec<int,2> bb = {b,b};
	arrvec<int,3> xbb = {x,b,b};
	arrvec<int,3> xob_chol = {x,o_chol,b};
	arrvec<int,3> xov_chol = {x,o_chol,v_chol};
	
	int nxbatches = m_eri3c2e_batched->nbatches(0);
	int nbbatches = m_eri3c2e_batched->nbatches(2);
	
	std::array<int,3> bdims = {nxbatches,nbbatches,nbbatches};
		
	auto blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
	auto blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
	
	vec<int> blkmap_o(o_chol.size()), blkmap_v(v_chol.size());
	std::iota(blkmap_o.begin(), blkmap_o.end(), 0);
	std::iota(blkmap_v.begin(), blkmap_v.end(), 0);
		
	arrvec<int,3> blkmaps = {blkmap_x, blkmap_o, blkmap_v};
	
	int nxbas = m_mol->c_dfbasis()->nbf();
	int nbas = m_mol->c_basis()->nbf();
	int no_chol = L_bo->nfullcols_total();
	int nv_chol = L_bv->nfullcols_total();
	
	auto spgrid2 = dbcsr::create_pgrid<2>(m_world.comm())
		.get();
	
	std::array<int,3> dims_xov_chol = {nxbas,no_chol,nv_chol};
	std::array<int,3> dims_xob_chol = {nxbas,no_chol,nbas};
	
	auto spgrid_xov_chol = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(dims_xov_chol)
		.get();
		
	auto spgrid_xob_chol = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(dims_xob_chol)
		.get();
	
	auto J_xov_batched = dbcsr::btensor_create<3>()
		.name("J_xov_batched")
		.pgrid(spgrid_xov_chol)
		.blk_sizes(xov_chol)
		.blk_map(blkmaps)
		.batch_dims(bdims)
		.btensor_type(dbcsr::btype::core)
		.print(LOG.global_plev())
		.get();
			
	auto eri_xov_batched = dbcsr::btensor_create<3>()
		.name("eri_xov_batched")
		.pgrid(spgrid_xov_chol)
		.blk_sizes(xov_chol)
		.blk_map(blkmaps)
		.batch_dims(bdims)
		.btensor_type(dbcsr::btype::core)
		.print(LOG.global_plev())
		.get();
		
	auto L_bo_01 = dbcsr::tensor_create<2>()
		.name("L_bo_01")
		.pgrid(spgrid2)
		.blk_sizes(bo_chol)
		.map1({0})
		.map2({1})
		.get();
		
	auto L_bv_01 = dbcsr::tensor_create<2>()
		.name("L_bv_01")
		.pgrid(spgrid2)
		.blk_sizes(bv_chol)
		.map1({0})
		.map2({1})
		.get();
		
	auto HT_xob_02_1 = dbcsr::tensor_create<3,double>()
		.name("HT_02_1")
		.pgrid(spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,2})
		.map2({1})
		.get();
		
	auto HT_xob_01_2 = dbcsr::tensor_create<3,double>()
		.name("HT_01_2")
		.pgrid(spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,1})
		.map2({2})
		.get();
		
	auto FT_xov_01_2 = dbcsr::tensor_create<3,double>()
		.name("FT_01_2")
		.pgrid(spgrid_xov_chol)
		.blk_sizes(xov_chol)
		.map1({0,1})
		.map2({2})
		.get();
		
	auto FT_xov_0_12 = dbcsr::tensor_create<3,double>()
		.name("FT_0_12")
		.pgrid(spgrid_xov_chol)
		.blk_sizes(xov_chol)
		.map1({0})
		.map2({1,2})
		.get();
		
	auto J_xov_0_12 = dbcsr::tensor_create<3,double>()
		.name("J_0_12")
		.pgrid(spgrid_xov_chol)
		.blk_sizes(xov_chol)
		.map1({0})
		.map2({1,2})
		.get();
	
	dbcsr::copy_matrix_to_tensor(*L_bo, *L_bo_01);
	dbcsr::copy_matrix_to_tensor(*L_bv, *L_bv_01);
	
	J_xov_batched->compress_init({0}, vec<int>{0}, vec<int>{1,2});
	eri_xov_batched->compress_init({0}, vec<int>{0}, vec<int>{1,2});
	m_eri3c2e_batched->decompress_init({0}, vec<int>{0,2}, vec<int>{1});
	
	// Form transformed u vectors
	
	auto SL_bo = dbcsr::create_template(*L_bo)
		.name("SL_bo")
		.get();
		
	auto SL_bv = dbcsr::create_template(*L_bv)
		.name("SL_bv")
		.get();
	
	auto up_ob = dbcsr::create<double>()
		.set_world(m_world)
		.name("u particle")
		.row_blk_sizes(o_chol)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto uh_bv = dbcsr::create<double>()
		.set_world(m_world)
		.name("u hole")
		.row_blk_sizes(b)
		.col_blk_sizes(v_chol)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	dbcsr::multiply('N', 'N', *m_s_bb, *L_bo, *SL_bo)
		.perform();
		
	dbcsr::multiply('N', 'N', *m_s_bb, *L_bv, *SL_bv)
		.perform();
		
	dbcsr::multiply('T', 'N', *SL_bo, *u_ao, *up_ob)
		.perform();
		
	dbcsr::multiply('N', 'N', *u_ao, *SL_bv, *uh_bv)
		.perform();
		
	SL_bo->release();
	SL_bv->release();
	
	auto up_ob_01 = dbcsr::tensor_create<2>()
		.name("u particle 01")
		.pgrid(spgrid2)
		.blk_sizes(ob_chol)
		.map1({0})
		.map2({1})
		.get();
		
	auto uh_bv_01 = dbcsr::tensor_create<2>()
		.name("u hole 01")
		.pgrid(spgrid2)
		.blk_sizes(bv_chol)
		.map1({0})
		.map2({1})
		.get();
	
	dbcsr::copy_matrix_to_tensor(*up_ob, *up_ob_01);
	dbcsr::copy_matrix_to_tensor(*uh_bv, *uh_bv_01);
	
	up_ob_01->filter(dbcsr::global::filter_eps);
	uh_bv_01->filter(dbcsr::global::filter_eps);
	
	up_ob->release();
	uh_bv->release();
	
	time_setup.finish();
	time_contr.start();
	
	for (int ix = 0; ix != nxbatches; ++ix) {
		
		m_eri3c2e_batched->decompress({ix});
		auto eri_02_1 = m_eri3c2e_batched->get_work_tensor();
		
		vec<vec<int>> xn_bounds = {
			m_eri3c2e_batched->bounds(0,ix),
			m_eri3c2e_batched->full_bounds(2)
		};
		
		LOG.os<>("Contraction 1\n");
		dbcsr::contract(*eri_02_1, *L_bo_01, *HT_xob_02_1)
			.bounds2(xn_bounds)
			.filter(dbcsr::global::filter_eps)
			.perform("Xmn, mi -> Xin");
		
		dbcsr::copy(*HT_xob_02_1, *HT_xob_01_2)
			.move_data(true)
			.perform();
		
		vec<vec<int>> xo_bounds = {
			m_eri3c2e_batched->bounds(0,ix),
			eri_xov_batched->full_bounds(1)
		};
		
		LOG.os<>("Contraction 2\n");
		dbcsr::contract(*HT_xob_01_2, *L_bv_01, *FT_xov_01_2)
			.bounds2(xo_bounds)
			.filter(dbcsr::global::filter_eps)
			.perform("Xin, na -> Xia");
			
		dbcsr::copy(*FT_xov_01_2, *FT_xov_0_12)
			.move_data(true)
			.perform();
			
		eri_xov_batched->compress({ix}, FT_xov_0_12);
		
		LOG.os<>("Contraction 3\n");
		dbcsr::contract(*HT_xob_01_2, *uh_bv_01, *FT_xov_01_2)
			.bounds2(xo_bounds)
			.filter(dbcsr::global::filter_eps)
			.alpha(1.0)
			.perform("Xim, ma -> Xia");
		
		HT_xob_01_2->clear();
		
		dbcsr::copy(*FT_xov_01_2, *J_xov_0_12)
			.move_data(true)
			.perform();
		
		LOG.os<>("Contraction 4\n");
		dbcsr:contract(*eri_02_1, *up_ob_01, *HT_xob_02_1)
			.bounds2(xn_bounds)
			.filter(dbcsr::global::filter_eps)
			.perform("Xmn, im -> Xin");
			
		dbcsr::copy(*HT_xob_02_1, *HT_xob_01_2)
			.move_data(true)
			.perform();
		
		LOG.os<>("Contraction 5\n");
		dbcsr::contract(*HT_xob_01_2, *L_bv_01, *FT_xov_01_2)
			.alpha(-1.0)
			.filter(dbcsr::global::filter_eps)
			.bounds2(xo_bounds)
			.perform("Xin, na -> Xia");
		
		HT_xob_01_2->clear();
			
		dbcsr::copy(*FT_xov_01_2, *J_xov_0_12)
			.sum(true)
			.move_data(true)
			.perform();
			
		J_xov_batched->compress({ix}, J_xov_0_12);
		FT_xov_01_2->clear();
		
	}
			
	J_xov_batched->compress_finalize();
	eri_xov_batched->compress_finalize();
	m_eri3c2e_batched->decompress_finalize();
	
	time_contr.finish();
	
	LOG.os<>("Occupancies: ", J_xov_batched->occupation(), ", ", 
		eri_xov_batched->occupation(), '\n');
	
	//MPI_Barrier(m_world.comm());
	//exit(0);
		
	time.finish();	
		
	return std::make_tuple(eri_xov_batched, J_xov_batched);
	
}

std::tuple<dbcsr::shared_tensor<2,double>,dbcsr::shared_tensor<2,double>>
	MVP_AOADC2::compute_F(dbcsr::sbtensor<3,double> eri_xov_batched,
	dbcsr::sbtensor<3,double> J_xov_batched)
{
	
	LOG.os<1>("Compute F\n");
	
	auto& time = TIME.sub("Computing F matrices");
	auto& time_setup = time.sub("Setup");
	auto& time_contr = time.sub("Contraction");

	time.start();
	time_setup.start();
	
	auto x = m_mol->dims().x();
	
	arrvec<int,2> xx = {x,x};
	
	auto V_xx_01 = dbcsr::tensor_create<2,double>()
		.name("v_xx_01")
		.pgrid(m_spgrid2)
		.blk_sizes(xx)
		.map1({0})
		.map2({1})
		.get();
		
	auto FA_xx_01 = dbcsr::tensor_create_template<2,double>(
		V_xx_01)
		.name("FA_xx_01")
		.get();
		
	auto FB_xx_01 = dbcsr::tensor_create_template<2,double>(
		V_xx_01)
		.name("FB_xx_01")
		.get();
	
	dbcsr::copy_matrix_to_tensor(*m_v_xx, *V_xx_01);
	
	int nvbatches = eri_xov_batched->nbatches(2);
	
	J_xov_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	eri_xov_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	
	time_setup.finish();
	time_contr.start();
	
	for (int iv = 0; iv != nvbatches; ++iv) {
		
		eri_xov_batched->decompress({iv});
		J_xov_batched->decompress({iv});
		
		vec<vec<int>> ov_bounds = {
			eri_xov_batched->full_bounds(1),
			eri_xov_batched->bounds(2,iv)
		};
		
		auto eri_0_12 = eri_xov_batched->get_work_tensor();
		auto J_0_12 = J_xov_batched->get_work_tensor();
		
		dbcsr::contract(*eri_0_12, *eri_0_12, *FA_xx_01)
			.bounds1(ov_bounds)
			.beta(1.0)
			.perform("Xia, Yia -> XY");
			
		dbcsr::contract(*J_0_12, *eri_0_12, *FB_xx_01)
			.bounds1(ov_bounds)
			.beta(1.0)
			.perform("Xia, Yia -> XY");
	
	}
	
#ifdef _DLOG
	dbcsr::print(*FA_xx_01);
	dbcsr::print(*FB_xx_01);
#endif
	
	auto temp = dbcsr::tensor_create_template<2,double>(
		V_xx_01)
		.name("temp")
		.get();
	
	auto transform = [&](auto& F) {
		
		dbcsr::contract(*V_xx_01, *F, *temp)
			.perform("XY, YZ -> XZ");
		
		dbcsr::contract(*temp, *V_xx_01, *F)
			.perform("XY, YZ -> XZ");
			
	};
	
	transform(FA_xx_01);
	transform(FB_xx_01);
	
	time_contr.finish();
	time.finish();
	
	return std::make_tuple(FA_xx_01, FB_xx_01);
	
}

void MVP_AOADC2::compute_I(dbcsr::sbtensor<3,double>& eri_xov_batched,
	dbcsr::sbtensor<3,double>& R_xov_batched, dbcsr::shared_tensor<2,double>& F_A,
	dbcsr::shared_tensor<2,double>& F_B, 
	dbcsr::sbtensor<3,double>& I_xov_batched)
{

	auto& time = TIME.sub("Computing I batched");
	time.start();

	LOG.os<>("Compute I\n");

	I_xov_batched->compress_init({2}, vec<int>{0}, vec<int>{1,2});
	R_xov_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	eri_xov_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	
	auto I_xov_0_12 = I_xov_batched->get_template("I_xbb_0_12", vec<int>{0},
		vec<int>{1,2});
	
	int nxbatches = eri_xov_batched->nbatches(0);
	int nvbatches = eri_xov_batched->nbatches(2);
	auto fullobds = eri_xov_batched->full_bounds(1);
	
	for (int iv = 0; iv != nvbatches; ++iv) {
		
		R_xov_batched->decompress({iv});
		auto R_0_12 = R_xov_batched->get_work_tensor();
		
		eri_xov_batched->decompress({iv});
		auto eri_0_12 = eri_xov_batched->get_work_tensor();
		
		auto vbds = eri_xov_batched->bounds(2,iv);
		
		for (int ix = 0; ix != nxbatches; ++ix) {		
			
			auto xbds = eri_xov_batched->bounds(0,ix);
			
			vec<vec<int>> xbounds = { 
				xbds
			};
			
			vec<vec<int>> kabounds = { 
				fullobds,
				vbds
			};
			
			dbcsr::contract(*F_A, *R_0_12, *I_xov_0_12)
				.bounds2(xbounds)
				.bounds3(kabounds)
				.filter(dbcsr::global::filter_eps)
				.beta(1.0)
				.perform("XY, Yka -> Xka");
				
			dbcsr::contract(*F_B, *eri_0_12, *I_xov_0_12)
				.bounds2(xbounds)
				.bounds3(kabounds)
				.filter(dbcsr::global::filter_eps)
				.beta(1.0)
				.perform("YX, Yka -> Xka");
								
		}
		
		I_xov_batched->compress({iv}, I_xov_0_12);
		
	}
	
	I_xov_batched->compress_finalize();
	R_xov_batched->decompress_finalize();
	eri_xov_batched->decompress_finalize();
	
	time.finish();
	
	return;
	
}

std::tuple<smat,smat> MVP_AOADC2::compute_sigma_ilap(
	dbcsr::sbtensor<3,double>& I_xov_batched, smat& L_bo, smat& L_bv,
	double omega)
{
	
	/*
	 * sigE1_(b,v_c)[ma] = I_(x,o_c,v_c)[Yka] * eri_(x,o_c,b)[Ykm]
	 * sigE2_(o_c,b)[im] = I_(x,o_c,v_c)[Xic] * eri_(x,v_c,b)[Ycm]
	 * 
	 * sigE1_bb = Po_bb * sigE1_bv * L_bv^t
	 * sigE2_bb = L_bo * sigE2_ob * Pv_bb
	 * 
	 * sigE_ia = C'_bo^t sigE C'_bv 
	 * 
	 */
	 
	auto& time = TIME.sub("Computing sigma ilap");
	auto& time_A = time.sub("Part A");
	auto& time_B = time.sub("Part B");
	
	time.start();
	
	auto o_chol = L_bo->col_blk_sizes();
	auto v_chol = L_bv->col_blk_sizes();
	auto x = m_mol->dims().x();
	auto b = m_mol->dims().b();
	
	arrvec<int,2> bo_chol = {b,o_chol};
	arrvec<int,2> bv_chol = {b,v_chol};
	arrvec<int,2> ob_chol = {o_chol,b};
	arrvec<int,3> xob_chol = {x,o_chol,b};
	arrvec<int,3> xvb_chol = {x,v_chol,b};
	
	auto L_bo_01 = dbcsr::tensor_create<2,double>()
		.name("L_bo_01")
		.pgrid(m_spgrid2)
		.blk_sizes(bo_chol)
		.map1({0})
		.map2({1})
		.get();
		
	auto L_bv_01 = dbcsr::tensor_create<2,double>()
		.name("L_bv_01")
		.pgrid(m_spgrid2)
		.blk_sizes(bv_chol)
		.map1({0})
		.map2({1})
		.get();
			
	dbcsr::copy_matrix_to_tensor(*L_bo, *L_bo_01);
	dbcsr::copy_matrix_to_tensor(*L_bv, *L_bv_01);
	
	int nxbas = m_mol->c_dfbasis()->nbf();
	int nbas = m_mol->c_basis()->nbf();
	int no_chol = L_bo->nfullcols_total();
	int nv_chol = L_bv->nfullcols_total();
	
	auto spgrid2 = dbcsr::create_pgrid<2>(m_world.comm())
		.get();
	
	std::array<int,3> dims_xob_chol = {nxbas,no_chol,nbas};
	std::array<int,3> dims_xvb_chol = {nxbas,nv_chol,nbas};
	
	auto spgrid_xov_chol = I_xov_batched->spgrid();
		
	auto spgrid_xob_chol = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(dims_xob_chol)
		.get();
		
	auto spgrid_xvb_chol = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(dims_xvb_chol)
		.get();
	
	int nxbatches = I_xov_batched->nbatches(0);
	int nobatches = I_xov_batched->nbatches(1);
	int nvbatches = I_xov_batched->nbatches(2);
	
	auto sigmaE1_HT_01 = dbcsr::tensor_create<2,double>()
		.name("sigmaE1_HT")
		.pgrid(m_spgrid2)
		.blk_sizes(bv_chol)
		.map1({0}).map2({1})
		.get();
		
	auto sigmaE2_HT_01 = dbcsr::tensor_create<2,double>()
		.name("sigmaE2_HT")
		.pgrid(m_spgrid2)
		.blk_sizes(ob_chol)
		.map1({0}).map2({1})
		.get();
	
	auto HT_xob_1_02 = dbcsr::tensor_create<3,double>()
		.name("HT_xob_1_02")
		.pgrid(spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({1}).map2({0,2})
		.get();
		
	auto HT_xob_01_2 = dbcsr::tensor_create<3,double>()
		.name("HT_xob_1_02")
		.pgrid(spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,1}).map2({2})
		.get();
		
	auto HT_xvb_1_02 = dbcsr::tensor_create<3,double>()
		.name("HT_xob_1_02")
		.pgrid(spgrid_xvb_chol)
		.blk_sizes(xvb_chol)
		.map1({1}).map2({0,2})
		.get();
		
	auto HT_xvb_01_2 = dbcsr::tensor_create<3,double>()
		.name("HT_xob_1_02")
		.pgrid(spgrid_xvb_chol)
		.blk_sizes(xvb_chol)
		.map1({0,1}).map2({2})
		.get();
	
	m_eri3c2e_batched->decompress_init({0}, vec<int>{1}, vec<int>{0,2});
	I_xov_batched->decompress_init({0}, vec<int>{0,1}, vec<int>{2});
	
	time_A.start();
	
	for (int ix = 0; ix != nxbatches; ++ix) {
			
			m_eri3c2e_batched->decompress({ix});
			auto eri_1_02 = m_eri3c2e_batched->get_work_tensor();
			
			I_xov_batched->decompress({ix});
			auto I_xov_01_2 = I_xov_batched->get_work_tensor();
			
			for (int io = 0; io != nobatches; ++io) {
				
				vec<vec<int>> xmbounds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->full_bounds(1)
				};
				vec<vec<int>> obounds = {
					I_xov_batched->bounds(1,io)
				};
				
				// form cbar
				dbcsr::contract(*eri_1_02, *L_bo_01, *HT_xob_1_02)
					.bounds2(xmbounds)
					.bounds3(obounds)
					//.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps/nxbatches)
					.perform("Xlm, li -> Xim");
										
				dbcsr::copy(*HT_xob_1_02, *HT_xob_01_2).move_data(true).perform();
				
				vec<vec<int>> xobounds = {
					m_eri3c2e_batched->bounds(0,ix),
					I_xov_batched->bounds(1,io)
				};
				
				// form sig_A
				dbcsr::contract(*I_xov_01_2, *HT_xob_01_2, *sigmaE1_HT_01)
					.bounds1(xobounds)
					//.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps/nxbatches)
					.beta(1.0)
					.perform("Xia, Xim -> ma");
				
				HT_xob_01_2->clear();
								
			}
			
		}
		
		//dbcsr::copy_tensor_to_matrix(*sigma_ilap_01, *sigma_ilap_A);
		
		//sigma_ilap_01->clear();
		
		m_eri3c2e_batched->decompress_finalize();
		I_xov_batched->decompress_finalize();
						
		m_eri3c2e_batched->decompress_init({0},vec<int>{1},vec<int>{0,2});
		I_xov_batched->decompress_init({0},vec<int>{0,2},vec<int>{1});
		
		time_A.finish();
		time_B.start();
		
		for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
			
			m_eri3c2e_batched->decompress({ix});
			auto eri_1_02 = m_eri3c2e_batched->get_work_tensor();
			
			I_xov_batched->decompress({ix});
			auto I_xov_02_1 = I_xov_batched->get_work_tensor();
			
			for (int iv = 0; iv != nvbatches; ++iv) {
				
				vec<vec<int>> xmbounds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->full_bounds(1)
				};
				vec<vec<int>> vbounds = {
					I_xov_batched->bounds(2,iv)
				};
				
				// form cbar
				dbcsr::contract(*eri_1_02, *L_bv_01, *HT_xvb_1_02)
					.bounds2(xmbounds)
					.bounds3(vbounds)
					//.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps/nxbatches)
					.perform("Xlm, la -> Xam");
										
				dbcsr::copy(*HT_xvb_1_02, *HT_xvb_01_2).move_data(true).perform();
				
				vec<vec<int>> xvbounds = {
					m_eri3c2e_batched->bounds(0,ix),
					I_xov_batched->bounds(2,iv)
				};
				
				// form sig_A
				dbcsr::contract(*I_xov_02_1, *HT_xvb_01_2, *sigmaE2_HT_01)
					.bounds1(xvbounds)
					//.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps 
						/ m_eri3c2e_batched->nbatches(0))
					.beta(1.0)
					.perform("Xia, Xam -> im");
				
				HT_xvb_01_2->clear();
								
			}
			
		}
		
		m_eri3c2e_batched->decompress_finalize();
		I_xov_batched->decompress_finalize();
		
		time_B.finish();
	
		auto o = m_mol->dims().oa();
		auto v = m_mol->dims().va();
		
		auto sigmaE1_HT = dbcsr::create<double>()
			.name("sigmaE1_HT")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(v_chol)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
			
		auto sigmaE2_HT = dbcsr::create<double>()
			.name("sigmaE2_HT")
			.set_world(m_world)
			.row_blk_sizes(o_chol)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
			
		dbcsr::copy_tensor_to_matrix(*sigmaE1_HT_01, *sigmaE1_HT);
		dbcsr::copy_tensor_to_matrix(*sigmaE2_HT_01, *sigmaE2_HT);
		
		sigmaE1_HT_01->destroy();
		sigmaE2_HT_01->destroy();
			
		auto Po_bb = dbcsr::create<double>()
			.name("Po_bb")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::symmetric)
			.get();
			
		auto Pv_bb = dbcsr::create<double>()
			.name("Pv_bb")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::symmetric)
			.get();
			
		dbcsr::multiply('N', 'T', *m_c_bo, *m_c_bo, *Po_bb)
			.perform();
			
		dbcsr::multiply('N', 'T', *m_c_bv, *m_c_bv, *Pv_bb)
			.perform();
			
		auto SC_bo = dbcsr::create_template<double>(*m_c_bo)
			.name("SC_bo")
			.get();
			
		auto SC_bv = dbcsr::create_template<double>(*m_c_bv)
			.name("SC_bo")
			.get();
		
		dbcsr::multiply('N', 'N', *m_s_bb, *m_c_bo, *SC_bo)
			.perform();
			
		dbcsr::multiply('N', 'N', *m_s_bb, *m_c_bv, *SC_bv)
			.perform();
			
		auto sigmaE1_bb = u_transform(sigmaE1_HT, 'N', Po_bb, 'T', L_bv);
		auto sigmaE2_bb = u_transform(sigmaE2_HT, 'N', L_bo, 'N', Pv_bb);	
		
		auto sigmaE1_ia = u_transform(sigmaE1_bb, 'T', SC_bo, 'N', SC_bv);
		auto sigmaE2_ia = u_transform(sigmaE2_bb, 'T', SC_bo, 'N', SC_bv);
	
		sigmaE1_ia->setname("sigma_E1");
		sigmaE2_ia->setname("sigma_E2");
		
		time.finish();
	
		return std::make_tuple(sigmaE1_ia, sigmaE2_ia);
	
} 

smat MVP_AOADC2::compute_sigma_2e(smat& u_ao, double omega) {
	
	/* IN AO:
	 * sig_e2 = - c_os_c ^2 [ sum_t,2>
	 * 	exp(omega t) C_μi C_αa * exp(-ε_a t) * I_{Xκα}(t) * (X|μκ') Po(t)_κκ'
	 * + exp(omega t) C_μi C_αa * exp(ε_i t) * I_{Xμγ}(t) * (X|γ'α) Pv(t)_γγ']
	 * 
	 * with
	 * 
	 * I_{Xκα}(t) = R_{Yκα} * F_{YX}(t) + (Y|κα) Ftilde_{XY}(t)
	 * 
	 * Ftilde_{X'Y'}(t) = (X'|X) (X|μν) * R_{Xμ'ν'} * Po(t)_{μμ'} * Pv(t)_{νν'} (YY')
	 * 
	 * R_{Χκα} = [ wv_κα (X|μκ) - wo_μγ (X|γα)]
	 * wv_κα = co_κi * S_αα' cv_α'a u_ia
	 * wo_μγ = S_μμ' * co_μ'i * cv_γa * u_ia
	 */
	
	LOG.os<1>("==== Computing ADC(2) SIGMA 2E ====\n");
	
	auto& time_2e = TIME.sub("Computing sigma(2e)");
	time_2e.start();
	
	double emin = m_eps_occ->front();
	double ehomo = m_eps_occ->back();
	double elumo = m_eps_vir->front();
	double emax = m_eps_vir->back();
	
	double ymin = 2*(elumo - ehomo) + omega;
	double ymax = 2*(emax - emin) + omega;
	
	LOG.os<1>("eps_min/eps_homo/eps_lumo/eps_max ", emin, " ", ehomo, " ", elumo, " ", emax, '\n');
	LOG.os<1>("ymin/ymax ", ymin, " ", ymax, '\n');
	
	math::laplace lp_dd(m_world.comm(), LOG.global_plev());
	
	lp_dd.compute(m_nlap, ymin, ymax);
		
	auto weights_dd = lp_dd.omega();
	auto xpoints_dd = lp_dd.alpha();
	
	auto o = m_mol->dims().oa();
	auto v = m_mol->dims().va();
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	int no = m_mol->nocc_alpha();
	int nv = m_mol->nvir_alpha();
	int nbas = m_mol->c_basis()->nbf();
	int nx = m_mol->c_dfbasis()->nbf();
	
	std::array<int,3> dims_xov = {nx,no,nv};
	arrvec<int,3> xov = {x,o,v};
	
	auto spgrid3_xov = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(dims_xov)
		.get();
		
	auto blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
	auto blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
	
	int nxbatches = m_eri3c2e_batched->nbatches(0);
	int nbbatches = m_eri3c2e_batched->nbatches(2);
 	
	auto sigma_2e_A = dbcsr::create<double>()
		.name("sigma_2e_A")
		.set_world(m_world)
		.row_blk_sizes(o)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto sigma_2e_B = dbcsr::create_template<double>(*sigma_2e_A)
		.name("sigma_2e_B").get();

	// Loop
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		double wght_dd = weights_dd[ilap];
		double xpt_dd = xpoints_dd[ilap];
		
		auto& time_chol = TIME.sub("Cholesky in sigma E");
		time_chol.start();
		
		auto L_bo = get_ortho_cholesky('O', wght_dd, xpt_dd, 0.125, 0.5);
		auto L_bv = get_ortho_cholesky('V', wght_dd, xpt_dd, 0.125, 0.5);
		
		std::string file_o = std::string(std::filesystem::current_path()) + 
			"/chol_occ_" + std::to_string(ilap);
		std::string file_v = std::string(std::filesystem::current_path()) + 
			"/chol_vir_" + std::to_string(ilap);
		
		//util::plot(L_bo, 1e-5, file_o);
		//util::plot(L_bv, 1e-5, file_v);
		
		time_chol.finish();
		
		auto o_chol = L_bo->col_blk_sizes();
		auto v_chol = L_bv->col_blk_sizes();
		
		vec<int> blkmap_o(o_chol.size()), blkmap_v(v_chol.size());
		std::iota(blkmap_o.begin(), blkmap_o.end(), 0);
		std::iota(blkmap_v.begin(), blkmap_v.end(), 0);
		
		std::array<int,3> bdims = {nxbatches,nbbatches,nbbatches};
		arrvec<int,3> blkmaps = {blkmap_x, blkmap_o, blkmap_v};
		arrvec<int,3> xov_chol = {x, o_chol, v_chol};
		
		auto I_xov_batched = dbcsr::btensor_create<3>()
			.name("I_xov_batched")
			.pgrid(spgrid3_xov)
			.blk_sizes(xov_chol)
			.blk_map(blkmaps)
			.batch_dims(bdims)
			.btensor_type(dbcsr::btype::core)
			.print(LOG.global_plev())
			.get();
		
		//auto pseudo_o = get_density(c_bo_scaled);
		//auto pseudo_v = get_density(c_bv_scaled);
		
		//pseudo_o->scale(pow(wght_dd,0.25));
		//pseudo_v->scale(pow(wght_dd,0.25));
		
		L_bo->filter(dbcsr::global::filter_eps);
		L_bv->filter(dbcsr::global::filter_eps);
		
		auto [eri_xov_batched, J_xov_batched] 
			= compute_laplace_batchtensors(u_ao, L_bo, L_bv);
			
		// Form F matrices
		auto [FA_xx, FB_xx] = compute_F(eri_xov_batched, J_xov_batched);
		
		compute_I(eri_xov_batched, J_xov_batched, FA_xx, FB_xx, I_xov_batched);
	
		J_xov_batched->reset();
		eri_xov_batched->reset();
	
		auto [sig_ilap_E1, sig_ilap_E2] = compute_sigma_ilap(I_xov_batched, 
			L_bo, L_bv, omega);
		
		double xpt = xpoints_dd[ilap];
		
		sig_ilap_E1->scale(exp(omega * xpt));
		sig_ilap_E2->scale(exp(omega * xpt));

#ifdef _DLOG
		dbcsr::print(*sig_ilap_E1);
		dbcsr::print(*sig_ilap_E2);
#endif
		
		sigma_2e_A->add(1.0, 1.0, *sig_ilap_E1);
		sigma_2e_B->add(1.0, 1.0, *sig_ilap_E2);
		
	} // end loop over laplace points
		
	sigma_2e_A->scale(-pow(m_c_os_coupling, 2));
	sigma_2e_B->scale(pow(m_c_os_coupling, 2));

#ifdef _DLOG
	std::cout << "SIGMA E A" << std::endl;
	dbcsr::print(*sigma_2e_A);
	
	std::cout << "SIGMA E B" << std::endl;
	dbcsr::print(*sigma_2e_B);
#endif

	sigma_2e_A->setname("sigma_2e");
	sigma_2e_A->add(1.0, 1.0, *sigma_2e_B);
	
	time_2e.finish();
	
	return sigma_2e_A;
	
}
#endif

/* =====================================================================
 *                         ADC(2) SIGMA CONSTRUCTOR
 * ====================================================================*/

smat MVP_AOADC2::compute(smat u_ia, double omega) {
	
	auto& time_com = TIME.sub("Computing sigma ADC(2)");
	
	TIME.start();
	time_com.start();
	
	LOG.os<1>("Computing AO-ADC(2) MVP product... \n");
	LOG.os<1>("Computing sigma_0 of AO-ADC(2) ... \n");
	
	auto sigma_0 = compute_sigma_0(u_ia, *m_eps_occ, *m_eps_vir);
	
#ifdef _DLOG
	LOG.os<>("SIGMA 0");
	dbcsr::print(*sigma_0);
#endif

	LOG.os<1>("Computing sigma_1 of AO-ADC(2) ... \n");

	auto u_ao = u_transform(u_ia, 'N', m_c_bo, 'T', m_c_bv); 
	auto jkpair = compute_jk(u_ao);
	auto sigma_1 = compute_sigma_1(jkpair.first, jkpair.second);
	
#ifdef _DLOG
	LOG.os<>("SIGMA 1");
	dbcsr::print(*sigma_1);
#endif

	LOG.os<1>("Computing sigma_2 of AO-ADC(2) ... \n");
	
	auto sigma_2a = compute_sigma_2a(u_ia);

	auto sigma_2b = compute_sigma_2b(u_ia);
	
	auto sigma_2c = compute_sigma_2c(jkpair.first, jkpair.second);
	
	auto sigma_2d = compute_sigma_2d(u_ia);
	
	auto sigma_2e = compute_sigma_2e(u_ao, omega);
	
#ifdef _DLOG
	LOG.os<>("SIGMA 2 A");
	dbcsr::print(*sigma_2a);
	
	LOG.os<>("SIGMA 2 B");
	dbcsr::print(*sigma_2b);
	
	LOG.os<>("SIGMA 2 C");
	dbcsr::print(*sigma_2c);
	
	LOG.os<>("SIGMA 2 D");
	dbcsr::print(*sigma_2d);
	
	LOG.os<>("SIGMA 2 E");
	dbcsr::print(*sigma_2e);
#endif
	
	sigma_0->add(1.0, 1.0, *sigma_1);
	sigma_0->add(1.0, 1.0, *sigma_2a);
	sigma_0->add(1.0, 1.0, *sigma_2b);
	sigma_0->add(1.0, 1.0, *sigma_2c);
	sigma_0->add(1.0, 1.0, *sigma_2d);
	sigma_0->add(1.0, 1.0, *sigma_2e);
	
	sigma_0->setname("sigma_2");
	
#ifdef _DLOG
	LOG.os<>("SIGMA 2 FULL");
	dbcsr::print(*sigma_0);
#endif

	time_com.finish();
	TIME.finish();
	
	TIME.print_info();
	exit(0);
	
	return sigma_0;
	
}

} // end namespace
