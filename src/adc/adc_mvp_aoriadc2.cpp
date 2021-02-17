#include "adc/adc_mvp.hpp"
#include "math/laplace/laplace.hpp"
#include "math/linalg/piv_cd.hpp"
#include "math/linalg/LLT.hpp"
#include "adc/adc_defaults.hpp"
#include "ints/fitting.hpp"
#include "utils/matrix_plot.hpp"

//#define _DLOG

namespace adc {

/* =====================================================================
 *                         INITIALIZING FUNCTIONS
 * ====================================================================*/
	
void MVP_AORISOSADC2::init() {
	
	LOG.os<1>("Initializing AO-ADC(2)\n");
	
	auto& t_init = TIME.sub("Initializing ADC(2)");
	
	TIME.start();
	t_init.start();

	LOG.os<1>("Computing Cholesky decomposition of S\n");
	math::LLT chol(m_s_bb, LOG.global_plev());
	
	chol.compute();
	auto b = m_mol->dims().b();
	
	m_s_sqrt_bb = chol.L(b);
	m_s_invsqrt_bb = chol.L_inv(b);
	
	// laplace
	LOG.os<1>("Computing laplace points for ground state densities.\n");
	
	auto& t_chol = TIME.sub("Computing cholesky decomposition (ss)");
	t_chol.start();
	
	m_laphelper_ss = std::make_shared<math::laplace_helper>(m_nlap, m_c_bo, 
		m_c_bv, m_s_sqrt_bb, m_s_invsqrt_bb, m_eps_occ, m_eps_vir, 
		LOG.global_plev());
		
	m_laphelper_dd = std::make_shared<math::laplace_helper>(m_nlap, m_c_bo, 
		m_c_bv, m_s_sqrt_bb, m_s_invsqrt_bb, m_eps_occ, m_eps_vir, 
		LOG.global_plev());
		
	m_laphelper_ss->compute(true, false, 0.0, m_mol->mo_split());
	
	t_chol.finish();
	
	LOG.os<1>("Setting up J,K,Z builders.\n");
	
	int nprint = LOG.global_plev();
	
	// J builder
	switch (m_jmethod) {
		case fock::jmethod::dfao:
		{
			m_jbuilder = fock::DF_J::create()
				.world(m_world)
				.molecule(m_mol)
				.print(nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.metric_inv(m_v_xx)
				.build();
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
			m_kbuilder = fock::DFAO_K::create()
				.world(m_world)
				.molecule(m_mol)
				.print(nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.fitting_batched(m_fitting_batched)
				.build();
			break;
		}
		case fock::kmethod::dfmem:
		{
			m_kbuilder = fock::DFMEM_K::create()
				.world(m_world)
				.molecule(m_mol)
				.print(nprint)
				.eri3c2e_batched(m_eri3c2e_batched)
				.metric_inv(m_v_xx)
				.build();
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
			m_zbuilder = mp::LLMP_FULL_Z::create()
				.world(m_world)
				.molecule(m_mol)
				.print(LOG.global_plev())
				.eri3c2e_batched(m_eri3c2e_batched)
				.intermeds(m_btype)
				.build();
			break;
		}
		case mp::zmethod::llmp_mem:
		{
			m_zbuilder = mp::LLMP_MEM_Z::create()
				.world(m_world)
				.molecule(m_mol)
				.print(LOG.global_plev())
				.eri3c2e_batched(m_eri3c2e_batched)
				.build();
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
	
	m_spgrid2 = dbcsr::pgrid<2>::create(m_world.comm()).build();
	
	t_init.finish();
	TIME.finish();
	
	LOG.os<1>("Done with setting up.\n");
	
}

/* =====================================================================
 *                          MVP FUNCTIONS (ADC1)
 * ====================================================================*/

// Computes the pseudo J and K matrices with the excited state density
std::pair<smat,smat> MVP_AORISOSADC2::compute_jk(smat& u_ao) {
	
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
smat MVP_AORISOSADC2::compute_sigma_1(smat& jmat, smat& kmat) {
	
	auto j = u_transform(jmat, 'T', m_c_bo, 'N', m_c_bv);
	auto k = u_transform(kmat, 'T', m_c_bo, 'N', m_c_bv);
	
	smat sig_ao = dbcsr::matrix<>::create_template(*jmat)
		.name("sig_ao")
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
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
 
void MVP_AORISOSADC2::compute_intermeds() {
	
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
	
	m_i_oo = dbcsr::matrix<>::create()
		.name("I_ij")
		.set_world(m_world)
		.row_blk_sizes(o)
		.col_blk_sizes(o)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	m_i_vv = dbcsr::matrix<>::create()
		.name("I_ab")
		.set_world(m_world)
		.row_blk_sizes(v)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto i_ob = dbcsr::matrix<>::create()
		.name("I_ij_part")
		.set_world(m_world)
		.row_blk_sizes(o)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto i_vb = dbcsr::matrix<>::create()
		.name("I_ab_part")
		.set_world(m_world)
		.row_blk_sizes(v)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto i_oo_tmp = dbcsr::matrix<>::create_template(*m_i_oo)
		.name("i_oo_temp").build();
		
	auto i_vv_tmp = dbcsr::matrix<>::create_template(*m_i_vv)
		.name("i_vv_temp").build();
	
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		LOG.os<1>("ADC(2) intermediates, laplace point nr. ", ilap, '\n');
		
		t_intermeds_1.start();
		
		auto po = m_laphelper_ss->pseudo_densities_occ()[ilap];
		auto pv = m_laphelper_ss->pseudo_densities_vir()[ilap];
		auto L_bu = m_laphelper_ss->pseudo_cholesky_occ()[ilap];
		
		L_bu->filter(dbcsr::global::filter_eps);
		po->filter(dbcsr::global::filter_eps);
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
				LOG.os<1>("Occupancy of Ifit: ", I_fit_xbb->occupation() * 100, '\n');
				k_inter = fock::DFAO_K::create()
					.world(m_world)
					.molecule(m_mol)
					.print(nprint)
					.eri3c2e_batched(m_eri3c2e_batched)
					.fitting_batched(I_fit_xbb)
					.build();
				break;
			}
			case fock::kmethod::dfmem:
			{
				k_inter = fock::DFMEM_K::create()
					.world(m_world)
					.molecule(m_mol)
					.print(nprint)
					.eri3c2e_batched(m_eri3c2e_batched)
					.metric_inv(f_xx_ilap)
					.build();
				break;
			}
		}
			
		k_inter->set_sym(false);
		k_inter->init();
		k_inter->set_density_alpha(pv);
		
		LOG.os<1>("Computing K_ilap.\n");
		
		k_inter->compute_K();
		auto ko_ilap = k_inter->get_K_A();
		
		auto c_bo_scaled = m_laphelper_ss->get_scaled_coeff('O', ilap, 0.25, 1.0);
		auto c_bv_scaled = m_laphelper_ss->get_scaled_coeff('V', ilap, 0.25, -1.0);
		
		LOG.os<1>("Forming partly-transformed intermediates.\n");
		
		dbcsr::multiply('T', 'N', 1.0, *c_bo_scaled, *ko_ilap, 1.0, *i_ob)
			.perform();
			
		k_inter->set_density_alpha(po);
		k_inter->compute_K();
		
		auto kv_ilap = k_inter->get_K_A();
			
		dbcsr::multiply('T', 'N', 1.0, *c_bv_scaled, *kv_ilap, 1.0, *i_vb)
			.perform();
			
		t_intermeds_2.finish();
								
	}
	
	LOG.os<1>("Forming fully transformed intermediates.\n");
	
	dbcsr::multiply('N', 'N', 1.0, *i_ob, *m_c_bo, 0.0, *m_i_oo)
			.perform();
	dbcsr::multiply('N', 'N', 1.0, *i_vb, *m_c_bv, 0.0, *m_i_vv)
			.perform();
			
	auto i_oo_tr = dbcsr::matrix<>::transpose(*m_i_oo).build();
	auto i_vv_tr = dbcsr::matrix<>::transpose(*m_i_vv).build();
	
	m_i_oo->add(1.0, 1.0, *i_oo_tr);
	m_i_vv->add(1.0, 1.0, *i_vv_tr);
	
	m_i_oo->scale(-0.5 * m_c_os);
	m_i_vv->scale(-0.5 * m_c_os);
	
	t_intermeds.finish();
			
}

/* =====================================================================
 *                         ADC(2) P-H CONTRIBUTIONS
 * ====================================================================*/

smat MVP_AORISOSADC2::compute_sigma_2a(smat& u_ia) {
	
	auto& t_sig = TIME.sub("Computing sigma (A)");
	t_sig.start();
	
	LOG.os<1>("==== Computing ADC(2) SIGMA 2A ====\n");
	
	// sig_2a = i_vv_ab * u_ib
	auto sig_2a = dbcsr::matrix<>::create_template(*u_ia)
		.name("sig_2a")
		.build();
		
	dbcsr::multiply('N', 'T', 1.0, *u_ia, *m_i_vv, 0.0, *sig_2a)
		.perform();
	
	t_sig.finish();
	
	return sig_2a;
	
}

smat MVP_AORISOSADC2::compute_sigma_2b(smat& u_ia) {
	
	auto& t_sig = TIME.sub("Computing sigma (B)");
	t_sig.start();
	
	LOG.os<1>("==== Computing ADC(2) SIGMA 2B ====\n");
	
	// sig_2b = i_oo_ij * u_ja
	auto sig_2b = dbcsr::matrix<>::create_template(*u_ia)
		.name("sig_2b")
		.build();
		
	dbcsr::multiply('N', 'N', 1.0, *m_i_oo, *u_ia, 0.0, *sig_2b)
		.perform();
	
	t_sig.finish();
	
	return sig_2b;
	
} 

smat MVP_AORISOSADC2::compute_sigma_2c(smat& jmat, smat& kmat) {
	
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
	 		
	auto I_ao = dbcsr::matrix<>::create_template(*jmat)
		.name("I_ao").build();
	auto jmat_t = dbcsr::matrix<>::transpose(*jmat)
		.build();
	auto kmat_t = dbcsr::matrix<>::transpose(*kmat)
		.build();
	
	I_ao->add(0.0, 1.0, *jmat_t);
	I_ao->add(1.0, 1.0, *kmat_t);
	
	jmat_t->release();
	kmat_t->release();
	
	auto o = m_mol->dims().oa();
	auto v = m_mol->dims().va();
	
	auto sig_2c = dbcsr::matrix<>::create()
		.set_world(m_world)
		.name("sig_2c")
		.row_blk_sizes(o)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	LOG.os<1>("-- Contracting over laplace points...\n");
	
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		auto pseudo_o = m_laphelper_ss->pseudo_densities_occ()[ilap];
		auto pseudo_v = m_laphelper_ss->pseudo_densities_vir()[ilap];
		
		auto I_pseudo = u_transform(I_ao, 'T', pseudo_o, 'N', pseudo_v);
		
		m_jbuilder->set_density_alpha(I_pseudo);		
		m_jbuilder->compute_J();
		
		auto jmat_pseudo = m_jbuilder->get_J();
		
		auto c_bo_scaled = m_laphelper_ss->get_scaled_coeff('O', ilap, 0.25, 1.0);
		auto c_bv_scaled = m_laphelper_ss->get_scaled_coeff('V', ilap, 0.25, -1.0);
		
		auto jmat_trans = u_transform(jmat_pseudo, 
			'T', c_bo_scaled, 'N', c_bv_scaled);
		
		sig_2c->add(1.0, 1.0, *jmat_trans);
		
	}
				
	sig_2c->scale(-0.25 * m_c_os);
	
	t_sig.finish();
	
	return sig_2c;
}

smat MVP_AORISOSADC2::compute_sigma_2d(smat& u_ia) {
	
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
	 
	auto I_ao = dbcsr::matrix<>::create_template(*m_s_bb)
		.name("I_ao")
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	 
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		auto c_bo_scaled = m_laphelper_ss->get_scaled_coeff('O', ilap, 0.25, 1.0);
		auto c_bv_scaled = m_laphelper_ss->get_scaled_coeff('V', ilap, 0.25, -1.0);
		
		auto u_pseudo = u_transform(u_ia, 'N', c_bo_scaled, 
			'T', c_bv_scaled);
		
		c_bo_scaled->release();
		c_bv_scaled->release();
		
		m_jbuilder->set_density_alpha(u_pseudo);
		m_jbuilder->compute_J();
		
		auto jpseudo1 = m_jbuilder->get_J();
		
		auto po = m_laphelper_ss->pseudo_densities_occ()[ilap];
		auto pv = m_laphelper_ss->pseudo_densities_vir()[ilap];;
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
	
	auto jmat_trans = dbcsr::matrix<>::transpose(*jmat).build();
	auto kmat_trans = dbcsr::matrix<>::transpose(*kmat).build();
	
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

std::tuple<dbcsr::sbtensor<3,double>,dbcsr::sbtensor<3,double>>
	MVP_AORISOSADC2::compute_laplace_batchtensors_OB(smat& u_ao, smat& L_bo, smat& pv_bb)
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
	
	arrvec<int,2> bo_chol = {b,o_chol};
	arrvec<int,2> ob_chol = {o_chol,b};
	arrvec<int,2> bb = {b,b};
	arrvec<int,3> xbb = {x,b,b};
	arrvec<int,3> xob_chol = {x,o_chol,b};
	
	int nxbatches = m_eri3c2e_batched->nbatches(0);
	int nbbatches = m_eri3c2e_batched->nbatches(2);
	
	std::array<int,3> bdims = {nxbatches,nbbatches,nbbatches};
		
	auto blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
	auto blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
	
	vec<int> blkmap_o(o_chol.size());
	std::iota(blkmap_o.begin(), blkmap_o.end(), 0);
		
	arrvec<int,3> blkmaps = {blkmap_x, blkmap_o, blkmap_b};
	
	int nxbas = m_mol->c_dfbasis()->nbf();
	int nbas = m_mol->c_basis()->nbf();
	int no_chol = L_bo->nfullcols_total();
	
	auto spgrid2 = dbcsr::pgrid<2>::create(m_world.comm())
		.build();
	
	std::array<int,3> dims_xob_chol = {nxbas,no_chol,nbas};
	
	auto spgrid_xob_chol = dbcsr::pgrid<3>::create(m_world.comm())
		.tensor_dims(dims_xob_chol)
		.build();
	
	auto J_xob_batched = dbcsr::btensor<3>::create()
		.name("J_xob_batched")
		.set_pgrid(spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.blk_maps(blkmaps)
		.batch_dims(bdims)
		.btensor_type(m_btype)
		.print(LOG.global_plev())
		.build();
			
	auto eri_xob_batched = dbcsr::btensor<3>::create()
		.name("eri_xov_batched")
		.set_pgrid(spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.blk_maps(blkmaps)
		.batch_dims(bdims)
		.btensor_type(m_btype)
		.print(LOG.global_plev())
		.build();
		
	auto L_bo_01 = dbcsr::tensor<2>::create()
		.name("L_bo_01")
		.set_pgrid(*spgrid2)
		.blk_sizes(bo_chol)
		.map1({0})
		.map2({1})
		.build();
		
	auto pv_bb_01 = dbcsr::tensor<2>::create()
		.name("pv_bb_01")
		.set_pgrid(*spgrid2)
		.blk_sizes(bb)
		.map1({0})
		.map2({1})
		.build();
		
	auto HT_xob_02_1 = dbcsr::tensor<3>::create()
		.name("HT_02_1")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,2})
		.map2({1})
		.build();
		
	auto HT_xob_01_2 = dbcsr::tensor<3>::create()
		.name("HT_01_2")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,1})
		.map2({2})
		.build();
		
	auto FT_xob_01_2 = dbcsr::tensor<3>::create()
		.name("FT_01_2")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,1})
		.map2({2})
		.build();
		
	auto FT_xob_02_1 = dbcsr::tensor<3>::create()
		.name("FT_1_02")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,2})
		.map2({1})
		.build();
		
	auto J_xob_02_1 = dbcsr::tensor<3>::create()
		.name("J_02_1")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,2})
		.map2({1})
		.build();
	
	dbcsr::copy_matrix_to_tensor(*L_bo, *L_bo_01);
	dbcsr::copy_matrix_to_tensor(*pv_bb, *pv_bb_01);
	
	// Form transformed u vectors
	
	auto SL_bo = dbcsr::matrix<>::create_template(*L_bo)
		.name("SL_bo")
		.build();
		
	auto SPv_bb = dbcsr::matrix<>::create_template(*pv_bb)
		.matrix_type(dbcsr::type::no_symmetry)
		.name("SPv_bb")
		.build();
	
	auto up_ob = dbcsr::matrix<>::create()
		.set_world(m_world)
		.name("u particle")
		.row_blk_sizes(o_chol)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto uh_bb = dbcsr::matrix<>::create()
		.set_world(m_world)
		.name("u hole")
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	dbcsr::multiply('N', 'N', 1.0, *m_s_bb, *L_bo, 0.0, *SL_bo)
		.perform();
		
	dbcsr::multiply('N', 'N', 1.0, *m_s_bb, *pv_bb, 0.0, *SPv_bb)
		.perform();
		
	dbcsr::multiply('T', 'N', 1.0, *SL_bo, *u_ao, 0.0, *up_ob)
		.perform();
		
	dbcsr::multiply('N', 'N', 1.0, *u_ao, *SPv_bb, 0.0, *uh_bb)
		.perform();
		
	SL_bo->release();
	SPv_bb->release();
	
	auto up_ob_01 = dbcsr::tensor<2>::create()
		.name("u particle 01")
		.set_pgrid(*spgrid2)
		.blk_sizes(ob_chol)
		.map1({0})
		.map2({1})
		.build();
		
	auto uh_bb_01 = dbcsr::tensor<2>::create()
		.name("u hole 01")
		.set_pgrid(*spgrid2)
		.blk_sizes(bb)
		.map1({0})
		.map2({1})
		.build();
	
	dbcsr::copy_matrix_to_tensor(*up_ob, *up_ob_01);
	dbcsr::copy_matrix_to_tensor(*uh_bb, *uh_bb_01);
	
	up_ob_01->filter(dbcsr::global::filter_eps);
	uh_bb_01->filter(dbcsr::global::filter_eps);
	
	up_ob->release();
	uh_bb->release();
	
	time_setup.finish();
	
	J_xob_batched->compress_init({0}, vec<int>{0,2}, vec<int>{1});
	eri_xob_batched->compress_init({0}, vec<int>{0,2}, vec<int>{1});
	m_eri3c2e_batched->decompress_init({0}, vec<int>{0,2}, vec<int>{1});
	
	time_contr.start();
	
	for (int ix = 0; ix != nxbatches; ++ix) {
		
		m_eri3c2e_batched->decompress({ix});
		auto eri_02_1 = m_eri3c2e_batched->get_work_tensor();
		
		vec<vec<int>> xn_bounds = {
			m_eri3c2e_batched->bounds(0,ix),
			m_eri3c2e_batched->full_bounds(2)
		};
		
		dbcsr::contract(*eri_02_1, *L_bo_01, *HT_xob_02_1)
			.bounds2(xn_bounds)
			.filter(dbcsr::global::filter_eps)
			.perform("Xmn, mi -> Xin");
		
		dbcsr::copy(*HT_xob_02_1, *HT_xob_01_2)
			.move_data(true)
			.perform();
		
		vec<vec<int>> xo_bounds = {
			m_eri3c2e_batched->bounds(0,ix),
			eri_xob_batched->full_bounds(1)
		};
		
		dbcsr::contract(*HT_xob_01_2, *pv_bb_01, *FT_xob_01_2)
			.bounds2(xo_bounds)
			.filter(dbcsr::global::filter_eps)
			.perform("Xin, nm -> Xim");
			
		dbcsr::copy(*FT_xob_01_2, *FT_xob_02_1)
			.move_data(true)
			.perform();
			
		eri_xob_batched->compress({ix}, FT_xob_02_1);
		
		dbcsr::contract(*HT_xob_01_2, *uh_bb_01, *FT_xob_01_2)
			.bounds2(xo_bounds)
			.filter(dbcsr::global::filter_eps)
			.alpha(1.0)
			.perform("Xim, mn -> Xin");
		
		HT_xob_01_2->clear();
		
		dbcsr::copy(*FT_xob_01_2, *J_xob_02_1)
			.move_data(true)
			.perform();
		
		dbcsr:contract(*eri_02_1, *up_ob_01, *HT_xob_02_1)
			.bounds2(xn_bounds)
			.filter(dbcsr::global::filter_eps)
			.perform("Xmn, im -> Xin");
			
		dbcsr::copy(*HT_xob_02_1, *HT_xob_01_2)
			.move_data(true)
			.perform();
		
		dbcsr::contract(*HT_xob_01_2, *pv_bb_01, *FT_xob_01_2)
			.alpha(-1.0)
			.filter(dbcsr::global::filter_eps)
			.bounds2(xo_bounds)
			.perform("Xin, nm -> Xim");
		
		HT_xob_01_2->clear();
			
		dbcsr::copy(*FT_xob_01_2, *J_xob_02_1)
			.sum(true)
			.move_data(true)
			.perform();
			
		J_xob_batched->compress({ix}, J_xob_02_1);
		FT_xob_01_2->clear();
		
	}
			
	J_xob_batched->compress_finalize();
	eri_xob_batched->compress_finalize();
	m_eri3c2e_batched->decompress_finalize();
	
	time_contr.finish();
	
	LOG.os<1>("Occupancies: ", J_xob_batched->occupation(), ", ", 
		eri_xob_batched->occupation(), '\n');
	
	//MPI_Barrier(m_world.comm());
	//exit(0);
		
	time.finish();	
		
	return std::make_tuple(eri_xob_batched, J_xob_batched);
	
}

std::tuple<dbcsr::shared_tensor<2,double>,dbcsr::shared_tensor<2,double>>
	MVP_AORISOSADC2::compute_F_OB(dbcsr::sbtensor<3,double> eri_xob_batched,
	dbcsr::sbtensor<3,double> J_xob_batched, 
	dbcsr::shared_matrix<double> L_bo)
{
	
	LOG.os<1>("Compute F\n");
	
	auto& time = TIME.sub("Computing F matrices");
	auto& time_setup = time.sub("Setup");
	auto& time_contr = time.sub("Contraction");
	auto& time_init = time.sub("Batch tensor initialization");

	time.start();
	time_setup.start();
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	auto o_chol = L_bo->col_blk_sizes();
	
	arrvec<int,2> xx = {x,x};
	arrvec<int,2> bo_chol = {b,o_chol};
	
	auto FT_xbb_02_1 = m_eri3c2e_batched->get_template("FT_xbb_02_1",
		vec<int>{0,2}, vec<int>{1});
		
	auto FT_xbb_0_12 = m_eri3c2e_batched->get_template("FT_xbb_0_12",
		vec<int>{0}, vec<int>{1,2});
	
	auto V_xx_01 = dbcsr::tensor<2>::create()
		.name("v_xx_01")
		.set_pgrid(*m_spgrid2)
		.blk_sizes(xx)
		.map1({0})
		.map2({1})
		.build();
		
	auto FA_xx_01 = dbcsr::tensor<2>::create_template(*
		V_xx_01)
		.name("FA_xx_01")
		.build();
		
	auto FB_xx_01 = dbcsr::tensor<2>::create_template(*
		V_xx_01)
		.name("FB_xx_01")
		.build();
		
	auto L_bo_01 = dbcsr::tensor<2>::create()
		.name("L_bo_01")
		.set_pgrid(*m_spgrid2)
		.blk_sizes(bo_chol)
		.map1({0})
		.map2({1})
		.build();
	
	dbcsr::copy_matrix_to_tensor(*m_v_xx, *V_xx_01);
	dbcsr::copy_matrix_to_tensor(*L_bo, *L_bo_01);
	
	int nxbatches = eri_xob_batched->nbatches(0);
	int nbbatches = eri_xob_batched->nbatches(2);
	
	time_setup.finish();
	
	time_init.start();
	
	J_xob_batched->decompress_init({2}, vec<int>{0,2}, vec<int>{1});
	eri_xob_batched->decompress_init({2}, vec<int>{0,2}, vec<int>{1});
	m_eri3c2e_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	
	time_init.finish();
	
	time_contr.start();
	
	for (int ib = 0; ib != nbbatches; ++ib) {
		
		eri_xob_batched->decompress({ib});
		J_xob_batched->decompress({ib});
		m_eri3c2e_batched->decompress({ib});
		
		vec<vec<int>> ob_bounds = {
			eri_xob_batched->full_bounds(1),
			eri_xob_batched->bounds(2,ib)
		};
		
		auto eri_xob_02_1 = eri_xob_batched->get_work_tensor();
		auto J_xob_02_1 = J_xob_batched->get_work_tensor();
		auto eri_xbb_0_12 = m_eri3c2e_batched->get_work_tensor();
		
		for (int ix = 0; ix != nxbatches; ++ix) {
		
			// allocate full transformed trensor
			auto& shellmat = *m_shellpairs;
					
			arrvec<int,3> res;
					
			auto xblkbounds = m_eri3c2e_batched->blk_bounds(0,ix);
			auto bblkbounds = m_eri3c2e_batched->blk_bounds(2,ib);

			for (int mublk = 0; mublk != b.size(); ++mublk) {
				for (int nublk = bblkbounds[0]; nublk != bblkbounds[1]+1; ++nublk) {
					
					if (!shellmat(mublk,nublk)) continue;
					
					for (int xblk = xblkbounds[0]; xblk != xblkbounds[1]+1; ++xblk) {
						
						std::array<int,3> idx = {xblk,mublk,nublk};
						if (m_world.rank() != FT_xbb_02_1->proc(idx)) continue;

						res[0].push_back(xblk);
						res[1].push_back(mublk);
						res[2].push_back(nublk);
					}
				}
			}
			
			FT_xbb_02_1->reserve(res);
			
			vec<vec<int>> xn_bounds = {
				m_eri3c2e_batched->bounds(0,ix),
				m_eri3c2e_batched->bounds(2,ib)
			};
			
			vec<vec<int>> mn_bounds = {
				m_eri3c2e_batched->full_bounds(1),
				m_eri3c2e_batched->bounds(2,ib)
			};
			
			vec<vec<int>> x_bounds = {
				m_eri3c2e_batched->bounds(0,ix)
			};
			
			dbcsr::contract(*eri_xob_02_1, *L_bo_01, *FT_xbb_02_1)
				.bounds2(xn_bounds)
				.filter(dbcsr::global::filter_eps)
				.retain_sparsity(true)
				.perform("Xin, mi -> Xmn");
				
			dbcsr::copy(*FT_xbb_02_1, *FT_xbb_0_12)
				.move_data(true)
				.perform();
				
			dbcsr::contract(*FT_xbb_0_12, *eri_xbb_0_12, *FA_xx_01)
				.bounds1(mn_bounds)
				.bounds2(x_bounds)
				.beta(1.0)
				.perform("Xmn, Ymn -> XY");
				
			FT_xbb_0_12->clear();
			FT_xbb_02_1->reserve(res);
			
			dbcsr::contract(*J_xob_02_1, *L_bo_01, *FT_xbb_02_1)
				.bounds2(xn_bounds)
				.filter(dbcsr::global::filter_eps)
				.retain_sparsity(true)
				.perform("Xin, mi -> Xmn");
				
			dbcsr::copy(*FT_xbb_02_1, *FT_xbb_0_12)
				.move_data(true)
				.perform();
				
			dbcsr::contract(*FT_xbb_0_12, *eri_xbb_0_12, *FB_xx_01)
				.bounds1(mn_bounds)
				.bounds2(x_bounds)
				.beta(1.0)
				.perform("Xmn, Ymn -> XY");
				
			FT_xbb_0_12->clear();
			
		} // end loop over x batches	
	} // end loop over b batches
	
	time_contr.finish();
	
	J_xob_batched->decompress_finalize();
	eri_xob_batched->decompress_finalize();
	m_eri3c2e_batched->decompress_finalize();
	
#ifdef _DLOG
	dbcsr::print(*FA_xx_01);
	dbcsr::print(*FB_xx_01);
#endif
	
	auto temp = dbcsr::tensor<2>::create_template(*
		V_xx_01)
		.name("temp")
		.build();
	
	auto transform = [&](auto& F) {
		
		dbcsr::contract(*V_xx_01, *F, *temp)
			.perform("XY, YZ -> XZ");
		
		dbcsr::contract(*temp, *V_xx_01, *F)
			.perform("XY, YZ -> XZ");
			
	};
	
	transform(FA_xx_01);
	transform(FB_xx_01);
	
	time.finish();
	
	return std::make_tuple(FA_xx_01, FB_xx_01);
	
}

dbcsr::sbtensor<3,double> MVP_AORISOSADC2::compute_I_OB(dbcsr::sbtensor<3,double>& eri_xob_batched,
	dbcsr::sbtensor<3,double>& R_xob_batched, dbcsr::shared_tensor<2,double>& F_A,
	dbcsr::shared_tensor<2,double>& F_B)
{

	auto& time = TIME.sub("Computing I batched");
	auto& time_init = time.sub("Batch tensor initialization");
	auto& time_contr = time.sub("Contraction");
	
	time.start();

	LOG.os<1>("Computing I\n");
	
	auto I_xob_batched = 
		dbcsr::btensor<3>::create_template(*eri_xob_batched)
			.name("I_xob_batched")
			.btensor_type(m_btype) 
			.print(LOG.global_plev())
			.build();
	
	time_init.start();
	
	I_xob_batched->compress_init({2}, vec<int>{0,1}, vec<int>{2});
	R_xob_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	eri_xob_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	
	time_init.finish();
	
	auto I_xob_01_2 = I_xob_batched->get_template("I_xbb_01_2", 
		vec<int>{0,1}, vec<int>{2});
		
	auto I_xob_0_12 = I_xob_batched->get_template("I_xbb_0_12", 
		vec<int>{0}, vec<int>{1,2});
	
	int nxbatches = eri_xob_batched->nbatches(0);
	int nbbatches = eri_xob_batched->nbatches(2);
	auto fullobds = eri_xob_batched->full_bounds(1);
	
	time_contr.start();
	
	for (int ib = 0; ib != nbbatches; ++ib) {
		
		R_xob_batched->decompress({ib});
		auto R_0_12 = R_xob_batched->get_work_tensor();
		
		eri_xob_batched->decompress({ib});
		auto eri_0_12 = eri_xob_batched->get_work_tensor();
		
		auto bbds = eri_xob_batched->bounds(2,ib);
		
		for (int ix = 0; ix != nxbatches; ++ix) {		
			
			auto xbds = eri_xob_batched->bounds(0,ix);
			
			vec<vec<int>> xbounds = { 
				xbds
			};
			
			vec<vec<int>> kabounds = { 
				fullobds,
				bbds
			};
			
			dbcsr::contract(*F_A, *R_0_12, *I_xob_0_12)
				.bounds2(xbounds)
				.bounds3(kabounds)
				.filter(dbcsr::global::filter_eps)
				.beta(1.0)
				.perform("XY, Yka -> Xka");
				
			dbcsr::contract(*F_B, *eri_0_12, *I_xob_0_12)
				.bounds2(xbounds)
				.bounds3(kabounds)
				.filter(dbcsr::global::filter_eps)
				.beta(1.0)
				.perform("YX, Yka -> Xka");
								
		}
		
		dbcsr::copy(*I_xob_0_12, *I_xob_01_2)
			.move_data(true)
			.perform();
		
		I_xob_batched->compress({ib}, I_xob_01_2);
		
	}
	
	time_contr.finish();
	
	I_xob_batched->compress_finalize();
	R_xob_batched->decompress_finalize();
	eri_xob_batched->decompress_finalize();
	
	time.finish();
	
	LOG.os<1>("Occupation of I_OB: ", I_xob_batched->occupation(), '\n');
	
	return I_xob_batched;
	
}

std::tuple<smat,smat> MVP_AORISOSADC2::compute_sigma_2e_ilap_OB(
	dbcsr::sbtensor<3,double>& I_xob_batched, smat& L_bo, double omega)
{
	
	/*
	 * sigE1_(b,b)[ma] = I_(x,o_c,b_t)[Yka] * eri_(x,o_c,b)[Ykm]
	 * sigE2_(o_c,b)[im] = I_(x,o_c,b_t)[Xic] * eri_(x,b,b)[Ycm]
	 * 
	 * sigE1_ia = Co * sigE1_bb * S * Cv
	 * sigE2_bb = Co * S * L_bo * sigE2_ob * Cv
	 * 
	 */
	 
	auto& time = TIME.sub("Computing sigma ilap");
	auto& time_A = time.sub("Part A");
	
	time.start();
	
	auto o_chol = L_bo->col_blk_sizes();
	auto x = m_mol->dims().x();
	auto b = m_mol->dims().b();
	
	arrvec<int,2> bo_chol = {b,o_chol};
	arrvec<int,2> ob_chol = {o_chol,b};
	arrvec<int,3> xob_chol = {x,o_chol,b};
	arrvec<int,2> bb = {b,b};
	
	auto L_bo_01 = dbcsr::tensor<2>::create()
		.name("L_bo_01")
		.set_pgrid(*m_spgrid2)
		.blk_sizes(bo_chol)
		.map1({0})
		.map2({1})
		.build();
	
	dbcsr::copy_matrix_to_tensor(*L_bo, *L_bo_01);
	
	int nxbas = m_mol->c_dfbasis()->nbf();
	int nbas = m_mol->c_basis()->nbf();
	int no_chol = L_bo->nfullcols_total();
	
	auto spgrid2 = dbcsr::pgrid<2>::create(m_world.comm())
		.build();
	
	std::array<int,3> dims_xob_chol = {nxbas,no_chol,nbas};
	
	auto spgrid_xob_chol = I_xob_batched->spgrid();
	
	int nxbatches = I_xob_batched->nbatches(0);
	int nobatches = I_xob_batched->nbatches(1);
	int nbbatches = I_xob_batched->nbatches(2);
	
	auto sig_pre_E1_bb_01 = dbcsr::tensor<2>::create()
		.name("sigmaE1_HT")
		.set_pgrid(*m_spgrid2)
		.blk_sizes(bb)
		.map1({0}).map2({1})
		.build();
		
	auto sig_pre_E2_ob_01 = dbcsr::tensor<2>::create()
		.name("sigmaE2_HT")
		.set_pgrid(*m_spgrid2)
		.blk_sizes(ob_chol)
		.map1({0}).map2({1})
		.build();
	
	auto HT_xob_1_02 = dbcsr::tensor<3>::create()
		.name("HT_xob_1_02")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({1}).map2({0,2})
		.build();
		
	auto HT_xob_01_2 = dbcsr::tensor<3>::create()
		.name("HT_xob_1_02")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,1}).map2({2})
		.build();
		
	auto I_xob_02_1 = I_xob_batched->get_template("I_xob_02_1", 
		vec<int>{0,2}, vec<int>{1});
		
	m_eri3c2e_batched->decompress_init({0}, vec<int>{1}, vec<int>{0,2});
	I_xob_batched->decompress_init({0}, vec<int>{0,1}, vec<int>{2});
	
	time_A.start();
	
	for (int ix = 0; ix != nxbatches; ++ix) {
			
			m_eri3c2e_batched->decompress({ix});
			auto eri_1_02 = m_eri3c2e_batched->get_work_tensor();
			
			I_xob_batched->decompress({ix});
			
			auto I_xob_01_2 = I_xob_batched->get_work_tensor();
			
			for (int ib = 0; ib != nbbatches; ++ib) {
				
				vec<vec<int>> xmbounds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->bounds(2,ib)
				};
			
				// form cbar
				dbcsr::contract(*eri_1_02, *L_bo_01, *HT_xob_1_02)
					.bounds2(xmbounds)
					//.print(LOG.global_plev() >= 3)
					.filter(dbcsr::global::filter_eps/nxbatches)
					.perform("Xlm, li -> Xim");
										
				dbcsr::copy(*HT_xob_1_02, *HT_xob_01_2).move_data(true).perform();
				
				vec<vec<int>> xobounds = {
					m_eri3c2e_batched->bounds(0,ix),
					I_xob_batched->full_bounds(1)
				};
				
				vec<vec<int>> mbounds = {
					m_eri3c2e_batched->bounds(2,ib)
				};
				
				// form sig_A
				dbcsr::contract(*I_xob_01_2, *HT_xob_01_2, *sig_pre_E1_bb_01)
					.bounds1(xobounds)
					.bounds3(mbounds)
					//.print(LOG.global_1plev() >= 3)
					.filter(dbcsr::global::filter_eps/nxbatches)
					.beta(1.0)
					.perform("Xin, Xim -> mn");
				
				HT_xob_01_2->clear();
				
				// form sig_B
				vec<vec<int>> xob_bds = {
					I_xob_batched->bounds(0,ix),
					I_xob_batched->full_bounds(1),
					I_xob_batched->bounds(2,ib)
				};
				
				dbcsr::copy(*I_xob_01_2, *I_xob_02_1)
					.bounds(xob_bds)
					.perform();
		
				dbcsr::contract(*I_xob_02_1, *eri_1_02, *sig_pre_E2_ob_01)
					.bounds1(xmbounds)
					.beta(1.0)
					.perform("Xin, Xmn -> im");
				
				I_xob_02_1->clear();
								
			}
			
		}
		
		time_A.finish();
		
		m_eri3c2e_batched->decompress_finalize();
		I_xob_batched->decompress_finalize();
	
		auto o = m_mol->dims().oa();
		auto v = m_mol->dims().va();
		
		auto sig_pre_E1_bb = dbcsr::matrix<>::create()
			.name("sigmaE1_bb")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();
			
		auto sig_pre_E2_ob = dbcsr::matrix<>::create()
			.name("sigmaE2_bb")
			.set_world(m_world)
			.row_blk_sizes(o_chol)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();
			
		dbcsr::copy_tensor_to_matrix(*sig_pre_E1_bb_01, *sig_pre_E1_bb);
		dbcsr::copy_tensor_to_matrix(*sig_pre_E2_ob_01, *sig_pre_E2_ob);
		
		sig_pre_E1_bb_01->destroy();
		sig_pre_E2_ob_01->destroy();
			
		auto SC_bv = dbcsr::matrix<>::create()
			.name("SC_bv")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(v)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();
			
		auto SC_bo = dbcsr::matrix<>::create()
			.name("SL_bo")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(o)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();
			
		auto LSC_co = dbcsr::matrix<>::create()
			.name("LSC_co")
			.set_world(m_world)
			.row_blk_sizes(o_chol)
			.col_blk_sizes(o)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();
			
		dbcsr::multiply('N', 'N', 1.0, *m_s_bb, *m_c_bv, 0.0, *SC_bv)
			.perform();
			
		dbcsr::multiply('N', 'N', 1.0, *m_s_bb, *m_c_bo, 0.0, *SC_bo)
			.perform();
			
		dbcsr::multiply('T', 'N', 1.0, *L_bo, *SC_bo, 0.0, *LSC_co)
			.perform();
			
		auto sigmaE1_ia = u_transform(sig_pre_E1_bb, 'T', m_c_bo, 'N', SC_bv);
		auto sigmaE2_ia = u_transform(sig_pre_E2_ob, 'T', LSC_co, 'N', m_c_bv);
	
		sigmaE1_ia->setname("sigma_E1");
		sigmaE2_ia->setname("sigma_E2");
		
		time.finish();
	
		return std::make_tuple(sigmaE1_ia, sigmaE2_ia);
	
} 

smat MVP_AORISOSADC2::compute_sigma_2e_OB(smat& u_ao, double omega) {
	
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
	
	LOG.os<1>("Computing lapalce matrices for doubles part.");
	if (std::fabs(m_old_omega - omega) 
		< std::numeric_limits<double>::epsilon() * 10)
	{
		LOG.os<1>("Omega is unchanged, no need to recompute.");
	
	} else {
		
		auto& time_chol = TIME.sub("Cholesky in sigma E");
		time_chol.start();
	
		LOG.os<1>("Omega changed, recomputing...\n");
		m_laphelper_dd->compute(true,false,omega,m_mol->mo_split());
		
		time_chol.finish();
			
	}
	
	auto o = m_mol->dims().oa();
	auto v = m_mol->dims().va();
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	auto sigma_2e_A = dbcsr::matrix<>::create()
		.name("sigma_2e_A")
		.set_world(m_world)
		.row_blk_sizes(o)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto sigma_2e_B = dbcsr::matrix<>::create_template(*sigma_2e_A)
		.name("sigma_2e_B").build();

	// Loop
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		auto L_bo = m_laphelper_dd->pseudo_cholesky_occ()[ilap];
		auto pseudo_v_bb = m_laphelper_dd->pseudo_densities_vir()[ilap];
		
		L_bo->filter(dbcsr::global::filter_eps);
		pseudo_v_bb->filter(dbcsr::global::filter_eps);
		
		auto [eri_xob_batched, J_xob_batched] 
			= compute_laplace_batchtensors_OB(u_ao, L_bo, pseudo_v_bb);
			
		// Form F matrices
		auto [FA_xx, FB_xx] = compute_F_OB(eri_xob_batched, J_xob_batched, L_bo);
		
		auto I_xob_batched = compute_I_OB(eri_xob_batched, J_xob_batched, 
			FA_xx, FB_xx);
	
		J_xob_batched->reset();
		eri_xob_batched->reset();
	
		auto [sig_ilap_E1, sig_ilap_E2] = compute_sigma_2e_ilap_OB(
			I_xob_batched, L_bo, omega);
			
		I_xob_batched->reset();
		
		double xpt = m_laphelper_dd->xpoints()[ilap];
		
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

std::tuple<dbcsr::sbtensor<3,double>,dbcsr::sbtensor<3,double>>
	MVP_AORISOSADC2::compute_laplace_batchtensors_OV(smat& u_ao, smat& L_bo, smat& L_bv)
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
	
	auto spgrid2 = dbcsr::pgrid<2>::create(m_world.comm())
		.build();
	
	std::array<int,3> dims_xov_chol = {nxbas,no_chol,nv_chol};
	std::array<int,3> dims_xob_chol = {nxbas,no_chol,nbas};
	
	auto spgrid_xov_chol = dbcsr::pgrid<3>::create(m_world.comm())
		.tensor_dims(dims_xov_chol)
		.build();
		
	auto spgrid_xob_chol = dbcsr::pgrid<3>::create(m_world.comm())
		.tensor_dims(dims_xob_chol)
		.build();
	
	auto J_xov_batched = dbcsr::btensor<3>::create()
		.name("J_xov_batched")
		.set_pgrid(spgrid_xov_chol)
		.blk_sizes(xov_chol)
		.blk_maps(blkmaps)
		.batch_dims(bdims)
		.btensor_type(m_btype)
		.print(LOG.global_plev())
		.build();
			
	auto eri_xov_batched = dbcsr::btensor<3>::create()
		.name("eri_xov_batched")
		.set_pgrid(spgrid_xov_chol)
		.blk_sizes(xov_chol)
		.blk_maps(blkmaps)
		.batch_dims(bdims)
		.btensor_type(m_btype)
		.print(LOG.global_plev())
		.build();
		
	auto L_bo_01 = dbcsr::tensor<2>::create()
		.name("L_bo_01")
		.set_pgrid(*spgrid2)
		.blk_sizes(bo_chol)
		.map1({0})
		.map2({1})
		.build();
		
	auto L_bv_01 = dbcsr::tensor<2>::create()
		.name("L_bv_01")
		.set_pgrid(*spgrid2)
		.blk_sizes(bv_chol)
		.map1({0})
		.map2({1})
		.build();
		
	auto HT_xob_02_1 = dbcsr::tensor<3>::create()
		.name("HT_02_1")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,2})
		.map2({1})
		.build();
		
	auto HT_xob_01_2 = dbcsr::tensor<3>::create()
		.name("HT_01_2")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,1})
		.map2({2})
		.build();
		
	auto FT_xov_01_2 = dbcsr::tensor<3>::create()
		.name("FT_01_2")
		.set_pgrid(*spgrid_xov_chol)
		.blk_sizes(xov_chol)
		.map1({0,1})
		.map2({2})
		.build();
		
	auto FT_xov_0_12 = dbcsr::tensor<3>::create()
		.name("FT_0_12")
		.set_pgrid(*spgrid_xov_chol)
		.blk_sizes(xov_chol)
		.map1({0})
		.map2({1,2})
		.build();
		
	auto J_xov_0_12 = dbcsr::tensor<3>::create()
		.name("J_0_12")
		.set_pgrid(*spgrid_xov_chol)
		.blk_sizes(xov_chol)
		.map1({0})
		.map2({1,2})
		.build();
	
	dbcsr::copy_matrix_to_tensor(*L_bo, *L_bo_01);
	dbcsr::copy_matrix_to_tensor(*L_bv, *L_bv_01);
	
	J_xov_batched->compress_init({0}, vec<int>{0}, vec<int>{1,2});
	eri_xov_batched->compress_init({0}, vec<int>{0}, vec<int>{1,2});
	m_eri3c2e_batched->decompress_init({0}, vec<int>{0,2}, vec<int>{1});
	
	// Form transformed u vectors
	
	auto SL_bo = dbcsr::matrix<>::create_template(*L_bo)
		.name("SL_bo")
		.build();
		
	auto SL_bv = dbcsr::matrix<>::create_template(*L_bv)
		.name("SL_bv")
		.build();
	
	auto up_ob = dbcsr::matrix<>::create()
		.set_world(m_world)
		.name("u particle")
		.row_blk_sizes(o_chol)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto uh_bv = dbcsr::matrix<>::create()
		.set_world(m_world)
		.name("u hole")
		.row_blk_sizes(b)
		.col_blk_sizes(v_chol)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	dbcsr::multiply('N', 'N', 1.0, *m_s_bb, *L_bo, 0.0, *SL_bo)
		.perform();
		
	dbcsr::multiply('N', 'N', 1.0, *m_s_bb, *L_bv, 0.0, *SL_bv)
		.perform();
		
	dbcsr::multiply('T', 'N', 1.0, *SL_bo, *u_ao, 0.0, *up_ob)
		.perform();
		
	dbcsr::multiply('N', 'N', 1.0, *u_ao, *SL_bv, 0.0, *uh_bv)
		.perform();
		
	SL_bo->release();
	SL_bv->release();
	
	auto up_ob_01 = dbcsr::tensor<2>::create()
		.name("u particle 01")
		.set_pgrid(*spgrid2)
		.blk_sizes(ob_chol)
		.map1({0})
		.map2({1})
		.build();
		
	auto uh_bv_01 = dbcsr::tensor<2>::create()
		.name("u hole 01")
		.set_pgrid(*spgrid2)
		.blk_sizes(bv_chol)
		.map1({0})
		.map2({1})
		.build();
	
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
		
		dbcsr::contract(*HT_xob_01_2, *L_bv_01, *FT_xov_01_2)
			.bounds2(xo_bounds)
			.filter(dbcsr::global::filter_eps)
			.perform("Xin, na -> Xia");
			
		dbcsr::copy(*FT_xov_01_2, *FT_xov_0_12)
			.move_data(true)
			.perform();
			
		eri_xov_batched->compress({ix}, FT_xov_0_12);
		
		dbcsr::contract(*HT_xob_01_2, *uh_bv_01, *FT_xov_01_2)
			.bounds2(xo_bounds)
			.filter(dbcsr::global::filter_eps)
			.alpha(1.0)
			.perform("Xim, ma -> Xia");
		
		HT_xob_01_2->clear();
		
		dbcsr::copy(*FT_xov_01_2, *J_xov_0_12)
			.move_data(true)
			.perform();
		
		dbcsr:contract(*eri_02_1, *up_ob_01, *HT_xob_02_1)
			.bounds2(xn_bounds)
			.filter(dbcsr::global::filter_eps)
			.perform("Xmn, im -> Xin");
			
		dbcsr::copy(*HT_xob_02_1, *HT_xob_01_2)
			.move_data(true)
			.perform();
		
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
	
	LOG.os<1>("Occupancies: ", J_xov_batched->occupation(), ", ", 
		eri_xov_batched->occupation(), '\n');
	
	//MPI_Barrier(m_world.comm());
	//exit(0);
		
	time.finish();	
		
	return std::make_tuple(eri_xov_batched, J_xov_batched);
	
}

std::tuple<dbcsr::shared_tensor<2,double>,dbcsr::shared_tensor<2,double>>
	MVP_AORISOSADC2::compute_F_OV(dbcsr::sbtensor<3,double> eri_xov_batched,
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
	
	auto V_xx_01 = dbcsr::tensor<2>::create()
		.name("v_xx_01")
		.set_pgrid(*m_spgrid2)
		.blk_sizes(xx)
		.map1({0})
		.map2({1})
		.build();
		
	auto FA_xx_01 = dbcsr::tensor<2>::create_template(*
		V_xx_01)
		.name("FA_xx_01")
		.build();
		
	auto FB_xx_01 = dbcsr::tensor<2>::create_template(*
		V_xx_01)
		.name("FB_xx_01")
		.build();
	
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
	
	auto temp = dbcsr::tensor<2>::create_template(*
		V_xx_01)
		.name("temp")
		.build();
	
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

dbcsr::sbtensor<3,double> MVP_AORISOSADC2::compute_I_OV(dbcsr::sbtensor<3,double>& eri_xov_batched,
	dbcsr::sbtensor<3,double>& R_xov_batched, dbcsr::shared_tensor<2,double>& F_A,
	dbcsr::shared_tensor<2,double>& F_B)
{

	auto& time = TIME.sub("Computing I batched");
	time.start();

	LOG.os<1>("Compute I\n");
	
	auto I_xov_batched = 
		dbcsr::btensor<3>::create_template(*eri_xov_batched)
			.name("I_xov_batched")
			.btensor_type(m_btype)
			.print(LOG.global_plev())
			.build();

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
	
	LOG.os<1>("Occupation of I_OV: ", I_xov_batched->occupation(), '\n');
	
	return I_xov_batched;
	
}

std::tuple<smat,smat> MVP_AORISOSADC2::compute_sigma_2e_ilap_OV(
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
	
	auto L_bo_01 = dbcsr::tensor<2>::create()
		.name("L_bo_01")
		.set_pgrid(*m_spgrid2)
		.blk_sizes(bo_chol)
		.map1({0})
		.map2({1})
		.build();
		
	auto L_bv_01 = dbcsr::tensor<2>::create()
		.name("L_bv_01")
		.set_pgrid(*m_spgrid2)
		.blk_sizes(bv_chol)
		.map1({0})
		.map2({1})
		.build();
			
	dbcsr::copy_matrix_to_tensor(*L_bo, *L_bo_01);
	dbcsr::copy_matrix_to_tensor(*L_bv, *L_bv_01);
	
	int nxbas = m_mol->c_dfbasis()->nbf();
	int nbas = m_mol->c_basis()->nbf();
	int no_chol = L_bo->nfullcols_total();
	int nv_chol = L_bv->nfullcols_total();
	
	auto spgrid2 = dbcsr::pgrid<2>::create(m_world.comm())
		.build();
	
	std::array<int,3> dims_xob_chol = {nxbas,no_chol,nbas};
	std::array<int,3> dims_xvb_chol = {nxbas,nv_chol,nbas};
	
	auto spgrid_xov_chol = I_xov_batched->spgrid();
		
	auto spgrid_xob_chol = dbcsr::pgrid<3>::create(m_world.comm())
		.tensor_dims(dims_xob_chol)
		.build();
		
	auto spgrid_xvb_chol = dbcsr::pgrid<3>::create(m_world.comm())
		.tensor_dims(dims_xvb_chol)
		.build();
	
	int nxbatches = I_xov_batched->nbatches(0);
	int nobatches = I_xov_batched->nbatches(1);
	int nvbatches = I_xov_batched->nbatches(2);
	
	auto sigmaE1_HT_01 = dbcsr::tensor<2>::create()
		.name("sigmaE1_HT")
		.set_pgrid(*m_spgrid2)
		.blk_sizes(bv_chol)
		.map1({0}).map2({1})
		.build();
		
	auto sigmaE2_HT_01 = dbcsr::tensor<2>::create()
		.name("sigmaE2_HT")
		.set_pgrid(*m_spgrid2)
		.blk_sizes(ob_chol)
		.map1({0}).map2({1})
		.build();
	
	auto HT_xob_1_02 = dbcsr::tensor<3>::create()
		.name("HT_xob_1_02")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({1}).map2({0,2})
		.build();
		
	auto HT_xob_01_2 = dbcsr::tensor<3>::create()
		.name("HT_xob_1_02")
		.set_pgrid(*spgrid_xob_chol)
		.blk_sizes(xob_chol)
		.map1({0,1}).map2({2})
		.build();
		
	auto HT_xvb_1_02 = dbcsr::tensor<3>::create()
		.name("HT_xob_1_02")
		.set_pgrid(*spgrid_xvb_chol)
		.blk_sizes(xvb_chol)
		.map1({1}).map2({0,2})
		.build();
		
	auto HT_xvb_01_2 = dbcsr::tensor<3>::create()
		.name("HT_xob_1_02")
		.set_pgrid(*spgrid_xvb_chol)
		.blk_sizes(xvb_chol)
		.map1({0,1}).map2({2})
		.build();
	
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
		
		auto sigmaE1_HT = dbcsr::matrix<>::create()
			.name("sigmaE1_HT")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(v_chol)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();
			
		auto sigmaE2_HT = dbcsr::matrix<>::create()
			.name("sigmaE2_HT")
			.set_world(m_world)
			.row_blk_sizes(o_chol)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();
			
		dbcsr::copy_tensor_to_matrix(*sigmaE1_HT_01, *sigmaE1_HT);
		dbcsr::copy_tensor_to_matrix(*sigmaE2_HT_01, *sigmaE2_HT);
		
		sigmaE1_HT_01->destroy();
		sigmaE2_HT_01->destroy();
			
		auto Po_bb = dbcsr::matrix<>::create()
			.name("Po_bb")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::symmetric)
			.build();
			
		auto Pv_bb = dbcsr::matrix<>::create()
			.name("Pv_bb")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::symmetric)
			.build();
			
		dbcsr::multiply('N', 'T', 1.0, *m_c_bo, *m_c_bo, 0.0, *Po_bb)
			.perform();
			
		dbcsr::multiply('N', 'T', 1.0, *m_c_bv, *m_c_bv, 0.0, *Pv_bb)
			.perform();
			
		auto SC_bo = dbcsr::matrix<>::create_template(*m_c_bo)
			.name("SC_bo")
			.build();
			
		auto SC_bv = dbcsr::matrix<>::create_template(*m_c_bv)
			.name("SC_bo")
			.build();
		
		dbcsr::multiply('N', 'N', 1.0, *m_s_bb, *m_c_bo, 0.0, *SC_bo)
			.perform();
			
		dbcsr::multiply('N', 'N', 1.0, *m_s_bb, *m_c_bv, 0.0, *SC_bv)
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

smat MVP_AORISOSADC2::compute_sigma_2e_OV(smat& u_ao, double omega) {
	
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
	
	LOG.os<1>("Computing lapalce matrices for doubles part.");
	if (std::fabs(m_old_omega - omega) 
		< std::numeric_limits<double>::epsilon() * 10)
	{
		LOG.os<1>("Omega is unchanged, no need to recompute.");
	
	} else {
	
		auto& time_chol = TIME.sub("Cholesky in sigma E");
		time_chol.start();
		
		LOG.os<1>("Omega changed, recomputing...\n");
		m_laphelper_dd->compute(true,true,omega,m_mol->mo_split());
		
		time_chol.finish();
			
	}
	
	auto o = m_mol->dims().oa();
	auto v = m_mol->dims().va();
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	auto sigma_2e_A = dbcsr::matrix<>::create()
		.name("sigma_2e_A")
		.set_world(m_world)
		.row_blk_sizes(o)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto sigma_2e_B = dbcsr::matrix<>::create_template(*sigma_2e_A)
		.name("sigma_2e_B").build();

	// Loop
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
		auto L_bo = m_laphelper_dd->pseudo_cholesky_occ()[ilap];
		auto L_bv = m_laphelper_dd->pseudo_cholesky_vir()[ilap];
		
		/*std::string file_o = std::string(std::filesystem::current_path()) + 
			"/chol_occ_" + std::to_string(ilap);
		std::string file_v = std::string(std::filesystem::current_path()) + 
			"/chol_vir_" + std::to_string(ilap);
		
		//util::plot(L_bo, 1e-5, file_o);
		//util::plot(L_bv, 1e-5, file_v);*/
		
		L_bo->filter(dbcsr::global::filter_eps);
		L_bv->filter(dbcsr::global::filter_eps);
		
		auto [eri_xov_batched, J_xov_batched] 
			= compute_laplace_batchtensors_OV(u_ao, L_bo, L_bv);
			
		// Form F matrices
		auto [FA_xx, FB_xx] = compute_F_OV(eri_xov_batched, J_xov_batched);
		
		auto I_xov_batched = compute_I_OV(eri_xov_batched, J_xov_batched, FA_xx, FB_xx);
	
		J_xov_batched->reset();
		eri_xov_batched->reset();
	
		auto [sig_ilap_E1, sig_ilap_E2] = compute_sigma_2e_ilap_OV(I_xov_batched, 
			L_bo, L_bv, omega);
		
		double xpt = m_laphelper_dd->xpoints()[ilap];
		
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

/* =====================================================================
 *                         ADC(2) SIGMA CONSTRUCTOR
 * ====================================================================*/

smat MVP_AORISOSADC2::compute(smat u_ia, double omega) {
	
	auto& time_com = TIME.sub("Computing sigma ADC(2)");
	
	TIME.start();
	time_com.start();
	
	LOG.os<1>("Computing AO-ADC(2) MVP product... \n");
	LOG.os<1>("Computing sigma_0 of AO-ADC(2) ... \n");
	
	auto sigma_0 = compute_sigma_0(u_ia, m_eps_occ, m_eps_vir);
	
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

	decltype(sigma_2d) sigma_2e;
	if constexpr(_use_doubles_ob) {
		sigma_2e = compute_sigma_2e_OB(u_ao, omega);
	} else {
		sigma_2e = compute_sigma_2e_OV(u_ao, omega);
	}
	

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
	
	LOG.os<>("DOT: ", u_ia->dot(*sigma_0), '\n');
	
	//TIME.print_info();
	//exit(0);
	
	return sigma_0;
	
}

} // end namespace
