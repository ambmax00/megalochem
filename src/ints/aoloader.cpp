#include "ints/aoloader.h"
#include "ints/fitting.h"
#include "math/linalg/LLT.h"
#include "math/solvers/hermitian_eigen_solver.h"

namespace ints {

using smatd = dbcsr::shared_matrix<double>;
using sbt3 = dbcsr::sbtensor<3,double>;
using sbt4 = dbcsr::sbtensor<4,double>;
	
std::pair<smatd,smatd> aoloader::invert(smatd in) {
	
	/*
	math::LLT chol(in, LOG.global_plev());
	chol.compute();
	
	
	
	auto linv = chol.L_inv(m);
	auto inv_sqrt = dbcsr::transpose(linv).get();
	auto inv = chol.inverse(m);
	*/
	
	auto m = in->row_blk_sizes();
	
	math::hermitian_eigen_solver herm(in, 'V', true);
	herm.compute();
	
	auto inv = herm.inverse();
	auto invsqrt = herm.inverse_sqrt();

	return std::make_pair<smatd,smatd>(
		std::move(inv),std::move(invsqrt));
	
}

void aoloader::compute() {
	
	LOG.os<>("Loading AO quantities.\n");
	
	TIME.start();
	
	ints::dfitting dfit(m_world, m_mol, LOG.global_plev());
	
	// setup all pgrids
	dbcsr::shared_pgrid<2> spgrid2;
	dbcsr::shared_pgrid<3> spgrid3;
	dbcsr::shared_pgrid<4> spgrid4;
	
	spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	m_reg.insert(key::pgrid2, spgrid2);
	
	int nbf = m_mol->c_basis()->nbf();
	
	if (m_mol->c_dfbasis()) {
		int naux = m_mol->c_dfbasis()->nbf();
		std::array<int,3> pdims3 = {naux,nbf,nbf};
		spgrid3 = dbcsr::create_pgrid<3>(m_world.comm())
			.tensor_dims(pdims3).get();
		m_reg.insert(key::pgrid3,spgrid3);
	}
	
	std::array<int,4> pdims4 = {nbf,nbf,nbf,nbf};
	spgrid4 = dbcsr::create_pgrid<4>(m_world.comm())
		.tensor_dims(pdims4).get();
	m_reg.insert(key::pgrid4,spgrid4);
	
	// compute s
	if (comp(key::ovlp_bb)) {
		LOG.os<>("Computing overlap integrals\n");
		auto& time = TIME.sub("Overlap integrals");
		time.start();
		auto s = m_aofac->ao_overlap();
		m_reg.insert(key::ovlp_bb,s);
		time.finish();
	}
	
	// compute k
	if (comp(key::kin_bb)) {
		LOG.os<>("Computing kinetic integrals\n");
		auto& time = TIME.sub("Kinetic integrals");
		time.start();
		auto k = m_aofac->ao_kinetic();
		m_reg.insert(key::kin_bb,k);
		time.finish();
	}
	
	// compute v
	if (comp(key::pot_bb)) {
		LOG.os<>("Computing nuclear integrals\n");
		auto& time = TIME.sub("Nuclear integrals");
		time.start();
		auto v = m_aofac->ao_nuclear();
		m_reg.insert(key::pot_bb,v);
		time.finish();
	}
	
	if (comp(key::ovlp_xx)) {
		LOG.os<>("Computing auxiliary overlap integrals\n");
		auto& time = TIME.sub("Auxiliary overlap integrals");
		time.start();
		auto s = m_aofac->ao_auxoverlap();
		m_reg.insert(key::ovlp_xx,s);
		time.finish();
	}
	
	if (comp(key::ovlp_bb_inv)) {
		LOG.os<>("Computing overlap inverse\n");
		auto& time = TIME.sub("Inverting overlap matrix");
		time.start();
		auto s = m_reg.get<smatd>(key::ovlp_bb);
		auto p = invert(s);
		auto s_inv = p.first;
		m_reg.insert(key::ovlp_bb_inv,s_inv);
		time.finish();
	}
	
	if (comp(key::ovlp_xx_inv)) {
		LOG.os<>("Computing auxiliary overlap inverse\n");
		auto& time = TIME.sub("Inverting auxiliary overlap matrix");
		time.start();
		auto s = m_reg.get<smatd>(key::ovlp_xx);		
		auto p = invert(s);
		auto s_inv = p.first;
		m_reg.insert(key::ovlp_xx_inv,s_inv);
		time.finish();
	}
	
	if (comp(key::coul_xx)) {
		LOG.os<>("Computing coulomb metric\n");
		auto& time = TIME.sub("Coulomb metric");
		time.start();
		auto c = m_aofac->ao_2c2e(ints::metric::coulomb);
		m_reg.insert(key::coul_xx,c);
		time.finish();
	}
		
	if (comp(key::erfc_xx)) {
		
		LOG.os<>("Computing erfc coulomb attenuated integrals\n");
		
		auto& time = TIME.sub("Erfc-attenuated coulomb metric");
		time.start();
		
		auto c = m_aofac->ao_2c2e(metric::coulomb);
		auto e = m_aofac->ao_2c2e(metric::erfc_coulomb);
		auto p = invert(c);
		auto cinv = p.first;
		
		smatd temp = 
			dbcsr::create_template<double>(c)
			.name("temp")
			.matrix_type(dbcsr::type::no_symmetry).get();
			
		dbcsr::multiply('N', 'N', *e, *cinv, *temp).perform();
		dbcsr::multiply('N', 'N', *temp, *e, *c).perform();
		
		m_reg.insert(key::erfc_xx,c);
		
		time.finish();
		
	}
		
	if (comp(key::coul_xx_inv) || comp(key::coul_xx_invsqrt)) {
		
		LOG.os<>("Computing metric inverse\n");
		
		auto& time = TIME.sub("Inverting metric");
		time.start();
		
		auto c = m_reg.get<smatd>(key::coul_xx);
		auto p = invert(c);
		auto cinv = p.first;
		auto cinvsqrt = p.second;
		if (comp(key::coul_xx_inv)) {
			m_reg.insert(key::coul_xx_inv,cinv);
		}
		if (comp(key::coul_xx_invsqrt)) {
			m_reg.insert(key::coul_xx_invsqrt,cinvsqrt);
		}
		
		time.finish();
		
	}
		
	if (comp(key::erfc_xx_inv) || comp(key::erfc_xx_invsqrt)) {
		
		LOG.os<>("Computing metric (attenuated) inverse\n");
		
		auto& time = TIME.sub("Inverting attenuated metric");
		time.start();
		
		auto c = m_reg.get<smatd>(key::erfc_xx);
		auto p = invert(c);
		auto cinv = p.first;
		auto cinvsqrt = p.second;
		if (comp(key::erfc_xx_inv)) {
			m_reg.insert(key::erfc_xx_inv,cinv);
		}
		if (comp(key::coul_xx_invsqrt)) {
			m_reg.insert(key::erfc_xx_invsqrt,cinvsqrt);
		}
		
		time.finish();
		
	}
		
	if (comp(key::coul_bbbb)) {
		
		auto& t_ints = TIME.sub("Computing eris");
		LOG.os<>("Computing 2e integrals.\n");
		
		t_ints.start();
		
		m_aofac->ao_eri_setup(metric::coulomb);
		
		auto b = m_mol->dims().b();
		arrvec<int,4> bbbb = {b,b,b,b};
		
		int nbatches_b = m_opt.get<int>("nbatches_x", 5);
		std::array<int,4> bdims = {nbatches_b,nbatches_b,nbatches_b,nbatches_b};
		
		auto blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
		arrvec<int,4> blkmaps = {blkmap_b, blkmap_b, blkmap_b, blkmap_b};
		
		auto eri_batched = dbcsr::btensor_create<4>()
			.name(m_mol->name() + "_eri_batched")
			.pgrid(spgrid4)
			.blk_sizes(bbbb)
			.blk_map(blkmaps)
			.batch_dims(bdims)
			.btensor_type(dbcsr::btype::core)
			.print(LOG.global_plev())
			.get();
			
		auto eris_gen = dbcsr::tensor_create<4,double>()
			.name("eris_4")
			.pgrid(spgrid4)
			.map1({0,1}).map2({2,3})
			.blk_sizes(bbbb)
			.get();
		
		vec<int> map1 = {0,1};
		vec<int> map2 = {2,3};
		eri_batched->compress_init({2,3},map1,map2);
		
		vec<vec<int>> bounds(4);
		
		for (int imu = 0; imu != eri_batched->nbatches(2); ++imu) {
			for (int inu = 0; inu != eri_batched->nbatches(3); ++inu) {
				
				bounds[0] = eri_batched->full_blk_bounds(0);
				bounds[1] = eri_batched->full_blk_bounds(1);
				bounds[2] = eri_batched->blk_bounds(2, imu);
				bounds[3] = eri_batched->blk_bounds(3, inu);
				
				m_aofac->ao_4c_fill(eris_gen, bounds, nullptr);
				
				eris_gen->filter(dbcsr::global::filter_eps);
				
				eri_batched->compress({imu,inu},eris_gen);
				
			}
		}
			
		eri_batched->compress_finalize();
		
		t_ints.finish();
		
		m_reg.insert(key::coul_bbbb, eri_batched);
		LOG.os<>("Done computing 2e integrals.\n\n");
		
	}
		
	if (comp(key::scr_xbb)) {
	
		LOG.os<>("Computing screener\n");
	
		auto& time = TIME.sub("Screener for 3c2e integrals");
		time.start();
		
		std::shared_ptr<ints::screener> scr;
		//if (comp(key::coul_xbb)) 
		//if (comp(key::erfc_xbb)) scr.reset(new ints::schwarz_screener(m_aofac,"erfc_coulomb"));
		scr.reset(new ints::schwarz_screener(m_aofac,"coulomb"));		
		scr->compute();
				
		m_reg.insert(key::scr_xbb,scr);
				
		time.finish();
				
	}		
	
	if (comp(key::coul_xbb) || comp(key::erfc_xbb)) {
		
		LOG.os<>("Computing 3c2e integrals.\n");
		
		//t_screen.start();
		
		auto scr = m_reg.get<ints::shared_screener>(key::scr_xbb);
		
		//t_screen.finish();
		
		auto& t_eri_batched = TIME.sub("3c2e integrals batched");
		auto& t_calc = t_eri_batched.sub("calc");
		auto& t_setup = t_eri_batched.sub("setup");
		auto& t_compress = t_eri_batched.sub("Compress");
		
		t_eri_batched.start();
		t_setup.start();
		
		ints::metric m;
		if (comp(key::coul_xbb)) m = ints::metric::coulomb;
		if (comp(key::erfc_xbb)) m = ints::metric::erfc_coulomb;
				
		m_aofac->ao_3c2e_setup(m);
		auto genfunc = m_aofac->get_generator(scr);
		
		std::string eris_mem = m_opt.get<std::string>("eris", "core");
		dbcsr::btype mytype = dbcsr::get_btype(eris_mem);
		
		int nbatches_x = m_opt.get<int>("nbatches_x", 5);
		int nbatches_b = m_opt.get<int>("nbatches_b", 5);
		
		auto b = m_mol->dims().b();
		auto x = m_mol->dims().x();
		arrvec<int,3> xbb = {x,b,b};
		
		std::array<int,3> bdims = {nbatches_x,nbatches_b,nbatches_b};
		
		auto blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
		auto blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
		
		arrvec<int,3> blkmaps = {blkmap_x, blkmap_b, blkmap_b};
		
		auto eri_batched = dbcsr::btensor_create<3>()
			.name(m_mol->name() + "_eri_batched")
			.pgrid(spgrid3)
			.blk_sizes(xbb)
			.batch_dims(bdims)
			.btensor_type(mytype)
			.blk_map(blkmaps)
			.print(LOG.global_plev())
			.get();
			
		auto eris_gen = dbcsr::tensor_create<3,double>()
			.name("eris_3")
			.pgrid(spgrid3)
			.map1({0}).map2({1,2})
			.blk_sizes(xbb)
			.get();
		
		eri_batched->set_generator(genfunc);

		vec<int> map1 = {0};
		vec<int> map2 = {1,2};
		eri_batched->compress_init({0},map1,map2);
		
		vec<vec<int>> bounds(3);
		
		t_setup.finish();

		for (int ix = 0; ix != eri_batched->nbatches(0); ++ix) {
				
				bounds[0] = eri_batched->blk_bounds(0,ix);
				bounds[1] = eri_batched->full_blk_bounds(1);
				bounds[2] = eri_batched->full_blk_bounds(2);
				
				t_calc.start();
				if (mytype != dbcsr::btype::direct) m_aofac->ao_3c_fill(eris_gen,bounds,scr);
				t_calc.finish();
				eris_gen->filter(dbcsr::global::filter_eps);
				t_compress.start();
				eri_batched->compress({ix}, eris_gen);
				t_compress.finish();
		}
		
		eri_batched->compress_finalize();
		
		//auto eri = eri_batched->get_work_tensor();
		//dbcsr::print(*eri);
		
		t_eri_batched.finish();
		
		if (comp(key::coul_xbb)) m_reg.insert(key::coul_xbb, eri_batched);
		if (comp(key::erfc_xbb)) m_reg.insert(key::erfc_xbb, eri_batched);
		
		double eri_occupation = eri_batched->occupation() * 100;
		
		LOG.os<1>("Occupation of 3c2e integrals: ", eri_occupation, "%\n");
		LOG.os<>("Done computing 3c2e integrals.\n\n");
		
		if (eri_occupation > 100) throw std::runtime_error(
			"3c2e integrals occupation more than 100%");
	
		auto mat = dfit.compute_idx(eri_batched);
		
	}
	
	if (comp(key::dfit_coul_xbb) || comp(key::dfit_erfc_xbb)) {
		
		LOG.os<>("Computing fitting coefficients\n");
		
		auto& time = TIME.sub("Density fitting coefficients");
		time.start();
		
		key k_eri, k_inv;
		
		if (comp(key::dfit_coul_xbb)) {
			k_eri = key::coul_xbb;
			k_inv = key::coul_xx_inv;
		}
		if (comp(key::dfit_erfc_xbb)) {
			k_eri = key::erfc_xbb;
			k_inv = key::erfc_xx_inv;
		}
		
		//LOG.os<1>("Computing fitting coefficients.\n");
		
		auto eri_batched = m_reg.get<sbt3>(k_eri);
		auto inv = m_reg.get<smatd>(k_inv);
		
		auto c_xbb_batched = dfit.compute(eri_batched, inv, 
			m_opt.get<std::string>("intermeds", "core"));
			
		auto mat = dfit.compute_idx(c_xbb_batched);
			
		if (comp(key::dfit_coul_xbb)) m_reg.insert(key::dfit_coul_xbb, c_xbb_batched);
		if (comp(key::dfit_erfc_xbb)) m_reg.insert(key::dfit_erfc_xbb, c_xbb_batched);
		
		time.finish();
		
	}
	
	if (comp(key::dfit_pari_xbb)) {
		
		LOG.os<>("Computing fitting coefficients (PARI)\n");
		
		auto& time = TIME.sub("Density fitting coefficients (PARI)");
		time.start();
				
		auto eri_batched = m_reg.get<sbt3>(key::coul_xbb);
		auto m_xx = m_reg.get<smatd>(key::coul_xx);
		auto scr = m_reg.get<ints::shared_screener>(key::scr_xbb);
		
		auto c_xbb_pari = dfit.compute_pari(eri_batched, m_xx, scr);
		m_reg.insert(key::dfit_pari_xbb, c_xbb_pari);
	
		time.finish();
	
	}
	
	if (comp(key::dfit_qr_xbb)) {
		
		LOG.os<>("Computing fitting coefficients (QR)\n");
		
		auto& time = TIME.sub("Density fitting coefficients (QR)");
		time.start();
		
		int nbatches_x = m_opt.get<int>("nbatches_x", 5);
		int nbatches_b = m_opt.get<int>("nbatches_b", 5);
		
		std::array<int,3> bdims = {nbatches_x, nbatches_b, nbatches_b};
		
		//auto eri_batched = m_reg.get<sbt3>(key::coul_xbb);
		auto m_xx = m_reg.get<smatd>(key::coul_xx);
		auto s_xx_inv = m_reg.get<smatd>(key::ovlp_xx_inv);
		auto scr = m_reg.get<ints::shared_screener>(key::scr_xbb);
		
		auto c_xbb_qr = dfit.compute_qr(s_xx_inv, m_xx, spgrid3, scr, 
			bdims, dbcsr::btype::core, true);
		m_reg.insert(key::dfit_qr_xbb, c_xbb_qr);
		
		auto mat = dfit.compute_idx(c_xbb_qr);
	
		time.finish();
	
	}
	
	TIME.finish();
	LOG.os<>("Finished loading AO quantities.\n");
	
}
	
} // end namespace
