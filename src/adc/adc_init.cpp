#include "adc/adcmod.h"
#include "ints/aoloader.h"
#include "ints/screening.h"
#include "math/linalg/LLT.h"
#include "locorb/locorb.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include <Eigen/Eigenvalues>

#include <type_traits>

namespace adc {
		
adcmod::adcmod(dbcsr::world w, hf::shared_hf_wfn hfref, desc::options& opt) :
	m_hfwfn(hfref), 
	m_opt(opt), 
	m_world(w),
	m_ao(w, hfref->mol(), opt),
	LOG(w.comm(), m_opt.get<int>("print", ADC_PRINT_LEVEL)),
	TIME(w.comm(), "ADC Module", LOG.global_plev())
{
	
	LOG.banner("ADC MODULE",50,'*');
	
	std::string dfbasname = m_opt.get<std::string>("dfbasis");
	int nsplit = m_hfwfn->mol()->c_basis()->nsplit();
	std::string splitmethod = m_hfwfn->mol()->c_basis()->split_method();
	auto atoms = m_hfwfn->mol()->atoms();
	
	bool augmented = m_opt.get<bool>("df_augmentation", false);
	auto dfbasis = std::make_shared<desc::cluster_basis>(
		dfbasname, atoms, splitmethod, nsplit, augmented);
	m_hfwfn->mol()->set_cluster_dfbasis(dfbasis);
	
	init_ao_tensors();
	
	LOG.os<>("--- Ready for launching computation. --- \n\n");
	
}

void adcmod::init_ao_tensors() {
	
	LOG.os<>("Setting up AO integral tensors.\n");
		
	// LOAD ALL ADC1 INTEGRALS
	auto adc1_opt = m_opt.subtext("adc1");
		
	auto jstr_adc1 = adc1_opt.get<std::string>("jmethod", ADC_ADC1_JMETHOD);
	auto kstr_adc1 = adc1_opt.get<std::string>("kmethod", ADC_ADC1_KMETHOD);
	auto mstr_adc1 = adc1_opt.get<std::string>("df_metric", ADC_ADC1_DF_METRIC);
	
	std::cout << "K: " << mstr_adc1 << std::endl;
	
	auto jmet_adc1 = fock::str_to_jmethod(jstr_adc1);
	auto kmet_adc1 = fock::str_to_kmethod(kstr_adc1);
	auto metr_adc1 = ints::str_to_metric(mstr_adc1);
	
	//fock::load_jints(jmet_adc1, metr_adc1, m_ao);
	//fock::load_kints(kmet_adc1, metr_adc1, m_ao);
	
	bool do_adc2 = m_opt.get<bool>("do_adc2",ADC_DO_ADC2);
	
	if (do_adc2) {
		
		auto adc2_opt = m_opt.subtext("adc2");
		bool local = adc2_opt.get<bool>("local", ADC_ADC2_LOCAL);
		
		// if not local, we are postponing integral evaluation for later
		if (!local) {
			auto jstr_adc2 = adc2_opt.get<std::string>("jmethod", ADC_ADC2_JMETHOD);
			auto kstr_adc2 = adc2_opt.get<std::string>("kmethod", ADC_ADC2_KMETHOD);
			auto zstr_adc2 = adc2_opt.get<std::string>("zmethod", ADC_ADC2_ZMETHOD);
			auto mstr_adc2 = adc2_opt.get<std::string>("df_metric", ADC_ADC2_DF_METRIC);
		
			auto jmet_adc2 = fock::str_to_jmethod(jstr_adc2);
			auto kmet_adc2 = fock::str_to_kmethod(kstr_adc2);
			auto zmet_adc2 = mp::str_to_zmethod(zstr_adc2);
			
			auto metr_adc2 = ints::str_to_metric(mstr_adc2);
		
			fock::load_jints(jmet_adc2, metr_adc2, m_ao);
			fock::load_kints(kmet_adc2, metr_adc2, m_ao);
		
		} else {
			
			m_ao.request(ints::key::coul_xx, true);
		
		}	
		
		// overlap always needed
		m_ao.request(ints::key::ovlp_bb, true);
		
	}
	
	m_ao.request(ints::key::ovlp_bb, true);
	int nprint = LOG.global_plev();

	m_ao.compute();
	
}

std::shared_ptr<MVP> adcmod::create_adc1() {
	
	dbcsr::shared_matrix<double> v_xx;
	dbcsr::sbtensor<3,double> eri3c2e, fitting;
	
	auto jmeth = fock::str_to_jmethod(m_opt.get<std::string>("adc1/jmethod", ADC_ADC1_JMETHOD));
	auto kmeth = fock::str_to_kmethod(m_opt.get<std::string>("adc1/kmethod", ADC_ADC1_KMETHOD));
	auto metr = ints::str_to_metric(m_opt.get<std::string>("adc1/df_metric", ADC_ADC1_DF_METRIC));
	
	std::cout << "K: " << m_opt.get<std::string>("adc1/df_metric", ADC_ADC1_DF_METRIC) << std::endl;
	
	auto aoreg = m_ao.get_registry();
	
	auto get = [&aoreg](auto& tensor, ints::key aokey) {
		if (aoreg.present(aokey)) {
			tensor = aoreg.get<typename std::remove_reference<decltype(tensor)>::type>(aokey);
		} else {
			std::cout << "NOT PRESENT :" << static_cast<int>(aokey) << std::endl;
			tensor = nullptr;
		}
	};
	
	switch (metr) {
		case ints::metric::coulomb:
		{
			get(eri3c2e, ints::key::coul_xbb);
			get(fitting, ints::key::dfit_coul_xbb);
			get(v_xx, ints::key::coul_xx_inv);
			break;
		}
		case ints::metric::erfc_coulomb:
		{
			get(eri3c2e, ints::key::erfc_xbb);
			get(fitting, ints::key::dfit_erfc_xbb);
			get(v_xx, ints::key::erfc_xx_inv);
			break;
		}
		case ints::metric::pari:
		{
			get(eri3c2e, ints::key::pari_xbb);
			if (!eri3c2e) std::cout << "NULL" << std::endl;
			get(fitting, ints::key::dfit_pari_xbb);
			get(v_xx, ints::key::coul_xx);
			break;
		}
		case ints::metric::qr_fit:
		{
			get(eri3c2e, ints::key::qr_xbb);
			get(fitting, ints::key::dfit_qr_xbb);
			get(v_xx, ints::key::coul_xx);
			break;
		}
	}
	
	auto ptr = create_MVP_AOADC1(m_world, m_hfwfn->mol(), LOG.global_plev())
		.c_bo(m_hfwfn->c_bo_A())
		.c_bv(m_hfwfn->c_bv_A())
		.eps_occ(*m_hfwfn->eps_occ_A())
		.eps_vir(*m_hfwfn->eps_vir_A())
		.eri3c2e_batched(eri3c2e)
		.fitting_batched(fitting)
		.v_xx(v_xx)
		.jmethod(jmeth)
		.kmethod(kmeth)
		.get();
		
	ptr->init();
	
	return ptr;
	
}

std::shared_ptr<MVP> adcmod::create_adc2(
	std::optional<std::vector<int>> atom_list) {
	
	dbcsr::shared_matrix<double> v_xx, s_bb, c_bo, c_bv;
	dbcsr::sbtensor<3,double> eri3c2e, fitting;
	
	auto jmeth = fock::str_to_jmethod(m_opt.get<std::string>("adc2/jmethod", ADC_ADC2_JMETHOD));
	auto kmeth = fock::str_to_kmethod(m_opt.get<std::string>("adc2/kmethod", ADC_ADC2_KMETHOD));
	auto zmeth = mp::str_to_zmethod(m_opt.get<std::string>("adc2/zmethod", ADC_ADC2_ZMETHOD));
	auto mytype = dbcsr::get_btype(m_opt.get<std::string>("adc2/intermeds", ADC_ADC2_INTERMEDS));
	double c_os = m_opt.get<double>("adc2/c_os", ADC_ADC2_C_OS);
	double c_os_coupling = m_opt.get<double>("adc2/c_os_coupling", ADC_ADC2_C_OS_COUPLING);
	int nlap = m_opt.get<int>("adc2/nlap", ADC_ADC2_NLAP);
	
	auto metr = ints::str_to_metric(m_opt.get<std::string>("adc2/df_metric", ADC_ADC2_DF_METRIC));
	auto aoreg = m_ao.get_registry();
	
	auto get = [&aoreg](auto& tensor, ints::key aokey) {
		if (aoreg.present(aokey)) {
			tensor = aoreg.get<typename std::remove_reference<decltype(tensor)>::type>(aokey);
		} else {
			tensor = nullptr;
		}
	};
	
	if (atom_list) {
		
		// ====== SCREENER =========
		auto mol = m_hfwfn->mol();

		ints::atomic_screener* ascr = new ints::atomic_screener(m_world, m_hfwfn->mol(), *atom_list);
		
		ascr->compute();
		
		auto blklist_b = ascr->blklist_b();
		auto blklist_x = ascr->blklist_x();
		
		ints::screener* scr_ptr = ascr;
		ints::shared_screener s_scr;
		s_scr.reset(scr_ptr);
				
		// ====== TRUNCATED 3C2E ERIS ========
		
		LOG.os<>("Computing truncated 3c2e integrals for ADC(2).\n");
		
		auto& t_eri_batched = TIME.sub("3c2e truncated integrals batched");
		
		t_eri_batched.start();
		
		ints::aofactory aofac(mol, m_world);
		
		aofac.ao_3c2e_setup(ints::metric::coulomb);
		auto genfunc = aofac.get_generator(s_scr);
		
		std::string eris_mem = m_opt.get<std::string>("adc2/eris", ADC_ADC2_ERIS);
		dbcsr::btype mytype = dbcsr::get_btype(eris_mem);
		
		int nbatches_x = m_opt.get<int>("nbatches_x", 5);
		int nbatches_b = m_opt.get<int>("nbatches_b", 5);
		
		auto b = mol->dims().b();
		auto x = mol->dims().x();
		arrvec<int,3> xbb = {x,b,b};
		
		std::array<int,3> bdims = {nbatches_x,nbatches_b,nbatches_b};
		
		auto blkmap_b = mol->c_basis()->block_to_atom(mol->atoms());
		auto blkmap_x = mol->c_dfbasis()->block_to_atom(mol->atoms());
		
		arrvec<int,3> blkmaps = {blkmap_x, blkmap_b, blkmap_b};
		
		auto spgrid3 = aoreg.get<dbcsr::shared_pgrid<3>>(ints::key::pgrid3);
		auto eri_batched = dbcsr::btensor_create<3>()
			.name(mol->name() + "_eritrunc_batched")
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
		
		for (int ix = 0; ix != eri_batched->nbatches(0); ++ix) {
				
				bounds[0] = eri_batched->blk_bounds(0,ix);
				bounds[1] = eri_batched->full_blk_bounds(1);
				bounds[2] = eri_batched->full_blk_bounds(2);
				
				if (mytype != dbcsr::btype::direct) aofac.ao_3c_fill(eris_gen,bounds,s_scr);
				//eris_gen->filter(dbcsr::global::filter_eps);
				eri_batched->compress({ix}, eris_gen);
		}
		
		eri_batched->compress_finalize();
				
		t_eri_batched.finish();
	
		double eri_occupation = eri_batched->occupation() * 100;
		
		LOG.os<1>("Occupation of truncated 3c2e integrals: ", eri_occupation, "%\n");
		LOG.os<>("Done computing truncated 3c2e integrals.\n\n");
		
		if (eri_occupation > 100) throw std::runtime_error(
			"3c2e integrals occupation more than 100%");
			
		eri3c2e = eri_batched;
		
		// ======== TRUNCATED METRIC ============
		
		int rank = m_world.rank();
		
		dbcsr::shared_matrix<double> m_xx, m_xx_trunc;
		
		get(m_xx, ints::key::coul_xx);
		m_xx_trunc = dbcsr::create_template<double>(*m_xx)
			.name("metric truncated")
			.get();
		
		std::vector<int> row_x, col_x;
		
		for (int iy = 0; iy != x.size(); ++iy) {
			if (!blklist_x[iy]) continue;
			for (int ix = 0; ix <= iy; ++ix) {
				if (!blklist_x[ix]) continue;
				if (m_xx_trunc->proc(ix,iy) != rank) continue;
				
				row_x.push_back(ix);
				col_x.push_back(iy);
			}
		}
		
		m_xx_trunc->reserve_blocks(row_x, col_x);
		m_xx_trunc->copy_in(*m_xx, true);
		//m_xx_trunc->filter(dbcsr::global::filter_eps);
				
		math::hermitian_eigen_solver solver(m_xx_trunc, 'V');
		solver.compute();
		
		v_xx = solver.inverse();
		v_xx->filter(dbcsr::global::filter_eps);
		
//		dbcsr::print(*v_xx);
				
		// ========== TRUNCATED OVERLAP ============
		
		dbcsr::shared_matrix<double> s_bb_full, s_bb_trunc;		
		get(s_bb_full, ints::key::ovlp_bb);
		
		s_bb_trunc = dbcsr::create_template<double>(*s_bb_full)
			.name("overlap truncated")
			.get();
		
		std::vector<int> row_b, col_b;
		
		for (int in = 0; in != b.size(); ++in) {
			//if (!blklist_b[in]) continue;
			for (int im = 0; im <= in; ++im) {
				//if (!blklist_b[im]) continue;
				if (s_bb_trunc->proc(im,in) != rank) continue;
				row_b.push_back(im);
				col_b.push_back(in);
			}
		}
		
		s_bb_trunc->reserve_blocks(row_b, col_b);
		s_bb_trunc->copy_in(*s_bb_full, true);
		s_bb_trunc->filter(dbcsr::global::filter_eps);
		
		s_bb = s_bb_trunc;
		
//		dbcsr::print(*s_bb);
		
		// ============= TRUNCATED COEFFICIENTS =============
		
		auto c_bo_full = m_hfwfn->c_bo_A();
		auto c_bv_full = m_hfwfn->c_bv_A();
		
		c_bo = dbcsr::create_template<double>(*c_bo_full)
			.name("truncated c_bo")
			.get();
			
		c_bv = dbcsr::create_template<double>(*c_bv_full)
			.name("truncated c_bv")
			.get();
		
		auto o = mol->dims().oa();
		auto v = mol->dims().va();
		
		std::vector<int> row_o, col_o, row_v, col_v;
	
		for (int ib = 0; ib != b.size(); ++ib) {
			//if (!blklist_b[ib]) continue;
			for (int io = 0; io != o.size(); ++io) {
				if (c_bo->proc(ib,io) != rank) continue;
				row_o.push_back(ib);
				col_o.push_back(io);
			}
			for (int iv = 0; iv != v.size(); ++iv) {
				if (c_bv->proc(ib,iv) != rank) continue;
				row_v.push_back(ib);
				col_v.push_back(iv);
			}
		}
		
		c_bo->reserve_blocks(row_o, col_o);
		c_bv->reserve_blocks(row_v, col_v);
			
		c_bo->copy_in(*c_bo_full, true);
		c_bv->copy_in(*c_bv_full, true);
		
		c_bo->filter(dbcsr::global::filter_eps);
		c_bv->filter(dbcsr::global::filter_eps);
		
//		dbcsr::print(*c_bo_full);
		
//		dbcsr::print(*c_bo);
		
//		dbcsr::print(*c_bv);
		
		// ======== TRUNCATED FITTING COEFFS ==================
		
		if (kmeth == fock::kmethod::dfao) {
			ints::dfitting dfit(m_world, mol, LOG.global_plev());
			fitting = dfit.compute(eri3c2e, v_xx, mytype);
		}
		
	} else {
	
		switch (metr) {
			case ints::metric::coulomb:
			{
				get(eri3c2e, ints::key::coul_xbb);
				get(fitting, ints::key::dfit_coul_xbb);
				get(v_xx, ints::key::coul_xx_inv);
				break;
			}
			case ints::metric::erfc_coulomb:
			{
				get(eri3c2e, ints::key::erfc_xbb);
				get(fitting, ints::key::dfit_erfc_xbb);
				get(v_xx, ints::key::erfc_xx_inv);
				break;
			}
			case ints::metric::pari:
			{
				get(eri3c2e, ints::key::pari_xbb);
				get(fitting, ints::key::dfit_pari_xbb);
				get(v_xx, ints::key::coul_xx);
				break;
			}
			case ints::metric::qr_fit:
			{
				get(eri3c2e, ints::key::qr_xbb);
				get(fitting, ints::key::dfit_qr_xbb);
				get(v_xx, ints::key::coul_xx);
				break;
			}
		}
		
		s_bb = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::ovlp_bb);
		c_bo = m_hfwfn->c_bo_A();
		c_bv = m_hfwfn->c_bv_A();
	}
	
	auto ptr = create_MVP_AOADC2(m_world, m_hfwfn->mol(), LOG.global_plev())
		.c_bo(c_bo)
		.c_bv(c_bv)
		.s_bb(s_bb)
		.eps_occ(m_hfwfn->eps_occ_A())
		.eps_vir(m_hfwfn->eps_vir_A())
		.eri3c2e_batched(eri3c2e)
		.fitting_batched(fitting)
		.v_xx(v_xx)
		.jmethod(jmeth)
		.kmethod(kmeth)
		.zmethod(zmeth)
		.btype(mytype)
		.nlap(nlap)
		.c_os(c_os)
		.c_os_coupling(c_os_coupling)
		.get();
				
	ptr->init();
	
	return ptr;
	
}

} // end namespace
