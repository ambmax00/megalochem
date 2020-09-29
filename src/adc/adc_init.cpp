#include "adc/adcmod.h"
#include "ints/aofactory.h"
#include "ints/screening.h"
#include "ints/gentran.h"
#include "math/linalg/LLT.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include <libint2/basis.h>
#include <Eigen/Eigenvalues>

namespace adc {
		
adcmod::adcmod(hf::shared_hf_wfn hfref, desc::options& opt, dbcsr::world& w) :
	m_hfwfn(hfref), 
	m_opt(opt), 
	m_world(w),
	m_nroots(m_opt.get<int>("nroots", ADC_NROOTS)),
	m_diag_order(m_opt.get<int>("diag_order", m_order)),
	m_c_os(m_opt.get<double>("c_os", ADC_C_OS)),
	m_c_osc(m_opt.get<double>("c_os_coupling", ADC_C_OS_COUPLING)),
	LOG(w.comm(), m_opt.get<int>("print", ADC_PRINT_LEVEL)),
	TIME(w.comm(), "ADC Module", LOG.global_plev()),
	m_jmethod(m_opt.get<std::string>("build_J", ADC_BUILD_J)),
	m_kmethod(m_opt.get<std::string>("build_K", ADC_BUILD_K)),
	m_zmethod(m_opt.get<std::string>("build_Z", ADC_BUILD_Z)),
{
	
	LOG.banner("ADC MODULE",50,'*');
	
	auto mstr = m_opt.get<std::string>("method", ADC_METHOD);
	
	auto miter = method_map.find(mstr);
	if (miter == method_map.end()) {
		throw std::runtime_error("Invalid method for adc.");
	}
	
	m_method = miter->second;	
	
	std::string dfbasname = m_opt.get<std::string>("dfbasis");
	int nsplit = m_hfwfn->mol()->c_basis()->nsplit();
	std::string splitmethod = m_hfwfn->mol()->c_basis()->split_method();
	auto atoms = m_hfwfn->mol()->atoms();
	
	auto dfbasis = std::make_shared<desc::cluster_basis>(
		dfbasname, atoms, splitmethod, nsplit);
	m_hfwfn->mol()->set_cluster_dfbasis(dfbasis);
	
	init();
	
	LOG.os<>("--- Ready for launching computation. --- \n\n");
	
}

void adcmod::init() {
	
	// setup pgrids
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	int nbf = m_hfwfn->mol()->c_basis()->nbf();
	int nocc = m_hfwfn->mol()->nocc_alpha();
	int nvir = m_hfwfn->mol()->nvir_alpha();
	int xnbf = m_hfwfn->mol()->c_dfbasis()->nbf();
	
	std::array<int,2> bosizes = {nbf,nocc};
	std::array<int,2> bvsizes = {nbf,nvir};
	
	m_spgrid2_bo = dbcsr::create_pgrid<2>(m_world.comm())
		.tensor_dims(bosizes).get();
		
	m_spgrid2_bv = dbcsr::create_pgrid<2>(m_world.comm())
		.tensor_dims(bvsizes).get();

	std::array<int,3> xbbsizes = {xnbf,nbf,nbf};
	
	m_spgrid3_xbb = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(xbbsizes).get();
	
}

void adcmod::init_ao_tensors() {
	
	LOG.os<>("Setting up AO integral tensors.\n");
	
	// setting up what we need
	bool compute_3c2e(false), 
		compute_metric_inv(false), 
		compute_metric_invsqrt(false),
		compute_s_bb(false);
		
	switch(m_method) {
		case method::ao_adc_1:
			compute_3c2e = true;
			compute_metric_inv = true;
			break;
		case method::ao_adc_2:
			compute_3c2e = true;
			compute_metric_inv = true;
		#if 0
			compute_metric_invsqrt = true;
		#endif
			compute_s_bb = true;
			break;
	}
	
	std::string metric = m_opt.get<std::string>("metric", ADC_METRIC);
	std::string eris_mem = m_opt.get<std::string>("eris", ADC_ERIS);
	
	std::shared_ptr<ints::aofactory> aofac = 
		std::make_shared<ints::aofactory>(m_hfwfn->mol(), m_world);
	
	dbcsr::shared_matrix<double> ao3c2e_overlap, s_bb,
		ao3c2e_overlap_inv, ao3c2e_overlap_invsqrt;
	dbcsr::sbtensor<3,double> eri_batched;
	ints::shared_screener s_scr;
	
	auto dimb = m_hfwfn->mol()->dims().b();
	auto dimx = m_hfwfn->mol()->dims().x();
	
	if (compute_metric_inv || compute_metric_invsqrt) {
		
		auto& t_eri = TIME.sub("Metric");
		
		t_eri.start();
		
		if (metric == "coulomb") {
		
			auto out = aofac->ao_3coverlap("coulomb");
			out->filter(dbcsr::global::filter_eps);
			ao3c2e_overlap = out;
			
		} else {
			
			auto coul_metric = aofac->ao_3coverlap("coulomb");
			auto other_metric = aofac->ao_3coverlap(metric);
			
			math::hermitian_eigen_solver solver(coul_metric, 'V');
			solver.compute();
			auto coul_inv = solver.inverse();
			
			coul_metric->clear();
			
			dbcsr::shared_matrix<double> temp = 
				dbcsr::create_template<double>(coul_metric)
				.name("temp")
				.matrix_type(dbcsr::type::no_symmetry).get();
				
			dbcsr::multiply('N', 'N', *other_metric, *coul_inv, *temp).perform();
			dbcsr::multiply('N', 'N', *temp, *other_metric, *coul_metric).perform();
			
			//coul_metric->filter(dbcsr::global::filter_eps);
			ao3c2e_overlap = coul_metric;
			
		}	
			
		t_eri.finish();
		
	}
	
	if (compute_metric_inv || compute_metric_invsqrt) {
		
		auto& t_inv = TIME.sub("Inverting metric");
		
		LOG.os<1>("Computing metric cholesky decomposition...\n");
		
		t_inv.start();
		
		auto s_xx = ao3c2e_overlap;
		
		math::LLT chol(s_xx, LOG.global_plev());
		chol.compute();
		
		auto x = m_hfwfn->mol()->dims().x();
		auto Linv = chol.L_inv(x);
		
		arrvec<int,2> xx = {dimx,dimx};
		
		if (compute_metric_inv) {
			
			LOG.os<1>("Computing metric inverse...\n");
			
			auto c_s_xx_inv = dbcsr::create_template<double>(Linv)
				.name("s_xx_inv")
				.matrix_type(dbcsr::type::symmetric).get();
			
			dbcsr::multiply('T', 'N', *Linv, *Linv, *c_s_xx_inv)
				.filter_eps(dbcsr::global::filter_eps)
				.perform();
										
			c_s_xx_inv->filter(dbcsr::global::filter_eps);
			
			ao3c2e_overlap_inv = c_s_xx_inv;

		}
		
		if (compute_metric_invsqrt) {
			
			LOG.os<1>("Computing metric inverse square root...\n");

			std::string name = "s_xx_invsqrt_(0|1)";
			
			dbcsr::shared_matrix<double> c_s_xx_invsqrt = dbcsr::transpose(Linv).get();
			
			c_s_xx_invsqrt->filter(dbcsr::global::filter_eps);
				
			ao3c2e_overlap_invsqrt = c_s_xx_invsqrt;
			
		}
		
		t_inv.finish();
		
	}
	
	if (compute_s_bb) {
		
		s_bb = aofac->ao_overlap();
		
	}
	
	if (compute_3c2e) {
		
		auto& t_screen = TIME.sub("3c2e screening");
		
		LOG.os<1>("Computing screening.\n");
		
		t_screen.start();
		
		s_scr.reset(new ints::schwarz_screener(aofac,metric));
		s_scr->compute();
				
		t_screen.finish();
		
		auto& t_eri_batched = TIME.sub("3c2e integrals batched");
		
		t_eri_batched.start();
		
		aofac->ao_3c2e_setup(metric);
		
		auto genfunc = aofac->get_generator(scr_s);
		
		dbcsr::btype mytype = dbcsr::get_btype(eris_mem);
		
		int nbatches_x = m_opt.get<int>("nbatches_x", ADC_NBATCHES_X);
		int nbatches_b = m_opt.get<int>("nbatches_b", ADC_NBATCHES_B);
		
		std::array<int,3> bdims = {nbatches_x,nbatches_b,nbatches_b};
		
		eri_batched = dbcsr::btensor_create<3>()
			.pgrid(m_spgrid3_xbb)
			.blk_sizes(xbb)
			.batch_dims(bdims)
			.btensor_type(mytype)
			.print(LOG.global_plev())
			.get();
			
		auto eri_hold = dbcsr::tensor_create<3,double>()
			.name("eri_calc")
			.pgrid(m_spgrid3_xbb)
			.blk_sizes(xbb)
			.map1({0}).map2({1,2})
			.get();
		
		eri_batched->set_generator(genfunc);
		
		eri_batched->compress_init({0}, vec<int>{0}, vec<int>{1,2});
		
		vec<vec<int>> bounds(3);
		
		for (int ix = 0; ix != eri_batched->nbatches(0); ++ix) {
				
				bounds[0] = eri_batched->blk_bounds(0,ix);
				bounds[1] = eri_batched->full_blk_bounds(1);
				bounds[2] = eri_batched->full_blk_bounds(2);
				
				if (mytype != dbcsr::btype::direct) aofac->ao_3c2e_fill(eri_hold,bounds,s_scr);
				
				eri_hold->filter(dbcsr::global::filter_eps);
				
				eri_batched->compress({ix}, eri_hold);
		}
		
		eribatch->compress_finalize();
		
		t_eri_batched.finish();
				
		LOG.os<1>("Occupation of 3c2e integrals: ", eri_batched->occupation() * 100, "%\n");
			
	}
	
	m_dfit = std::make_shared<ints::dfitting>(
				m_world, m_hfwfn->mol(), LOG.global_plev());
	auto intermeds = m_opt.get<std::string>("intermeds", ADC_INTERMEDS);
	
	// fitting coefficients
	if (m_kmethod == "batchdfao") {
		auto cfit = m_dfit->compute(eri_batched, ao3c2e_overlap_inv, intermeds);
		m_reg.insert_btensor<3,double>("c_xbb_batched", cfit);
	} else if (m_kmethod == "batchpari") {
		auto cfit = m_dfit->compute_pari(eri_batched, ao3c2e_overlap, s_scr);
		m_reg.insert_tensor<3,double>("c_xbb_pari", cfit);
	}
	
	if (eri_batched) m_reg.insert_btensor<3,double>("i_xbb_batched", eri_batched);
	if (ao3c2e_overlap_inv) m_reg.insert_matrix<double>("s_xx_inv", ao3c2e_overlap_inv);
	if (ao3c2e_overlap_invsqrt) 
		m_reg.insert_matrix<double>("s_xx_invsqrt", ao3c2e_oberlap_invsqrt);
	if (ao3c2e_overlap) m_reg.insert_matrix<double>("s_xx", ao3c2e_overlap);
	if (s_scr) m_reg.insert_screener("screener", s_scr);
	if (s_bb) m_reg.insert_matrix<double>("s_bb", s_bb);
	
	auto po = m_hfwfn->po_bb_A();
	auto pv = m_hfwfn->pv_bb_A();
	
	m_reg.insert_matrix<double>("po_bb", po);
	m_reg.insert_matrix<double>("pv_bb", pv);
	
	LOG.os<>("Finished setting up AO integral tensors.\n");
	
}

void adcmod::init_mo_tensors() {
	
	LOG.os<>("Setting up MO tensors.\n");
	
	auto c_bo = m_hfwfn->c_bo_A();
	auto c_bv = m_hfwfn->c_bv_A();
	
	auto o = m_hfwfn->mol()->dims().oa();
	auto v = m_hfwfn->mol()->dims().va();
	auto b = m_hfwfn->mol()->dims().b();
	
	arrvec<int,2> bo = {b,o};
	arrvec<int,2> bv = {b,v};
	
	m_reg.insert_matrix<double>("c_bo", c_bo);
	m_reg.insert_matrix<double>("c_bv", c_bv);
	
}


} // end namespace
