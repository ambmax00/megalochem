#include "adc/adcmod.h"
#include "ints/aofactory.h"
#include "ints/screening.h"
#include "ints/gentran.h"
#include "math/linalg/LLT.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include <libint2/basis.h>
#include <Eigen/Eigenvalues>

namespace adc {
/*
void adcmod::mo_load() {
	
	LOG.os<>("Constructing MO integrals...\n");
	
	dbcsr::pgrid<2> grid2(m_comm);
	
	auto epso = m_hfwfn->eps_occ_A();
	auto epsv = m_hfwfn->eps_vir_A();
	
	m_mo.eps_o = epso;
	m_mo.eps_v = epsv;

	ints::aofactory aofac(*m_hfwfn->mol(), m_comm);
	
	LOG.os<>("-- Loading 3c2e AO integrals\n"); 
	auto aoints = aofac.op("coulomb").dim("xbb").map1({0}).map2({1,2}).compute<3>(); 
	
	LOG.os<>("-- Loading df overlap integrals\n");
	auto metric = aofac.op("coulomb").dim("xx").map1({0}).map2({1}).compute<2>();
	
	int nocc = m_hfwfn->mol()->nocc_alpha();
	int nvir = m_hfwfn->mol()->nvir_alpha();
	
	auto c_bo = m_hfwfn->c_bo_A();
	auto c_bv = m_hfwfn->c_bv_A();
	
	vec<int> d = {1};
	
	LOG.os<>("-- Inverting metric matrix and contracting...\n");
	auto PQSQRT = aofac.invert(metric,2);
	
	dbcsr::pgrid<3> grid3(m_comm);
	
	// contract X
	arrvec<int,3> xbb = {m_dims.x, m_dims.b, m_dims.b};
	
	dbcsr::stensor<3> d_xbb = dbcsr::make_stensor<3>(
		dbcsr::tensor<3>::create().name("i_xbb(xx)^-1/2").ngrid(grid3) 
			.map1({0}).map2({1,2}).blk_sizes(xbb)); 
			
	dbcsr::contract(*aoints, *PQSQRT, *d_xbb).move(true).perform("XMN, XY -> YMN");
	
	aoints.reset();
	
	dbcsr::print(*c_bo);
	dbcsr::print(*c_bv);
	
	LOG.os<>("-- AO -> MO transformation...\n");
	
	m_mo.b_xoo = ints::transform3(d_xbb, c_bo, c_bo, "i_xoo(xx)^-1/2");
	m_mo.b_xov = ints::transform3(d_xbb, c_bo, c_bv, "i_xov(xx)^-1/2");
	m_mo.b_xvv = ints::transform3(d_xbb, c_bv, c_bv, "i_xvv(xx)^-1/2");
	
	d_xbb->destroy();

	mo_amplitudes();
	
	LOG.os<>("Finished computing MO quantities.\n\n");
		
}*/
		
adcmod::adcmod(desc::shf_wfn hfref, desc::options& opt, dbcsr::world& w) :
	m_hfwfn(hfref), 
	m_opt(opt), 
	m_world(w),
	m_nroots(m_opt.get<int>("nroots", ADC_NROOTS)),
	m_diag_order(m_opt.get<int>("diag_order", m_order)),
	m_c_os(m_opt.get<double>("c_os", ADC_C_OS)),
	m_c_osc(m_opt.get<double>("c_os_coupling", ADC_C_OS_COUPLING)),
	LOG(w.comm(), m_opt.get<int>("print", ADC_PRINT_LEVEL)),
	TIME(w.comm(), "ADC Module", LOG.global_plev())
{
	
	LOG.banner("ADC MODULE",50,'*');
	
	auto mstr = m_opt.get<std::string>("method", ADC_METHOD);
	
	auto miter = method_map.find(mstr);
	if (miter == method_map.end()) {
		throw std::runtime_error("Invalid method for adc.");
	}
	
	m_method = miter->second;	
	
	std::string dfbasname = m_opt.get<std::string>("dfbasis");
	
	libint2::BasisSet dfbas(dfbasname,m_hfwfn->mol()->atoms());
	m_hfwfn->mol()->set_dfbasis(dfbas);
	
	init();
	
	LOG.os<>("--- Ready for launching computation. --- \n\n");
	
}

void adcmod::init() {
	
	// setup pgrids
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	int nbf = m_hfwfn->mol()->c_basis().nbf();
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
		case method::ri_adc_1:
			compute_3c2e = true;
			compute_metric_invsqrt = true;
			break;
		case method::ri_adc_2:
			compute_3c2e = true;
			compute_metric_invsqrt = true;
			break;
		case method::sos_ri_adc_2:
			compute_3c2e = true;
			compute_metric_invsqrt = true;
			break;
		case method::ao_ri_adc_1:
			compute_3c2e = true;
			compute_metric_inv = true;
			break;
		case method::ao_ri_adc_2:
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
	
	dbcsr::shared_matrix<double> c_s_xx;
	
	auto dimb = m_hfwfn->mol()->dims().b();
	auto dimx = m_hfwfn->mol()->dims().x();
	
	if (compute_metric_inv || compute_metric_invsqrt) {
		
		auto& t_eri = TIME.sub("Metric");
		
		t_eri.start();
		
		if (metric == "coulomb") {
		
			auto out = aofac->ao_3coverlap("coulomb");
			out->filter(dbcsr::global::filter_eps);
			c_s_xx = out;
			
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
			c_s_xx = coul_metric;
			
		}
		
		auto x = m_hfwfn->mol()->dims().x();
		arrvec<int,2> xx = {dimx,dimx};
		
		auto s_xx_01 = dbcsr::tensor_create<2>().name("s_xx")
			.pgrid(m_spgrid2).map1({0}).map2({1}).blk_sizes(xx).get();
			
		dbcsr::copy_matrix_to_tensor(*c_s_xx, *s_xx_01);
		m_reg.insert_tensor<2,double>("s_xx", s_xx_01);		
			
		t_eri.finish();
		
	}
	
	if (compute_metric_inv || compute_metric_invsqrt) {
		
		auto& t_inv = TIME.sub("Inverting metric");
		
		LOG.os<1>("Computing metric cholesky decomposition...\n");
		
		t_inv.start();
		
		auto s_xx = c_s_xx;
		
		math::LLT chol(s_xx, LOG.global_plev());
		chol.compute();
		
		auto x = m_hfwfn->mol()->dims().x();
		auto Linv = chol.L_inv(x);
		
		arrvec<int,2> xx = {dimx,dimx};
		
		//math::hermitian_eigen_solver sol(s_xx, 'V', true);
		//sol.compute();
		
		if (compute_metric_inv) {
			
			LOG.os<1>("Computing metric inverse...\n");
			
			auto c_s_xx_inv = dbcsr::create_template<double>(Linv)
				.name("s_xx_inv")
				.matrix_type(dbcsr::type::symmetric).get();
			
			dbcsr::multiply('T', 'N', *Linv, *Linv, *c_s_xx_inv)
				.filter_eps(dbcsr::global::filter_eps)
				.perform();
				
			//auto c_s_xx_inv = sol.inverse();
			
			m_reg.insert_matrix<double>("s_xx_inv_mat", c_s_xx_inv); 
			
			auto s_xx_inv = dbcsr::tensor_create<2>().name("s_xx_inv")
				.pgrid(m_spgrid2).map1({0}).map2({1}).blk_sizes(xx).get();
			
			dbcsr::copy_matrix_to_tensor(*c_s_xx_inv, *s_xx_inv);
			
			s_xx_inv->filter(dbcsr::global::filter_eps);
			
			//dbcsr::print(*c_s_xx);
			//dbcsr::print(*s_xx_inv);
			
			m_reg.insert_tensor<2,double>("s_xx_inv", s_xx_inv);
			
		}
		
		if (compute_metric_invsqrt) {
			
			LOG.os<1>("Computing metric inverse square root...\n");

			std::string name = "s_xx_invsqrt_(0|1)";
			
			//dbcsr::print(*Linv);
			//auto M = dbcsr::matrix_to_eigen(s_xx);
			//Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> SAES(M);

			//Eigen::MatrixXd inv = SAES.operatorInverseSqrt();
			
			dbcsr::shared_matrix<double> c_s_xx_invsqrt = dbcsr::transpose(Linv).get();
			//auto c_s_xx_invsqrt = dbcsr::eigen_to_matrix(inv, 
			//	m_world, "inv", dimx, dimx, dbcsr::type::symmetric); // sol.inverse_sqrt();
			
			auto s_xx_invsqrt = dbcsr::tensor_create<2>().name("s_xx_invsqrt")
				.pgrid(m_spgrid2).map1({0}).map2({1}).blk_sizes(xx).get();
				
			dbcsr::copy_matrix_to_tensor(*c_s_xx_invsqrt, *s_xx_invsqrt);
			
			s_xx_invsqrt->filter(dbcsr::global::filter_eps);
				
			m_reg.insert_tensor<2,double>("s_xx_invsqrt", s_xx_invsqrt);
			
		}
		
		t_inv.finish();
		
	}
	
	if (compute_s_bb) {
		
		auto s = aofac->ao_overlap();
		m_reg.insert_matrix<double>("s_bb", s);
		
	}
	
	if (compute_3c2e) {
		
		auto& t_screen = TIME.sub("3c2e screening");
		
		LOG.os<1>("Computing screening.\n");
		
		t_screen.start();
		
		std::shared_ptr<ints::screener> scr_s(
			new ints::schwarz_screener(aofac,metric));
		scr_s->compute();
				
		t_screen.finish();
		
		auto& t_eri_batched = TIME.sub("3c2e integrals batched");
		
		t_eri_batched.start();
		
		aofac->ao_3c2e_setup(metric);
		auto eri = aofac->ao_3c2e_setup_tensor(m_spgrid3_xbb, 
			vec<int>{0}, vec<int>{1,2});
		auto genfunc = aofac->get_generator(scr_s);
		
		dbcsr::btype mytype = dbcsr::get_btype(eris_mem);
		
		int nbatches_x = m_opt.get<int>("nbatches_x", ADC_NBATCHES_X);
		int nbatches_b = m_opt.get<int>("nbatches_b", ADC_NBATCHES_B);
		
		std::array<int,3> bdims = {nbatches_x,nbatches_b,nbatches_b};
		
		auto eribatch = dbcsr::btensor_create<3>(eri)
			.batch_dims(bdims)
			.btensor_type(mytype)
			.print(LOG.global_plev())
			.get();
		
		eribatch->set_generator(genfunc);
		
		auto xfullblkbounds = eribatch->full_blk_bounds(0);
		auto mufullblkbounds = eribatch->full_blk_bounds(1);
		auto nublkbounds = eribatch->blk_bounds(2);
		
		eribatch->compress_init({2});
		
		vec<vec<int>> bounds(3);
		
		for (int inu = 0; inu != nublkbounds.size(); ++inu) {
				
				bounds[0] = xfullblkbounds;
				bounds[1] = mufullblkbounds;
				bounds[2] = nublkbounds[inu];
				
				if (mytype != dbcsr::btype::direct) aofac->ao_3c2e_fill(eri,bounds,scr_s);
				
				//dbcsr::print(*eri);
				eri->filter(dbcsr::global::filter_eps);
				
				eribatch->compress({inu}, eri);
		}
		
		eribatch->compress_finalize();
		
		t_eri_batched.finish();
		
		m_reg.insert_btensor<3,double>("i_xbb_batched", eribatch);
		
		LOG.os<1>("Occupation of 3c2e integrals: ", eribatch->occupation() * 100, "%\n");
		
	}
	
	auto po = m_hfwfn->po_bb_A();
	auto pv = m_hfwfn->pv_bb_A();
	
	m_reg.insert_matrix<double>("po_bb", po);
	m_reg.insert_matrix<double>("pv_bb", pv);
	
	LOG.os<>("Finished setting up AO integral tensors.\n");
	
}

void adcmod::init_mo_tensors() {
	
	LOG.os<>("Setting up MO tensors.\n");
	
	bool compute_moints(false);
	
	switch(m_method) {
		case method::ri_adc_1:
			compute_moints = true;
			break;
		case method::ri_adc_2:
			compute_moints = true;
			break;
		case method::sos_ri_adc_2:
			compute_moints = true;
			break;
		default:
			compute_moints = false;
			break;
	}
	
	auto c_bo = m_hfwfn->c_bo_A();
	auto c_bv = m_hfwfn->c_bv_A();
	
	auto o = m_hfwfn->mol()->dims().oa();
	auto v = m_hfwfn->mol()->dims().va();
	auto b = m_hfwfn->mol()->dims().b();
	
	arrvec<int,2> bo = {b,o};
	arrvec<int,2> bv = {b,v};
	
	/*
	auto c_bo_01 = dbcsr::tensor_create<2,double>()
		.pgrid(m_spgrid2_bo).name("c_bo").map1({0})
		.map2({1}).blk_sizes(bo).get();
		
	auto c_bv_01 = dbcsr::tensor_create<2,double>()
		.pgrid(m_spgrid2_bv).name("c_bv").map1({0})
		.map2({1}).blk_sizes(bv).get();
		
	dbcsr::copy_matrix_to_tensor(*c_bo, *c_bo_01);
	dbcsr::copy_matrix_to_tensor(*c_bv, *c_bv_01);
	
	m_reg.insert_tensor<2,double>("c_bo",c_bo_01);
	m_reg.insert_tensor<2,double>("c_bv",c_bv_01);*/
	
	//dbcsr::print(*c_bo);
	//dbcsr::print(*c_bv);
	
	m_reg.insert_matrix<double>("c_bo", c_bo);
	m_reg.insert_matrix<double>("c_bv", c_bv);
	
	//auto cboe = dbcsr::matrix_to_eigen(c_bo);
	//auto cbve = dbcsr::matrix_to_eigen(c_bv);
	
	//LOG.os<>("OCC:", '\n', cboe, '\n');
	//LOG.os<>("VIR:", '\n', cbve, '\n');
	
	/*
	if (compute_moints) { 
		
		int nbatches = m_opt.get<int>("nbatches", ADC_NBATCHES);
	
		auto i_xbb_batched = m_reg.get_btensor<3,double>("i_xbb_batched");
		auto s_xx_invsqrt = m_reg.get_tensor<2,double>("s_xx_invsqrt");
		
		auto i_xoo_batched = ints::transform3(i_xbb_batched, c_bo_01,
			c_bo_01, m_spgrid3_xoo, 5, dbcsr::core, "i_xoo_batched");
			
		auto i_xov_batched = ints::transform3(i_xbb_batched, c_bo_01,
			c_bv_01, m_spgrid3_xov, 5, dbcsr::core, "i_xov_batched");
			
		auto i_xvv_batched = ints::transform3(i_xbb_batched, c_bv_01,
			c_bv_01, m_spgrid3_xvv, 5, dbcsr::core, "i_xvv_batched");
					
		auto i_xoo = i_xoo_batched->get_stensor();
		auto i_xov = i_xov_batched->get_stensor();
		auto i_xvv = i_xvv_batched->get_stensor();
		
		dbcsr::print(*i_xoo);
		
		auto d_xoo_batched = std::make_shared<dbcsr::btensor<3,double>>(
			i_xoo,nbatches,dbcsr::core,1);
			
		auto d_xov_batched = std::make_shared<dbcsr::btensor<3,double>>(
			i_xov,nbatches,dbcsr::core,1);
			
		auto d_xvv_batched = std::make_shared<dbcsr::btensor<3,double>>(
			i_xvv,nbatches,dbcsr::core,1);
		
		auto nxbatches = d_xov_batched->nbatches_dim(0);
		auto nobatches = d_xov_batched->nbatches_dim(1);
		auto nvbatches = d_xov_batched->nbatches_dim(2);
		
		auto xbounds = d_xov_batched->bounds(0);
		auto obounds = d_xov_batched->bounds(1);
		auto vbounds = d_xov_batched->bounds(2);
		
		auto d_xoo = d_xoo_batched->get_stensor();
		auto d_xov = d_xov_batched->get_stensor();
		auto d_xvv = d_xvv_batched->get_stensor();
				
		for (int ix = 0; ix != xbounds.size(); ++ix) {
			for (int io = 0; io != obounds.size(); ++io) {
				
				vec<vec<int>> x_b = {
					xbounds[ix]
				};
				
				vec<vec<int>> oo_b = {
					d_xoo_batched->full_bounds(1),
					obounds[io]
				};
				
				
					
			//}
		//}
		
		dbcsr::contract(*s_xx_invsqrt, *i_xoo, *d_xoo)
					//.bounds2(x_b)
					//.bounds3(oo_b)
					//.beta(1.0)
					.perform("XY, Yij -> Xij");
					
		dbcsr::contract(*s_xx_invsqrt, *i_xov, *d_xov)
					//.bounds2(x_b)
					//.bounds3(oo_b)
					//.beta(1.0)
					.perform("XY, Yij -> Xij");
					
		dbcsr::contract(*s_xx_invsqrt, *i_xvv, *d_xvv)
					//.bounds2(x_b)
					//.bounds3(oo_b)
					//.beta(1.0)
					.perform("XY, Yij -> Xij");
			
		dbcsr::print(*d_xoo);	
		
		m_reg.insert_btensor<3,double>("d_xoo", d_xoo_batched);
		m_reg.insert_btensor<3,double>("d_xov", d_xov_batched);
		m_reg.insert_btensor<3,double>("d_xvv", d_xvv_batched);
		
	}*/
	
}


} // end namespace
