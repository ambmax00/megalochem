#include "fock/fockmod.h"
#include "ints/screening.h"
#include "utils/registry.h"
#include "math/linalg/LLT.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include "fock/fock_defaults.h"
#include <dbcsr_btensor.hpp>

namespace fock {
	
fockmod::fockmod (dbcsr::world iworld, desc::smolecule imol, desc::options iopt) :
	m_world(iworld),
	m_mol(imol),
	m_opt(iopt),
	LOG(m_world.comm(),m_opt.get<int>("print", FOCK_PRINT_LEVEL)),
	TIME(m_world.comm(), "Fock Builder", LOG.global_plev())
{
	// set up tensors
	auto b = m_mol->dims().b();
	m_f_bb_A = dbcsr::create<double>()
		.set_world(m_world)
		.name("f_bb_A")
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::symmetric)
		.get();
		
	if (m_mol->nele_alpha() != m_mol->nele_beta() && m_mol->nele_beta() != 0) {
		m_f_bb_B = dbcsr::create_template<double>(m_f_bb_A)
			.name("f_bb_B").get();
	}
	
}

void fockmod::init() {
	
	std::string j_method = m_opt.get<std::string>("build_J", FOCK_BUILD_J);
	std::string k_method = m_opt.get<std::string>("build_K", FOCK_BUILD_K);
	std::string metric = m_opt.get<std::string>("df_metric", FOCK_METRIC);
	std::string eris_mem = m_opt.get<std::string>("eris", FOCK_ERIS);
	
	bool compute_eris_batched = false;
	bool compute_3c2e_batched = false;
	bool compute_s_xx = false;
	bool s_xx_tensor = false;
	bool compute_s_xx_inv = false;
	bool compute_s_xx_invsqrt = false;
	
	// set J
	if (j_method == "exact") {
		
		J* builder = new EXACT_J(m_world, m_opt);
		m_J_builder.reset(builder);
		
		compute_eris_batched = true;
		
	} else if (j_method == "batchdf") {
		
		J* builder = new BATCHED_DF_J(m_world,m_opt);
		m_J_builder.reset(builder);
		
		compute_s_xx = true;
		compute_s_xx_inv = true;
		compute_3c2e_batched = true;
		
	} else {
		
		throw std::runtime_error("Unknown J method: " + j_method);
		
	}
		
	
	// set K
	if (k_method == "exact") {
		
		K* builder = new EXACT_K(m_world, m_opt);
		m_K_builder.reset(builder);
		
		compute_eris_batched = true;
		
	} else if (k_method == "batchdfao") {
		
		K* builder = new BATCHED_DFAO_K(m_world,m_opt);
		m_K_builder.reset(builder);
		
		compute_s_xx = true;
		compute_s_xx_inv = true;
		compute_3c2e_batched = true;
		
	} else if (k_method == "batchdfmo") {
		
		K* builder = new BATCHED_DFMO_K(m_world,m_opt);
		m_K_builder.reset(builder);
		
		compute_s_xx = true;
		compute_s_xx_invsqrt = true;
		compute_3c2e_batched = true;
		
	} else if (k_method == "batchpari") {
		
		K* builder = new BATCHED_PARI_K(m_world,m_opt);
		m_K_builder.reset(builder);
		
		compute_s_xx = true;
		compute_s_xx_inv = true;
		compute_3c2e_batched = true;
		s_xx_tensor = true;
		
	} else {
		
		throw std::runtime_error("Unknown K method: " + k_method);
		
	}
	
	LOG.os<>("Setting up JK builder.\n");
	LOG.os<>("J method: ", j_method, '\n');
	LOG.os<>("K method: ", k_method, '\n');
	
	std::shared_ptr<ints::aofactory> aofac =
		std::make_shared<ints::aofactory>(m_mol,m_world);
	
	//dbcsr::print(*m_p_A);
	
	m_J_builder->set_density_alpha(m_p_A);
	m_J_builder->set_density_beta(m_p_B);
	m_J_builder->set_coeff_alpha(m_c_A);
	m_J_builder->set_coeff_beta(m_c_B);
	m_J_builder->set_mol(m_mol);
	
	m_K_builder->set_density_alpha(m_p_A);
	m_K_builder->set_density_beta(m_p_B);
	m_K_builder->set_coeff_alpha(m_c_A);
	m_K_builder->set_coeff_beta(m_c_B);
	m_K_builder->set_mol(m_mol);
	
	// initialize pgrids
	
	if (compute_s_xx) {
		spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	}
	
	if (compute_3c2e_batched) {
		int nbf_b = m_mol->c_basis()->nbf();
		int nbf_x = m_mol->c_dfbasis()->nbf();
		std::array<int,3> tsizes = {nbf_x, nbf_b, nbf_b};
		
		spgrid3_xbb = dbcsr::create_pgrid<3>(m_world.comm()).tensor_dims(tsizes).get();
		
	}
	
	if (compute_eris_batched) {
		spgrid4 = dbcsr::create_pgrid<4>(m_world.comm()).get();
	}
	
	// initialize integrals depending on method combination

	dbcsr::shared_matrix<double> c_s_xx;
	
	if (compute_s_xx) {
		
		auto& t_eri = TIME.sub("Metric");
		LOG.os<1>("Computing metric...\n");
		
		t_eri.start();
		
		if (metric == "coulomb") {
		
			auto out = aofac->ao_2c2e("coulomb");
			out->filter(dbcsr::global::filter_eps);
			c_s_xx = out;
			
		} else {
			
			auto coul_metric = aofac->ao_2c2e("coulomb");
			auto other_metric = aofac->ao_2c2e("erfc_coulomb");
			
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
		
		//dbcsr::print(*c_s_xx);
		
		m_reg.insert_matrix<double>("s_xx", c_s_xx);
		LOG.os<1>("Finished computing metric.\n");
		
		t_eri.finish();
		
	}
	
	if (compute_s_xx_inv || compute_s_xx_invsqrt) {
		
		auto& t_inv = TIME.sub("Inverting metric");
		
		LOG.os<1>("Computing metric cholesky decomposition...\n");
		
		t_inv.start();
		
		auto s_xx = c_s_xx;
		
		math::LLT chol(s_xx, LOG.global_plev());
		chol.compute();
		
		auto x = m_mol->dims().x();
		auto Linv = chol.L_inv(x);
		
		arrvec<int,2> xx = {m_mol->dims().x(), m_mol->dims().x()};
		
		if (compute_s_xx_inv) {
			
			LOG.os<1>("Computing metric inverse...\n");
			
			auto c_s_xx_inv = dbcsr::create_template<double>(Linv)
				.name(m_mol->name() + "_s_xx_inv")
				.matrix_type(dbcsr::type::symmetric).get();
			
			dbcsr::multiply('T', 'N', *Linv, *Linv, *c_s_xx_inv).perform();
			
			c_s_xx_inv->filter(dbcsr::global::filter_eps);
			
			m_reg.insert_matrix<double>("s_xx_inv", c_s_xx_inv);
			
		}
		
		if (compute_s_xx_invsqrt) {
			
			LOG.os<1>("Computing metric inverse square root...\n");

			dbcsr::shared_matrix<double> c_s_xx_invsqrt = dbcsr::transpose(Linv).get();
			
			c_s_xx_invsqrt->filter(dbcsr::global::filter_eps);
				
			m_reg.insert_matrix<double>("s_xx_invsqrt", c_s_xx_invsqrt);
			
		}
		
		LOG.os<1>("Done with inverting.\n");
		t_inv.finish();
		
	}
	
	if (compute_eris_batched) {
		
		auto& t_ints = TIME.sub("Computing eris");
		LOG.os<1>("Computing 2e integrals.\n");
		
		t_ints.start();
		
		aofac->ao_eri_setup("coulomb");
		
		auto b = m_mol->dims().b();
		arrvec<int,4> bbbb = {b,b,b,b};
		
		int nbatches_b = m_opt.get<int>("nbatches_x", FOCK_NBATCHES_B);
		std::array<int,4> bdims = {nbatches_b,nbatches_b,nbatches_b,nbatches_b};
		
		auto eri_batched = dbcsr::btensor_create<4>()
			.name(m_mol->name() + "_eri_batched")
			.pgrid(spgrid4)
			.blk_sizes(bbbb)
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
				
				aofac->ao_eri_fill(eris_gen, bounds, nullptr);
				
				eris_gen->filter(dbcsr::global::filter_eps);
				
				eri_batched->compress({imu,inu},eris_gen);
				
			}
		}
			
		eri_batched->compress_finalize();
		
		t_ints.finish();
		
		m_reg.insert_btensor<4,double>("i_bbbb_batched", eri_batched);
		LOG.os<1>("Done computing 2e integrals.\n");
		
	}
	
	std::shared_ptr<ints::screener> scr_s;
	
	if (compute_3c2e_batched) {
		
		auto& t_screen = TIME.sub("3c2e screening");
		LOG.os<1>("Computing 3c2e integrals.\n");
		
		LOG.os<1>("Computing screening.\n");
		
		t_screen.start();
		
		scr_s.reset(new ints::schwarz_screener(aofac,metric));
		scr_s->compute();
				
		t_screen.finish();
		
		auto& t_eri_batched = TIME.sub("3c2e integrals batched");
		auto& t_calc = t_eri_batched.sub("calc");
		auto& t_setup = t_eri_batched.sub("setup");
		auto& t_compress = t_eri_batched.sub("Compress");
		
		t_eri_batched.start();
		t_setup.start();
		
		aofac->ao_3c2e_setup(metric);
		auto genfunc = aofac->get_generator(scr_s);
		
		dbcsr::btype mytype = dbcsr::get_btype(eris_mem);
		
		int nbatches_x = m_opt.get<int>("nbatches_x", FOCK_NBATCHES_X);
		int nbatches_b = m_opt.get<int>("nbatches_b", FOCK_NBATCHES_B);
		
		auto b = m_mol->dims().b();
		auto x = m_mol->dims().x();
		arrvec<int,3> xbb = {x,b,b};
		
		std::array<int,3> bdims = {nbatches_x,nbatches_b,nbatches_b};
		
		auto eri_batched = dbcsr::btensor_create<3>()
			.name(m_mol->name() + "_eri_batched")
			.pgrid(spgrid3_xbb)
			.blk_sizes(xbb)
			.batch_dims(bdims)
			.btensor_type(mytype)
			.print(LOG.global_plev())
			.get();
			
		auto eris_gen = dbcsr::tensor_create<3,double>()
			.name("eris_3")
			.pgrid(spgrid3_xbb)
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
				if (mytype != dbcsr::btype::direct) aofac->ao_3c2e_fill(eris_gen,bounds,scr_s);
				t_calc.finish();
				//dbcsr::print(*eri);
				eris_gen->filter(dbcsr::global::filter_eps);
				t_compress.start();
				eri_batched->compress({ix}, eris_gen);
				t_compress.finish();
		}
		
		eri_batched->compress_finalize();
		
		//auto eri = eri_batched->get_work_tensor();
		//dbcsr::print(*eri);
		
		t_eri_batched.finish();
		
		m_reg.insert_btensor<3,double>("i_xbb_batched", eri_batched);
		
		LOG.os<1>("Occupation of 3c2e integrals: ", eri_batched->occupation() * 100, "%\n");
		LOG.os<1>("Done computing 3c2e integrals.\n");
		
	}
	
	if (k_method == "batchdfao") {
		
		LOG.os<1>("Computing fitting coefficients.\n");
		
		auto eri_batched = m_reg.get_btensor<3,double>("i_xbb_batched");
		auto inv = m_reg.get_matrix<double>("s_xx_inv");
		m_dfit = std::make_shared<ints::dfitting>(m_world, m_mol, LOG.global_plev());
		auto c_xbb_batched = m_dfit->compute(eri_batched, inv, 
			m_opt.get<std::string>("intermeds", FOCK_INTERMEDS));
		m_reg.insert_btensor<3,double>("c_xbb_batched", c_xbb_batched);
	
	}
	
	if (k_method == "batchpari") {
		
		LOG.os<1>("Computing fitting coefficients.\n");
		
		auto eri_batched = m_reg.get_btensor<3,double>("i_xbb_batched");
		auto s_xx = m_reg.get_matrix<double>("s_xx");
		m_dfit = std::make_shared<ints::dfitting>(m_world, m_mol, LOG.global_plev());
		auto c_xbb_pari = m_dfit->compute_pari(eri_batched, s_xx, scr_s);
		m_reg.insert_tensor<3,double>("c_xbb_pari", c_xbb_pari);
	
	}
	
	m_J_builder->set_reg(m_reg);
	m_K_builder->set_reg(m_reg);
	
	m_J_builder->init();
	m_K_builder->init();
		
	LOG.os<>("Finished setting up JK builder \n \n");
	
}

void fockmod::compute(bool SAD_iter, int rank) {
	
	TIME.start();
	
	m_J_builder->set_SAD(SAD_iter,rank);
	m_K_builder->set_SAD(SAD_iter,rank);
	
	auto& t_j = TIME.sub("J builder");
	auto& t_k = TIME.sub("K builder");
	
	LOG.os<1>("Computing coulomb matrix.\n");
	t_j.start();
	m_J_builder->compute_J();
	t_j.finish();
		
	LOG.os<1>("Computing exchange matrix.\n");
	t_k.start();
	m_K_builder->compute_K();
	t_k.finish();
	
	auto Jmat = m_J_builder->get_J();
	
	auto KmatA = m_K_builder->get_K_A();
	auto KmatB = m_K_builder->get_K_B();
	
	m_f_bb_A->add(0.0,1.0,*m_core);
	m_f_bb_A->add(1.0,1.0,*Jmat);
	m_f_bb_A->add(1.0,1.0,*KmatA);
	//dbcsr::copy_tensor_to_matrix(*Jtensor,*m_f_bb_A,true);
	//dbcsr::copy_tensor_to_matrix(*KtensorA,*m_f_bb_A,true);
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_f_bb_A);
	}
	
	if (m_f_bb_B) {
		m_f_bb_B->add(0.0,1.0,*m_core);
		m_f_bb_B->add(1.0,1.0,*Jmat);
		m_f_bb_B->add(1.0,1.0,*KmatB);
		//dbcsr::copy_tensor_to_matrix(*Jtensor,*m_f_bb_B,true);
		//dbcsr::copy_tensor_to_matrix(*KtensorB,*m_f_bb_B,true);
		
		if (LOG.global_plev() >= 2) {
			dbcsr::print(*m_f_bb_B);
		}
	}
	
	TIME.finish();
		
}

} // end namespace
