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
	dbcsr::mat_d fA = dbcsr::matrix<>::create().set_world(m_world).name("f_bb_A")
		.row_blk_sizes(b).col_blk_sizes(b).type(dbcsr_type_symmetric);
	
	m_f_bb_A = fA.get_smatrix();
	
	if (m_mol->nele_alpha() != m_mol->nele_beta() && m_mol->nele_beta() != 0) {
		dbcsr::mat_d fB = dbcsr::matrix<>::create_template(*m_f_bb_A).name("f_bb_B");
		m_f_bb_B = fB.get_smatrix();
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
		
	}  else if (k_method == "batchdfmo") {
		
		K* builder = new BATCHED_DFMO_K(m_world,m_opt);
		m_K_builder.reset(builder);
		
		compute_s_xx = true;
		compute_s_xx_invsqrt = true;
		compute_3c2e_batched = true;
		
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
	m_J_builder->set_factory(aofac);
	
	m_K_builder->set_density_alpha(m_p_A);
	m_K_builder->set_density_beta(m_p_B);
	m_K_builder->set_coeff_alpha(m_c_A);
	m_K_builder->set_coeff_beta(m_c_B);
	m_K_builder->set_factory(aofac);
	
	// initialize pgrids
	
	if (compute_s_xx) {
		spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	}
	
	if (compute_3c2e_batched) {
		int nbf_b = m_mol->c_basis().nbf();
		int nbf_x = m_mol->c_dfbasis()->nbf();
		std::array<int,3> tsizes = {nbf_x, nbf_b, nbf_b};
		
		spgrid3_xbb = dbcsr::create_pgrid<3>(m_world.comm()).tensor_dims(tsizes).get();
		
	}
	
	if (compute_eris_batched) {
		spgrid4 = dbcsr::create_pgrid<4>(m_world.comm()).get();
	}
	
	// initialize integrals depending on method combination

	dbcsr::smatrix<double> c_s_xx;
	
	if (compute_s_xx) {
		
		auto& t_eri = TIME.sub("Metric");
		
		t_eri.start();
		
		if (metric == "coulomb") {
		
			auto out = aofac->ao_3coverlap("coulomb");
			out->filter();
			c_s_xx = out;
			
		} else {
			
			auto coul_metric = aofac->ao_3coverlap("coulomb");
			auto other_metric = aofac->ao_3coverlap(metric);
			
			math::hermitian_eigen_solver solver(coul_metric, 'V');
			solver.compute();
			auto coul_inv = solver.inverse();
			
			coul_metric->clear();
			
			dbcsr::mat_d temp = dbcsr::mat_d::create_template(*coul_metric)
				.name("temp").type(dbcsr_type_no_symmetry);
				
			dbcsr::multiply('N', 'N', *other_metric, *coul_inv, temp).perform();
			dbcsr::multiply('N', 'N', temp, *other_metric, *coul_metric).perform();
			
			coul_metric->filter();
			c_s_xx = coul_metric;
			
		}
		
		auto x = m_mol->dims().x();
		arrvec<int,2> xx = {x,x};
		
		auto s_xx_01 = dbcsr::tensor_create<2>().name("s_xx")
			.pgrid(spgrid2).map1({0}).map2({1}).blk_sizes(xx).get();
			
		dbcsr::copy_matrix_to_tensor(*c_s_xx, *s_xx_01);
		m_reg.insert_tensor<2,double>("s_xx", s_xx_01);		
			
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
			
			dbcsr::mat_d s = dbcsr::mat_d::create_template(*Linv)
				.name(m_mol->name() + "_s_xx_inv")
				.type(dbcsr_type_symmetric);
			auto c_s_xx_inv = s.get_smatrix();
			
			dbcsr::multiply('T', 'N', *Linv, *Linv, *c_s_xx_inv).perform();
			
			auto s_xx_inv = dbcsr::tensor_create<2>().name("s_xx_inv")
				.pgrid(spgrid2).map1({0}).map2({1}).blk_sizes(xx).get();
				
			dbcsr::copy_matrix_to_tensor(*c_s_xx_inv, *s_xx_inv);
			
			//dbcsr::print(*c_s_xx);
			//dbcsr::print(*s_xx_inv);
			
			m_reg.insert_tensor<2,double>("s_xx_inv", s_xx_inv);
			
		}
		
		if (compute_s_xx_invsqrt) {
			
			LOG.os<1>("Computing metric inverse square root...\n");

			std::string name = m_mol->name() + "_s_xx_invsqrt_(0|1)";
			
			//dbcsr::print(*Linv);
			
			dbcsr::mat_d Linv_t = dbcsr::mat_d::transpose(*Linv);
			auto c_s_xx_invsqrt = Linv_t.get_smatrix();
			
			auto s_xx_invsqrt = dbcsr::tensor_create<2>().name("s_xx_invsqrt")
				.pgrid(spgrid2).map1({0}).map2({1}).blk_sizes(xx).get();
				
			dbcsr::copy_matrix_to_tensor(*c_s_xx_invsqrt, *s_xx_invsqrt);
				
			m_reg.insert_tensor<2,double>("s_xx_invsqrt", s_xx_invsqrt);
			
		}
		
		t_inv.finish();
		
	}
	
	if (compute_eris_batched) {
		
		auto& t_ints = TIME.sub("Computing eris");
		
		t_ints.start();
		
		aofac->ao_eri_setup("coulomb");
		
		auto eris = aofac->ao_eri_setup_tensor(spgrid4, vec<int>{0,1}, vec<int>{2,3});
		
		int nbatches = m_opt.get<int>("nbatches", 4);
		
		dbcsr::sbtensor<4,double> eri_batched =
			std::make_shared<dbcsr::btensor<4,double>>(eris,nbatches,dbcsr::core,50);
					
		eri_batched->compress_init({0,1});
		
		vec<vec<int>> bounds(4);
		
		for (int imu = 0; imu != eri_batched->nbatches_dim(0); ++imu) {
			for (int inu = 0; inu != eri_batched->nbatches_dim(1); ++inu) {
				
				bounds[0] = eri_batched->blk_bounds(0)[imu];
				bounds[1] = eri_batched->blk_bounds(1)[inu];
				bounds[2] = eri_batched->full_blk_bounds(2);
				bounds[3] = eri_batched->full_blk_bounds(3);
				
				aofac->ao_eri_fill(eris, bounds, nullptr);
				
				eri_batched->compress({imu,inu},eris);
				
			}
		}
			
		eri_batched->compress_finalize();
		
		t_ints.finish();
		
		m_reg.insert_btensor<4,double>("i_bbbb_batched", eri_batched);
		
	}
	
	
	if (compute_3c2e_batched) {
		
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
		auto eri = aofac->ao_3c2e_setup_tensor(spgrid3_xbb, vec<int>{0}, vec<int>{1,2});
		auto genfunc = aofac->get_generator(scr_s);
		
		dbcsr::btype mytype = dbcsr::core;
		
		if (eris_mem == "direct") mytype = dbcsr::direct;
		if (eris_mem == "disk") mytype = dbcsr::disk; 
		
		int nbatches = m_opt.get<int>("nbatches", 4);
		
		dbcsr::sbtensor<3,double> eribatch = 
			std::make_shared<dbcsr::btensor<3,double>>(eri,nbatches,mytype,50);
		
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
				
				if (mytype != dbcsr::direct) aofac->ao_3c2e_fill(eri,bounds,scr_s);
				
				//dbcsr::print(*eri);
				
				eribatch->compress({inu}, eri);
		}
		
		eribatch->compress_finalize();
		
		t_eri_batched.finish();
		
		m_reg.insert_btensor<3,double>("i_xbb_batched", eribatch);
		
		LOG.os<1>("Occupation of 3c2e integrals: ", eribatch->occupation() * 100, "%\n");
		
	}
	
	m_J_builder->set_reg(m_reg);
	m_K_builder->set_reg(m_reg);
	
	m_J_builder->init();
	m_K_builder->init();
	
	m_J_builder->init_tensors();
	m_K_builder->init_tensors();
	
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
