#include "fock/fockmod.h"
#include "ints/screening.h"
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
	bool direct = m_opt.get<bool>("direct", FOCK_DIRECT);
	
	bool compute_eris = false;
	bool compute_3c2e = false;
	bool compute_3c2e_batched = false;
	bool compute_s_xx = false;
	bool compute_s_xx_inv = false;
	bool compute_s_xx_invsqrt = false;
	bool setup_btensor = false;
	
	// set J
	if (j_method == "exact") {
		
		J* builder = new EXACT_J(m_world, m_opt);
		m_J_builder.reset(builder);
		
		compute_eris = true;
		
	} else if (j_method == "df") {
		
		J* builder = new DF_J(m_world, m_opt);
		m_J_builder.reset(builder);
		
		compute_3c2e = true;
		compute_s_xx = true;
		compute_s_xx_inv = true;
		
	} else if (j_method == "batchdf") {
		
		J* builder = new BATCHED_DF_J(m_world,m_opt,metric,direct);
		m_J_builder.reset(builder);
		
		compute_s_xx = true;
		compute_s_xx_inv = true;
		setup_btensor = true;
		if (!direct) compute_3c2e_batched = true;
		
	}
		
	
	// set K
	if (k_method == "exact") {
		
		K* builder = new EXACT_K(m_world, m_opt);
		m_K_builder.reset(builder);
		
		compute_eris = true;
		
	} else if (k_method == "df") {
		
		K* builder = new DF_K(m_world,m_opt);
		m_K_builder.reset(builder);
		
		compute_3c2e = true;
		compute_s_xx = true;
		compute_s_xx_invsqrt = true;
		
	} else if (k_method == "batchdfmo") {
		
		K* builder = new BATCHED_DFMO_K(m_world,m_opt,metric,direct);
		m_K_builder.reset(builder);
		
		compute_s_xx = true;
		compute_s_xx_invsqrt = true;
		setup_btensor = true;
		if (!direct) compute_3c2e_batched = true;
		
	} else if (k_method == "batchdfao") {
		
		K* builder = new BATCHED_DFAO_K(m_world,m_opt,metric,direct);
		m_K_builder.reset(builder);
		
		compute_s_xx = true;
		compute_s_xx_inv = true;
		setup_btensor = true;
		if (!direct) compute_3c2e_batched = true;
		
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
	
	// initialize integrals depending on method combination
	
	ints::registry reg;
	
	if (compute_s_xx) {
		
		auto& t_eri = TIME.sub("Metric");
		
		t_eri.start();
		
		if (metric == "coulomb") {
		
			auto out = aofac->ao_3coverlap("coulomb");
			out->filter();
			reg.insert_matrix<double>(m_mol->name() + "_s_xx", out);
			
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
			reg.insert_matrix<double>(m_mol->name() + "_s_xx", coul_metric);
			
		}
			
		t_eri.finish();
		
	}
	
	if (compute_s_xx_inv || compute_s_xx_invsqrt) {
		
		auto& t_inv = TIME.sub("Inverting metric");
		
		LOG.os<1>("Computing metric cholesky decomposition...\n");
		
		t_inv.start();
		
		auto s_xx = reg.get_matrix<double>(m_mol->name() + "_s_xx");
		
		math::LLT chol(s_xx, LOG.global_plev());
		chol.compute();
		
		auto x = m_mol->dims().x();
		auto Linv = chol.L_inv(x);
		
		dbcsr::pgrid<2> grid2(m_world.comm());
		arrvec<int,2> xx = {m_mol->dims().x(), m_mol->dims().x()};
		
		if (compute_s_xx_inv) {
			
			LOG.os<1>("Computing metric inverse...\n");
			
			dbcsr::mat_d s = dbcsr::mat_d::create_template(*Linv)
				.name(m_mol->name() + "_s_xx_inv")
				.type(dbcsr_type_symmetric);
			auto s_xx_inv = s.get_smatrix();
			
			dbcsr::multiply('T', 'N', *Linv, *Linv, *s_xx_inv).perform();
			
			std::string name = m_mol->name() + "_s_xx_inv_(0|1)";
			
			dbcsr::stensor2_d s_xx_inv_01 = dbcsr::make_stensor<2>(
				dbcsr::tensor2_d::create().name(name).ngrid(grid2)
				.map1({0}).map2({1}).blk_sizes(xx));
				
			reg.insert_matrix<double>(s_xx_inv->name(),s_xx_inv);
				
			dbcsr::copy_matrix_to_tensor(*s_xx_inv, *s_xx_inv_01); 
			
			s_xx_inv_01->filter();
			reg.insert_tensor<2,double>(name, s_xx_inv_01);
			
		}
		
		if (compute_s_xx_invsqrt) {
			
			LOG.os<1>("Computing metric inverse square root...\n");

			std::string name = m_mol->name() + "_s_xx_invsqrt_(0|1)";
			
			//dbcsr::print(*Linv);
			
			dbcsr::mat_d Linv_t = dbcsr::mat_d::transpose(*Linv);
			
			dbcsr::stensor2_d s_xx_invsqrt_01 = dbcsr::make_stensor<2>(
				dbcsr::tensor2_d::create().name(name).ngrid(grid2)
				.map1({0}).map2({1}).blk_sizes(xx));
			
			dbcsr::copy_matrix_to_tensor(Linv_t, *s_xx_invsqrt_01);
			Linv_t.clear();
			
			//dbcsr::print(*s_xx_invsqrt_01);
			s_xx_invsqrt_01->filter();
			reg.insert_tensor<2,double>(name, s_xx_invsqrt_01);
			
		}
		
		t_inv.finish();
		
	}
	
	if (compute_eris) {
		
		auto eris = aofac->ao_eri(vec<int>{0,1},vec<int>{2,3});
		reg.insert_tensor<4,double>(m_mol->name() + "_i_bbbb_(01|23)", eris);
		
	}
	
	if (compute_3c2e) {
		
		auto& t_screen = TIME.sub("3c2e screening");
		
		LOG.os<1>("Computing screening.\n");
		
		t_screen.start();
		
		ints::screener* scr = new ints::schwarz_screener(aofac, metric);
		auto shscr = ints::shared_screener(scr);
		
		shscr->compute();
		
		t_screen.finish();
		
		LOG.os<1>("Computing 3c2e inetgrals...\n");
		
		auto& t_eri = TIME.sub("3c2e integrals");
		t_eri.start();
		
		dbcsr::stensor3_d out = aofac->ao_3c2e(vec<int>{0}, vec<int>{1,2},metric,shscr.get());
		
		reg.insert_tensor<3,double>(m_mol->name() + "_i_xbb_(0|12)", out);
		reg.insert_screener(m_mol->name() + "_schwarz_screener",shscr);
		
		t_eri.finish();
		
	}
	
	if (setup_btensor || compute_3c2e_batched) {
		
		auto& t_screen = TIME.sub("3c2e screening");
		
		LOG.os<1>("Computing screening.\n");
		
		t_screen.start();
		
		ints::screener* scr = new ints::schwarz_screener(aofac,metric);
		std::shared_ptr<ints::screener> scr_s(scr);
		scr->compute();
		
		reg.insert_screener(m_mol->name() + "_schwarz_screener", scr_s);
		
		t_screen.finish();
		
		auto& t_eri_batched = TIME.sub("3c2e integrals batched");
		
		aofac->ao_3c2e_setup(metric);
		
		auto eri = aofac->ao_3c2e_setup_tensor(vec<int>{0}, vec<int>{1,2});
		
		tensor::sbatchtensor<3,double> eribatch = 
			std::make_shared<tensor::batchtensor<3,double>>
				(eri,tensor::global::default_batchsize,LOG.global_plev());
				
		dbcsr::btensor<3,double> eris(eri,4,dbcsr::disk,50);
		
		auto xbounds = eris.bounds(0);
		auto nubounds = eris.bounds(1);
		auto mubounds = eris.bounds(2);
		auto xblkbounds = eris.blk_bounds(0);
		auto nublkbounds = eris.blk_bounds(1);
		auto mublkbounds = eris.blk_bounds(2);
		
		auto nufullbounds = eris.full_bounds(1);
		auto nufullblkbounds = eris.full_blk_bounds(1);
		
		eris.compress_init(eri, {0,2});
		
		vec<vec<int>> bounds(3);
		
		for (int ix = 0; ix != xbounds.size(); ++ix) {
			for (int imu = 0; imu != mubounds.size(); ++imu) {
				
				bounds[0] = xblkbounds[ix];
				bounds[1] = nufullblkbounds;
				bounds[2] = mublkbounds[imu];
				
				aofac->ao_3c2e_fill(eri,bounds,scr);
				dbcsr::print(*eri);
				
				eris.compress({ix,imu});
				
			}
		}
		
		eris.compress_finalize();
		
		eris.decompress_init(eri,{0,2});
		
		for (int ix = 0; ix != xbounds.size(); ++ix) {
			for (int imu = 0; imu != mubounds.size(); ++imu) {
				
				eris.decompress({ix,imu});
				
				dbcsr::print(*eri);
				eri->clear();
				
			}
		}
		
		eris.decompress_finalize();
				
		exit(0);
		
		eribatch->setup_batch();
		eribatch->set_batch_dim(vec<int>{0});
		
		eribatch->create_file();
		
		if (compute_3c2e_batched) {
			
			LOG.os<1>("Computing batched 3c2e integrals.\n");
			
			t_eri_batched.start();
		
			int nbatches = eribatch->nbatches();
			
			vec<vec<int>> bounds(3);
			for (int i = 0; i != nbatches; ++i) {
				
				bounds[0] = eribatch->bounds_blk(i,0);
				bounds[1] = eribatch->bounds_blk(i,1);
				bounds[2] = eribatch->bounds_blk(i,2);
				
				aofac->ao_3c2e_fill(eri,bounds,scr);
				
				//dbcsr::print(*eri);
				
				eribatch->write(i);
				
				eribatch->clear_batch();
				
			}
			
			LOG.os<1>("Occupancy of 3c2e integrals: ", eri->occupation()*100, "%\n");
			t_eri_batched.finish();
			
		}
		
		reg.insert_btensor<3,double>(m_mol->name() + "_i_xbb_(0|12)_batched", eribatch);
		
	}	
	
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
