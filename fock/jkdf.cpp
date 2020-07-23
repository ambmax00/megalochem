#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include "math/linalg/LLT.h"
#include <dbcsr_tensor_ops.hpp>

namespace fock {

BATCHED_DF_J::BATCHED_DF_J(dbcsr::world& w, desc::options& iopt) 
	: J(w,iopt) {} 

void BATCHED_DF_J::init_tensors() {
	
	// initialize tensors
	dbcsr::pgrid<2> grid2(m_world.comm());
	dbcsr::pgrid<3> grid3(m_world.comm());
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	vec<int> d = {1};
	
	arrvec<int,3> bbd = {b,b,d};
	arrvec<int,2> xd = {x,d};
	
	m_gp_xd = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().ngrid(grid2).name("c_x")
		.map1({0}).map2({1}).blk_sizes(xd));
	
	m_gq_xd = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create_template(*m_gp_xd).name("c2_x"));
	
	m_J_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create().ngrid(grid3).name("J dummy")
		.map1({0,1}).map2({2}).blk_sizes(bbd));
	
	m_ptot_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create_template(*m_J_bbd).name("ptot dummy"));
	
	m_inv = m_reg.get_tensor<2,double>(m_mol->name() + "_s_xx_inv_(0|1)");
	
	m_eri_batched = m_reg.get_btensor<3,double>(m_mol->name() + "_i_xbb_(0|12)_batched");
	
	m_scr = m_reg.get_screener(m_mol->name() + "_schwarz_screener");
	
}

void BATCHED_DF_J::compute_J() {
	
	auto& con1 = TIME.sub("first contraction");
	auto& con2 = TIME.sub("second contraction");
	auto& fetch1 = TIME.sub("fetch ints (1)");
	auto& fetch2 = TIME.sub("fetch ints (2)");
	
	TIME.start();
	
	// copy over density
	
	dbcsr::mat_d ptot = dbcsr::mat_d::create_template(*m_p_A).name("ptot");
	
	if (m_p_A && !m_p_B) {
		ptot.copy_in(*m_p_A);
		ptot.scale(2.0);
		dbcsr::copy_matrix_to_3Dtensor_new(ptot,*m_ptot_bbd,true);
		ptot.clear();
	} else {
		ptot.copy_in(*m_p_A);
		ptot.add(1.0, 1.0, *m_p_B);
		dbcsr::copy_matrix_to_3Dtensor_new<double>(ptot,*m_ptot_bbd,true);
		ptot.clear();
	}
	
	//dbcsr::print(*m_ptot_bbd);
	
	m_gp_xd->batched_contract_init();
	m_ptot_bbd->batched_contract_init();
		
	m_eri_batched->decompress_init({2});
	
	auto eri_0_12 = m_eri_batched->get_stensor();
	
	auto x_full_b = m_eri_batched->full_bounds(0);
	auto mu_full_b = m_eri_batched->full_bounds(1);
	auto nu_b = m_eri_batched->bounds(2);
	
	for (int inu = 0; inu != nu_b.size(); ++inu) {
			
		fetch1.start();
		m_eri_batched->decompress({inu});
		fetch1.finish();
		
		con1.start();
		
		vec<vec<int>> bounds1 = {
			mu_full_b,
			nu_b[inu]
		};
		
		dbcsr::contract(*eri_0_12, *m_ptot_bbd, *m_gp_xd)
			.bounds1(bounds1).beta(1.0)
			.perform("XMN, MN_ -> X_");
					
		con1.finish();
			
	}
	
	m_eri_batched->decompress_finalize();
	
	m_gp_xd->batched_contract_finalize();
	m_ptot_bbd->batched_contract_finalize();
		
	//dbcsr::print(*m_gp_xd);
	
	LOG.os<1>("X_, XY -> Y_\n");
	
	dbcsr::contract(*m_gp_xd, *m_inv, *m_gq_xd).perform("X_, XY -> Y_");
	
	//dbcsr::print(*m_gq_xd);
	
	m_J_bbd->batched_contract_init();
	m_gq_xd->batched_contract_init();
	m_eri_batched->decompress_init({2});
	
	for (int inu = 0; inu != nu_b.size(); ++inu) {
			
		fetch2.start();
		m_eri_batched->decompress({inu});
		fetch2.finish();
	
		con2.start();
			
		vec<vec<int>> bounds3 = {
			mu_full_b,
			nu_b[inu]
		};
	
		dbcsr::contract(*m_gq_xd, *eri_0_12, *m_J_bbd)
			.bounds3(bounds3).beta(1.0)
			.perform("X_, XMN -> MN_");
					
		con2.finish();
				
	}
	
	m_eri_batched->decompress_finalize();
	
	m_J_bbd->batched_contract_finalize();
	m_gq_xd->batched_contract_finalize();
	
	LOG.os<1>("Copy over...\n");
	
	dbcsr::copy_3Dtensor_to_matrix_new(*m_J_bbd, *m_J);
	
	m_J_bbd->clear();
	m_gp_xd->clear();
	m_gq_xd->clear();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_J);
	}
	
	TIME.finish();
	
}

/*
BATCHED_DFMO_K::BATCHED_DFMO_K(dbcsr::world& w, desc::options& opt) 
	: K(w,opt) {}

void BATCHED_DFMO_K::init_tensors() {
	
	auto b = m_p_A->row_blk_sizes();
	arrvec<int,2> bb = {b,b};
	
	dbcsr::pgrid<2> grid2(m_world.comm());
	
	m_K_01 = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().ngrid(grid2).name("K_01")
		.map1({0}).map2({1}).blk_sizes(bb));
		
	m_invsqrt = m_reg.get_tensor<2,double>(m_mol->name() + "_s_xx_invsqrt_(0|1)");
	
	m_eri_batched = m_reg.get_btensor<3,double>(m_mol->name() + "_i_xbb_(0|12)_batched");
	
	m_scr = m_reg.get_screener(m_mol->name() + "_schwarz_screener");
	
}

void BATCHED_DFMO_K::fetch_integrals(int ibatch, dbcsr::stensor3_d& reo_ints) {
	
	if (!m_direct) {
		
		m_eri_batched->read(ibatch);
		auto eri = m_eri_batched->get_stensor();
		
		dbcsr::copy(*eri, *reo_ints).move_data(true).perform();
		
	} else {
		
		m_factory->ao_3c2e_setup(m_metric);
		
		vec<vec<int>> bounds = {
			m_eri_batched->bounds_blk(ibatch,0),
			m_eri_batched->bounds_blk(ibatch,1),
			m_eri_batched->bounds_blk(ibatch,2)
		};
		
		m_factory->ao_3c2e_fill(reo_ints, bounds, m_scr);
		
	}
	
}
			
void BATCHED_DFMO_K::return_integrals(dbcsr::stensor3_d& reo_ints) {
	
	int nbatches = m_eri_batched->nbatches();
	if (m_direct || nbatches != 1) {
		reo_ints->clear();
	} else {
		auto eri = m_eri_batched->get_stensor();
		dbcsr::copy(*reo_ints,*eri).move_data(true).perform();
	}
	
}

void BATCHED_DFMO_K::compute_K() {
	
	TIME.start();
	
	auto b = m_mol->dims().b();
	auto X = m_mol->dims().x();
	
	dbcsr::pgrid<2> grid2(m_world.comm());
	dbcsr::pgrid<3> grid3(m_world.comm());
	
	auto compute_K_single = 
	[&] (dbcsr::smat_d& c_bm, dbcsr::smat_d& k_bb, std::string x) {
		
		auto& fetch1 = TIME.sub("Fetch ints " + x);
		auto& retints = TIME.sub("Returning ints " + x);
		auto& con1 = TIME.sub("Contraction (1) " + x);
		auto& con2 = TIME.sub("Contraction (2) " + x);
		auto& con3 = TIME.sub("Contraction (3) " + x);
		auto& reo1 = TIME.sub("Reordering (1) " + x);
		auto& reo2 = TIME.sub("Reordering (2) " + x);
	
		vec<int> o, m;
		k_bb->clear();
		
		if (m_SAD_iter) {
			m = c_bm->col_blk_sizes();
			o = m;
		} else {
			m = c_bm->col_blk_sizes();
			o = (x == "A") ? m_mol->dims().oa() : m_mol->dims().ob();
		}
		
		arrvec<int,2> bm = {b,m};
		arrvec<int,3> xbm = {X,b,m};
		arrvec<int,3> xbo = {X,b,o};
		arrvec<int,3> xbb = {X,b,b};
		
		m_c_bm = dbcsr::make_stensor<2>(
			dbcsr::tensor2_d::create().ngrid(grid2)
			.name("c_bm_" + x + "_0_1").map1({0}).map2({1})
			.blk_sizes(bm));
			
		dbcsr::copy_matrix_to_tensor(*c_bm, *m_c_bm);
			
		//dbcsr::print(*c_bm);
		//dbcsr::print(*m_c_bm);	
			
		m_INTS_01_2 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create().name("INTS_xbb_01_2")
			.ngrid(grid3).map1({0,1}).map2({2}).blk_sizes(xbb));
				
		m_HT1_xbm_01_2 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create().name("HT1_xbm_01_2_" + x)
			.ngrid(grid3).map1({0,1}).map2({2}).blk_sizes(xbm));
			
		m_HT1_xbm_0_12 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create().name("HT1_xbm_0_12_" + x)
			.ngrid(grid3).map1({0}).map2({1,2}).blk_sizes(xbm));
			
		m_HT2_xbm_0_12 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create().name("HT2_xbm_0_12_" + x)
			.ngrid(grid3).map1({0}).map2({1,2}).blk_sizes(xbm));
			
		m_HT2_xbm_01_2 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create().name("HT2_xbm_01_2_" + x)
			.ngrid(grid3).map1({0,2}).map2({1}).blk_sizes(xbm));
			
		m_dummy_xbo_01_2 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create().name("dummy_xbo_01_2_" + x)
			.ngrid(grid3).map1({0,1}).map2({2}).blk_sizes(xbo));
			
		m_dummy_batched_xbo_01_2 = std::make_shared<tensor::batchtensor<3,double>>
			(m_dummy_xbo_01_2,tensor::global::default_batchsize,LOG.global_plev());
			
		// setup batching
		m_dummy_batched_xbo_01_2->setup_batch();
		m_dummy_batched_xbo_01_2->set_batch_dim(vec<int>{2});
		m_eri_batched->set_batch_dim({2});
		
		int n_batches_o = m_dummy_batched_xbo_01_2->nbatches();
		int n_batches_m = m_eri_batched->nbatches();
		
		//std::cout << "INVSQRT" << std::endl;
		//dbcsr::print(*m_invsqrt);
		
		// LOOP OVER BATCHES OF OCCUPIED ORBITALS
		
		for (int I = 0; I != n_batches_o; ++I) {
			//std::cout << "IBATCH: " << I << std::endl;
			
			auto boundso = m_dummy_batched_xbo_01_2->bounds(I,2);
			
			//std::cout << "BOUNDSO: " << boundso[0] << " " << boundso[1] << std::endl;
			
			vec<vec<int>> bounds(1);
			bounds[0] = boundso;
			
			for (int M = 0; M != n_batches_m; ++M) {
				
				//std::cout << "MBATCH: " << M << std::endl;
				
				fetch1.start();
				fetch_integrals(M,m_INTS_01_2); 
				fetch1.finish();
				
				con1.start();
				dbcsr::contract(*m_INTS_01_2,*m_c_bm,*m_HT1_xbm_01_2)
					.bounds3(bounds).beta(1.0).perform("XMN, Ni -> XMi");
				con1.finish();
				
				retints.start();
				return_integrals(m_INTS_01_2);
				retints.finish();
			}
			
			LOG.os<1>("Occupancy of HTI: ", m_HT1_xbm_01_2->occupation()*100, "%\n");
			
			// end for M
			reo1.start();
			dbcsr::copy(*m_HT1_xbm_01_2,*m_HT1_xbm_0_12).move_data(true).perform();
			reo1.finish();
			
			con2.start();
			dbcsr::contract(*m_HT1_xbm_0_12,*m_invsqrt,*m_HT2_xbm_0_12).perform("XMi, XY -> YMi");
			con2.finish();
			m_HT1_xbm_0_12->clear();
			
			//dbcsr::print(*m_HT2_xbm_0_12);
			
			reo2.start();
			dbcsr::copy(*m_HT2_xbm_0_12,*m_HT2_xbm_01_2).move_data(true).perform();
			reo2.finish();
			
			con3.start();
			dbcsr::contract(*m_HT2_xbm_01_2,*m_HT2_xbm_01_2,*m_K_01).move(true)
				.beta(1.0).perform("Xmi, Xni -> mn"); 
			con3.finish();
			
			//dbcsr::print(*m_K_01);
				
		} // end for I
		
		dbcsr::copy_tensor_to_matrix(*m_K_01,*k_bb);
		m_K_01->clear();
		k_bb->scale(-1.0);
		
	}; // end lambda function
	
	compute_K_single(m_c_A, m_K_A, "A");
	
	if (m_K_B) compute_K_single(m_c_B, m_K_B, "B");
	
	//dbcsr::print(*m_K_A);
	
	TIME.finish();
		
}*/

BATCHED_DFAO_K::BATCHED_DFAO_K(dbcsr::world& w, desc::options& opt) 
	: K(w,opt) {}

void BATCHED_DFAO_K::init_tensors() {
	
	auto inv = m_reg.get_tensor<2,double>(m_mol->name() + "_s_xx_inv_(0|1)");
	m_scr = m_reg.get_screener(m_mol->name() + "_schwarz_screener");
	
	m_eri_batched = m_reg.get_btensor<3,double>(m_mol->name() + "_i_xbb_(0|12)_batched");
	
	auto eri_0_12 = m_eri_batched->get_stensor();
	
	// ======== Compute inv_xx * i_xxb ==============
	
	dbcsr::stensor3_d c_xbb_0_12 = dbcsr::make_stensor<3>(
		dbcsr::tensor3_d::create_template(*eri_0_12).name("c_xbb_0_12")
		.map1({0}).map2({1,2}));
	
	m_c_xbb_1_02 = dbcsr::make_stensor<3>(
		dbcsr::tensor3_d::create_template(*eri_0_12).name("c_xbb_1_02")
		.map1({1}).map2({0,2}));
	
	dbcsr::btype mytype = dbcsr::invalid;
	
	std::string intermeds = m_opt.get<std::string>("intermeds", "core");
	
	if (intermeds == "core") mytype = dbcsr::core;
	if (intermeds == "disk") mytype = dbcsr::disk;
	
	int nbatches = m_opt.get<int>("nbatches", 4);
	
	m_c_xbb_batched = std::make_shared<dbcsr::btensor<3,double>>(
		m_c_xbb_1_02, nbatches, mytype, 50);
	
	auto& calc_c = TIME.sub("Computing (mu,nu|X)(X|Y)^-1");
	auto& con = calc_c.sub("Contraction");
	auto& reo = calc_c.sub("Reordering");
	auto& write = calc_c.sub("Writing");
	auto& fetch = calc_c.sub("Fetching ints");
	
	LOG.os<1>("Computing C_xbb.\n");
	
	calc_c.start();
	
	m_eri_batched->decompress_init({2});
	m_c_xbb_batched->compress_init({2,0});
		
	auto mu_full_b = m_c_xbb_batched->full_bounds(1);
	auto x_b = m_c_xbb_batched->bounds(0);
	auto nu_b = m_c_xbb_batched->bounds(2);
	
	//dbcsr::print(*inv);
	
	for (int inu = 0; inu != m_c_xbb_batched->nbatches_dim(2); ++inu) {
		
		fetch.start();
		m_eri_batched->decompress({inu});
		fetch.finish();
			
		for (int ix = 0; ix != m_c_xbb_batched->nbatches_dim(0); ++ix) {
			
				vec<vec<int>> b2 = {
					x_b[ix]
				};
				
				vec<vec<int>> b3 = {
					mu_full_b,
					nu_b[inu]
				};

				con.start();
				dbcsr::contract(*inv, *eri_0_12, *c_xbb_0_12)
					.bounds2(b2).bounds3(b3)
					.perform("XY, YMN -> XMN");
				con.finish();
						
				reo.start();
				dbcsr::copy(*c_xbb_0_12, *m_c_xbb_1_02).move_data(true).perform();
				reo.finish();
				
				//dbcsr::print(*m_c_xbb_1_02);
				
				m_c_xbb_batched->compress({inu,ix}, m_c_xbb_1_02);
				
		}
	}

	calc_c.finish();
	
	LOG.os<1>("Done.\n");
	
	// ========== END ==========
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,2> bb = {b,b};
	
	dbcsr::pgrid<2> grid2(m_world.comm());
	dbcsr::pgrid<3> grid3(m_world.comm());
	
	m_cbar_xbb_01_2 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create_template(*eri_0_12)
			.name("Cbar_xbb_01_2").map1({0,1}).map2({2}));
		
	m_cbar_xbb_02_1 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create_template(*eri_0_12)
			.name("Cbar_xbb_02_1").map1({0,2}).map2({1}));
	
	m_K_01 = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().ngrid(grid2).name("K_01")
		.map1({0}).map2({1}).blk_sizes(bb));
		
	m_p_bb = dbcsr::make_stensor<2>(
			dbcsr::tensor2_d::create_template(*m_K_01)
			.name("p_bb_0_1").map1({0}).map2({1}));

}

void BATCHED_DFAO_K::compute_K() {
	
	TIME.start();
	
	auto compute_K_single = 
	[&] (dbcsr::smat_d& p_bb, dbcsr::smat_d& k_bb, std::string x) {
		
		LOG.os<1>("Computing exchange part (", x, ")\n");
		
		dbcsr::copy_matrix_to_tensor(*p_bb, *m_p_bb);
			
		//dbcsr::print(*c_bm);
		//dbcsr::print(*m_c_bm);	
		
		// LOOP OVER X
		
		auto& reo_int = TIME.sub("Reordering ints " + x);
		auto& reo_1_batch = TIME.sub("Reordering (1)/batch " + x);
		auto& con_1_batch = TIME.sub("Contraction (1)/batch " + x);
		auto& con_2_batch = TIME.sub("Contraction (2)/batch " + x);
		auto& fetch = TIME.sub("Fetching integrals/batch " + x);
		auto& fetch2 = TIME.sub("Fetching fitting coeffs/batch " + x);
		auto& retint = TIME.sub("Returning integrals/batch " + x);
		
		reo_int.start();
		m_eri_batched->reorder(vec<int>{0,1}, vec<int>{2});
		reo_int.finish();
		
		auto eri_01_2 = m_eri_batched->get_stensor();
	
		m_eri_batched->decompress_init({2});
		m_c_xbb_batched->decompress_init({2,0});
		
		m_K_01->batched_contract_init();
		
		//vec<int> nubounds = m_eri_batched->bounds(1);
		
		auto x_b = m_c_xbb_batched->bounds(0);
		auto mu_full_b = m_c_xbb_batched->full_bounds(1);
		auto nu_b = m_c_xbb_batched->bounds(2);
		
		for (int inu = 0; inu != nu_b.size(); ++inu) {
			
			// fetch integrals
			fetch.start();
			m_eri_batched->decompress({inu});
			fetch.finish();
			
			//dbcsr::print(*eri_01_2);
			
			for (int ix = 0; ix != x_b.size(); ++ix) {
				
				std::cout << "BATCH (x/n): " << ix << " " << inu << std::endl;
	
				vec<vec<int>> xm_bounds = { x_b[ix], mu_full_b };
				vec<vec<int>> n_bounds = { nu_b[inu] };
			
				con_1_batch.start();
				dbcsr::contract(*eri_01_2, *m_p_bb, *m_cbar_xbb_01_2)
					.bounds1(n_bounds).bounds2(xm_bounds)
					.perform("XML, LS -> XMS");
				con_1_batch.finish();
			
				vec<vec<int>> copy_bounds = {
					x_b[ix],
					mu_full_b,
					nu_b[inu]
				};
			
				reo_1_batch.start();
				dbcsr::copy(*m_cbar_xbb_01_2, *m_cbar_xbb_02_1)
					.bounds(copy_bounds).move_data(true).perform();
				reo_1_batch.finish();
				
				//dbcsr::print(*m_cbar_xbb_02_1);
				
				//dbcsr::print(*m_cbar_xbb_02_1);
			
				// get c_xbb
				fetch2.start();
				m_c_xbb_batched->decompress({inu,ix});
				auto c_xbb_1_02 = m_c_xbb_batched->get_stensor();
				fetch2.finish();
				
				//dbcsr::print(*c_xbb_1_02);
				
				vec<vec<int>> xs_bounds = { x_b[ix], nu_b[inu] };
			
				con_2_batch.start();
				dbcsr::contract(*m_cbar_xbb_02_1, *c_xbb_1_02, *m_K_01)
					.bounds1(xs_bounds).beta(1.0)
					.perform("XMS, XNS -> MN");
				con_2_batch.finish();
								
				m_cbar_xbb_02_1->clear();
				
			}
			
		}
		
		retint.start();
		m_eri_batched->reorder(vec<int>{0},vec<int>{1,2});
		retint.finish();
		
		m_K_01->batched_contract_finalize();
		
		dbcsr::copy_tensor_to_matrix(*m_K_01,*k_bb);
		m_K_01->clear();
		m_p_bb->clear();
		k_bb->scale(-1.0);
		
		LOG.os<1>("Done with exchange.\n");
		
	}; // end lambda function
	
	compute_K_single(m_p_A, m_K_A, "A");
	
	if (m_K_B) compute_K_single(m_p_B, m_K_B, "B");
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_K_A);
		if (m_K_B) dbcsr::print(*m_K_B);
	}
	
	TIME.finish();
			
}
	
	
} // end namespace
