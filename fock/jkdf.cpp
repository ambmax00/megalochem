#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include "math/linalg/LLT.h"
#include <dbcsr_tensor_ops.hpp>

namespace fock {

DF_J::DF_J(dbcsr::world& w, desc::options& iopt) : J(w,iopt) {} 
DF_K::DF_K(dbcsr::world& w, desc::options& iopt) : K(w,iopt) {}

void DF_J::init_tensors() {
	
	// initialize tensors
	dbcsr::pgrid<2> grid2(m_world.comm());
	dbcsr::pgrid<3> grid3(m_world.comm());
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	vec<int> d = {1};
	
	arrvec<int,3> bbd = {b,b,d};
	arrvec<int,2> xd = {x,d};
	
	m_c_xd = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().ngrid(grid2).name("c_x")
		.map1({0}).map2({1}).blk_sizes(xd));
	
	m_c2_xd = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create_template(*m_c_xd).name("c2_x"));
	
	m_J_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create().ngrid(grid3).name("J dummy")
		.map1({0,1}).map2({2}).blk_sizes(bbd));
	
	m_ptot_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create_template(*m_J_bbd).name("ptot dummy"));
	
	m_inv = m_reg.get_tensor<2,double>(m_mol->name() + "_s_xx_inv_(0|1)");
	
}

void DF_J::compute_J() {
	
	// fetch integrals
	
	LOG.os<1>("Fetching integrals.\n");
	
	auto i_xbb_012 = m_reg.get_tensor<3,double>(m_mol->name() + "_i_xbb_(0|12)");

	// copy over density
	
	LOG.os<1>("Copy over density.\n");
	
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
	
	LOG.os<1>("XMN, MN_ -> X_\n");
	
	dbcsr::contract(*i_xbb_012, *m_ptot_bbd, *m_c_xd).perform("XMN, MN_ -> X_");
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_c_xd);
	}
	
	LOG.os<1>("X_, XY -> Y_\n");
	
	dbcsr::contract(*m_c_xd, *m_inv, *m_c2_xd).perform("X_, XY -> Y_");
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_c2_xd);
	}
	
	LOG.os<1>("X, XMN -> MN\n");
	
	dbcsr::contract(*m_c2_xd, *i_xbb_012, *m_J_bbd).perform("X_, XMN -> MN_");
	
	LOG.os<1>("Copy over...\n");
	
	dbcsr::copy_3Dtensor_to_matrix_new(*m_J_bbd, *m_J);
	
	m_c_xd->clear();
	m_c2_xd->clear();
	m_J_bbd->clear();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_J);
	}
	
}

void DF_K::init_tensors() {
		
	m_inv = m_reg.get_tensor<2,double>(m_mol->name() + "_s_xx_inv_(0|1)");
	
	auto b = m_p_A->row_blk_sizes();
	arrvec<int,2> bb = {b,b};
	
	dbcsr::pgrid<2> grid2(m_world.comm());
	
	m_K_01 = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().ngrid(grid2).name("K_01")
		.map1({0}).map2({1}).blk_sizes(bb));
		
}

void DF_K::compute_K() {
	
	auto b = m_mol->dims().b();
	auto X = m_mol->dims().x();
	
	auto compute_K_single = [&](dbcsr::smat_d& c_bm, dbcsr::smat_d& k_bb, std::string x) {
				
			LOG.os<1>("Computing exchange term (", x, ") ... \n");
			
			auto& reo_int_1 = TIME.sub("Reorder Integrals (1)");
			auto& reo_int_2 = TIME.sub("Reorder Integrals (2)");
			auto& con1 = TIME.sub("Contraction (1)");
			auto& con2 = TIME.sub("Contraction (2)");
			auto& con3 = TIME.sub("Contraction (3)");
			auto& reo_HT_1 = TIME.sub("Reordering HT (1)");
			auto& reo_HT_2 = TIME.sub("Reordering HT (2)");
			auto& reo_D_1 = TIME.sub("Reordering D (1)");
			
			// init all tensors
			
			auto m = c_bm->col_blk_sizes();
			arrvec<int,2> bm = {b,m};
			arrvec<int,3> xbm = {X,b,m};
			arrvec<int,3> xbb = {X,b,b};
	
			dbcsr::pgrid<2> grid2(m_world.comm());
			dbcsr::pgrid<3> grid3(m_world.comm());
			
			m_c_bm = dbcsr::make_stensor<2>(
				dbcsr::tensor2_d::create().name("c_bm").ngrid(grid2)
				.map1({0}).map2({1}).blk_sizes(bm));
			
			m_INTS_01_2 = dbcsr::make_stensor<3>(
				dbcsr::tensor3_d::create().name("_01_2").ngrid(grid3)
				.map1({0,1}).map2({2}).blk_sizes(xbb));
				
			m_HT_0_12 = dbcsr::make_stensor<3>(
				dbcsr::tensor3_d::create().name("HT_0_12").ngrid(grid3)
				.map1({0}).map2({1,2}).blk_sizes(xbm));
				
			m_HT_01_2 = dbcsr::make_stensor<3>(
				dbcsr::tensor3_d::create_template(*m_HT_0_12).name("HT_01_2")
				.map1({0,1}).map2({2}));
				
			m_HT_02_1 = dbcsr::make_stensor<3>(
				dbcsr::tensor3_d::create_template(*m_HT_0_12).name("HT_02_1")
				.map1({0,2}).map2({1}));
				
			m_D_0_12 = dbcsr::make_stensor<3>(
				dbcsr::tensor3_d::create_template(*m_HT_0_12).name("D_0_12")
				.map1({0}).map2({1,2}));
				
			m_D_02_1 = dbcsr::make_stensor<3>(
				dbcsr::tensor3_d::create_template(*m_HT_01_2).name("D_02_1")
				.map1({0,2}).map2({1}));
				
			dbcsr::copy_matrix_to_tensor(*k_bb, *m_K_01, false);
			
			reo_int_1.start();
				
			auto INTS_0_12 = m_reg.get_tensor<3,double>(m_mol->name() + "_i_xbb_(0|12)");
			
			//dbcsr::print(*INTS_0_12);
			
			dbcsr::copy(*INTS_0_12,*m_INTS_01_2).move_data(true).perform();
			
			reo_int_1.finish();
			
			con1.start();
			
			dbcsr::copy_matrix_to_tensor<double>(*c_bm, *m_c_bm, false);
			
			if (m_SAD_iter) {
				
				dbcsr::contract(*m_c_bm, *m_INTS_01_2, *m_HT_01_2).perform("Ni, XMN -> XMi");			
				
			} else {
				
				int nocc = (x == "A") ? m_mol->nocc_alpha() - 1 : m_mol->nocc_beta() - 1;	
				
				vec<vec<int>> occ_bounds = {{0,nocc}};
					
				dbcsr::contract(*m_c_bm, *m_INTS_01_2, *m_HT_01_2).bounds2(occ_bounds).perform("Ni, XMN -> XMi");
				
			}
			
			//dbcsr::print(*m_HT_01_2);
			
			con1.finish();
			
			reo_HT_1.start();
			dbcsr::copy(*m_HT_01_2, *m_HT_0_12).move_data(true).perform();
			reo_HT_1.finish();
			
			con2.start();
			dbcsr::contract(*m_HT_0_12, *m_inv, *m_D_0_12).perform("XMi, XY -> YMi");
			con2.finish();
			
			//ka("mu,nu") = HTa("M,mu,i") * Da("M,nu,i");
			
			reo_HT_2.start();
			dbcsr::copy(*m_HT_0_12, *m_HT_02_1).move_data(true).perform();
			reo_HT_2.finish();
			reo_D_1.start();
			dbcsr::copy(*m_D_0_12, *m_D_02_1).move_data(true).perform();
			reo_D_1.finish();
			
			con3.start();
			dbcsr::contract(*m_HT_02_1, *m_D_02_1, *m_K_01).move(true).alpha(-1.0).perform("XMi, XNi -> MN");
			con3.finish();
			
			if (LOG.global_plev() >= 2) {
				dbcsr::print(*m_K_01);
			}
			
			reo_int_2.start();
			dbcsr::copy(*m_INTS_01_2,*INTS_0_12).move_data(true).perform();
			reo_int_2.finish();
			
			dbcsr::copy_tensor_to_matrix(*m_K_01, *k_bb, false);

	};
	
	compute_K_single(m_c_A, m_K_A, "A");
	
	if (m_K_B) compute_K_single(m_c_B, m_K_B, "B");	
			
	
}

BATCHED_DF_J::BATCHED_DF_J(dbcsr::world& w, desc::options& iopt, 
	std::string metric, bool direct) : 
	m_direct(direct), m_metric(metric), J(w,iopt) {} 

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

void BATCHED_DF_J::fetch_integrals(int ibatch) {
	
	if (!m_direct) {
		m_eri_batched->read(ibatch);
	} else {
		m_factory->ao_3c2e_setup(m_metric);
		vec<vec<int>> blkbounds = {
			m_eri_batched->bounds_blk(ibatch,0),
			m_eri_batched->bounds_blk(ibatch,1),
			m_eri_batched->bounds_blk(ibatch,2)
		};
		auto eri = m_eri_batched->get_stensor();
		m_factory->ao_3c2e_fill(eri, blkbounds,m_scr.get());
	}
	
}

void BATCHED_DF_J::compute_J() {
	
	auto& con1 = TIME.sub("first contraction");
	auto& con2 = TIME.sub("second contraction");
	auto& fetch1 = TIME.sub("fetch ints (1)");
	auto& fetch2 = TIME.sub("fetch ints (2)");
	
	TIME.start();
	
	// fetch batchtensor
		
	m_eri_batched->set_batch_dim(0);
	int nbatches = m_eri_batched->nbatches();
	
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
	
	for (int ibatch = 0; ibatch != nbatches; ++ibatch) {
		
		LOG.os<1>("J builder (1), batch nr. ", ibatch, '\n');
		// Fetch integrals
		
		fetch1.start();
		fetch_integrals(ibatch);
		fetch1.finish();
		
		auto i_xbb_0_12 = m_eri_batched->get_stensor();
		//dbcsr::print(*i_xbb_0_12);
		
		LOG.os<1>("XMN, MN_ -> X_\n");
		
		con1.start();
	
		vec<vec<int>> bounds1 = {
			m_eri_batched->bounds(ibatch,1), // M
			m_eri_batched->bounds(ibatch,2) // N
		};
		
		vec<vec<int>> bounds2 = {
			m_eri_batched->bounds(ibatch,0)
		};
	
		dbcsr::contract(*i_xbb_0_12, *m_ptot_bbd, *m_gp_xd)
			.bounds1(bounds1).bounds2(bounds2)
			.beta(1.0).perform("XMN, MN_ -> X_");
		m_eri_batched->clear_batch();
		
		con1.finish();
		
	}
	
	m_gp_xd->batched_contract_finalize();
	m_ptot_bbd->batched_contract_finalize();
		
	//dbcsr::print(*m_gp_xd);
	
	LOG.os<1>("X_, XY -> Y_\n");
	
	dbcsr::contract(*m_gp_xd, *m_inv, *m_gq_xd).perform("X_, XY -> Y_");
	
	//dbcsr::print(*m_gq_xd);
	
	m_eri_batched->set_batch_dim(0);
	
	m_J_bbd->batched_contract_init();
	m_gq_xd->batched_contract_init();
	
	for (int ibatch = 0; ibatch != nbatches; ++ibatch) {
		
		LOG.os<1>("J builder (2), batch nr. ", ibatch, '\n');
		// Fetch integrals
		
		fetch2.start();
		fetch_integrals(ibatch);
		fetch2.finish();
		
		auto i_xbb_0_12 = m_eri_batched->get_stensor();
		
		LOG.os<1>("X, XMN -> MN\n");
		
		con2.start();
		
		vec<vec<int>> bounds1 = {
			m_eri_batched->bounds(ibatch,0)
		};
		
		std::cout << "XBOUNDS: " << bounds1[0][0] << " " << bounds1[0][1] << std::endl;
		
		vec<int> mbounds = m_eri_batched->bounds(ibatch,1);
		vec<int> nbounds = m_eri_batched->bounds(ibatch,2);
		
		std::cout << "BOUNDS: " << mbounds[0] << " " << mbounds[1] << " / " << nbounds[0] << " "
			<< nbounds[1] << std::endl;
		
		vec<vec<int>> bounds3 = {
			vec<int>{mbounds[0],nbounds[0]},
			vec<int>{mbounds[1],nbounds[1]}
		};
	
		dbcsr::contract(*m_gq_xd, *i_xbb_0_12, *m_J_bbd)
			.print(true).beta(1.0).perform("X_, XMN -> MN_");
		
		//dbcsr::print(*m_J_bbd);
		
		m_eri_batched->clear_batch();
		
		con2.finish();
		
	}
	
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

BATCHED_DFMO_K::BATCHED_DFMO_K(dbcsr::world& w, desc::options& opt, 
	std::string metric, bool direct) 
	: m_direct(direct), m_metric(metric), K(w,opt) {}

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
		
		m_factory->ao_3c2e_fill(reo_ints, bounds, m_scr.get());
		
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
		m_dummy_batched_xbo_01_2->set_batch_dim(2);
		m_eri_batched->set_batch_dim(2);
		
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
		
}

BATCHED_DFAO_K::BATCHED_DFAO_K(dbcsr::world& w, desc::options& opt, 
	std::string metric, bool direct) 
	: m_direct(direct), m_metric(metric), K(w,opt) {}
	
void BATCHED_DFAO_K::fetch_integrals(int ibatch, dbcsr::stensor3_d reo_ints) {
	
	if (!m_direct) {
		
		m_eri_batched->read(ibatch);
		auto eri = m_eri_batched->get_stensor();
		
		if (reo_ints) dbcsr::copy(*eri, *reo_ints).move_data(true).perform();
		
	} else {
		
		auto eri = m_eri_batched->get_stensor();
		
		m_factory->ao_3c2e_setup(m_metric);
		
		vec<vec<int>> bounds = {
			m_eri_batched->bounds_blk(ibatch,0),
			m_eri_batched->bounds_blk(ibatch,1),
			m_eri_batched->bounds_blk(ibatch,2)
		};
		
		if (reo_ints) {
			m_factory->ao_3c2e_fill(reo_ints, bounds, m_scr.get());
		} else {
			m_factory->ao_3c2e_fill(eri, bounds, m_scr.get());
		}
		
	}
	
}

void BATCHED_DFAO_K::init_tensors() {
	
	auto inv = m_reg.get_tensor<2,double>(m_mol->name() + "_s_xx_inv_(0|1)");
	m_scr = m_reg.get_screener(m_mol->name() + "_schwarz_screener");
	
	m_eri_batched = m_reg.get_btensor<3,double>(m_mol->name() + "_i_xbb_(0|12)_batched");
	auto eri = m_eri_batched->get_stensor();	
	
	// ======== Compute inv_xx * i_xxb ==============
	
	dbcsr::stensor3_d c_xbb_0_12 = dbcsr::make_stensor<3>(
		dbcsr::tensor3_d::create_template(*eri).name("c_xbb_0_12")
		.map1({0}).map2({1,2}));
	
	dbcsr::stensor3_d c_xbb_1_02 = dbcsr::make_stensor<3>(
		dbcsr::tensor3_d::create_template(*eri).name("c_xbb_0_12")
		.map1({1}).map2({0,2}));
	
	m_c_xbb_batched = std::make_shared<tensor::batchtensor<3,double>>(
		c_xbb_1_02, tensor::global::default_batchsize, LOG.global_plev());
	
	m_c_xbb_batched->create_file();
	m_c_xbb_batched->setup_batch();
	m_c_xbb_batched->set_batch_dim(1);
	
	m_eri_batched->set_batch_dim(1);
	
	// C_x,mu,nu -> loop over mu
	int mu_nbatches = m_c_xbb_batched->nbatches();
	
	auto& calc_c = TIME.sub("Computing (mu,nu|X)(X|Y)^-1");
	auto& con = calc_c.sub("Contraction");
	auto& reo = calc_c.sub("Reordering");
	auto& write = calc_c.sub("Writing");
	
	LOG.os<1>("Computing C_xbb.\n");
	
	calc_c.start();
	
	for (int MUBATCH = 0; MUBATCH!= mu_nbatches; ++MUBATCH) {
		
		fetch_integrals(MUBATCH,nullptr);
		
		// c_xbb->reserve_template(*eri);
		con.start();
		dbcsr::contract(*inv, *eri, *c_xbb_0_12).print(true).perform("XY, YMN -> XMN");
		con.finish();
		
		reo.start();
		dbcsr::copy(*c_xbb_0_12, *c_xbb_1_02).move_data(true).perform();
		reo.finish();
		
		//dbcsr::print(*c_xbb_1_02);
		
		write.start();
		m_c_xbb_batched->write(MUBATCH);
		write.finish();
		
		m_c_xbb_batched->clear_batch();
		m_eri_batched->clear_batch();
		
	}
	
	LOG.os<1>("Occupancy of fitting coefficients: ", c_xbb_1_02->occupation()*100, "%\n");
	
	calc_c.finish();
	
	LOG.os<1>("Done.\n");
	
	// ========== END ==========
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,2> bb = {b,b};
	
	dbcsr::pgrid<2> grid2(m_world.comm());
	dbcsr::pgrid<3> grid3(m_world.comm());
	
	m_cbar_xbb_01_2 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create_template(*eri)
			.name("Cbar_xbb_01_2").map1({0,1}).map2({2}));
		
	m_cbar_xbb_02_1 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create_template(*eri)
			.name("Cbar_xbb_02_1").map1({0,2}).map2({1}));
		
	m_eri_01_2 = dbcsr::make_stensor<3>(
			dbcsr::tensor3_d::create_template(*eri)
			.name("eri_01_2").map1({0,1}).map2({2}));
	
	m_K_01 = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().ngrid(grid2).name("K_01")
		.map1({0}).map2({1}).blk_sizes(bb));
		
	m_p_bb = dbcsr::make_stensor<2>(
			dbcsr::tensor2_d::create_template(*m_K_01)
			.name("p_bb_0_1").map1({0}).map2({1}));

}
			
void BATCHED_DFAO_K::return_integrals(dbcsr::stensor3_d& reo_ints) {
	
	int nbatches = m_eri_batched->nbatches();
	if (m_direct || nbatches != 1) {
		reo_ints->clear();
	} else {
		auto eri = m_eri_batched->get_stensor();
		dbcsr::copy(*reo_ints,*eri).move_data(true).perform();
	}
	
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
		
		auto& reo_1_batch = TIME.sub("Reordering (1)/batch " + x);
		auto& con_1_batch = TIME.sub("Contraction (1)/batch " + x);
		auto& con_2_batch = TIME.sub("Contraction (2)/batch " + x);
		auto& fetch = TIME.sub("Fetching integrals/batch " + x);
		auto& fetch2 = TIME.sub("Fetching fitting coeffs/batch " + x);
		auto& retint = TIME.sub("Returning integrals/batch " + x);
		
		m_c_xbb_batched->set_batch_dim(0);
		m_eri_batched->set_batch_dim(0);
		
		auto c_xbb_1_02 = m_c_xbb_batched->get_stensor();
		
		int nbatches_x = m_c_xbb_batched->nbatches();
		
		for (int XBATCH = 0; XBATCH != nbatches_x; ++XBATCH) {
			
			// fetch integrals
			fetch.start();
			fetch_integrals(XBATCH,m_eri_01_2);
			fetch.finish();
			
			//dbcsr::print(*m_eri_01_2);
			
			LOG.os<1>("Contract nr 1, batch nr ", XBATCH, '\n');
			
			con_1_batch.start();
			dbcsr::contract(*m_eri_01_2, *m_p_bb, *m_cbar_xbb_01_2)
				.print(true).perform("XML, LS -> XMS");
			con_1_batch.finish();	
			
			retint.start();
			return_integrals(m_eri_01_2);
			retint.finish();
			
			LOG.os<1>("Occupancy of (mu,nu|X) P_mu,nu: ", m_cbar_xbb_01_2->occupation()*100, "%\n");
			
			//dbcsr::print(*m_cbar_xbb_01_2);
			
			reo_1_batch.start();
			dbcsr::copy(*m_cbar_xbb_01_2, *m_cbar_xbb_02_1).move_data(true).perform();
			reo_1_batch.finish();
			
			// get c_xbb
			fetch2.start();
			m_c_xbb_batched->read(XBATCH);
			fetch2.finish();
			
			//dbcsr::print(*c_xbb_1_02);
			
			LOG.os<1>("Contract nr 2, batch nr ", XBATCH, '\n');
			
			con_2_batch.start();
			dbcsr::contract(*m_cbar_xbb_02_1, *c_xbb_1_02, *m_K_01)
				.print(true).beta(1.0).perform("XMS, XNS -> MN");
			con_2_batch.finish();
			
			m_c_xbb_batched->clear_batch();
			m_cbar_xbb_02_1->clear();
			
		}
		
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
