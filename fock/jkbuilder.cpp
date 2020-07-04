#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include <dbcsr_tensor_ops.hpp>

namespace fock {
	
JK_common::JK_common(dbcsr::world& w, desc::options opt) :
	m_world(w), m_opt(opt),
	LOG(m_world.comm(),m_opt.get<int>("print", FOCK_PRINT_LEVEL)),
	TIME(m_world.comm(), "JK Builder", LOG.global_plev()) {}
	
void J::init() {
	
	// set up J
	auto b = m_p_A->row_blk_sizes();
	
	m_J = std::make_shared<dbcsr::mat_d>(
		dbcsr::mat_d::create().name("J_bb").set_world(m_world)
		.row_blk_sizes(b).col_blk_sizes(b).type(dbcsr_type_symmetric));
	
}

void K::init() {
	
	// set up K's
	auto b = m_p_A->row_blk_sizes();
	
	m_K_A = std::make_shared<dbcsr::mat_d>(
		dbcsr::mat_d::create().name("K_bb_A").set_world(m_world)
		.row_blk_sizes(b).col_blk_sizes(b).type(dbcsr_type_symmetric));
		
	if (m_p_B) m_K_B = std::make_shared<dbcsr::mat_d>(
		dbcsr::mat_d::create().name("K_bb_B").set_world(m_world)
		.row_blk_sizes(b).col_blk_sizes(b).type(dbcsr_type_symmetric));
	
}

EXACT_J::EXACT_J(dbcsr::world& w, desc::options& iopt) : J(w,iopt) {} 
EXACT_K::EXACT_K(dbcsr::world& w, desc::options& iopt) : K(w,iopt) {}

void EXACT_J::init_tensors() {
	
	dbcsr::pgrid<3> grid3(m_world.comm());
	auto b = m_p_A->row_blk_sizes();
	vec<int> d = {1};
	
	arrvec<int,3> bbd = {b,b,d};
	
	m_J_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create().ngrid(grid3).name("J dummy")
		.map1({0,1}).map2({2}).blk_sizes(bbd));
	
	m_ptot_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create_template(*m_J_bbd).name("ptot dummy"));
	
}

void EXACT_K::init_tensors() {
	
	dbcsr::pgrid<3> grid3(m_world.comm());
	auto b = m_p_A->row_blk_sizes();
	vec<int> d = {1};
	
	arrvec<int,3> bbd = {b,b,d};
	
	m_K_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create().ngrid(grid3).name("K dummy")
		.map1({0,1}).map2({2}).blk_sizes(bbd));
	
	m_p_bbd = dbcsr::make_stensor<3>(dbcsr::tensor3_d::create_template(*m_K_bbd).name("p dummy"));
	
}	

void EXACT_J::compute_J() {
	
	// copy P 
	
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
	
	if (LOG.global_plev() >= 3) {
		dbcsr::print(*m_ptot_bbd);
	}
	
	// grab integrals
	auto eris = m_reg.get_tensor<4,double>(m_mol->name() + "_i_bbbb_(01|23)", true, true);
	
	dbcsr::contract(*m_ptot_bbd, *eris, *m_J_bbd).print(false).perform("LS_, MNLS -> MN_");
	
	if (LOG.global_plev() >= 3) {
		dbcsr::print(*m_J_bbd);
	}

	dbcsr::copy_3Dtensor_to_matrix_new<double>(*m_J_bbd,*m_J);
	
	m_ptot_bbd->clear();
	m_J_bbd->clear();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_J);
	}

}

void EXACT_K::compute_K() {
	
	auto eris = m_reg.get_tensor<4,double>(m_mol->name() + "_i_bbbb_(02|13)", true, true);
	
	auto compute_K_single = [&](dbcsr::smat_d& p, dbcsr::smat_d& k, std::string x) {
		
		dbcsr::copy_matrix_to_3Dtensor_new<double>(*p,*m_p_bbd,true);
		
		LOG.os<1>("Computing exchange term (", x, ") ... \n");
			
		dbcsr::contract(*m_p_bbd, *eris, *m_K_bbd).alpha(-1.0).perform("LS_, MLSN -> MN_");

		dbcsr::copy_3Dtensor_to_matrix_new(*m_K_bbd,*k);
		
		//dbcsr::print(*m_K_bbd);
		
		m_K_bbd->clear();
		m_p_bbd->clear();
		
		if (LOG.global_plev() >= 2) {
			dbcsr::print(*k);
		}
		
	};
	
	compute_K_single(m_p_A, m_K_A, "A");
	if (m_K_B) compute_K_single(m_p_B, m_K_B, "B");
	
}

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
	
	auto i_xbb_012 = m_reg.get_tensor<3,double>(m_mol->name() + "_i_xbb_(0|12)",true,true);

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

BATCHED_DF_J::BATCHED_DF_J(dbcsr::world& w, desc::options& iopt) : J(w,iopt) {} 

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
	
}

void BATCHED_DF_J::fetch_integrals(tensor::sbatchtensor<3,double>& btensor, int ibatch) {
	
	btensor->read(ibatch);
	
}

void BATCHED_DF_J::compute_J() {
	
	auto& con1 = TIME.sub("first contraction");
	auto& con2 = TIME.sub("second contraction");
	auto& fetch1 = TIME.sub("fetch ints (1)");
	auto& fetch2 = TIME.sub("fetch ints (2)");
	
	TIME.start();
	
	// fetch batchtensor
	
	auto eri_batched = m_reg.get_btensor<3,double>(m_mol->name() + "_i_xbb_(0|12)_batched");
	
	eri_batched->set_batch_dim(vec<int>{2});
	int nbatches = eri_batched->nbatches();
	
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
	
	for (int ibatch = 0; ibatch != nbatches; ++ibatch) {
		
		LOG.os<1>("J builder (1), batch nr. ", ibatch, '\n');
		// Fetch integrals
		
		fetch1.start();
		fetch_integrals(eri_batched,ibatch);
		fetch1.finish();
		
		auto i_xbb_0_12 = eri_batched->get_stensor();
		//dbcsr::print(*i_xbb_0_12);
		
		LOG.os<1>("XMN, MN_ -> X_\n");
		
		con1.start();
	
		dbcsr::contract(*i_xbb_0_12, *m_ptot_bbd, *m_gp_xd).beta(1.0).perform("XMN, MN_ -> X_");
		eri_batched->clear_batch();
		
		con1.finish();
		
	}
		
	//dbcsr::print(*m_gp_xd);
	
	LOG.os<1>("X_, XY -> Y_\n");
	
	dbcsr::contract(*m_gp_xd, *m_inv, *m_gq_xd).perform("X_, XY -> Y_");
	
	//dbcsr::print(*m_gq_xd);
	
	for (int ibatch = 0; ibatch != nbatches; ++ibatch) {
		
		LOG.os<1>("J builder (2), batch nr. ", ibatch, '\n');
		// Fetch integrals
		
		fetch2.start();
		fetch_integrals(eri_batched,ibatch);
		fetch2.finish();
		
		auto i_xbb_0_12 = eri_batched->get_stensor();
		
		LOG.os<1>("X, XMN -> MN\n");
		
		con2.start();
	
		dbcsr::contract(*m_gq_xd, *i_xbb_0_12, *m_J_bbd).beta(1.0).perform("X_, XMN -> MN_");
		
		//dbcsr::print(*m_J_bbd);
		
		eri_batched->clear_batch();
		
		con2.finish();
		
	}
	
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

BATCHED_DF_K::BATCHED_DF_K(dbcsr::world& w, desc::options& opt) : K(w,opt) {}

void BATCHED_DF_K::init_tensors() {
	
	auto b = m_p_A->row_blk_sizes();
	arrvec<int,2> bb = {b,b};
	
	dbcsr::pgrid<2> grid2(m_world.comm());
	
	m_K_01 = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().ngrid(grid2).name("K_01")
		.map1({0}).map2({1}).blk_sizes(bb));
		
	m_invsqrt = m_reg.get_tensor<2,double>(m_mol->name() + "_s_xx_invsqrt_(0|1)");
	
}

void BATCHED_DF_K::compute_K() {
	
	TIME.start();
	
	auto b = m_mol->dims().b();
	auto X = m_mol->dims().x();
	
	dbcsr::pgrid<2> grid2(m_world.comm());
	dbcsr::pgrid<3> grid3(m_world.comm());
	
	auto compute_K_single = 
	[&] (dbcsr::smat_d& c_bm, dbcsr::smat_d& k_bb, std::string x) {
		
		auto& fetch1 = TIME.sub("Fetch ints (1) " + x);
		auto& fetch2 = TIME.sub("Fetch ints (2) " + x);
		auto& con1 = TIME.sub("Contraction (1) " + x);
		auto& con2 = TIME.sub("Contraction (2) " + x);
		auto& con3 = TIME.sub("Contraction (3) " + x);
		auto& reo1 = TIME.sub("Reordering (1) " + x);
		auto& reo2 = TIME.sub("Reordering (2) " + x);
		auto& reo3 = TIME.sub("Reordering (3) " + x);
		auto& reo4 = TIME.sub("Reordering (4) " + x);
		
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
		
		auto eri_batched = m_reg.get_btensor<3,double>(m_mol->name() + "_i_xbb_(0|12)_batched");
		
		eri_batched->set_batch_dim({2});
		
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
			
		m_dummy_batched_xbo_01_2->setup_batch();
		m_dummy_batched_xbo_01_2->set_batch_dim(vec<int>{2});
		
		int n_batches_o = m_dummy_batched_xbo_01_2->nbatches();
		int n_batches_m = eri_batched->nbatches();
		
		//std::cout << "INVSQRT" << std::endl;
		//dbcsr::print(*m_invsqrt);
		
		// LOOP OVER BATCHES OF OCCUPIED ORBITALS
		
		auto eri_ptr = eri_batched->get_stensor();
		
		if (n_batches_m == 1) {
			reo1.start();
			dbcsr::copy(*eri_ptr,*m_INTS_01_2).move_data(true).perform();
			reo1.finish();
		}
		
		for (int I = 0; I != n_batches_o; ++I) {
			//std::cout << "IBATCH: " << I << std::endl;
			
			auto boundso = m_dummy_batched_xbo_01_2->bounds(I,2);
			
			//std::cout << "BOUNDSO: " << boundso[0] << " " << boundso[1] << std::endl;
			
			vec<vec<int>> bounds(1);
			bounds[0] = boundso;
			
			for (int M = 0; M != n_batches_m; ++M) {
				
				//std::cout << "MBATCH: " << M << std::endl;
				
				fetch1.start();
				eri_batched->read(M);
				fetch1.finish();
				
				//dbcsr::print(*eri_ptr);
				
				if (n_batches_m != 1) {
					reo1.start();
					dbcsr::copy(*eri_ptr,*m_INTS_01_2).move_data(true).perform();
					reo1.finish();
				}
				
				con1.start();
				dbcsr::contract(*m_INTS_01_2,*m_c_bm,*m_HT1_xbm_01_2)
					.bounds3(bounds).beta(1.0).perform("XMN, Ni -> XMi");
				con1.finish();
				
				//dbcsr::print(*m_HT1_xbm_01_2);
				
				if (n_batches_m != 1) m_INTS_01_2->clear();
				
			}
			
			// end for M
			reo2.start();
			dbcsr::copy(*m_HT1_xbm_01_2,*m_HT1_xbm_0_12).move_data(true).perform();
			reo2.finish();
			
			con2.start();
			dbcsr::contract(*m_HT1_xbm_0_12,*m_invsqrt,*m_HT2_xbm_0_12).perform("XMi, XY -> YMi");
			con2.finish();
			m_HT1_xbm_0_12->clear();
			
			//dbcsr::print(*m_HT2_xbm_0_12);
			
			reo3.start();
			dbcsr::copy(*m_HT2_xbm_0_12,*m_HT2_xbm_01_2).move_data(true).perform();
			reo3.finish();
			
			con3.start();
			dbcsr::contract(*m_HT2_xbm_01_2,*m_HT2_xbm_01_2,*m_K_01).move(true)
				.beta(1.0).perform("Xmi, Xni -> mn"); 
			con3.finish();
			
			//dbcsr::print(*m_K_01);
				
		} // end for I
		
		if (n_batches_m == 1) {
			reo4.start();
			dbcsr::copy(*m_INTS_01_2, *eri_ptr).move_data(true).perform();
			reo4.finish();
		}
		
		dbcsr::copy_tensor_to_matrix(*m_K_01,*k_bb);
		m_K_01->clear();
		k_bb->scale(-1.0);
		
	}; // end lambda function
	
	compute_K_single(m_c_A, m_K_A, "A");
	
	if (m_K_B) compute_K_single(m_c_B, m_K_B, "B");
	
	//dbcsr::print(*m_K_A);
	
	TIME.finish();
		
}
		
} // end namespace
