#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include <dbcsr_tensor_ops.hpp>

namespace fock {
	
JK_common::JK_common(dbcsr::world& w, desc::options opt) :
	m_world(w), m_opt(opt),
	LOG(m_world.comm(),m_opt.get<int>("print", FOCK_PRINT_LEVEL)),
	TIME(m_world.comm(), "Fock Builder", LOG.global_plev()) {}
	
void J::init() {
	
	// set up J
	dbcsr::pgrid<2> grid2(m_world.comm());
	auto b = m_p_A->row_blk_sizes();
	arrvec<int,2> bb = {b,b};
	m_J = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().name("J_bb")
		.ngrid(grid2).map1({0}).map2({1}).blk_sizes(bb));
	
}

void K::init() {
	
	// set up K's
	dbcsr::pgrid<2> grid2(m_world.comm());
	auto b = m_p_A->row_blk_sizes();
	arrvec<int,2> bb = {b,b};
	m_K_A = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create().name("K_bb_A")
		.ngrid(grid2).map1({0}).map2({1}).blk_sizes(bb));
	if (m_p_B) m_K_B = dbcsr::make_stensor<2>(dbcsr::tensor2_d::create_template(*m_K_A).name("K_bb_B"));
	
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
	if (m_p_A && !m_p_B) {
		std::cout << "SCALING." << std::endl;
		dbcsr::copy_matrix_to_3Dtensor<double>(*m_p_A,*m_ptot_bbd,false,true);
		m_ptot_bbd->scale(2.0);
	} else {
		std::cout << "ADDING" << std::endl;
		dbcsr::copy_matrix_to_3Dtensor<double>(*m_p_A,*m_ptot_bbd,false,true);
		dbcsr::copy_matrix_to_3Dtensor<double>(*m_p_B,*m_ptot_bbd,true,true);
	}
	
	if (LOG.global_plev() >= 3) {
		dbcsr::print(*m_ptot_bbd);
	}
	
	// grab integrals
	auto eris = m_reg.get_tensor<4,double>(m_mol->name() + "_i_bbbb_(01|23)", true, true);
	
	dbcsr::contract(*m_ptot_bbd, *eris, *m_J_bbd).print(true).perform("LS_, MNLS -> MN_");
	
	if (LOG.global_plev() >= 3) {
		dbcsr::print(*m_J_bbd);
	}

	dbcsr::copy_3Dtensor_to_2Dtensor<double>(*m_J_bbd,*m_J,false);
	
	m_ptot_bbd->clear();
	m_J_bbd->clear();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_J);
	}

}

void EXACT_K::compute_K() {
	
	auto eris = m_reg.get_tensor<4,double>(m_mol->name() + "_i_bbbb_(02|13)", true, true);
	
	auto compute_K_single = [&](dbcsr::smat_d& p, dbcsr::stensor2_d& k, std::string x) {
		
		dbcsr::copy_matrix_to_3Dtensor<double>(*p,*m_p_bbd,false,true);
		
		LOG.os<1>("Computing exchange term (", x, ") ... \n");
			
		dbcsr::contract(*m_p_bbd, *eris, *m_K_bbd).alpha(-1.0).perform("LS_, MLSN -> MN_");

		dbcsr::copy_3Dtensor_to_2Dtensor(*m_K_bbd,*k);
		
		dbcsr::print(*m_K_bbd);
		
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
	
	dbcsr::copy_matrix_to_3Dtensor<double>(*m_p_A, *m_ptot_bbd, false, true);
	
	if (!m_p_B) {
		m_ptot_bbd->scale(2.0);
	} else {
		dbcsr::copy_matrix_to_3Dtensor<double>(*m_p_B, *m_ptot_bbd, true, true);
	}
	
	LOG.os<1>("XMN, MN_ -> X_\n");
	
	dbcsr::contract(*i_xbb_012, *m_ptot_bbd, *m_c_xd).perform("XMN, MN_ -> X_");
	
	LOG.os<1>("X_, XY -> Y_\n");
	
	dbcsr::contract(*m_c_xd, *m_inv, *m_c2_xd).perform("X_, XY -> Y_");
	
	LOG.os<1>("X, XMN -> MN\n");
	
	dbcsr::contract(*m_c2_xd, *i_xbb_012, *m_J_bbd).perform("X_, XMN -> MN_");
	
	LOG.os<1>("Copy over...\n");
	
	dbcsr::copy_3Dtensor_to_2Dtensor(*m_J_bbd, *m_J, false);
	
	m_c_xd->clear();
	m_c2_xd->clear();
	m_J_bbd->clear();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_J);
	}
	
}

void DF_K::init_tensors() {
		
	m_inv = m_reg.get_tensor<2,double>(m_mol->name() + "_s_xx_inv_(0|1)");
		
}

void DF_K::compute_K() {
	
	auto b = m_mol->dims().b();
	auto X = m_mol->dims().x();
	
	auto compute_K_single = [&](dbcsr::smat_d& c_bm, dbcsr::stensor2_d& k_bb, std::string x) {
				
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
			dbcsr::contract(*m_HT_02_1, *m_D_02_1, *k_bb).move(true).alpha(-1.0).perform("XMi, XNi -> MN");
			con3.finish();
			
			if (LOG.global_plev() >= 2) {
				dbcsr::print(*k_bb);
			}
			
			reo_int_2.start();
			dbcsr::copy(*m_INTS_01_2,*INTS_0_12).move_data(true).perform();
			reo_int_2.finish();

	};
	
	compute_K_single(m_c_A, m_K_A, "A");
	
	if (m_K_B) compute_K_single(m_c_B, m_K_B, "B");	
			
	
}
		
		
	
} // end namespace
