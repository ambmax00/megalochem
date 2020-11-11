#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include "ints/fitting.h"
#include "math/linalg/LLT.h"
#include <dbcsr_tensor_ops.hpp>

namespace fock {

BATCHED_DF_J::BATCHED_DF_J(dbcsr::world w, desc::smolecule mol, int print) 
	: J(w,mol,print,"BATCHED_DF_J") {} 

void BATCHED_DF_J::init() {
	
	init_base();
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	vec<int> d = {1};
	
	int nbf = std::accumulate(b.begin(), b.end(), 0);
	int xnbf = std::accumulate(x.begin(), x.end(), 0);
	
	arrvec<int,3> bbd = {b,b,d};
	arrvec<int,2> xd = {x,d};
	arrvec<int,2> xx = {x,x};
	
	std::array<int,2> tsizes2 = {xnbf,1};
	std::array<int,3> tsizes3 = {nbf,nbf,1};
	
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	m_spgrid_xd = dbcsr::create_pgrid<2>(m_world.comm())
		.tensor_dims(tsizes2).get();
		
	m_spgrid_bbd = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(tsizes3).get();
	
	m_gp_xd = dbcsr::tensor_create<2>().name("gp_xd").pgrid(m_spgrid_xd)
		.map1({0}).map2({1}).blk_sizes(xd).get();
		
	m_gq_xd = dbcsr::tensor_create_template<2>(m_gp_xd).name("gq_xd").get();
	
	m_J_bbd = dbcsr::tensor_create<3>().name("J_bbd").pgrid(m_spgrid_bbd)
		.map1({0,1}).map2({2}).blk_sizes(bbd).get();
	
	m_ptot_bbd = dbcsr::tensor_create_template<3>(m_J_bbd).name("ptot_bbd").get();
	
	m_v_inv_01 = dbcsr::tensor_create<2,double>()
		.name("inv")
		.pgrid(m_spgrid2)
		.blk_sizes(xx)
		.map1({0}).map2({1})
		.get();
	
}

void BATCHED_DF_J::compute_J() {
	
	auto& con1 = TIME.sub("first contraction");
	auto& con2 = TIME.sub("second contraction");
	auto& fetch1 = TIME.sub("fetch ints (1)");
	auto& fetch2 = TIME.sub("fetch ints (2)");
	auto& reoint = TIME.sub("Reordering ints");
	
	TIME.start();
	
	// copy over density
	
	auto ptot = dbcsr::create_template<double>(m_p_A)
		.name("ptot").get();
	
	if (m_p_A && !m_p_B) {
		ptot->copy_in(*m_p_A);
		ptot->scale(2.0);
		bool sym = ptot->has_symmetry();
		//if (!sym) std::cout << "HAS NO SYMMETRY" << std::endl;
		dbcsr::copy_matrix_to_3Dtensor_new(*ptot,*m_ptot_bbd,sym);
		//dbcsr::print(ptot);
		ptot->clear();
	} else {
		ptot->copy_in(*m_p_A);
		ptot->add(1.0, 1.0, *m_p_B);
		bool sym = ptot->has_symmetry();
		dbcsr::copy_matrix_to_3Dtensor_new<double>(*ptot,*m_ptot_bbd,sym);
		ptot->clear();
	}
	
	m_ptot_bbd->filter(dbcsr::global::filter_eps);
	
	m_gp_xd->batched_contract_init();
	m_ptot_bbd->batched_contract_init();
		
	reoint.start();
	m_eri3c2e_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	reoint.finish();
	
	int nbatches = m_eri3c2e_batched->nbatches(2);
	
	for (int inu = 0; inu != nbatches; ++inu) {
			
		fetch1.start();
		m_eri3c2e_batched->decompress({inu});
		auto eri_0_12 = m_eri3c2e_batched->get_work_tensor();
		fetch1.finish();
		
		con1.start();
		
		vec<vec<int>> bounds1 = {
			m_eri3c2e_batched->full_bounds(1),
			m_eri3c2e_batched->bounds(2, inu)
		};
		
		dbcsr::contract(*eri_0_12, *m_ptot_bbd, *m_gp_xd)
			.bounds1(bounds1).beta(1.0)
			.filter(dbcsr::global::filter_eps)
			.perform("XMN, MN_ -> X_");
					
		con1.finish();
			
	}
	
	m_eri3c2e_batched->decompress_finalize();
	
	m_gp_xd->batched_contract_finalize();
	m_ptot_bbd->batched_contract_finalize();
			
	LOG.os<1>("X_, XY -> Y_\n");
	
	dbcsr::copy_matrix_to_tensor(*m_v_inv, *m_v_inv_01);
	
	dbcsr::contract(*m_gp_xd, *m_v_inv_01, *m_gq_xd)
		.filter(dbcsr::global::filter_eps).perform("X_, XY -> Y_");
	
	m_v_inv_01->clear();
	//dbcsr::print(*m_gq_xd);
	
	//dbcsr::print(*m_inv);
	
	m_J_bbd->batched_contract_init();
	m_gq_xd->batched_contract_init();
	
	m_eri3c2e_batched->decompress_init({2},vec<int>{0},vec<int>{1,2});
	
	for (int inu = 0; inu != nbatches; ++inu) {
			
		fetch2.start();
		m_eri3c2e_batched->decompress({inu});
		auto eri_0_12 = m_eri3c2e_batched->get_work_tensor();
		fetch2.finish();
	
		con2.start();
			
		vec<vec<int>> bounds3 = {
			m_eri3c2e_batched->full_bounds(1),
			m_eri3c2e_batched->bounds(2, inu)
		};
	
		dbcsr::contract(*m_gq_xd, *eri_0_12, *m_J_bbd)
			.bounds3(bounds3).beta(1.0)
			.filter(dbcsr::global::filter_eps / nbatches)
			.perform("X_, XMN -> MN_");
					
		con2.finish();
				
	}
	
	m_eri3c2e_batched->decompress_finalize();
	
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

BATCHED_DFMO_K::BATCHED_DFMO_K(dbcsr::world w, desc::smolecule mol, int print)
	: K(w,mol,print,"BATCHED_DFMO_K") {}

void BATCHED_DFMO_K::init() {
	
	init_base();
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	arrvec<int,2> bb = {b,b};
	arrvec<int,2> xx = {x,x};
	
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	m_K_01 = dbcsr::tensor_create<2>().pgrid(m_spgrid2).name("K_01")
		.map1({0}).map2({1}).blk_sizes(bb).get();
	
	m_v_invsqrt_01 = dbcsr::tensor_create<2>().pgrid(m_spgrid2).name("s_xx_invsqrt")
		.map1({0}).map2({1}).blk_sizes(xx).get();
	dbcsr::copy_matrix_to_tensor(*m_v_invsqrt, *m_v_invsqrt_01);
			
}

void BATCHED_DFMO_K::compute_K() {
	
	TIME.start();
	
	auto b = m_mol->dims().b();
	auto X = m_mol->dims().x();
	
	auto compute_K_single = 
	[&] (dbcsr::smat_d& c_bm, dbcsr::smat_d& k_bb, std::string x) {
		
		auto& reo0 = TIME.sub("Reordering ints (1) " + x);
		auto& fetch1 = TIME.sub("Fetch ints " + x);
		//auto& retints = TIME.sub("Reordering ints (2) " + x);
		auto& con1 = TIME.sub("Contraction (1) " + x);
		auto& con2 = TIME.sub("Contraction (2) " + x);
		auto& con3 = TIME.sub("Contraction (3) " + x);
		auto& reo1 = TIME.sub("Reordering (1) " + x);
		auto& reo2 = TIME.sub("Reordering (2) " + x);
	
		vec<int> o, m, m_off;
		k_bb->clear();
		
		if (m_SAD_iter) {
			m = c_bm->col_blk_sizes();
			o = m;
		} else {
			m = c_bm->col_blk_sizes();
			o = (x == "A") ? m_mol->dims().oa() : m_mol->dims().ob();
		}
		
		// split it
		
		int occ_nbatches = m_occ_nbatches;
		vec<vec<int>> o_bounds = dbcsr::make_blk_bounds(o, occ_nbatches);
		
		vec<int> o_offsets(o.size());
		int off = 0;	
	
		for (int i = 0; i != o.size(); ++i) {
			o_offsets[i] = off;
			off += o[i];
		}
			
		for (int i = 0; i != o_bounds.size(); ++i) { 
			o_bounds[i][0] = o_offsets[o_bounds[i][0]];
			o_bounds[i][1] = o_offsets[o_bounds[i][1]]
				+ o[o_bounds[i][1]] - 1;
		}
		
		if (LOG.global_plev() >= 1) {
			LOG.os<1>("OCC bounds: ");
			for (auto p : o_bounds) {
				LOG.os<1>(p[0], " -> ", p[1]);
			} LOG.os<1>('\n');
		}
		
		arrvec<int,2> bm = {b,m};
		arrvec<int,3> xmb = {X,m,b};
		
		int nocc = std::accumulate(m.begin(), m.end(), 0);
		int nbf = std::accumulate(b.begin(), b.end(), 0);
		int xnbf = std::accumulate(x.begin(), x.end(), 0);
		
		std::array<int,2> tsizes2 = {nbf,nocc};
		std::array<int,3> tsizes3 = {xnbf,nocc,nbf};
		
		dbcsr::shared_pgrid<2> grid2 =
			dbcsr::create_pgrid<2>(m_world.comm()).tensor_dims(tsizes2).get();
			
		dbcsr::shared_pgrid<3> grid3 =
			dbcsr::create_pgrid<3>(m_world.comm()).tensor_dims(tsizes3).get();
		
		m_c_bm = dbcsr::tensor_create<2,double>().pgrid(grid2)
			.name("c_bm_" + x + "_0_1").map1({0}).map2({1})
			.blk_sizes(bm).get();
			
		dbcsr::copy_matrix_to_tensor(*c_bm, *m_c_bm);
							
		m_HT1_xmb_02_1 = dbcsr::tensor_create<3,double>().name("HT1_xmb_02_1_" + x)
			.pgrid(grid3).map1({0,2}).map2({1}).blk_sizes(xmb).get();
			
		m_HT1_xmb_0_12 = dbcsr::tensor_create_template<3>(m_HT1_xmb_02_1)
			.name("HT1_xmb_0_12_" + x).map1({0}).map2({1,2}).get();
			
		m_HT2_xmb_0_12 = dbcsr::tensor_create_template<3>(m_HT1_xmb_02_1)
			.name("HT2_xmb_0_12_" + x).map1({0}).map2({1,2}).get();
			
		m_HT2_xmb_01_2 = dbcsr::tensor_create_template<3>(m_HT1_xmb_02_1)
			.name("HT2_xmb_01_2_" + x).map1({0,1}).map2({2}).get();
		
		int64_t nze_HTI = 0;
		
		auto full = m_HT1_xmb_02_1->nfull_total();
		int64_t nze_HTI_tot = (int64_t)full[0] * (int64_t)full[1] * (int64_t)full[2];
		
		for (int iocc = 0; iocc != o_bounds.size(); ++iocc) {
			
			LOG.os<1>("IOCC = ", iocc, " ", o_bounds[iocc][0],
				" -> ", o_bounds[iocc][1], '\n');
			
			vec<vec<int>> o_tbounds = {
				o_bounds[iocc]
			};
			
			reo0.start();
			m_eri3c2e_batched->decompress_init({2}, vec<int>{0,2},vec<int>{1});
			reo0.finish();
			
			//m_c_bm->batched_contract_init();
			m_HT1_xmb_02_1->batched_contract_init();
			
			for (int inu = 0; inu != m_eri3c2e_batched->nbatches(2); ++inu) {
				
				//std::cout << "MBATCH: " << M << std::endl;
				
				fetch1.start();
				m_eri3c2e_batched->decompress({inu});
				auto eri_xbb_02_1 = m_eri3c2e_batched->get_work_tensor();
				fetch1.finish();
				
				vec<vec<int>> xn_bounds = {
					m_eri3c2e_batched->full_bounds(0),
					m_eri3c2e_batched->bounds(2, inu)
				};
				
				con1.start();
				dbcsr::contract(*eri_xbb_02_1,*m_c_bm,*m_HT1_xmb_02_1)
					.bounds2(xn_bounds).bounds3(o_tbounds).beta(1.0)
					.filter(dbcsr::global::filter_eps / m_eri3c2e_batched->nbatches(2))
					.perform("XMN, Mi -> XiN");
				con1.finish();
			
			}
			
			nze_HTI += m_HT1_xmb_02_1->num_nze_total();
			
			//m_c_bm->batched_contract_finalize();
			m_HT1_xmb_02_1->batched_contract_finalize();
			m_eri3c2e_batched->decompress_finalize();
			
			// end for M
			reo1.start();
			dbcsr::copy(*m_HT1_xmb_02_1,*m_HT1_xmb_0_12).move_data(true).perform();
			reo1.finish();
			
			vec<vec<int>> nu_o_bounds = {
				o_bounds[iocc],
				m_eri3c2e_batched->full_bounds(2)
			};
			
			con2.start();
			dbcsr::contract(*m_HT1_xmb_0_12,*m_v_invsqrt_01,*m_HT2_xmb_0_12)
				.bounds2(nu_o_bounds)
				.filter(dbcsr::global::filter_eps)
				.perform("XiN, XY -> YiN");
			con2.finish();
			m_HT1_xmb_0_12->clear();
			
			reo2.start();
			dbcsr::copy(*m_HT2_xmb_0_12,*m_HT2_xmb_01_2).move_data(true).perform();
			reo2.finish();
			
			auto HT2_xmb_01_2_copy = 
				dbcsr::tensor_create_template<3,double>(m_HT2_xmb_01_2)
				.name("HT2_xmb_01_2_copy").get();
				
			dbcsr:copy(*m_HT2_xmb_01_2, *HT2_xmb_01_2_copy).perform();
			
			//dbcsr::print(*m_HT2_xmb_01_2);
			//dbcsr::print(*HT2_xmb_01_2_copy);
			
			LOG.os<1>("Computing K_mn = HT_xim * HT_xin\n");
			
			m_K_01->batched_contract_init();
			m_HT2_xmb_01_2->batched_contract_init();
			HT2_xmb_01_2_copy->batched_contract_init();
			
			for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {
			
				vec<vec<int>> x_o_bounds = {
					m_eri3c2e_batched->bounds(0,ix),
					o_bounds[iocc]
				};
					
				con3.start();
				dbcsr::contract(*m_HT2_xmb_01_2,*HT2_xmb_01_2_copy,*m_K_01)
					.bounds1(x_o_bounds).beta(1.0)
					.filter(dbcsr::global::filter_eps/
						m_eri3c2e_batched->nbatches(0))
					.perform("XiM, XiN -> MN"); 
				con3.finish();
				
			}
			
			m_K_01->batched_contract_finalize();
			m_HT2_xmb_01_2->batched_contract_finalize();
			HT2_xmb_01_2_copy->batched_contract_finalize();
			
			m_HT2_xmb_01_2->clear();
			HT2_xmb_01_2_copy->clear();
							
		} // end for I
		
		double HTI_occupancy = (double)nze_HTI / (double)nze_HTI_tot;
		LOG.os<1>("Occupancy of HTI: ", HTI_occupancy * 100, "%\n");
				
		dbcsr::copy_tensor_to_matrix(*m_K_01,*k_bb);
		m_K_01->clear();
		k_bb->scale(-1.0);
		
		m_HT1_xmb_02_1->destroy();
		m_HT1_xmb_0_12->destroy();
		m_HT2_xmb_01_2->destroy();
		m_HT2_xmb_0_12->destroy();
		m_c_bm->destroy();
		
		grid2->destroy();
		grid3->destroy();
		
	}; // end lambda function
	
	compute_K_single(m_c_A, m_K_A, "A");
	
	if (m_K_B) compute_K_single(m_c_B, m_K_B, "B");
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_K_A);
		if (m_K_B) dbcsr::print(*m_K_B);
	}
	
	TIME.finish();
		
}

BATCHED_DFAO_K::BATCHED_DFAO_K(dbcsr::world w, desc::smolecule mol, int print)
	: K(w,mol,print,"BATCHED_DFAO_K") {}
void BATCHED_DFAO_K::init() {
	
	init_base();
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,3> xbb = {x,b,b};
	
	m_spgrid3_xbb = m_eri3c2e_batched->spgrid();
	
	// ========== END ==========
	
	arrvec<int,2> bb = {b,b};
	
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	m_cbar_xbb_01_2 = dbcsr::tensor_create<3,double>()
		.name("Cbar_xbb_01_2")
		.pgrid(m_spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0,1}).map2({2})
		.get();
	
	m_cbar_xbb_1_02 = 
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("Cbar_xbb_1_02").map1({1}).map2({0,2}).get();
	
	m_K_01 = dbcsr::tensor_create<2,double>().pgrid(m_spgrid2).name("K_01")
		.map1({0}).map2({1}).blk_sizes(bb).get();
		
	m_p_bb = dbcsr::tensor_create_template<2,double>(m_K_01)
			.name("p_bb_0_1").map1({0}).map2({1}).get();

}

void BATCHED_DFAO_K::compute_K() {
	
	TIME.start();
	
	auto compute_K_single = 
	[&] (dbcsr::smat_d& p_bb, dbcsr::smat_d& k_bb, std::string x) {
		
		LOG.os<1>("Computing exchange part (", x, ")\n");
		
		dbcsr::copy_matrix_to_tensor(*p_bb, *m_p_bb);
		m_p_bb->filter(dbcsr::global::filter_eps);			
		//dbcsr::print(*c_bm);
		//dbcsr::print(*m_c_bm);	
		
		// LOOP OVER X
		
		auto& reo_int = TIME.sub("Reordering ints " + x);
		auto& reo_1_batch = TIME.sub("Reordering (1)/batch " + x);
		auto& con_1_batch = TIME.sub("Contraction (1)/batch " + x);
		auto& con_2_batch = TIME.sub("Contraction (2)/batch " + x);
		auto& fetch = TIME.sub("Fetching integrals/batch " + x);
		auto& fetch2 = TIME.sub("Fetching fitting coeffs/batch " + x);
		//auto& retint = TIME.sub("Returning integrals/batch " + x);
	
		m_fitting_batched->decompress_init({2,0}, vec<int>{1}, vec<int>{0,2});
		
		m_K_01->batched_contract_init();
		m_cbar_xbb_01_2->batched_contract_init();
		m_cbar_xbb_1_02->batched_contract_init();
		
		int64_t nze_cbar = 0;
		//auto full = eri_01_2->nfull_total();
		//int64_t nze_cbar_tot = (int64_t)full[0] * (int64_t)full[1] * (int64_t)full[2];
		
		reo_int.start();
		m_eri3c2e_batched->decompress_init({0}, vec<int>{0,1}, vec<int>{2});
		reo_int.finish();
		
		for (int ix = 0; ix != m_eri3c2e_batched->nbatches(0); ++ix) {	
			
			// fetch integrals
			fetch.start();
			m_eri3c2e_batched->decompress({ix});
			auto eri_01_2 = m_eri3c2e_batched->get_work_tensor();
			fetch.finish();
			
			//m_p_bb->batched_contract_init();
			
			for (int inu = 0; inu != m_eri3c2e_batched->nbatches(2); ++inu) {
				
				vec<vec<int>> xm_bounds = { 
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->full_bounds(1)
				};
				
				vec<vec<int>> n_bounds = { 
					m_eri3c2e_batched->bounds(2,inu)
				};
			
				con_1_batch.start();
				dbcsr::contract(*eri_01_2, *m_p_bb, *m_cbar_xbb_01_2)
					.bounds2(xm_bounds)
					.bounds3(n_bounds)
					.filter(dbcsr::global::filter_eps)
					.perform("XMN, NL -> XML");
				con_1_batch.finish();
			
				nze_cbar += m_cbar_xbb_01_2->num_nze_total();
			
				//m_cbar_xbb_01_2->filter(dbcsr::global::filter_eps);
			
				vec<vec<int>> copy_bounds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->full_bounds(1),
					m_eri3c2e_batched->bounds(2,inu)
				};
			
				reo_1_batch.start();
				dbcsr::copy(*m_cbar_xbb_01_2, *m_cbar_xbb_1_02)
					.bounds(copy_bounds).move_data(true).perform();
				reo_1_batch.finish();
				
				//dbcsr::print(*m_cbar_xbb_02_1);
				
				//dbcsr::print(*m_cbar_xbb_02_1);
			
				// get c_xbb
				fetch2.start();
				m_fitting_batched->decompress({inu,ix});
				auto c_xbb_1_02 = m_fitting_batched->get_work_tensor();
				fetch2.finish();
				
				//dbcsr::print(*c_xbb_1_02);
				
				vec<vec<int>> xs_bounds = { 
					m_eri3c2e_batched->bounds(0,ix), 
					m_eri3c2e_batched->bounds(2, inu) 
				};
			
				con_2_batch.start();
				dbcsr::contract(*c_xbb_1_02, *m_cbar_xbb_1_02, *m_K_01)
					.bounds1(xs_bounds).beta(1.0)
					.filter(dbcsr::global::filter_eps / m_eri3c2e_batched->nbatches(2))
					.perform("XNS, XMS -> MN");
				con_2_batch.finish();
								
				m_cbar_xbb_1_02->clear();
				
			}
			
			//m_p_bb->batched_contract_finalize();
			
		}
		
		m_eri3c2e_batched->decompress_finalize();
		m_fitting_batched->decompress_finalize();
		
		m_cbar_xbb_01_2->batched_contract_finalize();
		m_cbar_xbb_1_02->batched_contract_finalize();
		
		//double occ_cbar = (double) nze_cbar / (double) nze_cbar_tot;
		//LOG.os<1>("Occupancy of cbar: ", occ_cbar, "%\n");
		
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

BATCHED_DFMEM_K::BATCHED_DFMEM_K(dbcsr::world w, desc::smolecule mol, int print)
	: K(w,mol,print,"BATCHED_DFMEM_K") {}
void BATCHED_DFMEM_K::init() {
	
	init_base();
			
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,3> xbb = {x,b,b};
	arrvec<int,2> bb = {b,b};
	arrvec<int,2> xx = {x,x};
	
	m_spgrid3_xbb = m_eri3c2e_batched->spgrid();
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	m_cbar_xbb_01_2 = dbcsr::tensor_create<3,double>()
		.name("Cbar_xbb_01_2")
		.pgrid(m_spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0,1}).map2({2})
		.get();
		
	m_cbar_xbb_02_1 = 
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("Cbar_xbb_0_12")
		.map1({0,2}).map2({1})
		.get();
		
	m_c_xbb_02_1 = 
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("c_xbb_02_1")
		.map1({0,2}).map2({1})
		.get();
	
	m_cpq_xbb_0_12 =
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("cbarpq_xbb_0_12")
		.map1({0}).map2({1,2})
		.get();
		
	m_cpq_xbb_01_2 = 
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("cpq_xbb_02_1")
		.map1({0,1}).map1({2})
		.get();
	
	m_K_01 = dbcsr::tensor_create<2,double>()
		.pgrid(m_spgrid2)
		.name("K_01")
		.map1({0}).map2({1})
		.blk_sizes(bb)
		.get();
		
	m_p_bb = dbcsr::tensor_create_template<2,double>(m_K_01)
			.name("p_bb_0_1")
			.map1({0})
			.map2({1})
			.get();
			
	m_v_xx_01 = dbcsr::tensor_create<2,double>()
		.name("v_xx_01")
		.pgrid(m_spgrid2)
		.blk_sizes(xx)
		.map1({0}).map2({1})
		.get();
		
}

void BATCHED_DFMEM_K::compute_K() {
	
	TIME.start();
	
	dbcsr::copy_matrix_to_tensor(*m_v_xx, *m_v_xx_01);
		
	auto compute_K_single = 
	[&] (dbcsr::smat_d& p_bb, dbcsr::smat_d& k_bb, std::string x) {
		
		LOG.os<1>("Computing exchange part (", x, ")\n");
		
		dbcsr::copy_matrix_to_tensor(*p_bb, *m_p_bb);
		m_p_bb->filter(dbcsr::global::filter_eps);				
		//dbcsr::print(*c_bm);
		//dbcsr::print(*m_c_bm);	
				
		auto& reo_int = TIME.sub("Reordering ints " + x);
		auto& reo_1 = TIME.sub("Reordering (1)/batch " + x);
		auto& reo_2 = TIME.sub("Reordering (2)/batch " + x);
		auto& reo_3 = TIME.sub("Reordering (3)/batch " + x);
		auto& con_1 = TIME.sub("Contraction (1)/batch " + x);
		auto& con_2 = TIME.sub("Contraction (2)/batch " + x);
		auto& con_3 = TIME.sub("Contraction (3)/batch " + x);
		auto& fetch = TIME.sub("Fetching coeffs/batch " + x);
		//auto& retint = TIME.sub("Returning integrals/batch " + x);
		
		reo_int.start();
		m_eri3c2e_batched->decompress_init({0}, vec<int>{0}, vec<int>{1,2});
		reo_int.finish();
		
		int nxbatches = m_eri3c2e_batched->nbatches(0);
		int nnbatches = m_eri3c2e_batched->nbatches(2);
		
		for (int ix = 0; ix != nxbatches; ++ix) {
			
			LOG.os<1>("BATCH X: ", ix, '\n');
			
			for (int iy = 0; iy != nxbatches; ++iy) {
				
				LOG.os<1>("BATCH Y: ", iy, '\n');
				
				fetch.start();
				m_eri3c2e_batched->decompress({iy});
				auto c_xbb_0_12 = m_eri3c2e_batched->get_work_tensor();
				fetch.finish();
				
				vec<vec<int>> ybds = {
					m_eri3c2e_batched->bounds(0,iy)
				};
				
				vec<vec<int>> xbds = {
					m_eri3c2e_batched->bounds(0,ix)
				};
				
				con_1.start();
				dbcsr::contract(*m_v_xx_01, *c_xbb_0_12, *m_cpq_xbb_0_12)
					.bounds1(ybds)
					.bounds2(xbds)
					.filter(dbcsr::global::filter_eps)
					.perform("XY, Ymr -> Xmr");
				con_1.finish();
				
				reo_1.start();
				dbcsr::copy(*m_cpq_xbb_0_12, *m_cpq_xbb_01_2)
					.move_data(true)
					.sum(true)
					.perform();
				reo_1.finish();
				
			}
			
			m_eri3c2e_batched->decompress({ix});
			auto c_xbb_0_12 = m_eri3c2e_batched->get_work_tensor();
			
			for (int isig = 0; isig != m_eri3c2e_batched->nbatches(2); ++isig) {
				
				LOG.os<1>("BATCH SIG: ", isig, '\n');
				
				vec<vec<int>> xmbds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->full_bounds(1)
				};
				
				vec<vec<int>> sbds = {
					m_eri3c2e_batched->bounds(2,isig)
				};
				
				con_2.start();
				dbcsr::contract(*m_cpq_xbb_01_2, *m_p_bb, *m_cbar_xbb_01_2)
					.bounds2(xmbds)
					.bounds3(sbds)
					.filter(dbcsr::global::filter_eps)
					.perform("Xmr, rs -> Xms");
				con_2.finish();
				
				vec<vec<int>> cpybds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->full_bounds(1),
					m_eri3c2e_batched->bounds(2,isig)
				};
				
				reo_2.start();
				dbcsr::copy(*m_cbar_xbb_01_2, *m_cbar_xbb_02_1)
					.bounds(cpybds)
					.move_data(true)
					.perform();
				reo_2.finish();
				
				reo_3.start();
				dbcsr:copy(*c_xbb_0_12, *m_c_xbb_02_1)
					.bounds(cpybds)
					.perform();
				reo_3.finish();
				
				vec<vec<int>> xsbds = {
					m_eri3c2e_batched->bounds(0,ix),
					m_eri3c2e_batched->bounds(2,isig)
				};
				
				con_3.start();
				dbcsr::contract(*m_c_xbb_02_1, *m_cbar_xbb_02_1, *m_K_01)
					.bounds1(xsbds)
					.filter(dbcsr::global::filter_eps/nxbatches)
					.beta(1.0)
					.perform("Xns, Xms -> mn");
				con_3.finish();
					
			}
						
			m_cpq_xbb_01_2->clear();
			
		}
		
		m_eri3c2e_batched->decompress_finalize();
			
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
	
	m_v_xx_01->clear();
	
	TIME.finish();
			
}
	
} // end namespace
