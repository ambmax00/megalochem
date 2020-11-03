#include "fock/jkbuilder.h"

namespace fock {
	
BATCHED_DFSPARSE_K::BATCHED_DFSPARSE_K(dbcsr::world& w, desc::options& opt) 
	: K(w,opt,"BATCHED_DFSPARSE_K") {}
void BATCHED_DFSPARSE_K::init() {
	
	init_base();
		
	m_c_xbb_batched = m_reg.get<dbcsr::sbtensor<3,double>>(Kkey::dfit_xbb);
	
	m_v_xx = m_reg.get<dbcsr::shared_matrix<double>>(Kkey::v_xx);
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,3> xbb = {x,b,b};
	arrvec<int,2> bb = {b,b};
	arrvec<int,2> xx = {x,x};
	
	m_spgrid3_xbb = m_c_xbb_batched->spgrid();
	m_spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	m_cbar_xbb_01_2 = dbcsr::tensor_create<3,double>()
		.name("cbar_xbb_01_2")
		.pgrid(m_spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0,1}).map2({2})
		.get();
		
	m_cbar_xbb_0_12 = 
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("cbar_xbb_0_12")
		.map1({0}).map2({1,2})
		.get();
		
	m_c_xbb_02_1 = 
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("c_xbb_02_1")
		.map1({0,2}).map2({1})
		.get();
	
	m_cbarpq_xbb_0_12 =
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("cbarpq_xbb_0_12")
		.map1({0}).map2({1,2})
		.get();
		
	m_cbarpq_xbb_02_1 = 
		dbcsr::tensor_create_template<3>(m_cbar_xbb_01_2)
		.name("cpq_xbb_02_1")
		.map1({0,2}).map1({1})
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
		
	// indices of blocks of (X|ms) where Xs are relevant
	int nbf = std::accumulate(b.begin(), b.end(), 0);
	int nxbf = std::accumulate(x.begin(), x.end(), 0);
	
	Eigen::MatrixXi idx_list_local = Eigen::MatrixXi::Zero(nxbf,nbf);
	m_idx_list = idx_list_local;
	
	m_c_xbb_batched->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	int nbatches = (m_c_xbb_batched->get_type() == dbcsr::btype::core) ? 1 : 
		m_c_xbb_batched->nbatches(2);
	
	for (int inu = 0; inu != nbatches; ++inu) {
		
		m_c_xbb_batched->decompress({inu});
		auto c_xbb = m_c_xbb_batched->get_work_tensor();
		
		dbcsr::iterator_t<3,double> iter(*c_xbb);
		iter.start();
		
		while (iter.blocks_left()) {
			iter.next_block();
			auto& idx = iter.idx();
			
			idx_list_local(idx[0], idx[1]) = 1;
		}
		iter.stop();
	}
	
	m_c_xbb_batched->decompress_finalize();	
	MPI_Allreduce(idx_list_local.data(), m_idx_list.data(), nbf * nxbf, 
		MPI_INT, MPI_LOR, m_world.comm());
		
}

void BATCHED_DFSPARSE_K::compute_K() {
	
	TIME.start();
	
	dbcsr::copy_matrix_to_tensor(*m_v_xx, *m_v_xx_01);
	
	auto compute_K_single = 
	[&] (dbcsr::smat_d& p_bb, dbcsr::smat_d& k_bb, std::string x) {
		
		LOG.os<1>("Computing exchange part (", x, ")\n");
		
		dbcsr::copy_matrix_to_tensor(*p_bb, *m_p_bb);
				
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
		m_c_xbb_batched->decompress_init({0}, vec<int>{0,1}, vec<int>{2});
		reo_int.finish();
		
		int nxbatches = m_c_xbb_batched->nbatches(0);
		int nnbatches = m_c_xbb_batched->nbatches(2);
		
		for (int ix = 0; ix != nxbatches; ++ix) {
			
			LOG.os<1>("BATCH X: ", ix, '\n');
			
			for (int iy = 0; iy != nxbatches; ++iy) {
				
				LOG.os<1>("BATCH Y: ", iy, '\n');
				
				fetch.start();
				m_c_xbb_batched->decompress({iy});
				auto c_xbb_01_2 = m_c_xbb_batched->get_work_tensor();
				fetch.finish();
				
				vec<vec<int>> ymbds = {
					m_c_xbb_batched->bounds(0,iy),
					m_c_xbb_batched->full_bounds(1)
				};
				
				con_1.start();
				dbcsr::contract(*c_xbb_01_2, *m_p_bb, *m_cbar_xbb_01_2)
					.bounds2(ymbds)
					.filter(dbcsr::global::filter_eps)
					.perform("Ymr, rs -> Yms");
				con_1.finish();
				
				vec<vec<int>> cpybds = {
					m_c_xbb_batched->bounds(0,iy),
					m_c_xbb_batched->full_bounds(1),
					m_c_xbb_batched->full_bounds(2)
				};
				
				reo_1.start();
				dbcsr::copy(*m_cbar_xbb_01_2, *m_cbar_xbb_0_12)
					.bounds(cpybds)
					.move_data(true)
					.perform();
				reo_1.finish();
				
				vec<vec<int>> ybds = {
					m_c_xbb_batched->bounds(0,iy)
				};
				
				vec<vec<int>> xbds = {
					m_c_xbb_batched->bounds(0,ix)
				};
				
				con_2.start();
				dbcsr::contract(*m_v_xx_01, *m_cbar_xbb_0_12, *m_cbarpq_xbb_0_12)
					.bounds1(ybds)
					.bounds2(xbds)
					.filter(dbcsr::global::filter_eps)
					.perform("XY, Yms -> Xms");
				con_2.finish();
			
				vec<vec<int>> cpybds2 = {
					m_c_xbb_batched->bounds(0,ix),
					m_c_xbb_batched->full_bounds(1),
					m_c_xbb_batched->full_bounds(2)
				};
				
				reo_2.start();
				dbcsr::copy(*m_cbarpq_xbb_0_12, *m_cbarpq_xbb_02_1)
					.move_data(true)
					.bounds(cpybds2)
					.sum(true)
					.perform();
				reo_2.finish();
					
			}
						
			m_c_xbb_batched->decompress({ix});
			auto c_xbb_01_2 = m_c_xbb_batched->get_work_tensor();
						
			for (int isig = 0; isig != nnbatches; ++isig) {
				
				LOG.os<1>("BATCH ISIG: ", isig, '\n');
				
				vec<vec<int>> cpybds = {
					m_c_xbb_batched->bounds(0,ix),
					m_c_xbb_batched->full_bounds(1),
					m_c_xbb_batched->bounds(2,isig)
				};
				
				reo_3.start();
				dbcsr::copy(*c_xbb_01_2, *m_c_xbb_02_1)
					.bounds(cpybds)
					.perform();
				reo_3.finish();
					
				vec<vec<int>> xsbds = {
					m_c_xbb_batched->bounds(0,ix),
					m_c_xbb_batched->bounds(2,isig)
				};
				
				con_3.start();
				dbcsr::contract(*m_c_xbb_02_1, *m_cbarpq_xbb_02_1, *m_K_01)
					.bounds1(xsbds)
					.filter(dbcsr::global::filter_eps/nxbatches)
					.beta(1.0)
					.perform("Xns, Xms -> mn");
				con_3.finish();
		
				m_c_xbb_02_1->clear();
			
			}
			
			m_cbarpq_xbb_02_1->clear();
			
		}
		
		m_c_xbb_batched->decompress_finalize();
			
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
