#include "hf/fock_builder.h"
#include "hf/hfdefaults.h"
#include "ints/aofactory.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include <dbcsr_conversions.hpp>
#include <dbcsr_tensor_ops.hpp>

#include <utility>
#include <iomanip>

namespace hf {
	
fockbuilder::fockbuilder(desc::smolecule mol, desc::options opt, dbcsr::world& w, int print,
	util::mpi_time& TIME_IN) :
	m_mol(mol), 
	m_opt(opt), 
	m_world(w),
	LOG(m_world.comm(),print),
	TIME(TIME_IN),
	m_use_df(opt.get<bool>("use_df", HF_USE_DF)) {

	// build the integrals 
	ints::aofactory ao(*m_mol, m_world);
	
	m_restricted = (m_mol->nele_alpha() == m_mol->nele_beta()) ? true : false;
	
	//std::cout << std::scientific;
	//std::cout << std::setprecision(12);
	
	if (!m_use_df) {
		
		auto& t_int2ele = TIME.sub("2e electron integrals");
		
		t_int2ele.start();
	
		LOG.os<>("Computing 2e integrals...\n");
		m_2e_ints = ao.op("coulomb").dim("bbbb").compute_4(vec<int>{0,1}, vec<int>{2,3});
		
		t_int2ele.finish();
		
		LOG.os<>("Done with 2e integrals...\n");
		
	} else {
		
		auto& t_int3c2e = TIME.sub("3-centre-2-electron integrals");
		
		t_int3c2e.start();
		
		LOG.os<>("Computing 3c2e integrals...\n");
		m_3c2e_ints = ao.op("coulomb").dim("xbb").compute_3(vec<int>{0},vec<int>{1,2});
		
		t_int3c2e.finish();
		
		LOG.os<>("Done with 3c2e integrals...\n");
		
		if (LOG.global_plev() >= 3) {
			//dbcsr::print(*m_3c2e_ints);
		}
		
		auto& t_metric = TIME.sub("Metric Matrix");
		
		t_metric.start();
		
		LOG.os<>("Computing metric...\n");
		auto metric_xx = ao.op("coulomb").dim("xx").compute();
		
		t_metric.finish();
		
		LOG.os<>("Done.\n");
		
		//m_xx = math::symmetrize(m_xx, "xx");
		
		auto& t_inv = TIME.sub("Inverse of metric matrix");
		
		t_inv.start();
		
		LOG.os<>("Computing inverse...\n");
		
		math::hermitian_eigen_solver solver(metric_xx, 'V', (LOG.global_plev() >= 2) ? true : false);
		
		solver.compute();
		
		auto inv_xx = solver.inverse();
		
		auto x = m_mol->dims().x();
		arrvec<int,2> xx = {x,x};
		
		dbcsr::pgrid<2> grid2(m_world.comm());
		
		m_inv_xx = dbcsr::make_stensor<2>(
			dbcsr::tensor<2>::create().name("metric inverse").ngrid(grid2)
				.map1({0}).map2({1}).blk_sizes(xx));
				
		dbcsr::copy_matrix_to_tensor(*inv_xx,*m_inv_xx);
		
		inv_xx->release();
		
		LOG.os<>("Done.\n");
		
		t_inv.finish();
		grid2.destroy();
		
		if (LOG.global_plev() >= 1) {
			LOG.os<1>("Block Distribution of 3c2e tensor: ");
			auto nblk = m_3c2e_ints->nblks_local();
			LOG.os<1>(nblk[0], " ", nblk[1], " ", nblk[2]);
			LOG.os<1>("Sparsity: ", m_3c2e_ints->occupation());
		}
		
		//m_xx->destroy();
		
	}
	
	auto b = m_mol->dims().b();
	mat f_bb_A = mat::create().name("f_bb_A").set_world(m_world)
		.row_blk_sizes(b).col_blk_sizes(b).type(dbcsr_type_hermitian);
		
	m_f_bb_A = f_bb_A.get_smatrix();
	
	if (!m_restricted && m_mol->nele_beta() != 0) {
		mat f_bb_B = mat::create_template(*m_f_bb_A).name("f_bb_B");
		m_f_bb_B = f_bb_B.get_smatrix();
	} else {
		m_f_bb_B = nullptr;
	}

}

void fockbuilder::build_j(stensor<2>& p_A, stensor<2>& p_B) {
	
	// build PT = PA + PB
	dbcsr::tensor<2> pt = dbcsr::tensor<2>::create_template().tensor_in(*p_A).name("ptot");
	dbcsr::copy(*p_A, pt).perform();
	
	if (p_B) {
		dbcsr::copy(*p_B, pt).sum(true).perform();
	} else {
		pt.scale(2.0);
	}
	
	if (LOG.global_plev() >= 2) dbcsr::print(pt);
	
	auto& t_j = TIME.sub("Coulomb Term");
	t_j.start();
	
	LOG.os<1>("Computing coulomb term... \n");
	
	if (!m_use_df) {
		
		auto pt_bbY = dbcsr::add_dummy(pt);
		
		dbcsr::pgrid<3> grid3(m_world.comm());
		arrvec<int,3> bbD = {m_mol->dims().b(),m_mol->dims().b(),vec<int>{1}};
		dbcsr::tensor<3> j_bbY = dbcsr::tensor<3>::create().name("j dummy").ngrid(grid3).map1({0,1}).map2({2}).blk_sizes(bbD);
	
		dbcsr::contract(pt_bbY, *m_2e_ints, j_bbY).perform("LS_, MNLS -> MN_");
		
		m_j_bb = (dbcsr::remove_dummy(j_bbY, vec<int>{0}, vec<int>{1}, "j_bb")).get_stensor();
		
		pt_bbY.destroy();
		j_bbY.destroy();
		
	} else {
		
		dbcsr::pgrid<2> grid2(m_world.comm());
		dbcsr::pgrid<3> grid3(m_world.comm());
		
		arrvec<int,2> xD = {m_mol->dims().x(),vec<int>{1}};
		arrvec<int,3> bbD = {m_mol->dims().b(),m_mol->dims().b(),vec<int>{1}};
		
		dbcsr::tensor<2> c_xD = dbcsr::tensor<2>::create().name("c_xD").ngrid(grid2).map1({0}).map2({1}).blk_sizes(xD);
		dbcsr::tensor<2> d_xD = dbcsr::tensor<2>::create().name("d_xD").ngrid(grid2).map1({0}).map2({1}).blk_sizes(xD);
		
		auto pt_bbD = dbcsr::add_dummy(pt);
		dbcsr::tensor<3> j_bbD = dbcsr::tensor<3>::create().name("j dummy").ngrid(grid3).map1({0,1}).map2({2}).blk_sizes(bbD);
		
		//cX("M") = PT("mu,nu") * B("M,mu,nu");
		dbcsr::contract(pt_bbD, *m_3c2e_ints, c_xD).perform("MN_, XMN -> X_");
		
		//dX("M") = Jinv("M,N") * cX("N");
		dbcsr::contract(*m_inv_xx, c_xD, d_xD).perform("XY, Y_ -> X_");
		
		//j("mu,nu") = dX("M") * B("M,mu,nu");
		dbcsr::contract(d_xD, *m_3c2e_ints, j_bbD).perform("X_, XMN -> MN_");
		
		m_j_bb = (dbcsr::remove_dummy(j_bbD, vec<int>{0}, vec<int>{1}, "j_bb")).get_stensor();
	
		c_xD.destroy();
		d_xD.destroy();
		j_bbD.destroy();
		pt_bbD.destroy();
		grid2.destroy();
		grid3.destroy();
	
	}
	
	t_j.finish();
	
	LOG.os<1>("Done.\n");
	
	if (LOG.global_plev() >= 2) dbcsr::print(*m_j_bb);
	
	pt.destroy();
	
}

void fockbuilder::build_k(stensor<2>& p_A, stensor<2>& p_B, stensor<2>& c_A, stensor<2>& c_B, bool SAD) {
	
	bool log = false;
	int u = 0;
	
	auto& t_k = TIME.sub("Exchange Term");
	
	dbcsr::pgrid<3> grid3(m_world.comm());
	
	t_k.start();
	
	if (!m_use_df) {
		
		auto make_k = [&](dbcsr::stensor<2,double>& p_bb, dbcsr::stensor<2,double>& k_bb, std::string x) {
		
			LOG.os<1>("Computing exchange term (", x, ") ... \n");
		
			auto p_bbY = dbcsr::add_dummy(*p_bb);
			arrvec<int,3> bbD = {m_mol->dims().b(),m_mol->dims().b(),vec<int>{1}};
		
			tensor<3> k_bbY = tensor<3>::create().name("k dummy").ngrid(grid3).map1({0,1}).map2({2}).blk_sizes(bbD);
			
			dbcsr::contract(p_bbY, *m_2e_ints, k_bbY).alpha(-1.0).perform("LS_, MLSN -> MN_");
		
			k_bb = (dbcsr::remove_dummy(k_bbY, vec<int>{0}, vec<int>{1}, "k_bb_" + x)).get_stensor();
			
			p_bbY.destroy();
			k_bbY.destroy();
		
		};
		
		make_k(p_A, m_k_bb_A, "A");
		
		if (p_B && c_B) 
			make_k(p_B, m_k_bb_B, "B");
			
		
		
	} else {
		
		auto make_k = [&](dbcsr::stensor<2,double>& p_bb, dbcsr::stensor<2,double>& c_bm, 
			dbcsr::stensor<2,double>& k_bb, std::string x, bool SAD_iter) {
				
			LOG.os<1>("Computing exchange term (", x, ") ... \n");
			
			if (!k_bb) {
				k_bb = dbcsr::make_stensor<2>(
				tensor<2>::create_template().tensor_in(*p_bb).name("k_bb_"+x));
			}
			
			vec<int> o = c_bm->blk_sizes()[1];
			
			arrvec<int,3> HTsizes = {m_mol->dims().x(), m_mol->dims().b(), o};
			
			dbcsr::tensor<3> HT = dbcsr::tensor<3>::create().name("HT_"+x).ngrid(grid3)
				.map1({0,1}).map2({2}).blk_sizes(HTsizes);
			dbcsr::tensor<3> D = dbcsr::tensor<3>::create().name("D_"+x).ngrid(grid3)
				.map1({0}).map2({1,2}).blk_sizes(HTsizes);
			
			// HTa("M,mu,i") = Coa("nu,i") * B("M,mu,nu");
			if (!SAD_iter) {
				
				int nocc = (x == "A") ? m_mol->nocc_alpha() - 1 : m_mol->nocc_beta() - 1;	
				
				//std::cout << "NOCC: " << nocc << std::endl;
							
				vec<vec<int>> occ_bounds = {{0,nocc}};
				
				dbcsr::contract(*c_bm, *m_3c2e_ints, HT).bounds2(occ_bounds).perform("Ni, XMN -> XMi");
				
			} else {
				
				dbcsr::contract(*c_bm, *m_3c2e_ints, HT).perform("Ni, XMN -> XMi");
				
			}
			
			HT.filter();
			
			//Da("M,mu,i") = HTa("N,mu,i") * Jinv("N,M");
			dbcsr::contract(HT, *m_inv_xx, D).perform("XMi, XY -> YMi");
			
			D.filter();
			
			//ka("mu,nu") = HTa("M,mu,i") * Da("M,nu,i");
			dbcsr::contract(HT, D, *k_bb).alpha(-1.0).perform("XMi, XNi -> MN");
						
			HT.destroy();
			D.destroy();
		
		};
		
		make_k(p_A, c_A, m_k_bb_A, "A", SAD);
		
		if (p_B && c_B) 
			make_k(p_B, c_B, m_k_bb_B, "B", SAD);
			
	}
	
	t_k.finish();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_k_bb_A);
		if (p_B && c_B) dbcsr::print(*m_k_bb_B);
	}
	
	grid3.destroy();
	
	
}

void fockbuilder::compute(smat& core, smat& p_A, smat& c_A, smat& p_B, smat& c_B, bool SAD) {
	
	dbcsr::pgrid<2> grid2(m_world.comm());
	
	auto b = c_A->row_blk_sizes();
	auto m = c_A->col_blk_sizes();
	
	arrvec<int,2> bb = {b,b};
	arrvec<int,2> bm = {b,m};
	
	stensor<2> t_p_A = dbcsr::make_stensor<2>(
		tensor<2>::create().name("pA tensor").ngrid(grid2).map1({0}).map2({1}).blk_sizes(bb));
		
	stensor<2> t_c_A = dbcsr::make_stensor<2>(
		tensor<2>::create().name("cA tensor").ngrid(grid2).map1({0}).map2({1}).blk_sizes(bm));
		
	dbcsr::copy_matrix_to_tensor(*p_A, *t_p_A);
	dbcsr::copy_matrix_to_tensor(*c_A, *t_c_A);
		
	stensor<2> t_p_B = nullptr;
	stensor<2> t_c_B = nullptr;
	
	if (p_B) {
		t_p_B = dbcsr::make_stensor<2>(
			tensor<2>::create_template().tensor_in(*t_p_A).name("pB tensor"));
		dbcsr::copy_matrix_to_tensor(*p_B, *t_p_B);
	}
	
	if (c_B) {
		
		auto mB = c_B->col_blk_sizes();
		arrvec<int,2> bmB = {b,mB};
		
		t_c_B = dbcsr::make_stensor<2>(
			tensor<2>::create().name("cB tensor").ngrid(grid2).map1({0}).map2({1}).blk_sizes(bmB));

		dbcsr::copy_matrix_to_tensor(*c_B, *t_c_B);
	}
	
	build_j(t_p_A, t_p_B);
	build_k(t_p_A, t_p_B, t_c_A, t_c_B, SAD);
	
	//dbcsr::print(*m_f_bb_A);
	
	m_f_bb_A->clear();
	m_f_bb_A->copy_in(*core);
	dbcsr::copy_tensor_to_matrix(*m_j_bb, *m_f_bb_A, true);
	dbcsr::copy_tensor_to_matrix(*m_k_bb_A, *m_f_bb_A, true);
	
	m_k_bb_A->clear();
	
	if (m_f_bb_B) {
		m_f_bb_B->clear();
		m_f_bb_B->copy_in(*core);
		dbcsr::copy_tensor_to_matrix(*m_j_bb, *m_f_bb_B, true);
		dbcsr::copy_tensor_to_matrix(*m_k_bb_B, *m_f_bb_B, true);
		m_k_bb_B->clear();
	}
	
	m_j_bb->clear();
	
	LOG.os<2>("Finished Fock Matrix:\n");
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_f_bb_A);
		if (m_f_bb_B) dbcsr::print(*m_f_bb_B);
	}
	
}



		
} // end namespace
