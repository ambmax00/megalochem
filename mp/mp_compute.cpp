#include "mp/mpmod.h"
#include "mp/mp_defaults.h"
#include "math/laplace/laplace.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include "math/linalg/piv_cd.h"
#include "ints/aofactory.h"
#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_tensor_ops.hpp>

#include <libint2/basis.h>

using mat_d = dbcsr::mat_d;
using tensor2_d = dbcsr::tensor2_d;
using tensor3_d = dbcsr::tensor3_d;

using smat_d = dbcsr::smat_d;
using stensor2_d = dbcsr::stensor2_d;
using stensor3_d = dbcsr::stensor3_d;

namespace mp {
	
mpmod::mpmod(desc::shf_wfn& wfn_in, desc::options& opt_in, dbcsr::world& w_in) :
	m_hfwfn(wfn_in),
	m_opt(opt_in),
	m_world(w_in),
	LOG(m_world.comm(),m_opt.get<int>("print", MP_PRINT_LEVEL)),
	TIME(m_world.comm(), "Moller Plesset", LOG.global_plev())
{
	std::string dfbasname = m_opt.get<std::string>("dfbasis");
	
	libint2::BasisSet dfbas(dfbasname,m_hfwfn->mol()->atoms());
	m_hfwfn->mol()->set_dfbasis(dfbas);
	
}

void mpmod::compute() {
	
	LOG.banner<>("CD-LT-SOS-RI-MP2", 50, '*');
	
	// to do:
	// 1. Insert Screening
	// 2. Impose sparsity on B_XBB
	// 3. LOGging and TIMEing
	
	// get energies
	auto eps_o = m_hfwfn->eps_occ_A();
	auto eps_v = m_hfwfn->eps_vir_A();
	
	auto mol = m_hfwfn->mol();
	
	int nlap = m_opt.get<int>("nlap",MP_NLAP);
	double c_os = m_opt.get<double>("c_os",MP_C_OS);
	
	// laplace
	double emin = eps_o->front();
	double ehomo = eps_o->back();
	double elumo = eps_v->front();
	double emax = eps_v->back();
	
	LOG.os<>("eps_min/eps_homo/eps_lumo/eps_max ", emin, " ", ehomo, " ", elumo, " ", emax, '\n');
	
	math::laplace lp(nlap, emin, ehomo, elumo, emax);
	
	lp.compute();
	
	auto lp_omega = lp.omega();
	auto lp_alpha = lp.alpha();
	
	// integrals
	ints::aofactory fac(m_hfwfn->mol(), m_world);
	
	auto B_xbb_1_02 = fac.ao_3c2e(vec<int>{1},vec<int>{0,2}); // check screening!
	auto M_xy = fac.ao_3coverlap();
	
	math::hermitian_eigen_solver hsolver(M_xy, 'V');
	hsolver.compute();
	
	auto inv_xx = hsolver.inverse();
	
	auto b = mol->dims().b();
	auto x = mol->dims().x();
	auto o = mol->dims().oa();
	
	// load coefficient matrices
	auto p_occ = m_hfwfn->po_bb_A();
	auto p_vir = m_hfwfn->pv_bb_A();
	
	auto c_occ = m_hfwfn->c_bo_A();
	auto c_vir = m_hfwfn->c_bv_A();
	
	// matrices and tensors
	
	smat_d c_occ_exp = std::make_shared<mat_d>(
		mat_d::create_template(*c_occ).name("Scaled Occ Coeff"));
		
	smat_d c_vir_exp = std::make_shared<mat_d>(
		mat_d::create_template(*c_vir).name("Scaled Vir Coeff"));
		
	smat_d pseudo_occ = std::make_shared<mat_d>(
		mat_d::create_template(*p_occ).name("Pseudo Density (OCC)"));
		
	smat_d pseudo_vir = std::make_shared<mat_d>(
		mat_d::create_template(*p_vir).name("Pseudo Density (VIR)"));
		
	smat_d Z_XX = std::make_shared<mat_d>(
		mat_d::create_template(*inv_xx).name("Z_xx").type(dbcsr_type_no_symmetry));
		
	smat_d Ztilde_XX = std::make_shared<mat_d>(
		mat_d::create_template(*inv_xx).name("Ztilde_xx").type(dbcsr_type_no_symmetry));
	
	arrvec<int,2> bo = {b,o};
	arrvec<int,2> bb = {b,b};
	arrvec<int,2> xx = {x,x};
	
	dbcsr::pgrid<2> grid2(m_world.comm());
	
	stensor2_d L_bo_0_1 = dbcsr::make_stensor<2>(
		tensor2_d::create().ngrid(grid2).name("L_bo_0_1")
		.map1({0}).map2({1}).blk_sizes(bo));
		
	stensor2_d pseudo_vir_0_1 = dbcsr::make_stensor<2>(
		tensor2_d::create().ngrid(grid2).name("pseudo_vir_0_1")
		.map1({0}).map2({1}).blk_sizes(bb));
		
	stensor2_d Z_XX_0_1 = dbcsr::make_stensor<2>(
		tensor2_d::create().ngrid(grid2).name("Z_xx_0_1")
		.map1({0}).map2({1}).blk_sizes(xx));
		
	dbcsr::pgrid<3> grid3(m_world.comm());
	
	arrvec<int,3> xob = {x,o,b};
	
	stensor3_d B_xob_1_02 = dbcsr::make_stensor<3>(
		tensor3_d::create().ngrid(grid3).name("B_Xob_1_02").map1({1}).map2({0,2})
		.blk_sizes(xob));
		
	stensor3_d B_xob_2_01 = dbcsr::make_stensor<3>(
		tensor3_d::create_template(*B_xob_1_02).name("B_Xob_2_01").map1({2}).map2({0,1}));
		
	stensor3_d B_xoB_2_01 = dbcsr::make_stensor<3>(
		tensor3_d::create_template(*B_xob_1_02).name("B_XoB_2_01").map1({2}).map2({0,1}));
		
	stensor3_d B_xBB_1_02 = dbcsr::make_stensor<3>(
		tensor3_d::create_template(*B_xbb_1_02).name("B_XBB_1_02").map1({1}).map2({0,2}));
		
	stensor3_d B_xBB_0_12 = dbcsr::make_stensor<3>(
		tensor3_d::create_template(*B_xbb_1_02).name("B_XBB_0_12").map1({0}).map2({1,2}));
		
	stensor3_d B_xbb_12_0 = dbcsr::make_stensor<3>(
		tensor3_d::create_template(*B_xbb_1_02).name("B_xbb_12_0").map1({1,2}).map2({0}));
	
	double mp2_energy = 0.0;
	
	// loop over laplace points 
	for (int ilap = 0; ilap != nlap; ++ilap) {
		
		LOG.os<>("LAPLACE POINT ", ilap, '\n');
		
		LOG.os<>("Forming pseudo densities.\n");
		
		std::vector<double> exp_occ = *eps_o;
		std::vector<double> exp_vir = *eps_v;
		
		double alpha = lp_alpha[ilap];
		double omega = lp_omega[ilap];
		
		std::for_each(exp_occ.begin(),exp_occ.end(),
			[alpha](double& eps) {
				eps = exp(0.5 * eps * alpha);
			});
			
		std::for_each(exp_vir.begin(),exp_vir.end(),
			[alpha](double& eps) {
				eps = exp(-0.5 * eps * alpha);
			});
			
		c_occ_exp->copy_in(*c_occ);
		c_vir_exp->copy_in(*c_vir);
		
		dbcsr::print(*c_occ_exp);
		
		for (auto x : exp_occ) {
			std::cout << x << " ";
		} std::cout << std::endl;
		
		c_occ_exp->scale(exp_occ, "right");
		c_vir_exp->scale(exp_vir, "right");
		
		dbcsr::print(*c_occ_exp);
		
		c_occ_exp->filter();
		c_vir_exp->filter();
		
		dbcsr::multiply('N', 'T', *c_occ_exp, *c_occ_exp, *pseudo_occ).alpha(pow(omega,0.25)).perform();
		dbcsr::multiply('N', 'T', *c_vir_exp, *c_vir_exp, *pseudo_vir).alpha(pow(omega,0.25)).perform();
		
		if (LOG.global_plev() >= 2) {
			dbcsr::print(*pseudo_occ);
			dbcsr::print(*pseudo_vir);
		}
		
		// make chol decomposition
		
		math::pivinc_cd chol(pseudo_occ, 0);
		
		chol.compute();
		
		int rank = chol.rank();
		
		LOG.os<>("Cholesky decomposition rank: ", rank, '\n');
	
		auto L_bo = chol.L(b, o);
		
		//copy over
		dbcsr::copy_matrix_to_tensor(*L_bo, *L_bo_0_1);
		
		// first transform 
		LOG.os<1>("First transform.\n");
		dbcsr::contract(*L_bo_0_1, *B_xbb_1_02, *B_xob_1_02).print(true).perform("mi, Xmn -> Xin");
		
		// reorder
		LOG.os<1>("Reordering B_xob.\n");
		dbcsr::copy(*B_xob_1_02, *B_xob_2_01).move_data(true).perform();
		
		//copy over vir density
		dbcsr::copy_matrix_to_tensor(*pseudo_vir, *pseudo_vir_0_1);
		
		// second transform
		LOG.os<1>("Second transform.\n");
		dbcsr::contract(*pseudo_vir_0_1, *B_xob_2_01, *B_xoB_2_01).move(true).print(true).perform("Nn, Xin -> XiN");
		
		// reorder
		LOG.os<1>("Reordering B_xoB.\n");
		dbcsr::copy(*B_xoB_2_01, *B_xob_1_02).move_data(true).perform();
		
		// final contraction
		LOG.os<1>("Final transform.\n");
		dbcsr::contract(*L_bo_0_1, *B_xob_1_02, *B_xBB_1_02).move(true).print(true).perform("Mi, XiN -> XMN");
		
		// reorder
		LOG.os<1>("Reordering B_xbb and B_xBB.\n");
		dbcsr::copy(*B_xBB_1_02, *B_xBB_0_12).move_data(true).perform();
		dbcsr::copy(*B_xbb_1_02, *B_xbb_12_0).move_data(true).perform();
		
		// form Z
		LOG.os<1>("Forming Z.\n");
		dbcsr::contract(*B_xBB_0_12, *B_xbb_12_0, *Z_XX_0_1).print(true).perform("Mmn, Nmn -> MN");
		
		B_xBB_0_12->clear();
		
		// reorder back integrals
		dbcsr::copy(*B_xbb_12_0, *B_xbb_1_02).move_data(true).perform();
		
		// copy
		dbcsr::copy_tensor_to_matrix(*Z_XX_0_1, *Z_XX);
		Z_XX_0_1->clear();
		
		// multiply
		LOG.os<1>("Ztilde = Z * Jinv\n");
		dbcsr::multiply('N', 'N', *Z_XX, *inv_xx, *Ztilde_XX).perform();
		
		dbcsr::print(*Ztilde_XX);
		
		LOG.os<1>("Local reduction.\n");
		
		dbcsr::iter_d iter(*Ztilde_XX);
		
		double sum = 0.0;
		
		int nblks = x.size();

		smat_d Ztilde_XX_t = std::make_shared<mat_d>(
			mat_d::transpose(*Ztilde_XX));
			
		dbcsr::print(*Ztilde_XX);
		dbcsr::print(*Ztilde_XX_t);
		
		auto loc_rows = Ztilde_XX->local_rows();
		auto loc_cols = Ztilde_XX->local_cols();

#pragma omp parallel for collapse(2) reduction(+:sum)
		for (int i = 0; i != loc_rows.size(); ++i) {
			for (int j = 0; j != loc_cols.size(); ++j) {
		//for (auto iblk : loc_rows) {
		//	for (auto jblk : loc_cols) {
				int iblk = loc_rows[i];
				int jblk = loc_cols[j];
				
				//std::cout << iblk << " " << jblk << std::endl;
				
				bool found1 = false;
				bool found2 = false;
				
				auto blk = Ztilde_XX->get_block_p(iblk,jblk,found1);
				auto blk_t = Ztilde_XX_t->get_block_p(iblk,jblk,found2);
				
				for (int i = 0; i != blk.ntot(); ++i) {
					std::cout << blk.data()[i] << " " << blk_t.data()[i] << std::endl;
				}
				
				if (!found1 || !found2) continue;
				
				//std::cout << "COMPUTE." << std::endl;
				
				sum += std::inner_product(blk.data(), blk.data() + blk.ntot(),
					blk_t.data(), 0.0);
					
			}
		}

		double total = 0.0;
		LOG.os<1>("Global reduction.\n");

		MPI_Allreduce(&sum, &total, 1, MPI_DOUBLE, MPI_SUM, m_world.comm());
		
		LOG.os<>("Partial sum: ", total, '\n');
		
		mp2_energy += total;
		
	}
	
	//mp2_energy *= c_os;
	
	LOG.os<>("Final MP2 energy: ", mp2_energy, '\n');

}

} // end namespace 
