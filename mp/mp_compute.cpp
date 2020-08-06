#include "mp/mpmod.h"
#include "mp/mp_defaults.h"
#include "math/laplace/laplace.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include "math/linalg/LLT.h"
#include "math/linalg/piv_cd.h"
#include "ints/aofactory.h"
#include "ints/screening.h"
#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_tensor_ops.hpp>
#include <dbcsr_btensor.hpp>

#include <libint2/basis.h>

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

void mpmod::compute_batch() {

	LOG.banner<>("Batched CD-LT-SOS-RI-MP2", 50, '*');
	
	auto& laptime = TIME.sub("Computing laplace points");
	auto& scrtime = TIME.sub("Computing screener");
	auto& calcints = TIME.sub("Computing integrals");
	auto& invtime = TIME.sub("Inverting metric");
	auto& fetchints1 = TIME.sub("Fetching 3c2e ints (1)");
	auto& fetchints2 = TIME.sub("Fetching 3c2e ints (2)");
	auto& pseudotime = TIME.sub("Forming pseudo densities");
	auto& pcholtime = TIME.sub("Pivoted cholesky decomposition");
	auto& firsttran = TIME.sub("First transform");
	auto& sectran = TIME.sub("Second transform");
	auto& fintran = TIME.sub("Final transform");
	auto& reo1 = TIME.sub("Reordering (1)");
	auto& reo2 = TIME.sub("Reordering (2)");
	auto& reo3 = TIME.sub("Reordering (3)");
	auto& reo_ints1 = TIME.sub("Reordering ints (1)");
	auto& reo_ints2 = TIME.sub("Reordering ints (2)");
	auto& readtime = TIME.sub("Decompressing B_xBB");
	auto& writetime = TIME.sub("Decompressing B_xBB");
	auto& viewtime = TIME.sub("Setting view");
	auto& formZ = TIME.sub("Forming Z");
	auto& formZtilde = TIME.sub("Forming Z tilde");
	auto& redtime = TIME.sub("Reduction");
	
	TIME.start();
	
	// to do:
	// 1. Insert Screening -> done
	// 2. Impose sparsity on B_XBB
	// 3. LOGging and TIMEing
	
	// get energies
	auto eps_o = m_hfwfn->eps_occ_A();
	auto eps_v = m_hfwfn->eps_vir_A();
	
	auto mol = m_hfwfn->mol();
	
	auto o = mol->dims().oa();
	auto v = mol->dims().va();
	auto b = 
	mol->dims().b();
	auto x = mol->dims().x();
	
	int nbf = std::accumulate(b.begin(), b.end(), 0);
	int dfnbf = std::accumulate(x.begin(), x.end(), 0);
	
	// options
	int nlap = m_opt.get<int>("nlap",MP_NLAP);
	double c_os = m_opt.get<double>("c_os",MP_C_OS);
	int nbatches = m_opt.get<int>("nbatches", MP_NBATCHES);
	std::string eri_method = m_opt.get<std::string>("eris", MP_ERIS);
	std::string intermed_method = m_opt.get<std::string>("intermeds", MP_INTERMEDS);
	double cholprec = m_opt.get<double>("cholprec", MP_CHOLPREC);
	
	// laplace
	double emin = eps_o->front();
	double ehomo = eps_o->back();
	double elumo = eps_v->front();
	double emax = eps_v->back();
	
	double ymin = 2*(elumo - ehomo);
	double ymax = 2*(emax - emin);
	
	LOG.os<>("eps_min/eps_homo/eps_lumo/eps_max ", emin, " ", ehomo, " ", elumo, " ", emax, '\n');
	LOG.os<>("ymin/ymax ", ymin, " ", ymax, '\n');
	
	math::laplace lp(m_world.comm(), LOG.global_plev());
	
	laptime.start();
	lp.compute(nlap, ymin, ymax);
	laptime.finish();
	
	auto lp_omega = lp.omega();
	auto lp_alpha = lp.alpha();
	
	//==================================================================
	//                        PGRIDS
	//==================================================================
	
	auto spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	std::array<int,3> xbb_sizes = {dfnbf, nbf, nbf};
	
	auto spgrid3_xbb = dbcsr::create_pgrid<3>(m_world.comm()).tensor_dims(xbb_sizes).get();
	
	//==================================================================
	//                        INTEGRALS
	//==================================================================
	
	std::shared_ptr<ints::aofactory> aofac 
	 = std::make_shared<ints::aofactory>(m_hfwfn->mol(), m_world);
	
	// screening
	ints::shared_screener s_scr(new ints::schwarz_screener(aofac, "erfc_coulomb"));
	
	scrtime.start();
	s_scr->compute();
	scrtime.finish();
	
	aofac->ao_3c2e_setup("erfc_coulomb");
	
	auto B_xbb = aofac->ao_3c2e_setup_tensor(spgrid3_xbb, vec<int>{1},vec<int>{0,2});
	
	dbcsr::btype eri_type = dbcsr::invalid;
	dbcsr::btype intermed_type = dbcsr::invalid;
	
	if (eri_method == "core") {
		eri_type = dbcsr::core;
	} else if (eri_method == "direct") {
		eri_type = dbcsr::direct;
	} else if (eri_method == "disk") {
		eri_type = dbcsr::disk;
	} 
	
	if (intermed_method == "core") {
		intermed_type = dbcsr::core;
	} else if (intermed_method == "disk") {
		intermed_type = dbcsr::disk;
	} 
	
	dbcsr::sbtensor<3,double> B_xbb_batch = 
		std::make_shared<dbcsr::btensor<3,double>>(B_xbb,nbatches,eri_type,1);
	
	auto gen_func = aofac->get_generator(s_scr);
	B_xbb_batch->set_generator(gen_func);
	
	calcints.start();
	
	B_xbb_batch->compress_init({0});
	
	auto x_blk_bounds = B_xbb_batch->blk_bounds(0);
	auto nu_fullblk_bounds = B_xbb_batch->full_blk_bounds(1);
	auto mu_fullblk_bounds = B_xbb_batch->full_blk_bounds(2);
	
	for (int ix = 0; ix != x_blk_bounds.size(); ++ix) {
		
		vec<vec<int>> blkbounds = {
			x_blk_bounds[ix],
			nu_fullblk_bounds,
			mu_fullblk_bounds
		};
	
		if (eri_type != dbcsr::direct) {
			aofac->ao_3c2e_fill(B_xbb, blkbounds, s_scr);
			B_xbb->filter(dbcsr::global::filter_eps);
		}
				
		B_xbb_batch->compress({ix}, B_xbb);
		
	}
	
	auto eri = B_xbb_batch->get_stensor();
	
	//dbcsr::print(*eri);
	
	B_xbb_batch->compress_finalize();
	
	calcints.finish();
	
	//==================================================================
	//                          METRIC
	//==================================================================
	
	invtime.start();
	
	auto C_xx = aofac->ao_3coverlap("coulomb");
	auto S_erfc_xx = aofac->ao_3coverlap("erfc_coulomb");
	
	// Ctilde = (S C-1 S)-1
	
	// invert C
	//dbcsr::print(*S_erfc_xx);
	
	LOG.os<>("Inverting erfc overlap metric...\n");
	
	math::hermitian_eigen_solver solver(C_xx, 'V', true);
	
	solver.compute();
	
	auto C_inv_xx = solver.inverse();
	//auto Ctilde_xx = solver.inverse();
	
	LOG.os<>("Forming tilde inv ...\n");
	
	auto Ctilde_inv_xx = dbcsr::create_template(C_xx)
		.name("Ctilde_inv_xx").get();
		
	auto temp = dbcsr::create_template(C_xx)
		.name("temp")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	C_xx->clear();
		
	dbcsr::multiply('N', 'N', *S_erfc_xx, *C_inv_xx, *temp).perform();
	dbcsr::multiply('N', 'N', *temp, *S_erfc_xx, *Ctilde_inv_xx).perform();
	
	S_erfc_xx->release();
	C_inv_xx->release();
	temp->release();
	
	math::hermitian_eigen_solver solver2(Ctilde_inv_xx,'V',true);
	
	solver2.compute();
	
	auto Ctilde_xx = solver2.inverse();
	
	Ctilde_inv_xx->release();
	
	Ctilde_xx->filter(dbcsr::global::filter_eps);
	
	invtime.finish();
	
	//==================================================================
	//                         SETUP OTHER TENSORS
	//==================================================================
	
	auto p_occ = m_hfwfn->po_bb_A();
	auto p_vir = m_hfwfn->pv_bb_A();
	
	auto c_occ = m_hfwfn->c_bo_A();
	auto c_vir = m_hfwfn->c_bv_A();
	
	// matrices and tensors
	
	auto c_occ_exp = dbcsr::create_template(c_occ)
		.name("Scaled Occ Coeff").get();
		
	auto c_vir_exp = dbcsr::create_template(c_vir)
		.name("Scaled Vir Coeff").get();
		
	auto pseudo_occ = dbcsr::create_template(p_occ)
		.name("Pseudo Density (OCC)").get();
		
	auto pseudo_vir = dbcsr::create_template(p_vir)
		.name("Pseudo Density (VIR)").get();
		
	auto Z_XX = dbcsr::create_template(Ctilde_xx)
		.name("Z_xx")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto Ztilde_XX = dbcsr::create_template(Ctilde_xx)
		.name("Ztilde_xx")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	arrvec<int,2> bb = {b,b};
	arrvec<int,2> xx = {x,x};
	
	auto pseudo_vir_0_1 = dbcsr::tensor_create<2,double>()
		.pgrid(spgrid2).name("pseudo_vir_0_1")
		.map1({0}).map2({1}).blk_sizes(bb).get();
		
	auto Z_XX_0_1 = dbcsr::tensor_create<2,double>()
		.pgrid(spgrid2).name("Z_xx_0_1")
		.map1({0}).map2({1}).blk_sizes(xx).get();
				
	auto B_xBB_0_12_wr = dbcsr::tensor_create_template<3,double>(B_xbb)
		.name("B_XBB_0_12").map1({0}).map2({1,2}).get();
		
	auto B_xBB_1_02 = dbcsr::tensor_create_template<3,double>(B_xbb)
		.name("B_XBB_1_02").map1({1}).map2({0,2}).get();
	
	dbcsr::sbtensor<3,double> B_xBB_batch = 
		std::make_shared<dbcsr::btensor<3,double>>(B_xBB_0_12_wr,nbatches,intermed_type,1);
	
	double mp2_energy = 0.0;
	
	//==================================================================
	//                   BEGIN LAPLACE QUADRATURE
	//==================================================================
	
	for (int ilap = 0; ilap != nlap; ++ilap) {
		
		LOG.os<>("LAPLACE POINT ", ilap, '\n');
		
		LOG.os<>("Forming pseudo densities.\n");
		
		// ================== PSEUDO DENSITIES =========================
		pseudotime.start();
		
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
		
		c_occ_exp->scale(exp_occ, "right");
		c_vir_exp->scale(exp_vir, "right");
				
		//c_occ_exp->filter();
		//c_vir_exp->filter();
		
		dbcsr::multiply('N', 'T', *c_occ_exp, *c_occ_exp, *pseudo_occ)
			.alpha(pow(omega,0.25)).perform();
		dbcsr::multiply('N', 'T', *c_vir_exp, *c_vir_exp, *pseudo_vir)
			.alpha(pow(omega,0.25)).perform();
		
		pseudotime.finish();
		
		//=============== CHOLESKY DECOMPOSITION =======================
		pcholtime.start();
		math::pivinc_cd chol(pseudo_occ, LOG.global_plev());
		chol.reorder("value");
		
		chol.compute();
		
		int rank = chol.rank();
		
		auto u = dbcsr::split_range(rank, mol->mo_split());
		
		LOG.os<>("Cholesky decomposition rank: ", rank, '\n');
	
		auto L_bu = chol.L(b, u);
		
		L_bu->filter(dbcsr::global::filter_eps);

		pseudo_occ->filter(dbcsr::global::filter_eps);
		pseudo_vir->filter(dbcsr::global::filter_eps);
		
		LOG.os<>("Occupancy of L: ", L_bu->occupation()*100, "%\n");
		
		arrvec<int,2> bu = {b,u};
		arrvec<int,3> xub = {x,u,b};
		
		std::array<int,2> bu_sizes = {nbf, rank};
		
		std::array<int,3> xub_sizes = {dfnbf, rank, nbf};
		
		auto spgrid2_bu = dbcsr::create_pgrid<2>(m_world.comm())
			.tensor_dims(bu_sizes).get();
			
		auto spgrid3_xub = dbcsr::create_pgrid<3>(m_world.comm())
			.tensor_dims(xub_sizes).get();
		
		auto L_bu_0_1 = dbcsr::tensor_create<2,double>()
			.pgrid(spgrid2_bu).name("L_bu_0_1")
			.map1({0}).map2({1}).blk_sizes(bu).get();
	
		auto B_xub_1_02 = dbcsr::tensor_create<3,double>()
			.pgrid(spgrid3_xub).name("B_Xub_1_02").map1({1}).map2({0,2})
			.blk_sizes(xub).get();
		
		auto B_xub_2_01 = 
			dbcsr::tensor_create_template<3,double>(B_xub_1_02)
			.name("B_Xub_2_01").map1({2}).map2({0,1}).get();
		
		auto B_xuB_2_01 = 
			dbcsr::tensor_create_template<3,double>(B_xub_1_02)
			.name("B_XuB_2_01").map1({2}).map2({0,1}).get();
		
		//copy over
		dbcsr::copy_matrix_to_tensor(*L_bu, *L_bu_0_1);
		pcholtime.finish();
				
		//============== B_X,B,B = B_x,b,b * Lo_o,b * Pv_b,b 
		
		B_xbb_batch->decompress_init({0});
		B_xBB_batch->compress_init({0,2});
		
		LOG.os<1>("Starting batching over auxiliary functions.\n");
		
		// LOOP OVER BATCHES OF AUXILIARY FUNCTIONS 
		for (int ix = 0; ix != B_xbb_batch->nbatches_dim(0); ++ix) {
			
			LOG.os<1>("-- (X) Batch ", ix, "\n");
			
			LOG.os<1>("-- Fetching ao ints...\n");
			fetchints1.start();
			B_xbb_batch->decompress({ix});
			fetchints1.finish();
			
			auto B_xbb_1_02 = B_xbb_batch->get_stensor();
						
			// first transform 
		    LOG.os<1>("-- First transform.\n");
		    
		    vec<vec<int>> x_nu_bounds = { 
				B_xbb_batch->bounds(0)[ix], 
				B_xbb_batch->full_bounds(2)
			};
		    
		    firsttran.start();
		    dbcsr::contract(*L_bu_0_1, *B_xbb_1_02, *B_xub_1_02)
				.bounds3(x_nu_bounds).perform("mi, Xmn -> Xin");
		    firsttran.finish();
		    
		    // reorder
			LOG.os<1>("-- Reordering B_xob.\n");
			reo1.start();
			dbcsr::copy(*B_xub_1_02, *B_xub_2_01).move_data(true).perform();
			reo1.finish();
			
			//dbcsr::print(*B_xob_2_01);
			
			//copy over vir density
			dbcsr::copy_matrix_to_tensor(*pseudo_vir, *pseudo_vir_0_1);
		
			// new loop over nu
			for (int inu = 0; inu != B_xBB_batch->nbatches_dim(2); ++inu) {
				
				LOG.os<1>("---- (NU) Batch ", inu, "\n");
				
				// second transform
				LOG.os<1>("---- Second transform.\n");
				
				vec<vec<int>> nu_bounds = { B_xBB_batch->bounds(2)[inu] };
				vec<vec<int>> x_u_bounds = { 
					B_xBB_batch->bounds(0)[ix],
					vec<int>{0, rank - 1}
				};
				
				sectran.start();
				dbcsr::contract(*pseudo_vir_0_1, *B_xub_2_01, *B_xuB_2_01)
					.bounds2(nu_bounds).bounds3(x_u_bounds)
					.perform("Nn, Xin -> XiN");
				sectran.finish();
			
				// reorder
				LOG.os<1>("---- Reordering B_xoB.\n");
				reo2.start();
				dbcsr::copy(*B_xuB_2_01, *B_xub_1_02).move_data(true).perform();
				reo2.finish();
				
				//dbcsr::print(*B_xob_1_02);
		
				// final contraction
				//B_xBB_1_02->reserve_template(*B_xbb_1_02);
			
				LOG.os<1>("-- Final transform.\n");
				
				vec<vec<int>> x_nu_bounds = {
					B_xBB_batch->bounds(0)[ix],
					B_xBB_batch->bounds(2)[inu]
				};
									
				fintran.start();
				dbcsr::contract(*L_bu_0_1, *B_xub_1_02, *B_xBB_1_02)
					.bounds3(x_nu_bounds).perform("Mi, XiN -> XMN");
				fintran.finish();
		
				// reorder
				LOG.os<1>("---- B_xBB.\n");
				reo3.start();
				dbcsr::copy(*B_xBB_1_02, *B_xBB_0_12_wr).move_data(true).perform();
				reo3.finish();
				
				//dbcsr::print(*B_xBB_0_12_wr);
				
				//dbcsr::print(*B_xBB_0_12);
				
				LOG.os<1>("---- Writing B_xBB to disk.\n");
				writetime.start();
				B_xBB_batch->compress({ix,inu},B_xBB_0_12_wr);
				writetime.finish();
				
			}
			
		}
		
		B_xbb_batch->decompress_finalize();
		B_xBB_batch->compress_finalize();
		
		LOG.os<>("Occupation of B_xBB: ", B_xBB_batch->occupation()*100, "%\n"); 
		
		LOG.os<>("Finished batching.\n");
		
		LOG.os<>("Reordering ints 1|02 -> 0|12 \n");
		
		reo_ints1.start();
		B_xbb_batch->reorder(vec<int>{0},vec<int>{1,2});
		reo_ints1.finish();
		
		LOG.os<>("Setting up decompression.\n");
		
		viewtime.start();
		B_xbb_batch->decompress_init({1,2});
		B_xBB_batch->decompress_init({1,2});
		viewtime.finish();
		
		auto mu_bbounds = B_xbb_batch->bounds(1);
		auto nu_bbounds = B_xbb_batch->bounds(2);
		
		LOG.os<>("Computing Z_XY.\n");
		
		Z_XX_0_1->batched_contract_init();
		
		for (int imu = 0; imu != mu_bbounds.size(); ++imu) {
			for (int inu = 0; inu != nu_bbounds.size(); ++inu) {
				
				LOG.os<>("-- Batch: ", imu, " ", inu, '\n');
				
				LOG.os<>("-- Fetching integrals...\n");
				
				fetchints2.start();
				B_xbb_batch->decompress({imu,inu});
				fetchints2.finish();
				
				auto B_xbb_0_12 = B_xbb_batch->get_stensor();
				
				LOG.os<>("-- Fetching intermediate...\n");
				
				readtime.start();
				B_xBB_batch->decompress({imu,inu});
				readtime.finish();
				
				auto B_xBB_0_12 = B_xBB_batch->get_stensor();
				
				vec<vec<int>> mn_bounds = {
					mu_bbounds[imu],
					nu_bbounds[inu]
				};
				
				// form Z
				LOG.os<>("-- Forming Z.\n");
				
				formZ.start();
				dbcsr::contract(*B_xBB_0_12, *B_xbb_0_12, *Z_XX_0_1)
					.beta(1.0).bounds1(mn_bounds)
					.perform("Mmn, Nmn -> MN");
				formZ.finish();
				
			}
		}
		
		B_xbb_batch->decompress_finalize();
		B_xBB_batch->decompress_finalize();
		
		B_xBB_batch->reset();
		
		LOG.os<>("Reordering ints 0|12 -> 1|02 \n");
		
		reo_ints2.start();
		B_xbb_batch->reorder(vec<int>{1},vec<int>{0,2});
		reo_ints2.finish();
		
		Z_XX_0_1->batched_contract_finalize();	
		
		Z_XX_0_1->filter(dbcsr::global::filter_eps);	
		
		LOG.os<1>("Finished batching.\n");

		formZtilde.start();
		
		// copy
		dbcsr::copy_tensor_to_matrix(*Z_XX_0_1, *Z_XX);
		Z_XX_0_1->clear();
		
		// multiply
		LOG.os<1>("Ztilde = Z * Jinv\n");
		dbcsr::multiply('N', 'N', *Z_XX, *Ctilde_xx, *Ztilde_XX).perform();
		
		formZtilde.finish();
		
		//dbcsr::print(*Ztilde_XX);
		
		redtime.start();
		
		LOG.os<1>("Local reduction.\n");
		
		dbcsr::iter_d iter(*Ztilde_XX);
		
		double sum = 0.0;
		
		int nblks = x.size();
		
		Ztilde_XX->filter(dbcsr::global::filter_eps);

		auto Ztilde_XX_t = dbcsr::transpose(Ztilde_XX).get();
			
		//dbcsr::print(*Ztilde_XX);
		//dbcsr::print(*Ztilde_XX_t);
		
		const auto loc_rows = Ztilde_XX->local_rows();
		const auto loc_cols = Ztilde_XX->local_cols();

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
				
				//for (int i = 0; i != blk.ntot(); ++i) {
				//	std::cout << blk.data()[i] << " " << blk_t.data()[i] << std::endl;
				//}
				
				if (!found1 || !found2) continue;
				
				//std::cout << "COMPUTE." << std::endl;
				
				sum += std::inner_product(blk.data(), blk.data() + blk.ntot(),
					blk_t.data(), 0.0);
					
			}
		}

		redtime.finish();

		double total = 0.0;
		LOG.os<1>("Global reduction.\n");

		MPI_Allreduce(&sum, &total, 1, MPI_DOUBLE, MPI_SUM, m_world.comm());
		
		LOG.os<>("Partial sum: ", total, '\n');
		
		mp2_energy += total;
		
	}
	
	//mp2_energy *= c_os;
	
	LOG.os<>("Final MP2 energy: ", mp2_energy, '\n');
	
	TIME.finish();
	
	TIME.print_info();
	
}

} // end namespace 
