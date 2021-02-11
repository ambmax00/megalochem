#include "mp/mpmod.h"
#include "mp/mp_defaults.h"
#include "mp/z_builder.h"
#include "math/laplace/laplace.h"
#include "math/linalg/piv_cd.h"
#include "math/linalg/LLT.h"
#include "ints/aoloader.h"
#include <omp.h>
#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_tensor_ops.hpp>
#include <dbcsr_btensor.hpp>

#define _ORTHOGONALIZE

namespace mp {
	
mpmod::mpmod(dbcsr::world w, hf::shared_hf_wfn wfn_in, desc::options& opt_in) :
	m_hfwfn(wfn_in),
	m_world(w),
	m_opt(opt_in),
	LOG(m_world.comm(),m_opt.get<int>("print", MP_PRINT_LEVEL)),
	TIME(m_world.comm(), "Moller Plesset", LOG.global_plev())
{
	std::string dfbasname = m_opt.get<std::string>("dfbasis");
	
	int nsplit = m_hfwfn->mol()->c_basis()->nsplit();
	std::string splitmethod = m_hfwfn->mol()->c_basis()->split_method();
	auto atoms = m_hfwfn->mol()->atoms(); 
	
	bool augmented = m_opt.get<bool>("df_augmentation", false);
	auto dfbasis = std::make_shared<desc::cluster_basis>(
		dfbasname, atoms, splitmethod, nsplit, augmented);
	
	m_hfwfn->mol()->set_cluster_dfbasis(dfbasis);
	
	//std::cout << "SIZE: " << dfbasis->size() << std::endl;
	
}

void mpmod::compute() {

	LOG.banner<>("Batched CD-LT-SOS-RI-MP2", 50, '*');
	
	auto& laptime = TIME.sub("Computing laplace points");
	auto& scrtime = TIME.sub("Computing screener");
	auto& calcints = TIME.sub("Computing integrals");
	auto& spinfotime = TIME.sub("Shellpair info");
	auto& invtime = TIME.sub("Inverting metric");
	auto& pseudotime = TIME.sub("Forming pseudo densities");
	auto& pcholtime = TIME.sub("Pivoted cholesky decomposition");
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
	auto b = mol->dims().b();
	auto x = mol->dims().x();
	
	arrvec<int,3> xbb = {x,b,b};
	
	int nbf = std::accumulate(b.begin(), b.end(), 0);
	int dfnbf = std::accumulate(x.begin(), x.end(), 0);
	
	//std::cout << "NBFS: " << nbf << " " << dfnbf << std::endl;
	
	// options
	int nlap = m_opt.get<int>("nlap",MP_NLAP);
	double c_os = m_opt.get<double>("c_os",MP_C_OS);
	
	int nbatches_x = m_opt.get<int>("nbatches_x", MP_NBATCHES_X);
	int nbatches_b = m_opt.get<int>("nbatches_b", MP_NBATCHES_B);
	
	std::string eri_method = m_opt.get<std::string>("eris", MP_ERIS);
	std::string intermed_method = m_opt.get<std::string>("intermeds", MP_INTERMEDS);
		
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

	std::string metric_str = m_opt.get<std::string>("df_metric", MP_METRIC);
	ints::metric coulmet = ints::str_to_metric(metric_str);
	
	ints::aoloader aoload(m_world, m_hfwfn->mol(), m_opt);
	
	if (coulmet == ints::metric::coulomb) {
		
		aoload.request(ints::key::coul_xx, false);
		aoload.request(ints::key::coul_xx_inv, true);
		aoload.request(ints::key::scr_xbb, true);
		aoload.request(ints::key::coul_xbb, true);
		
	} else if (coulmet == ints::metric::erfc_coulomb) {
		
		aoload.request(ints::key::erfc_xx, false);
		aoload.request(ints::key::coul_xx, false);
		aoload.request(ints::key::erfc_xx_inv, true);
		aoload.request(ints::key::scr_xbb, true);
		aoload.request(ints::key::erfc_xbb, true);
		
	} else if (coulmet == ints::metric::qr_fit) {
		
		aoload.request(ints::key::coul_xx, true);
		aoload.request(ints::key::ovlp_xx, false);
		aoload.request(ints::key::ovlp_xx_inv, false);
		aoload.request(ints::key::scr_xbb, true);
		aoload.request(ints::key::qr_xbb, true);
		
	}
	
	#ifdef _ORTHOGONALIZE
	aoload.request(ints::key::ovlp_bb, true);
	#endif

	aoload.compute();
	auto aoreg = aoload.get_registry();
	
	dbcsr::sbtensor<3,double> eri3c2e_batched;
	dbcsr::shared_matrix<double> metric_matrix;
	
	if (coulmet == ints::metric::coulomb) {
		
		eri3c2e_batched = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::coul_xbb);
		metric_matrix = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::coul_xx_inv);
		
	} else if (coulmet == ints::metric::erfc_coulomb) {
		
		eri3c2e_batched = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::erfc_xbb);
		metric_matrix = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::erfc_xx_inv);
		
	} else if (coulmet == ints::metric::qr_fit) {
		
		eri3c2e_batched = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::qr_xbb);
		metric_matrix = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::coul_xx);
		
	}
	
	#ifdef _ORTHOGONALIZE
	auto s_bb = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::ovlp_bb);
	
	LOG.os<1>("Computing square root and square root inverse of S using LLT.\n");
	
	math::LLT lltsolver(s_bb, LOG.global_plev());
	lltsolver.compute();
	
	auto Sllt_bb = lltsolver.L(b);
	auto Sllt_inv_bb = lltsolver.L_inv(b);
	s_bb->release();
	#endif
	
	spinfotime.start();
	SMatrixXi spinfo = nullptr;
	spinfo = get_shellpairs(eri3c2e_batched);
	spinfotime.finish();
	
	//==================================================================
	//                         SETUP OTHER TENSORS
	//==================================================================
	
	auto c_occ = m_hfwfn->c_bo_A();
	auto c_vir = m_hfwfn->c_bv_A();
	
	// matrices and tensors
	
	auto c_occ_exp = dbcsr::create_template(*c_occ)
		.name("Scaled Occ Coeff").get();
		
	auto c_vir_exp = dbcsr::create_template(*c_vir)
		.name("Scaled Vir Coeff").get();
		
	auto pseudo_occ = dbcsr::create_template(*s_bb)
		.name("Pseudo Density (OCC)").get();
		
	auto pseudo_vir = dbcsr::create_template(*s_bb)
		.name("Pseudo Density (VIR)").get();
		
	auto ztilde_XX = dbcsr::create_template(*metric_matrix)
		.name("ztilde_xx")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	double mp2_energy = 0.0;
	
	//==================================================================
	//                          SETUP Z BUILDER 
	//==================================================================
	
	std::string zmethod_str = m_opt.get<std::string>("build_Z", MP_BUILD_Z);
	std::string intermeds_str = m_opt.get<std::string>("intermeds", MP_INTERMEDS);
	
	auto zmeth = str_to_zmethod(zmethod_str);
	auto intermeds = dbcsr::get_btype(intermeds_str);
	
	std::shared_ptr<Z> zbuilder;
	
	if (zmeth == zmethod::llmp_full) {
		
		zbuilder = create_LLMP_FULL_Z(m_world, m_hfwfn->mol(), LOG.global_plev())
			.eri3c2e_batched(eri3c2e_batched)
			.intermeds(intermeds)
			.get();
		
	} else if (zmeth == zmethod::llmp_mem) {
		
		zbuilder = create_LLMP_MEM_Z(m_world, m_hfwfn->mol(), LOG.global_plev())
			.eri3c2e_batched(eri3c2e_batched)
			.get();
		
	}
	
	if (zbuilder == nullptr) throw std::runtime_error("Invalid z builder!");
	
	zbuilder->init();
	
	zbuilder->set_shellpair_info(spinfo);
	
	//==================================================================
	//                      BEGIN LAPLACE QUADRATURE
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
		
		#ifdef _ORTHOGONALIZE
		
		auto c_occ_ortho = dbcsr::create_template<double>(*c_occ_exp)
			.name("c_occ_ortho")
			.get();
			
		dbcsr::multiply('N', 'N', *Sllt_inv_bb, *c_occ_exp, *c_occ_ortho)
			.perform();
				
		dbcsr::multiply('N', 'T', *c_occ_ortho, *c_occ_ortho, *pseudo_occ)
			.alpha(pow(omega,0.25)).perform();
			
		#else 
		
		dbcsr::multiply('N', 'T', *c_occ_exp, *c_occ_exp, *pseudo_occ)
			.alpha(pow(omega,0.25)).perform();
			
		#endif
		
		dbcsr::multiply('N', 'T', *c_vir_exp, *c_vir_exp, *pseudo_vir)
			.alpha(pow(omega,0.25)).perform();
		
		pseudotime.finish();
		
		//=============== CHOLESKY DECOMPOSITION =======================
		pcholtime.start();
		
		math::pivinc_cd chol(pseudo_occ, LOG.global_plev());
		//chol.reorder("value");
		
		chol.compute();
		
		int rank = chol.rank();
		
		auto u = dbcsr::split_range(rank, mol->mo_split());
		
		LOG.os<>("Cholesky decomposition rank: ", rank, '\n');
	
		#ifdef _ORTHOGONALIZE
	
		auto L_bu_ortho = chol.L(b, u);
		auto L_bu = dbcsr::create_template<double>(*L_bu_ortho)
			.name("L_bu")
			.get();
			
		dbcsr::multiply('N', 'N', *Sllt_bb, *L_bu_ortho, *L_bu)
			.perform();
			
		#else 
		
		auto L_bu = chol.L(b, u);
		
		#endif
		
		L_bu->filter(dbcsr::global::filter_eps);

		pseudo_occ->filter(dbcsr::global::filter_eps);
		pseudo_vir->filter(dbcsr::global::filter_eps);
		
		LOG.os<>("Occupancy of L: ", L_bu->occupation()*100, "%\n");
		
		pcholtime.finish();
				
		//============== B_X,B,B = B_x,b,b * Lo_o,b * Pv_b,b 
		
		zbuilder->set_occ_coeff(L_bu);
		zbuilder->set_vir_density(pseudo_vir);
		
		zbuilder->compute();
		auto Z_XX = zbuilder->zmat();
		
		//dbcsr::print(*Z_XX);
		
		formZtilde.start();
		
		// multiply
		LOG.os<1>("Ztilde = Z * Jinv\n");
		dbcsr::multiply('N', 'N', *Z_XX, *metric_matrix, *ztilde_XX)
			.filter_eps(dbcsr::global::filter_eps).perform();
		
		formZtilde.finish();
		
		//dbcsr::print(*Ztilde_XX);
		
		redtime.start();
		
		LOG.os<1>("Local reduction.\n");
		
		dbcsr::iter_d iter(*ztilde_XX);
		
		double sum = 0.0;
		
		int nblks = x.size();
		
		auto ztilde_XX_t = dbcsr::transpose(*ztilde_XX).get();
			
		//dbcsr::print(*Ztilde_XX);
		//dbcsr::print(*Ztilde_XX_t);
		
		const auto loc_rows = ztilde_XX->local_rows();
		const auto loc_cols = ztilde_XX->local_cols();
		
		const int isize = loc_rows.size();
		const int jsize = loc_cols.size();

#pragma omp parallel for collapse(2) reduction(+:sum)
		for (int i = 0; i != isize; ++i) {
			for (int j = 0; j != jsize; ++j) {
		//for (auto iblk : loc_rows) {
		//	for (auto jblk : loc_cols) {
				int iblk = loc_rows[i];
				int jblk = loc_cols[j];
				
				//std::cout << iblk << " " << jblk << std::endl;
				
				bool found1 = false;
				bool found2 = false;
				
				auto blk = ztilde_XX->get_block_p(iblk,jblk,found1);
				auto blk_t = ztilde_XX_t->get_block_p(iblk,jblk,found2);
				
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
	
	aoload.print_info();
	zbuilder->print_info();
	TIME.print_info();
		
	m_mpwfn = std::make_shared<mp_wfn>(*m_hfwfn);
	m_mpwfn->m_mp_ss_energy = 0.0;
	m_mpwfn->m_mp_os_energy = mp2_energy;
	m_mpwfn->m_mp_energy = c_os * mp2_energy;
	
}

} // end namespace 
