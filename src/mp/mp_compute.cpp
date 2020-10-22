#include "mp/mpmod.h"
#include "mp/mp_defaults.h"
#include "mp/z_builder.h"
#include "math/laplace/laplace.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include "math/linalg/LLT.h"
#include "math/linalg/piv_cd.h"
#include "ints/aofactory.h"
#include "ints/screening.h"
#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_tensor_ops.hpp>
#include <dbcsr_btensor.hpp>

namespace mp {
	
mpmod::mpmod(hf::shared_hf_wfn& wfn_in, desc::options& opt_in, dbcsr::world& w_in) :
	m_hfwfn(wfn_in),
	m_opt(opt_in),
	m_world(w_in),
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

void mpmod::compute_batch() {

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
	bool force_sparsity = m_opt.get<bool>("force_sparsity", MP_FORCE_SPARSITY);
		
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
	auto B_xbb = dbcsr::tensor_create<3,double>()
		.name("i_xbb")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	dbcsr::btype eri_type = dbcsr::get_btype(eri_method);
	dbcsr::btype intermed_type = dbcsr::get_btype(intermed_method);
	
	std::array<int,3> bdims = {nbatches_x,nbatches_b,nbatches_b};
	
	dbcsr::sbtensor<3,double> B_xbb_batch = 
		dbcsr::btensor_create<3>()
		.name(mol->name() + "_eri_batched")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.batch_dims(bdims)
		.btensor_type(eri_type)
		.print(LOG.global_plev())
		.get();
	
	auto gen_func = aofac->get_generator(s_scr);
	B_xbb_batch->set_generator(gen_func);
	
	calcints.start();
	
	B_xbb_batch->compress_init({0}, vec<int>{0}, vec<int>{1,2});
	
	for (int ix = 0; ix != B_xbb_batch->nbatches(0); ++ix) {
			
			vec<vec<int>> blkbounds = {
				B_xbb_batch->blk_bounds(0,ix),
				B_xbb_batch->full_blk_bounds(1),
				B_xbb_batch->full_blk_bounds(2)
			};
		
			if (eri_type != dbcsr::btype::direct) {
				aofac->ao_3c2e_fill(B_xbb, blkbounds, s_scr);
				B_xbb->filter(dbcsr::global::filter_eps);
			}
			
			B_xbb_batch->compress({ix}, B_xbb);
	}
	
	B_xbb_batch->compress_finalize();
	
	util::registry reg;
	
	reg.insert_btensor<3,double>("i_xbb_batched", B_xbb_batch);
	
	calcints.finish();
	
	spinfotime.start();
	SMatrixXi spinfo = nullptr;
	if (m_opt.get<bool>("force_sparsity", MP_FORCE_SPARSITY)) {
		spinfo = get_shellpairs(B_xbb_batch);
	}
	spinfotime.finish();
	
	//==================================================================
	//                          METRIC
	//==================================================================
	
	invtime.start();
	
	auto C_xx = aofac->ao_2c2e("coulomb");
	auto S_erfc_xx = aofac->ao_2c2e("erfc_coulomb");
	
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
		
	auto Ztilde_XX = dbcsr::create_template(Ctilde_xx)
		.name("Ztilde_xx")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	double mp2_energy = 0.0;
	
	//==================================================================
	//                          SETUP Z BUILDER 
	//==================================================================
	
	std::string zmethod = m_opt.get<std::string>("build_Z", MP_BUILD_Z);
	
	Z* zbuilder = nullptr;
	
	if (zmethod == "LLMPFULL") {
		zbuilder = new LLMP_FULL_Z(m_world, mol, m_opt);
	} else if (zmethod == "LLMPMEM") {
		zbuilder = new LLMP_MEM_Z(m_world, mol, m_opt);
	}
	
	if (zbuilder == nullptr) throw std::runtime_error("Invalid z builder!");
	
	zbuilder->set_reg(reg);
	
	zbuilder->init_tensors();
	
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
		dbcsr::multiply('N', 'N', *Z_XX, *Ctilde_xx, *Ztilde_XX)
			.filter_eps(dbcsr::global::filter_eps).perform();
		
		formZtilde.finish();
		
		//dbcsr::print(*Ztilde_XX);
		
		redtime.start();
		
		LOG.os<1>("Local reduction.\n");
		
		dbcsr::iter_d iter(*Ztilde_XX);
		
		double sum = 0.0;
		
		int nblks = x.size();
		
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
	
	zbuilder->print_info();
	TIME.print_info();
	
	delete zbuilder;
	
	m_mpwfn = std::make_shared<mp_wfn>(*m_hfwfn);
	m_mpwfn->m_mp_ss_energy = 0.0;
	m_mpwfn->m_mp_os_energy = mp2_energy;
	m_mpwfn->m_mp_energy = c_os * mp2_energy;
	
}

} // end namespace 
