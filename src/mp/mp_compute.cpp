#include "mp/mpmod.hpp"
#include "mp/mp_defaults.hpp"
#include "mp/z_builder.hpp"
#include "math/laplace/laplace.hpp"
#include "math/linalg/piv_cd.hpp"
#include "math/linalg/LLT.hpp"
#include "ints/aoloader.hpp"
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
	
	auto spgrid2 = dbcsr::pgrid<2>::create(m_world.comm()).build();
	
	std::array<int,3> xbb_sizes = {dfnbf, nbf, nbf};
	
	auto spgrid3_xbb = dbcsr::pgrid<3>::create(m_world.comm()).tensor_dims(xbb_sizes).build();
	
	//==================================================================
	//                        INTEGRALS
	//==================================================================
	
	// integral machine
	
	std::optional<int> nbatches_b_opt = m_opt.present("nbatches_b") ? 
		std::make_optional<int>(m_opt.get<int>("nbatches_b")) : 
		std::nullopt;
		
	std::optional<int> nbatches_x_opt = m_opt.present("nbatches_x") ? 
		std::make_optional<int>(m_opt.get<int>("nbatches_x")) : 
		std::nullopt;
		
	std::optional<dbcsr::btype> btype_e = m_opt.present("eris") ?
		std::make_optional<dbcsr::btype>(dbcsr::get_btype(m_opt.get<std::string>("eris"))) :
		std::nullopt;
		
	std::optional<dbcsr::btype> btype_i = m_opt.present("intermeds") ?
		std::make_optional<dbcsr::btype>(dbcsr::get_btype(m_opt.get<std::string>("intermeds"))) :
		std::nullopt;
	
	std::shared_ptr<ints::aoloader> ao
		= ints::aoloader::create()
		.set_world(m_world)
		.set_molecule(mol)
		.print(LOG.global_plev())
		.nbatches_b(nbatches_b_opt)
		.nbatches_x(nbatches_x_opt)
		.btype_eris(btype_e)
		.btype_intermeds(btype_i)
		.build();

	auto zmeth = str_to_zmethod(
		m_opt.get<std::string>("build_Z", MP_BUILD_Z));
		
	auto zmetr = ints::str_to_metric(
		m_opt.get<std::string>("df_metric", MP_METRIC));
	
	#ifdef _ORTHOGONALIZE
	ao->request(ints::key::ovlp_bb, true);
	#endif
	
	load_zints(zmeth, zmetr, *ao);

	ao->compute();
	
	auto zbuilder = create_z()
		.set_world(m_world)
		.set_molecule(mol)
		.print(LOG.global_plev())
		.aoloader(*ao)
		.method(zmeth)
		.metric(zmetr)
		.build();
	
	auto& aoreg = ao->get_registry();
	dbcsr::shared_matrix<double> metric_matrix;
	
	switch (zmetr) {
		case ints::metric::coulomb:
			metric_matrix = aoreg.get<dbcsr::shared_matrix<double>>(
				ints::key::coul_xx_inv);
			break;	
				
		case ints::metric::erfc_coulomb:
			metric_matrix = aoreg.get<dbcsr::shared_matrix<double>>(
				ints::key::erfc_xx_inv);
			break;
		
		case ints::metric::qr_fit:
			metric_matrix = aoreg.get<dbcsr::shared_matrix<double>>(
				ints::key::coul_xx);
			break;
			
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
	
	//==================================================================
	//                         SETUP OTHER TENSORS
	//==================================================================
	
	auto c_occ = m_hfwfn->c_bo_A();
	auto c_vir = m_hfwfn->c_bv_A();
	
	// matrices and tensors
	
	auto c_occ_exp = dbcsr::matrix<>::create_template(*c_occ)
		.name("Scaled Occ Coeff").build();
		
	auto c_vir_exp = dbcsr::matrix<>::create_template(*c_vir)
		.name("Scaled Vir Coeff").build();
		
	auto pseudo_occ = dbcsr::matrix<>::create()
		.name("Pseudo Density (OCC)")
		.set_world(m_world)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::symmetric)
		.build();
		
	auto pseudo_vir = dbcsr::matrix<>::create_template(*pseudo_occ)
		.name("Pseudo Density (VIR)").build();
		
	auto ztilde_XX = dbcsr::matrix<>::create_template(*metric_matrix)
		.name("ztilde_xx")
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	double mp2_energy = 0.0;
	
	//==================================================================
	//                          SETUP Z BUILDER 
	//==================================================================
		
	zbuilder->init();
		
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
		
		auto c_occ_ortho = dbcsr::matrix<>::create_template(*c_occ_exp)
			.name("c_occ_ortho")
			.build();
			
		dbcsr::multiply('N', 'N', 1.0, *Sllt_inv_bb, *c_occ_exp, 0.0, 
			*c_occ_ortho).perform();
				
		dbcsr::multiply('N', 'T', pow(omega,0.25), *c_occ_ortho, *c_occ_ortho, 
			0.0, *pseudo_occ).perform();
			
		#else 
		
		dbcsr::multiply('N', 'T', pow(omega,0.25), *c_occ_exp, *c_occ_exp, 
			0.0, *pseudo_occ).perform();
			
		#endif
		
		dbcsr::multiply('N', 'T', pow(omega,0.25), *c_vir_exp, *c_vir_exp, 
			0.0, *pseudo_vir).perform();
				
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
		auto L_bu = dbcsr::matrix<>::create_template(*L_bu_ortho)
			.name("L_bu")
			.build();
			
		dbcsr::multiply('N', 'N', 1.0, *Sllt_bb, *L_bu_ortho, 0.0, *L_bu)
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
		dbcsr::multiply('N', 'N', 1.0, *Z_XX, *metric_matrix, 0.0, 
			*ztilde_XX)
			.filter_eps(dbcsr::global::filter_eps)
			.perform();
		
		formZtilde.finish();
		
		//dbcsr::print(*Ztilde_XX);
		
		redtime.start();
		
		LOG.os<1>("Local reduction.\n");
								
		auto ztilde_XX_t = dbcsr::matrix<>::transpose(*ztilde_XX)
			.build();
			
		double sum = ztilde_XX->dot(*ztilde_XX_t);
			
		redtime.finish();
		
		LOG.os<>("Partial sum: ", sum, '\n');
		
		mp2_energy += sum;
		
	}
	
	//mp2_energy *= c_os;
	
	LOG.setprecision(12);
	LOG.os<>("Final MP2 energy: ", mp2_energy, '\n');
	LOG.os<>("Final MP2 energy (scaled): ", mp2_energy * c_os, '\n');
	LOG.reset();

	TIME.finish();
	
	ao->print_info();
	zbuilder->print_info();
	TIME.print_info();
		
	m_mpwfn = std::make_shared<mp_wfn>(*m_hfwfn);
	m_mpwfn->m_mp_ss_energy = 0.0;
	m_mpwfn->m_mp_os_energy = mp2_energy;
	m_mpwfn->m_mp_energy = c_os * mp2_energy;
	
}

} // end namespace 
