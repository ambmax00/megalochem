#include "mp/mpmod.hpp"
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

namespace megalochem {

namespace mp {
	
void mpmod::init() {
	
	m_wfn->mol->set_cluster_dfbasis(m_df_basis);
	
	std::cout << "NLAP: " << m_nlap << std::endl;
	
	std::cout << "NBATCHES: " << m_nbatches_b << " " << m_nbatches_x << std::endl;
		
}

desc::shared_wavefunction mpmod::compute() {

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
	auto eps_o = m_wfn->hf_wfn->eps_occ_A();
	auto eps_v = m_wfn->hf_wfn->eps_vir_A();
	
	auto mol = m_wfn->mol;
	
	auto o = mol->dims().oa();
	auto v = mol->dims().va();
	auto b = mol->dims().b();
	auto x = mol->dims().x();
	
	arrvec<int,3> xbb = {x,b,b};
	
	int nbf = std::accumulate(b.begin(), b.end(), 0);
	int dfnbf = std::accumulate(x.begin(), x.end(), 0);
	
	//std::cout << "NBFS: " << nbf << " " << dfnbf << std::endl;
		
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
	lp.compute(m_nlap, ymin, ymax);
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
		
	dbcsr::btype btype_e = dbcsr::get_btype(m_eris);
	dbcsr::btype btype_i = dbcsr::get_btype(m_imeds);
		
	std::shared_ptr<ints::aoloader> ao
		= ints::aoloader::create()
		.set_world(m_world)
		.set_molecule(mol)
		.print(LOG.global_plev())
		.nbatches_b(m_nbatches_b)
		.nbatches_x(m_nbatches_x)
		.btype_eris(btype_e)
		.btype_intermeds(btype_i)
		.build();

	auto zmeth = str_to_zmethod(m_build_Z);
		
	auto zmetr = ints::str_to_metric(m_df_metric);
	
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
	
	math::LLT lltsolver(m_world, s_bb, LOG.global_plev());
	lltsolver.compute();
	
	auto Sllt_bb = lltsolver.L(b);
	auto Sllt_inv_bb = lltsolver.L_inv(b);
	s_bb->release();
	#endif
	
	//==================================================================
	//                         SETUP OTHER TENSORS
	//==================================================================
	
	auto c_occ = m_wfn->hf_wfn->c_bo_A();
	auto c_vir = m_wfn->hf_wfn->c_bv_A();
	
	// matrices and tensors
	
	auto c_occ_exp = dbcsr::matrix<>::create_template(*c_occ)
		.name("Scaled Occ Coeff").build();
		
	auto c_vir_exp = dbcsr::matrix<>::create_template(*c_vir)
		.name("Scaled Vir Coeff").build();
		
	auto pseudo_occ = dbcsr::matrix<>::create()
		.name("Pseudo Density (OCC)")
		.set_cart(m_world.dbcsr_grid())
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::symmetric)
		.build();
		
	auto pseudo_vir = dbcsr::matrix<>::create_template(*pseudo_occ)
		.name("Pseudo Density (VIR)").build();
		
	auto pseudo_ortho_occ = dbcsr::matrix<>::create_template(*pseudo_occ)
		.name("Pseudo Orth. Density (OCC)").build();
		
	auto pseudo_ortho_vir = dbcsr::matrix<>::create_template(*pseudo_occ)
		.name("Pseudo Orth. Density (VIR)").build();
		
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
	
	for (int ilap = 0; ilap != m_nlap; ++ilap) {
		
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
		
		auto c_occ_ortho = dbcsr::matrix<>::create_template(*c_occ_exp)
			.name("c_occ_ortho")
			.build();
			
		dbcsr::multiply('N', 'N', 1.0, *Sllt_inv_bb, *c_occ_exp, 0.0, 
			*c_occ_ortho).perform();
				
		dbcsr::multiply('N', 'T', pow(omega,0.25), *c_occ_ortho, *c_occ_ortho, 
			0.0, *pseudo_ortho_occ).perform();
		
		dbcsr::multiply('N', 'T', pow(omega,0.25), *c_vir_exp, *c_vir_exp, 
			0.0, *pseudo_vir).perform();
				
		pseudotime.finish();
		
		//=============== CHOLESKY DECOMPOSITION =======================
		pcholtime.start();
		
		dbcsr::shared_matrix<double> Locc_bu, Lvir_br;
		
		math::pivinc_cd chol(m_world, pseudo_ortho_occ, LOG.global_plev());
		//chol.reorder("value");
		
		chol.compute();
		
		int rank = chol.rank();
		
		auto u = dbcsr::split_range(rank, mol->mo_split());
		
		LOG.os<>("Cholesky decomposition rank: ", rank, '\n');
	
		auto Locc_bu_ortho = chol.L(b, u);
		
		Locc_bu = dbcsr::matrix<>::create_template(*Locc_bu_ortho)
			.name("L_bu")
			.build();
			
		dbcsr::multiply('N', 'N', 1.0, *Sllt_bb, *Locc_bu_ortho, 0.0, *Locc_bu)
			.perform();
		
		Locc_bu->filter(dbcsr::global::filter_eps);

		if (zmeth == zmethod::ll_full) {
			
			auto c_vir_ortho = dbcsr::matrix<>::create_template(*c_vir_exp)
				.name("c_vir_ortho")
				.build();
			
			dbcsr::multiply('N', 'N', 1.0, *Sllt_inv_bb, *c_vir_exp, 0.0, 
				*c_vir_ortho).perform();
					
			dbcsr::multiply('N', 'T', pow(omega,0.25), *c_vir_ortho, *c_vir_ortho, 
				0.0, *pseudo_ortho_vir).perform();
			
			math::pivinc_cd chol_v(m_world, pseudo_ortho_vir, LOG.global_plev());
			
			chol_v.compute();
			
			int rank_v = chol_v.rank();
			
			auto r = dbcsr::split_range(rank_v, mol->mo_split());
			
			LOG.os<>("Cholesky decomposition rank: ", rank_v, '\n');
		
			auto Lvir_br_ortho = chol_v.L(b, r);
			
			Lvir_br = dbcsr::matrix<>::create_template(*Lvir_br_ortho)
				.name("L_br")
				.build();
				
			dbcsr::multiply('N', 'N', 1.0, *Sllt_bb, *Lvir_br_ortho, 0.0, *Lvir_br)
				.perform();
			
			Lvir_br->filter(dbcsr::global::filter_eps);

		}

		pseudo_occ->filter(dbcsr::global::filter_eps);
		pseudo_vir->filter(dbcsr::global::filter_eps);
		
		LOG.os<>("Occupancy of L: ", Locc_bu->occupation()*100, "%\n");
		
		pcholtime.finish();
				
		//============== B_X,B,B = B_x,b,b * Lo_o,b * Pv_b,b 
		
		zbuilder->set_occ_coeff(Locc_bu);
		zbuilder->set_vir_coeff(Lvir_br);
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
	LOG.os<>("Final MP2 energy (scaled): ", mp2_energy * m_c_os, '\n');
	LOG.reset();

	TIME.finish();
	
	ao->print_info();
	zbuilder->print_info();
	TIME.print_info();
		
	auto mpwfn = std::make_shared<desc::mp_wavefunction>(0.0, mp2_energy, m_c_os * mp2_energy);
	
	auto out = std::make_shared<desc::wavefunction>();
	
	out->mol = m_wfn->mol;
	out->hf_wfn = m_wfn->hf_wfn;
	out->mp_wfn = mpwfn;
	
	return out;
	
}

} // end namespace 

} // end mega
