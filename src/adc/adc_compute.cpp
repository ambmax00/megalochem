#include "adc/adcmod.hpp"
#include "adc/adc_mvp.hpp"
#include "math/solvers/davidson.hpp"
#include "math/linalg/piv_cd.hpp"
#include "math/linalg/SVD.hpp"
#include "math/solvers/hermitian_eigen_solver.hpp"
#include "locorb/locorb.hpp"

#include <dbcsr_conversions.hpp>

namespace adc {

dbcsr::shared_matrix<double> canonicalize(dbcsr::shared_matrix<double> u_lm,
	std::vector<double> eps_m)
{
	
	std::cout << "CANONICALIZING!!!" << std::endl;
	
	auto w = u_lm->get_world();
	
	auto l = u_lm->row_blk_sizes();
	auto m = u_lm->col_blk_sizes();
	
	auto f_mm = dbcsr::matrix<>::create()
		.name("f_mm")
		.set_world(w)
		.row_blk_sizes(m)
		.col_blk_sizes(m)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	f_mm->reserve_all();
	
	f_mm->set_diag(eps_m);
	
	dbcsr::print(*u_lm);
	
	dbcsr::print(*f_mm);
	
	auto f_ll = u_transform(f_mm, 'N', u_lm, 'T', u_lm);
	
	dbcsr::print(*f_ll);
	
	math::hermitian_eigen_solver solver(f_ll, 'V', 1);
	solver.compute();
	
	auto u_cl = solver.eigvecs();
	
	auto eval = solver.eigvals();
	
	if (w.rank() == 0) {
		for (auto e : eval) {
			std::cout << e << " ";
		} std::cout << std::endl;
		for (auto e : eps_m) {
			std::cout << e << " ";
		} std::cout << std::endl;
	}
	dbcsr::print(*u_cl);
	
	return u_cl;
	
}

void adcmod::compute() {
	
	// Generate guesses
	
	compute_diag();
	
	LOG.os<>("--- Starting Computation ---\n\n");
			
	int nocc = m_hfwfn->mol()->nocc_alpha();
	int nvir = m_hfwfn->mol()->nvir_alpha();
	auto epso = m_hfwfn->eps_occ_A();
	auto epsv = m_hfwfn->eps_vir_A();
	
	LOG.os<>("Computing guess vectors...\n");
	// now order it : there is probably a better way to do it
	auto eigen_ia = dbcsr::matrix_to_eigen(*m_d_ov);
	
	std::vector<int> index(eigen_ia.size(), 0);
	for (int i = 0; i!= index.size(); ++i) {
		index[i] = i;
	}
	
	std::sort(index.begin(), index.end(), 
		[&](const int& a, const int& b) {
			return (eigen_ia.data()[a] < eigen_ia.data()[b]);
	});
		
	// generate the guesses
	
	auto o = m_hfwfn->mol()->dims().oa();
	auto v = m_hfwfn->mol()->dims().va();
	auto b = m_hfwfn->mol()->dims().b();
	
	int nguesses = m_opt.get<int>("nguesses", ADC_NGUESSES);
	std::vector<dbcsr::shared_matrix<double>> dav_guess(nguesses);
	
	for (int i = 0; i != nguesses; ++i) {
		
		LOG.os<>("Guess ", i, '\n');
		
		Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(nocc,nvir);
		mat.data()[index[i]] = 1.0;
		
		std::string name = "guess_" + std::to_string(i);
		
		auto guessmat = dbcsr::eigen_to_matrix(mat, m_world, name,
			o, v, dbcsr::type::no_symmetry);
		
		dav_guess[i] = guessmat;
		
		//dbcsr::print(*guessmat);
		
	}
	
	// set up ADC(1) and davidson 
	
	bool do_balancing = m_opt.get<bool>("balanced", ADC_BALANCING);
	bool do_block = m_opt.get<bool>("block", ADC_BLOCK);
	double conv_adc1 = m_opt.get<double>("adc1/dav_conv", ADC_ADC1_DAV_CONV);
	int maxiter_adc1 = m_opt.get<double>("adc1/maxiter", ADC_ADC1_MAXITER);
	
	auto adc1_mvp = create_adc1();
	math::davidson<MVP> dav(m_world.comm(), LOG.global_plev());
	
	dav.set_factory(adc1_mvp);
	dav.set_diag(m_d_ov);
	dav.pseudo(false);
	dav.balancing(do_balancing);
	dav.block(do_block);
	dav.conv(conv_adc1);
	dav.maxiter(maxiter_adc1);	
	
	int nroots = m_opt.get<int>("nroots", ADC_NROOTS);
	
	auto& t_davidson = TIME.sub("Davidson diagonalization");
	
	/*
	auto r = filio::read_matrix("lauric.dat", "name", m_world, o, v,
		dbcsr::type::no_symmetry);
		
	auto nto = get_canon_pao(r, m_hfwfn->c_bo_A(),
		m_hfwfn->c_bv_A(), *m_hfwfn->eps_occ_A(), *m_hfwfn->eps_vir_A(),
		0.995);*/
		
	/*

	auto cbo = m_hfwfn->c_bo_A();
	auto cbv = m_hfwfn->c_bv_A(); 	
	
	auto rcopy = dbcsr::copy(*r).build();
	rcopy->scale(1.0/sqrt(rcopy->dot(*rcopy)));
	
	math::SVD svdcomp(rcopy, 'V', 'V', 10);
	svdcomp.compute();
	
	auto fullrank = svdcomp.rank();
	auto fullvals = svdcomp.s();
	
	LOG.os<>("RANK FULL: ", fullrank, '\n');
	LOG.os<>("SING VAL FULL\n");
	for (auto sval : fullvals) {
		LOG.os<>(sval, " ");
	}
	LOG.os<>('\n');
	
	auto [atomblks, sigblks] = get_significant_blocks(r, 0.95, nullptr, 0.0);
	
	LOG.os<>("Significant atoms:\n");
	for (auto blk : atomblks) {
		LOG.os<>(blk, " ");
	}
	LOG.os<>('\n');
	
	LOG.os<>("Significant blocks:\n");
	for (auto blk : sigblks) {
		LOG.os<>(blk, " ");
	}
	LOG.os<>('\n');
	
	locorb::mo_localizer moloc(m_world, m_hfwfn->mol());
	
	auto reg = m_ao.get_registry();
	auto s_bb = reg.get<dbcsr::shared_matrix<double>>(ints::key::ovlp_bb);
	
	auto [co_pr, u_ro, epso_r] = moloc.compute_truncated_pao(
		cbo, s_bb, *epso, sigblks);
	auto [cv_ps, u_sv, epsv_s] = moloc.compute_truncated_pao(
		cbv, s_bb, *epsv, sigblks);
	
	auto pv = dbcsr::create_template<double>(*s_bb)
		.name("pv")
		.build();
		
	auto pvloc = dbcsr::create_template<double>(*s_bb)
		.name("pv")
		.build();
		
	dbcsr::multiply('N', 'T', *cbv, *cbv, *pv).perform();
	dbcsr::multiply('N', 'T', *cv_ps, *cv_ps, *pvloc).perform();
	
	pv->filter(1e-3);
	pvloc->filter(1e-3);
	
	LOG.os<>("OCCUP: ", pv->occupation(), " ", pvloc->occupation(), '\n');
	
	MPI_Barrier(m_world.comm());
	exit(0);
	
	int nbas = m_hfwfn->mol()->c_basis()->nbf();
	int nbas_t = co_pr->nfullrows_total();
	int nocc_t = co_pr->nfullcols_total();
	int nvir_t = cv_ps->nfullcols_total();
	
	LOG.os<>("AO: ", nbas, " -> ", nbas_t, '\n');
	LOG.os<>("MO (O): ", nocc, " -> ", nocc_t, '\n');
	LOG.os<>("MO (V): ", nvir, " -> ", nvir_t, '\n');
	LOG.os<>("NXbas(prev): ", m_hfwfn->mol()->c_dfbasis()->nbf(), '\n');

	int natoms = m_hfwfn->mol()->atoms().size();
	std::vector<int> fullatoms(natoms);
	std::iota(fullatoms.begin(), fullatoms.end(), 0);
	
	LOG.os<>("Forming fragment...\n");
	auto mol_frag = m_hfwfn->mol()->fragment(nocc_t, nocc_t, nvir_t, 
		nvir_t, fullatoms);
		
	LOG.os<>("Fragment info:\n");
	LOG.os<>("Nelec alpha: ", mol_frag->nele_alpha(), '\n');
	LOG.os<>("Nelec beta: ", mol_frag->nele_beta(), '\n');
	LOG.os<>("Occ: ", mol_frag->nocc_alpha(), '\n');
	LOG.os<>("Vir: ", mol_frag->nvir_alpha(), '\n');
	LOG.os<>("Nbas: ", mol_frag->c_basis()->nbf(), '\n');
	LOG.os<>("NXbas: ", mol_frag->c_dfbasis()->nbf(), '\n');
		
	//dbcsr::print(*ctrunc);
	
	auto x = mol_frag->dims().x();
	LOG.os<>("THIS IS X: \n");
	for (auto f : x) {
		std::cout << f << std::endl;
	}
	
	
	ints::aoloader fragloader(m_world, mol_frag, m_opt);
	
	fock::load_jints(fock::jmethod::dfao, ints::metric::coulomb, fragloader);
	fock::load_kints(fock::kmethod::dfao, ints::metric::coulomb, fragloader);
	
	fragloader.compute();
	
	auto freg = fragloader.get_registry();
	auto eribatched = freg.get<dbcsr::sbtensor<3,double>>(ints::key::coul_xbb);
	auto v_xx = freg.get<dbcsr::shared_matrix<double>>(ints::key::coul_xx_inv);
	auto fitbatched = freg.get<dbcsr::sbtensor<3,double>>(ints::key::dfit_coul_xbb);
	
	auto ptr = create_MVP_AOADC1(m_world, mol_frag, LOG.global_plev())
		.c_bo(co_pr)
		.c_bv(cv_ps)
		.eps_occ(epso_r)
		.eps_vir(epsv_s)
		.eri3c2e_batched(eribatched)
		.fitting_batched(fitbatched)
		.v_xx(v_xx)
		.jmethod(fock::jmethod::dfao)
		.kmethod(fock::kmethod::dfao)
		.build();
		
	ptr->init();
		
	auto u_rs = u_transform(r, 'N', u_ro, 'T', u_sv);
	
	dbcsr::print(*u_rs);
	
	double n = sqrt(u_rs->dot(*u_rs));
	u_rs->scale(1.0/n);
	
	auto sig = ptr->compute(u_rs, 0.0);
	
	double energy = u_rs->dot(*sig);
	
	LOG.os<>("ENERGY: ", energy, '\n');
		
	MPI_Barrier(m_world.comm());
	exit(0);*/
	
	LOG.os<>("==== Starting ADC(1) Computation ====\n\n"); 
	
	t_davidson.start();
	dav.compute(dav_guess, nroots);
	t_davidson.finish();
	
	adc1_mvp->print_info();
			
	//auto rvecs = std::vector<dbcsr::shared_matrix<double>>{r}; //dav.ritz_vectors();
	//auto ex = std::vector<double>{0.21392}; //dav.eigvals();
	
	auto rvecs = dav.ritz_vectors();
	auto ex = dav.eigvals();
	
	LOG.os<>("==== Finished ADC(1) Computation ====\n\n"); 
	
	int nstart = do_block ? 0 : nroots-1;
	
	LOG.os<>("ADC(1) Excitation energies:\n");
	for (int iroot = nstart; iroot != nroots; ++iroot) {
		LOG.os<>("Excitation nr. ", iroot+1, " : ", ex[iroot], '\n');
	}
	
	/*
	auto r1 = rvecs[0];
	
	auto c_bo = m_hfwfn->c_bo_A();
	auto c_bv = m_hfwfn->c_bv_A(); 
	
	auto u_bb_a = u_transform(r1, 'N', c_bo, 'T', c_bv);
	
	math::SVD solver(u_bb_a, 'V', 'V', 9999);
	solver.compute();
	
	int rank = solver.rank();
	auto r = dbcsr::split_range(rank, 5);
	
	auto U = solver.U(b,r);
	auto Vt = solver.Vt(r,b);
	
	exit(0);*/
	
	bool do_adc2 = m_opt.get<bool>("do_adc2", ADC_DO_ADC2);
	
	if (!do_adc2) return; 
	
	//int theta = m_opt.get<int>("adc2/theta", -1.0);
	
	auto paos = get_canon_pao(rvecs[0], m_hfwfn->c_bo_A(), m_hfwfn->c_bv_A(), 
		*m_hfwfn->eps_occ_A(), *m_hfwfn->eps_vir_A(), 0.995);
		
	auto dpao_ov = u_transform(m_d_ov, 'T', paos.u_or, 'N', paos.u_vs);
	
	LOG.os<>("==== Starting ADC(2) Computation ====\n\n"); 
	
	math::diis_davidson<MVP> mdav(m_world.comm(), LOG.global_plev());
	
	double conv_micro_adc2 = m_opt.get<double>("adc2/micro_conv", ADC_ADC2_MICRO_CONV);
	double conv_macro_adc2 = m_opt.get<double>("adc2(macro_conv", ADC_ADC2_MACRO_CONV);
	int micro_maxiter_adc2 = m_opt.get<int>("adc2/micro_maxiter", ADC_ADC2_MICRO_MAXITER);
	int macro_maxiter_adc2 = m_opt.get<int>("adc2/macro_maxiter", ADC_ADC2_MACRO_MAXITER);
	
	mdav.macro_maxiter(macro_maxiter_adc2);
	mdav.macro_conv(conv_macro_adc2);
	
	mdav.set_diag(m_d_ov); //dpao_ov);
	mdav.balancing(do_balancing);
	mdav.micro_maxiter(micro_maxiter_adc2);
	
	int istart = (do_block) ? 0 : nroots-1;
	std::vector<double> ex_adc2(nroots,0.0);
	
	bool local = m_opt.get<bool>("adc2/local", ADC_ADC2_LOCAL);
	bool is_init = false;
	std::shared_ptr<MVP> adc2_mvp; 
	
	for (int iroot = istart; iroot != nroots; ++iroot) {
		
		LOG.os<>("============================================\n");
		LOG.os<>("    Computing excited state nr. ", iroot+1, '\n');
		LOG.os<>("============================================\n\n");
		
		LOG.os<>("Setting up ADC(2) MVP builder.\n");
		
		if (local) {
			//auto atomlist = get_significant_blocks(rvecs[0], 0.9975, nullptr, 0.0);
			//adc2_mvp = create_adc2(atomlist);
		} else if (!is_init) {
			adc2_mvp = create_adc2(); //paos);
			//rvecs[0] = u_transform(rvecs[0], 'T', paos.u_or, 'N', paos.u_vs); 
		}
		
		mdav.set_factory(adc2_mvp);
		mdav.compute(rvecs, iroot+1, ex[iroot]);
		
		double en = mdav.eigval()[iroot];
		ex_adc2[iroot] = en;
		
		if (local) {
			adc2_mvp.reset();
		}
		
	}
	
	LOG.os<>("==== Finished ADC(2) Computation ====\n\n"); 
			
	LOG.os<>("ADC(2) Excitation energies:\n");
	for (int iroot = nstart; iroot != nroots; ++iroot) {
		LOG.os<>("Excitation nr. ", iroot+1, " : ", ex_adc2[iroot], '\n');
	}
	
	TIME.print_info();
		
}

std::tuple<std::vector<int>, std::vector<int>> 
	adcmod::get_significant_blocks(dbcsr::shared_matrix<double> u_ia, 
	double theta, dbcsr::shared_matrix<double> metric_bb, double gamma) 
{
	
	auto c_bo = m_hfwfn->c_bo_A();
	auto c_bv = m_hfwfn->c_bv_A(); 
	
	auto u_bb_a = u_transform(u_ia, 'N', c_bo, 'T', c_bv);
	
	auto b = m_hfwfn->mol()->dims().b();
	auto atoms = m_hfwfn->mol()->atoms();
    int natoms = atoms.size();

	auto retvec = b;
	std::iota(retvec.begin(), retvec.end(), 0);
	
	std::vector<int> retmol(natoms);
	std::iota(retmol.begin(), retmol.end(), 0);
		
	//return std::make_tuple(retmol,retvec);
	
	auto dims = m_world.dims();
    auto dist = dbcsr::default_dist(b.size(), m_world.size(), b);
    
    double norm = u_bb_a->norm(dbcsr_norm_frobenius);
    
    u_bb_a->scale(1.0/norm);
        
    u_bb_a->replicate_all();

    std::vector<double> occ_norms(b.size(),0.0);
    std::vector<double> vir_norms(b.size(),0.0);
    std::vector<int> idx_occ(b.size());
    std::vector<int> idx_vir(b.size());
    
    std::iota(idx_occ.begin(), idx_occ.end(), 0);
    std::iota(idx_vir.begin(), idx_vir.end(), 0);
    
    auto blkmap = m_hfwfn->mol()->c_basis()->block_to_atom(atoms);
    
    // loop over blocks rows/cols
    for (int iblk = 0; iblk != b.size(); ++iblk) {
		
		if (dist[iblk] == m_world.rank()) {
			
			int blksize_i = b[iblk];
			
			std::vector<double> blknorms_r(blksize_i,0.0);
			std::vector<double> blknorms_c(blksize_i,0.0);
			
			for (int jblk = 0; jblk != b.size(); ++jblk) {
				
				bool foundr = false;
				bool foundc = false;
				
				auto blkr_p = u_bb_a->get_block_p(iblk,jblk,foundr);
				auto blkc_p = u_bb_a->get_block_p(jblk,iblk,foundc);
				
				int blksize_j = b[jblk];
				
				if (foundr) { 
					
					for (int i = 0; i != blksize_i; ++i) {
						for (int j = 0; j != blksize_j; ++j) {
							blknorms_r[i] += pow(blkr_p(i,j),2.0);
						}
					}
				}
				
				if (foundc) {
					
					for (int i = 0; i != blksize_i; ++i) {
						for (int j = 0; j != blksize_j; ++j) {
							blknorms_c[i] += pow(blkc_p(j,i),2.0);
						}
					}
				}
				
				
				
			}
			
			occ_norms[iblk] = std::accumulate(blknorms_r.begin(),blknorms_r.end(),0.0);
			vir_norms[iblk] = std::accumulate(blknorms_c.begin(),blknorms_c.end(),0.0);
			
		}
		
	}
	
	// communicate to all processes
	MPI_Allreduce(MPI_IN_PLACE, occ_norms.data(), b.size(), MPI_DOUBLE,
		MPI_SUM, m_world.comm());
		
	MPI_Allreduce(MPI_IN_PLACE, vir_norms.data(), b.size(), MPI_DOUBLE,
		MPI_SUM, m_world.comm());
	
	LOG.os<2>("NORMS ALL (OCC): \n");
	for (auto v : occ_norms) {
		LOG.os<2>(v, " ");
	} LOG.os<2>('\n');
	
	LOG.os<2>("NORMS ALL (VIR): \n");
	for (auto v : vir_norms) {
		LOG.os<2>(v, " ");
	} LOG.os<2>('\n');
	
	std::sort(idx_occ.begin(), idx_occ.end(), 
		[&occ_norms](const int a, const int b) {
			return (occ_norms[a] > occ_norms[b]);
		});
		
	std::sort(idx_vir.begin(), idx_vir.end(), 
		[&vir_norms](const int a, const int b) {
			return (vir_norms[a] > vir_norms[b]);
		});
	
	double totnorm_occ = 0.0;
	double totnorm_vir = 0.0;
	
	std::vector<int> atoms_check(natoms,0);
	std::vector<int> atoms_all;
	
	for (auto idx : idx_occ) {
		//std::cout << totnorm << std::endl;
		if (totnorm_occ < theta) {
			totnorm_occ += occ_norms[idx];
			int iatom = blkmap[idx];
			atoms_check[iatom] = 1;
		} else {
			break;
		}
	}
	
	for (auto idx : idx_vir) {
		//std::cout << totnorm << std::endl;
		if (totnorm_vir < theta) {
			totnorm_vir += vir_norms[idx];
			int iatom = blkmap[idx];
			atoms_check[iatom] = 1;
		} else {
			break;
		}
	}
	
	LOG.os<1>("ATOM CHECK: \n");
	for (auto v : atoms_check) {
		LOG.os<1>(v, " ");
	} LOG.os<1>('\n');
	
	u_bb_a->distribute();
	u_bb_a->clear();
	
	// add all blocks centered on atoms
	vec<int> idx_check(b.size(),0);
	vec<int> idx_all;
	
	for (int iblk = 0; iblk != b.size(); ++iblk) {
		for (int iatom = 0; iatom != natoms; ++iatom) {
			if (!atoms_check[iatom]) continue;
			if (blkmap[iblk] == iatom) idx_check[iblk] = 1;
		}
	}
	
	LOG.os<1>("IDX CHECK: \n");
	for (auto v : idx_check) {
		LOG.os<1>(v, " ");
	} LOG.os<1>('\n');
	
	int idx = 0;
	for (auto b : idx_check) {
		if (b) idx_all.push_back(idx);
		idx++;
	}
	
	idx = 0;
	for (auto a : atoms_check) {
		if (a) atoms_all.push_back(idx);
		idx++;
	}
		
	LOG.os<>("IDX ALL: \n");
	
	MPI_Barrier(m_world.comm());
	for (int ip = 0; ip != m_world.size(); ++ip) {
		if (ip == m_world.rank()) {
			for (auto v : idx_all) {
				std::cout << v << " ";
			} std::cout << std::endl;
		}
		MPI_Barrier(m_world.comm());
	}
	
	return std::make_tuple(atoms_all, idx_all);
	
	/*
	if (metric_bb == nullptr) return idx_all;
	
	// now add blocks connected to indices by metric
		
	MPI_Comm comm = m_world.comm();
	
	auto idx_check_aug = idx_check;
	int nblk = idx_check.size();
		
	for (int iblk = 0; iblk != nblk; ++iblk) {
		if (!idx_check[iblk]) continue;
		for (int jblk = 0; jblk != nblk; ++jblk) {
			bool found = false;
			
			int i = (iblk <= jblk) ? iblk : jblk;
			int j = (iblk <= jblk) ? jblk : iblk;
			
			auto blk_p = metric_bb->get_block_p(i,j,found);
			if (!found) continue;
			double max = blk_p.max_abs();
			if (max > gamma) idx_check_aug[jblk] = 1;
		}
	}
	
	MPI_Allreduce(MPI_IN_PLACE, idx_check_aug.data(), idx_check_aug.size(), 
		MPI_INT, MPI_LOR, comm);
			
	LOG.os<1>("IDX CHECK NEW: \n");
	for (auto v : idx_check_aug) {
		LOG.os<1>(v, " ");
	} LOG.os<1>('\n');
	
	vec<int> idx_all_aug;
	idx = 0;
	for (auto b : idx_check_aug) {
		if (b) idx_all_aug.push_back(idx);
		++idx;
	}
	
	LOG.os<1>("IDX ALL (AUG): \n");
	for (auto v : idx_all_aug) {
		LOG.os<1>(v, " ");
	} LOG.os<1>('\n');
	
	return idx_all_aug;	*/
	
}

adcmod::canon_lmo adcmod::get_canon_nto(dbcsr::shared_matrix<double> u_ia, 
	dbcsr::shared_matrix<double> c_bo, dbcsr::shared_matrix<double> c_bv, 
	std::vector<double> eps_o, std::vector<double> eps_v,
	double theta)
{
	
	LOG.os<>("Computing canonicalized NTO coefficient matrices with eps = ", theta, '\n');
	
	// Preliminaries
	auto b = c_bo->row_blk_sizes();
	auto o = c_bo->col_blk_sizes();
	auto v = c_bv->col_blk_sizes();
	
	// Step 1: SVD decomposition of u_ia'
	
	LOG.os<>("-- Computing SVD decomposition of u_ia\n"); 
	
	auto u_ia_copy = dbcsr::matrix<>::copy(*u_ia).build();
	
	double norm = 1.0/sqrt(u_ia->dot(*u_ia));
	u_ia_copy->scale(norm);
	
	math::SVD svd_decomp(u_ia_copy, 'V', 'V', 0);
	svd_decomp.compute(theta);
	int rank = svd_decomp.rank();
	
	auto r = dbcsr::split_range(rank, o[0]);
	
	auto u_ir = svd_decomp.U(o,r);
	auto vt_rv = svd_decomp.Vt(r,v);

	// Step 2: Orthogonalize MOs
	
	LOG.os<>("-- Orthogonalizing new MOs\n"); 
	
	math::SVD svd_o(u_ir, 'V', 'V', 0);
	math::SVD svd_v(vt_rv, 'V', 'V', 0);
	
	svd_o.compute(1e-10);
	svd_v.compute(1e-10);
	
	auto t = dbcsr::split_range(svd_o.rank(), o[0]);
	auto s = dbcsr::split_range(svd_v.rank(), o[1]);
	
	auto uortho_ot = svd_o.U(o,t);
	auto vtortho_sv = svd_v.Vt(s,v);
	
	// Step 3: Form fock matrices
	
	LOG.os<>("-- Forming Fock matrices\n"); 
	
	auto wrd = c_bo->get_world();
	
	auto form_fock = [&](std::vector<int> m, std::vector<double> eps_m) {
		
		auto f = dbcsr::matrix<>::create()
			.set_world(wrd)
			.name("fock")
			.row_blk_sizes(m)
			.col_blk_sizes(m)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();
			
		f->reserve_diag_blocks();
		f->set_diag(eps_m);
		
		return f;
		
	};
	
	auto f_oo = form_fock(o,eps_o);
	auto f_vv = form_fock(v,eps_v);
	
	auto f_tt = u_transform(f_oo, 'T', uortho_ot, 'N', uortho_ot);
	auto f_ss = u_transform(f_vv, 'N', vtortho_sv, 'T', vtortho_sv);
	
	// Step 4 : Canonicalize
	
	LOG.os<>("-- Canonicalizing NTOs\n"); 
	
	math::hermitian_eigen_solver hsolver_o(f_tt, 'V', false);
	math::hermitian_eigen_solver hsolver_v(f_ss, 'V', false);
	
	hsolver_o.compute();
	hsolver_v.compute();
	
	auto canon_tt = hsolver_o.eigvecs();
	auto canon_ss = hsolver_v.eigvecs();
	
	auto eps_t = hsolver_o.eigvals();
	auto eps_s = hsolver_v.eigvals();
	
	auto trans_ot = dbcsr::matrix<>::create()
		.name("trans ot")
		.set_world(wrd)
		.row_blk_sizes(o)
		.col_blk_sizes(t)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto trans_vs = dbcsr::matrix<>::create()
		.name("trans vs")
		.set_world(wrd)
		.row_blk_sizes(v)
		.col_blk_sizes(s)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	auto c_bt = dbcsr::matrix<>::create()
		.name("SVD c_bo")
		.set_world(wrd)
		.row_blk_sizes(b)
		.col_blk_sizes(t)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto c_bs = dbcsr::matrix<>::create()
		.name("SVD c_bv")
		.set_world(wrd)
		.row_blk_sizes(b)
		.col_blk_sizes(s)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	LOG.os<>("-- Forming final NTO coefficient matrices\n"); 

	dbcsr::multiply('N', 'N', 1.0, *uortho_ot, *canon_tt, 0.0, *trans_ot)
		.perform();
	dbcsr::multiply('T', 'N', 1.0, *vtortho_sv, *canon_ss, 0.0, 
		*trans_vs).perform();
	
	dbcsr::multiply('N', 'N', 1.0, *c_bo, *trans_ot, 0.0, *c_bt).perform();
	dbcsr::multiply('N', 'N', 1.0, *c_bv, *trans_vs, 0.0, *c_bs).perform();

	auto print = [&](auto v) {
		for (auto d : v) {
			LOG.os<>(d, " ");
		} LOG.os<>('\n');
	};
	
	LOG.os<>("Occupied energies.\n");
	print(eps_o);
	print(eps_t);

	LOG.os<>("Virtual energies.\n");
	print(eps_v);
	print(eps_s);
	
	int no_t = c_bo->nfullcols_total();
	int nv_t = c_bv->nfullcols_total();
	int no = c_bt->nfullcols_total();
	int nv = c_bs->nfullcols_total();
	
	LOG.os<>("DIMENSIONS REDUCED FROM: ", no_t, "/", nv_t, " -> ",
		no, "/", nv, '\n');

	return canon_lmo{c_bt, c_bs, trans_ot, trans_vs, eps_t, eps_s};

}

adcmod::canon_lmo adcmod::get_canon_pao(dbcsr::shared_matrix<double> u_ia, 
	dbcsr::shared_matrix<double> c_bo, dbcsr::shared_matrix<double> c_bv, 
	std::vector<double> eps_o, std::vector<double> eps_v,
	double theta)
{
	
	auto [atom_blks, basis_blks] = get_significant_blocks(u_ia, theta, nullptr, 0.0);
	
	locorb::mo_localizer moloc(m_world, m_hfwfn->mol());
	
	auto reg = m_aoloader->get_registry();
	auto s_bb = reg.get<dbcsr::shared_matrix<double>>(ints::key::ovlp_bb);
	
	auto [c_br, u_or, eps_r] = moloc.compute_truncated_pao(c_bo, s_bb, eps_o, basis_blks, nullptr);
	auto [c_bs, u_vs, eps_s] = moloc.compute_truncated_pao(c_bv, s_bb, eps_v, basis_blks, nullptr);
	
	int no_t = c_bo->nfullcols_total();
	int nv_t = c_bv->nfullcols_total();
	int no = c_br->nfullcols_total();
	int nv = c_bs->nfullcols_total();
	
	LOG.os<>("DIMENSIONS REDUCED FROM: ", no_t, "/", nv_t, " -> ",
		no, "/", nv, '\n');

	
	return canon_lmo{c_br, c_bs, u_or, u_vs, eps_r, eps_s};
	
}

} // end namespace
