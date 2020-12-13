#include "adc/adcmod.h"
#include "adc/adc_mvp.h"
#include "math/solvers/davidson.h"
#include "math/linalg/piv_cd.h"
#include "locorb/locorb.h"

#include <dbcsr_conversions.hpp>

namespace adc {

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
	auto eigen_ia = dbcsr::matrix_to_eigen(m_d_ov);
	
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
	
	LOG.os<>("==== Starting ADC(1) Computation ====\n\n"); 
	
	t_davidson.start();
	dav.compute(dav_guess, nroots);
	t_davidson.finish();
	
	adc1_mvp->print_info();
			
	auto rvecs = dav.ritz_vectors();
	auto ex = dav.eigvals();
	
	LOG.os<>("==== Finished ADC(1) Computation ====\n\n"); 
	
	int nstart = do_block ? 0 : nroots-1;
	
	LOG.os<>("ADC(1) Excitation energies:\n");
	for (int iroot = nstart; iroot != nroots; ++iroot) {
		LOG.os<>("Excitation nr. ", iroot+1, " : ", ex[iroot], '\n');
	}
	
	bool do_adc2 = m_opt.get<bool>("do_adc2", ADC_DO_ADC2);
	
	if (!do_adc2) return; 
	
	LOG.os<>("==== Starting ADC(2) Computation ====\n\n"); 
	
	math::modified_davidson<MVP> mdav(m_world.comm(), LOG.global_plev());
	
	double conv_micro_adc2 = m_opt.get<double>("adc2/micro_conv", ADC_ADC2_MICRO_CONV);
	double conv_macro_adc2 = m_opt.get<double>("adc2(macro_conv", ADC_ADC2_MACRO_CONV);
	int micro_maxiter_adc2 = m_opt.get<int>("adc2/micro_maxiter", ADC_ADC2_MICRO_MAXITER);
	int macro_maxiter_adc2 = m_opt.get<int>("adc2/macro_maxiter", ADC_ADC2_MACRO_MAXITER);
	
	mdav.macro_maxiter(macro_maxiter_adc2);
	mdav.macro_conv(conv_macro_adc2);
	
	mdav.sub().set_diag(m_d_ov);
	mdav.sub().pseudo(true);
	mdav.sub().block(false);
	mdav.sub().balancing(do_balancing);
	mdav.sub().conv(conv_micro_adc2);
	mdav.sub().maxiter(micro_maxiter_adc2);
	
	int istart = (do_block) ? 0 : nroots-1;
	std::vector<double> ex_adc2(nroots,0.0);
	
	bool local = m_opt.get<bool>("adc2/local", ADC_ADC2_LOCAL);
	bool is_init = false;
	std::shared_ptr<MVP> adc2_mvp; 
	
	for (int iroot = istart; iroot != nroots; ++iroot) {
		
		LOG.os<>("============================================\n");
		LOG.os<>("=== Computing excited state nr. ", iroot+1, '\n');
		LOG.os<>("============================================\n\n");
		
		LOG.os<>("Setting up ADC(2) MVP builder.\n");
		
		if (local) {
			auto atomlist = get_significant_blocks(rvecs[0], 0.9975, nullptr, 0.0);
			adc2_mvp = create_adc2(atomlist);
		} else if (!is_init) {
			adc2_mvp = create_adc2();
		}
		
		mdav.sub().set_factory(adc2_mvp);
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

std::vector<int> adcmod::get_significant_blocks(dbcsr::shared_matrix<double> u_ia, 
	double theta, dbcsr::shared_matrix<double> metric_bb, double gamma) 
{
	
	auto c_bo = m_hfwfn->c_bo_A();
	auto c_bv = m_hfwfn->c_bv_A(); 
	
	auto u_bb_a = u_transform(u_ia, 'N', c_bo, 'T', c_bv);
	
	auto b = m_hfwfn->mol()->dims().b();
	
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
    
    auto atoms = m_hfwfn->mol()->atoms();
    int natoms = atoms.size();
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
	
	return idx_all_aug;	
	
}


}
