#include "adc/adcmod.h"
#include "adc/adc_mvp.h"
#include "math/solvers/davidson.h"
#include "math/linalg/piv_cd.h"
#include "locorb/locorb.h"

#include <dbcsr_conversions.hpp>

namespace adc {

void adcmod::compute() {
	
		// BEFORE: init tensors base (ints, metrics, etc..., put into m_reg)
		
		init_ao_tensors();
		
		// AFTER: init tensors (2) (mo-ints, diags, amplitudes ...) 
		
		init_mo_tensors();
				
		// SECOND: Generate guesses
		
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
		
		math::davidson<MVP> dav(m_world.comm(), LOG.global_plev());
		
		dav.set_factory(m_adc1_mvp);
		dav.set_diag(m_d_ov);
		dav.pseudo(false);
		dav.conv(m_opt.get<double>("dav_conv", ADC_DAV_CONV));
		dav.maxiter(100);	
		
		int nroots = m_opt.get<int>("nroots", ADC_NROOTS);
		
		auto& t_davidson = TIME.sub("Davidson diagonalization");
		
		t_davidson.start();
		dav.compute(dav_guess, nroots);
		t_davidson.finish();
		
		m_adc1_mvp->print_info();
		
		LOG.os<>("Excitation energy of state nr. ", m_nroots, ": ", dav.eigvals()[m_nroots-1], '\n');
		
		auto rvecs = dav.ritz_vectors();
		auto vec_k = rvecs[m_nroots-1];
		
		auto c_bo = m_hfwfn->c_bo_A();
		auto c_bv = m_hfwfn->c_bv_A();
		
		LOG.os<>("AOs\n");
		
		auto v_ht = dbcsr::create<double>()
			.name("ht")
			.set_world(m_world)
			.row_blk_sizes(o)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
			
		auto v_bb = dbcsr::create<double>()
			.name("v_bb")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
		
		
		dbcsr::multiply('N', 'T', *vec_k, *c_bv, *v_ht).perform();
		dbcsr::multiply('N', 'N', *c_bo, *v_ht, *v_bb).perform();
		
		auto p = m_hfwfn->po_bb_A();
		
		//dbcsr::print(*v_bb);
		
		LOG.os<>("Occupation: ", v_bb->occupation() * 100, "%\n");
		LOG.os<>("Compared to ", p->occupation()*100, "% for density matrix.\n");
		
		// list of exitations
		
		// Which blocks are absent?
		vec<std::vector<int>> blk_list(10, std::vector<int>(b.size(),0));
		
		dbcsr::iterator<double> iter(*v_bb);
		iter.start();
				
		while (iter.blocks_left()) {
			
			iter.next_block();
			
			for (int i = 0; i != iter.row_size(); ++i) {
				for (int j = 0; j != iter.col_size(); ++j) {
					
					for (int e = 0; e != 10; ++e) {
						if (fabs(iter(i,j)) > pow(10.0,-e)) {
							blk_list[e][iter.row()] += 1;
							blk_list[e][iter.col()] += 1;
						}
					}
				}
			} 
			
		}
		
		iter.stop();
			
		int nblk = 0;
		
		for (int i = 0; i != 10; ++i) {
			MPI_Allreduce(MPI_IN_PLACE, blk_list[i].data(), blk_list[i].size(), 
				MPI_INT, MPI_SUM, m_world.comm());
		}
		
		for (int i = 0; i != 10; ++i) {
			int nblk = 0;
			for (auto& e : blk_list[i]) {
				if (e != 0) nblk++;
			}
			LOG.os<>("Basis blocks (", pow(10.0,-i), "): ", nblk, " out of ", b.size(), '\n'); 
		}
		
		auto v1 = get_significant_blocks(v_bb,0.9975,nullptr,0);
		auto v2 = get_significant_blocks(v_bb,0.99975,nullptr,0);
		auto v3 = get_significant_blocks(v_bb,0.999975,nullptr,0);
		auto v4 = get_significant_blocks(v_bb,0.9999975,nullptr,0);
		auto v5 = get_significant_blocks(v_bb,0.9975,m_s_bb,1e-4);

		TIME.print_info();
		
}

std::vector<int> adcmod::get_significant_blocks(dbcsr::shared_matrix<double> u_bb, 
	double theta, dbcsr::shared_matrix<double> metric_bb, double gamma) 
{
	
	//dbcsr::print(*u_bb);
	//dbcsr::print(*metric_bb);
	
	auto u_bb_a = dbcsr::copy<double>(u_bb).get();
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
	
	LOG.os<>("NORMS ALL (OCC): \n");
	for (auto v : occ_norms) {
		LOG.os<>(v, " ");
	} LOG.os<>('\n');
	
	LOG.os<>("NORMS ALL (VIR): \n");
	for (auto v : vir_norms) {
		LOG.os<>(v, " ");
	} LOG.os<>('\n');
	
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
	
	LOG.os<>("ATOM CHECK: \n");
	for (auto v : atoms_check) {
		LOG.os<>(v, " ");
	} LOG.os<>('\n');
	
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
	
	LOG.os<>("IDX CHECK: \n");
	for (auto v : idx_check) {
		LOG.os<>(v, " ");
	} LOG.os<>('\n');
	
	int idx = 0;
	for (auto b : idx_check) {
		if (b) idx_all.push_back(idx);
		idx++;
	}
		
	LOG.os<>("IDX ALL: \n");
	for (auto v : idx_all) {
		LOG.os<>(v, " ");
	} LOG.os<>('\n');
	
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
			
	LOG.os<>("IDX CHECK NEW: \n");
	for (auto v : idx_check_aug) {
		LOG.os<>(v, " ");
	} LOG.os<>('\n');
	
	vec<int> idx_all_aug;
	idx = 0;
	for (auto b : idx_check_aug) {
		if (b) idx_all_aug.push_back(idx);
		++idx;
	}
	
	LOG.os<>("IDX ALL (AUG): \n");
	for (auto v : idx_all_aug) {
		LOG.os<>(v, " ");
	} LOG.os<>('\n');
	
	exit(0);
	return idx_all_aug;	
	
}


}
