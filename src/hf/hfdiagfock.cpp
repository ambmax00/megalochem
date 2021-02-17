#include "hf/hfmod.hpp"
#include <dbcsr_conversions.hpp>
#include <dbcsr_matrix_ops.hpp>
#include "math/solvers/hermitian_eigen_solver.hpp"
#include "math/linalg/piv_cd.hpp"
#include <algorithm> 

namespace hf { 
	
void hfmod::diag_fock() {
	
	//updates coeffcient matrices (c_bo_A, c_bo_B) and densities (p_bo_A, p_bo_B)
	
	auto& t_diag = TIME.sub("Fock Diagonalization");
	
	t_diag.start();
	
	auto diagonalize = [&](smat_d& f_bb, smat_d& c_bm, std::vector<double>& eps, std::string x) {
		
		LOG.os<2>("Orthogonalizing Fock Matrix: ", x, '\n');
		
		auto FX = dbcsr::matrix<>::create_template(*f_bb)
			.name("FX").matrix_type(dbcsr::type::no_symmetry).build();
		auto XFX = dbcsr::matrix<>::create_template(*f_bb)
			.name("XFX").build();
		
		dbcsr::multiply('N','N',1.0,*f_bb,*m_x_bb,0.0,*FX).perform();
		
		//dbcsr::print(*f_bb);
		
		//dbcsr::print(FX);
		
		dbcsr::multiply('T','N',1.0,*m_x_bb,*FX,0.0,*XFX).perform(); 
		
		FX->release();
		
		if (LOG.global_plev() >= 2) 
			dbcsr::print(*XFX);
		
		math::hermitian_eigen_solver solver(XFX, 'V', (LOG.global_plev() >= 2) ? true : false);
		
		vec<int> m = (x == "A") ? m_mol->dims().ma() : m_mol->dims().mb();
		
		solver.eigvec_colblks(m).compute();
		
		auto eigval = solver.eigvals();
		auto c_bm_x = solver.eigvecs();
		
		LOG.os<3>("Eigenvalues: \n");
		if (LOG.global_plev() >= 3 && m_world.rank() == 0) {
			for (auto e : eigval) {
				LOG.os<>(e, " ");
			} LOG.os<>('\n');
		}
		
		eps.resize(eigval.size());
		std::copy(eigval.data(), eigval.data() + eigval.size(), eps.begin());
		
		LOG.os<3>("Eigenvectors: \n");
		if (LOG.global_plev() >= 3) {
			dbcsr::print(*c_bm_x);
		}
		
		auto new_c_bm = dbcsr::matrix<>::create_template(*c_bm_x)
			.name(c_bm_x->name()).build();
		*c_bm = std::move(*new_c_bm); 
	
		//Transform back
		dbcsr::multiply('N','N',1.0,*m_x_bb,*c_bm_x,0.0,*c_bm).perform();
		
		if (LOG.global_plev() >= 2) 
			dbcsr::print(*c_bm);
		
		XFX->release();
		c_bm_x->release();
			
	};
	
	auto form_density = [&] (dbcsr::shared_matrix<double>& p_bb, 
		dbcsr::shared_matrix<double>& c_bm, std::string x) {
		
		int limit = 0;
		
		if (x == "A") limit = m_mol->nocc_alpha() - 1;
		if (x == "B") limit = m_mol->nocc_beta() - 1;
		
		
		dbcsr::multiply('N', 'T', 1.0, *c_bm, *c_bm, 0.0, *p_bb)
			.first_k(0).last_k(limit)
			.perform();
		
		if (LOG.global_plev() >= 2) 
			dbcsr::print(*p_bb);
					
		//dbcsr::print(*p_bb);
		
		LOG.os<1>("Occupancy of ", p_bb->name(), " : ", p_bb->occupation()*100, "%\n"); 
		
	};
	
	auto localize = [&] (dbcsr::shared_matrix<double>& p_bb, 
		dbcsr::shared_matrix<double>& c_bm, std::string x) {
		
		LOG.os<1>("Localizing occupied ", x, " MO orbitals.\n");
		
		math::pivinc_cd pcd(p_bb, LOG.global_plev());
		
		pcd.compute();
		
		int rank = pcd.rank();
		auto o = dbcsr::split_range(rank, m_mol->mo_split());
		
		auto m = c_bm->col_blk_sizes();
		auto b = c_bm->row_blk_sizes();
		
		auto L = pcd.L(b,m);
		
		// move cvir into L
		#pragma omp parallel for
		for (int im = o.size(); im != m.size(); ++im) {
			for (int ib = 0; ib != b.size(); ++ib) {
			
				bool found = false;
				auto blk_c = c_bm->get_block_p(ib,im,found);
				if (!found) continue;
				L->put_block(ib,im,blk_c);
		
			}	
		}
		
		*c_bm = std::move(*L);
				
		LOG.os<1>("Done with localization.\n");
		
	};
	
	diagonalize(m_f_bb_A, m_c_bm_A, *m_eps_A, "A");
	if (m_f_bb_B) {
		diagonalize(m_f_bb_B, m_c_bm_B, *m_eps_B, "B");
	}
	
	t_diag.finish();
	
	auto fraca = m_mol->frac_occ_alpha();
	if (fraca) {
		
		vec<double> occs(m_mol->nocc_alpha() + m_mol->nvir_alpha(),1.0);
		std::copy(fraca->begin(),fraca->end(),occs.begin());
		
		m_c_bm_A->scale(occs, "right");
		
		if (LOG.global_plev() >= 1) {
			LOG.os<1>("Fractional coeff A\n");
			dbcsr::print(*m_c_bm_A);
		}
		
	}
	
	auto fracb = m_mol->frac_occ_beta();
	if (fracb && !m_restricted) {
		vec<double> occs(m_mol->nocc_beta() + m_mol->nvir_beta(),1.0);
		std::copy(fracb->begin(),fracb->end(),occs.begin());
		m_c_bm_B->scale(occs, "right");
		
		if (LOG.global_plev() >= 1) {
			LOG.os<1>("Fractional coeff B\n");
			dbcsr::print(*m_c_bm_B);
		}
		
	}
	
	auto& t_density = TIME.sub("Form Density Matrix.");
	
	t_density.start();
	
	form_density(m_p_bb_A, m_c_bm_A, "A");
	
	if (!m_restricted && !m_nobetaorb) {
		form_density(m_p_bb_B, m_c_bm_B, "B");
	} else if (!m_restricted && m_nobetaorb) {
		auto p_bb_B = 
			dbcsr::matrix<>::create_template(*m_p_bb_A)
			.name("p_bb_B").build();
		m_p_bb_B->set(0.0);
		
		if (LOG.global_plev() >= 2) 
			dbcsr::print(*m_p_bb_B);
		
	}
	
	t_density.finish();
	
	if (m_locc) {
		
		auto& t_loc = TIME.sub("MO localization");
		t_loc.start();
		
		localize(m_p_bb_A, m_c_bm_A, "A");
		if (m_p_bb_B) localize(m_p_bb_B, m_c_bm_B, "B");
		
		t_loc.finish();
		
	}
	
	m_c_bm_A->filter(dbcsr::global::filter_eps);
	m_p_bb_A->filter(dbcsr::global::filter_eps);
	
	if (m_c_bm_B) m_c_bm_B->filter(dbcsr::global::filter_eps);
	if (m_p_bb_B) m_p_bb_B->filter(dbcsr::global::filter_eps);
	
}

void hfmod::compute_virtual_density() {
	
	auto form_density = [&] (dbcsr::shared_matrix<double>& pv_bb, 
		dbcsr::shared_matrix<double>& c_bm, std::string x) {
		
		int lobound, upbound;
		
		pv_bb = dbcsr::matrix<>::create_template(*m_p_bb_A)
			.name("pv_bb_"+x).build();
		
		//std::cout << "HERE: " << x << std::endl;
		
		if (x == "A") {
			lobound = m_mol->nocc_alpha();
			upbound = lobound + m_mol->nvir_alpha() - 1;
		}
		if (x == "B") {
			lobound = m_mol->nocc_beta();
			upbound = lobound + m_mol->nvir_beta() - 1;
		}
		
		//auto emat = dbcsr::matrix_to_eigen(*c_bm);
		//std::cout << emat << std::endl;
		
		//std::cout << "BOUNDS" << lobound << " " << upbound << std::endl;
		dbcsr::multiply('N','T',1.0,*c_bm,*c_bm,0.0,*pv_bb)
			.first_k(lobound).last_k(upbound)
			.perform();
		//std::cout << "OUT" << std::endl;
		
		if (LOG.global_plev() >= 2) 
			dbcsr::print(*pv_bb);
			
		pv_bb->filter(dbcsr::global::filter_eps);
		
		//std::cout << "DONE." << std::endl;
		
	};
	
	if (m_mol->nvir_alpha() != 0) {
		form_density(m_pv_bb_A, m_c_bm_A, "A");
	} else {
		
		m_pv_bb_A = dbcsr::matrix<>::create_template(*m_p_bb_A)
			.name("pv_bb_A").build();
		m_pv_bb_A->reserve_all();
		m_pv_bb_A->set(0.0);
		//m_pv_bb_A->filter();
	}
	
	if (!m_restricted && m_mol->nele_beta() != 0) {
		form_density(m_pv_bb_B, m_c_bm_B, "B");
	} else {
		
		m_pv_bb_B = dbcsr::matrix<>::create_template(*m_p_bb_A)
			.name("pv_bb_B").build();
		
		m_pv_bb_B->reserve_all();
		m_pv_bb_B->set(0.0);
		//m_pv_bb_B->filter();
	}
	
	//std::cout << "Done with density." << std::endl;
	
}
	

} //end namespace
