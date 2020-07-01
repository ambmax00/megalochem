#include "hf/hfmod.h"
#include <dbcsr_conversions.hpp>
#include <dbcsr_matrix_ops.hpp>
#include "math/solvers/hermitian_eigen_solver.h"
#include <algorithm> 

namespace hf { 
	
void hfmod::diag_fock() {
	
	//updates coeffcient matrices (c_bo_A, c_bo_B) and densities (p_bo_A, p_bo_B)
	
	auto& t_diag = TIME.sub("Fock Diagonalization");
	
	t_diag.start();
	
	auto diagonalize = [&](smat_d& f_bb, smat_d& c_bm, std::vector<double>& eps, std::string x) {
		
		LOG.os<2>("Orthogonalizing Fock Matrix: ", x, '\n');
		
		mat_d FX = mat_d::create_template(*f_bb).name("FX").type(dbcsr_type_no_symmetry);
		mat_d XFX = mat_d::create_template(*f_bb).name("XFX");
		
		dbcsr::multiply('N','N',*f_bb,*m_x_bb,FX).perform();
		
		//dbcsr::print(*f_bb);
		
		//dbcsr::print(FX);
		
		dbcsr::multiply('T','N',*m_x_bb,FX,XFX).perform(); 
		
		FX.release();
		
		if (LOG.global_plev() >= 1) 
			dbcsr::print(XFX);
		
		auto XFXs = XFX.get_smatrix();
		
		math::hermitian_eigen_solver solver(XFXs, 'V', (LOG.global_plev() >= 2) ? true : false);
		
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
		
		dbcsr::mat_d new_c_bm = dbcsr::mat_d::create_template(*c_bm_x).name(c_bm_x->name());
		*c_bm = std::move(new_c_bm); 
	
		//Transform back
		dbcsr::multiply('N','N',*m_x_bb,*c_bm_x,*c_bm).perform();
		//dbcsr::einsum<2,2,2>({"IJ, Ji -> Ii", *m_x_bb, c_bm_x, c_bm, .unit_nr = u, .log = log});
		
		if (LOG.global_plev() >= 1) 
			dbcsr::print(*c_bm);
		
		XFXs->release();
		c_bm_x->release();
			
	};
	
	auto form_density = [&] (smat_d& p_bb, smat_d& c_bm, std::string x) {
		
		int limit = 0;
		
		if (x == "A") limit = m_mol->nocc_alpha() - 1;
		if (x == "B") limit = m_mol->nocc_beta() - 1;
		
		
		dbcsr::multiply('N', 'T', *c_bm, *c_bm, *p_bb).first_k(0).last_k(limit).perform();
		
		if (LOG.global_plev() >= 1) 
			dbcsr::print(*p_bb);
		
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
		mat_d p_bb_B = mat_d::create_template(*m_p_bb_A).name("p_bb_B");
		m_p_bb_B = p_bb_B.get_smatrix();
		m_p_bb_B->set(0.0);
		
		if (LOG.global_plev() >= 1) 
			dbcsr::print(*m_p_bb_B);
		
	}
	
	t_density.finish();
	
}

void hfmod::compute_virtual_density() {
	
	auto form_density = [&] (smat_d& pv_bb, smat_d& c_bm, std::string x) {
		
		int lobound, upbound;
		
		mat_d p = mat_d::create_template(*m_p_bb_A).name("pv_bb_"+x);
		pv_bb = p.get_smatrix();
		
		std::cout << "HERE: " << x << std::endl;
		
		if (x == "A") {
			lobound = m_mol->nocc_alpha();
			upbound = lobound + m_mol->nvir_alpha() - 1;
		}
		if (x == "B") {
			lobound = m_mol->nocc_beta();
			upbound = lobound + m_mol->nvir_beta() - 1;
		}
		
		//std::cout << "GOOD" << lobound << " " << upbound << std::endl;
		dbcsr::multiply('N','T',*c_bm,*c_bm,*pv_bb).first_k(lobound).last_k(upbound).perform();
		//std::cout << "OUT" << std::endl;
		
		if (LOG.global_plev() >= 2) 
			dbcsr::print(*pv_bb);
		
		std::cout << "DONE." << std::endl;
		
	};
	
	if (m_mol->nvir_alpha() != 0) {
		form_density(m_pv_bb_A, m_c_bm_A, "A");
	} else {
		
		mat_d p = mat_d::create_template(*m_p_bb_A).name("pv_bb_A");
		m_pv_bb_A = p.get_smatrix();
		m_pv_bb_A->reserve_all();
		m_pv_bb_A->set(0.0);
		//m_pv_bb_A->filter();
	}
	
	if (!m_restricted && m_mol->nele_beta() != 0) {
		form_density(m_pv_bb_B, m_c_bm_B, "B");
	} else {
		mat_d p = mat_d::create_template(*m_p_bb_A).name("pv_bb_B");
		m_pv_bb_B = p.get_smatrix();
		m_pv_bb_B->reserve_all();
		m_pv_bb_B->set(0.0);
		//m_pv_bb_B->filter();
	}
	
	//std::cout << "Done with density." << std::endl;
	
}
	

} //end namespace
