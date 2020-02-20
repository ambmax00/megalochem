#include "hf/hfmod.h"
#include "math/tensor/dbcsr_conversions.hpp"
#include "math/linalg/symmetrize.h"
#include "math/other/scale.h"

#include <algorithm> 

#include <Eigen/Eigenvalues>

namespace hf { 
	
void hfmod::diag_fock() {
	
	//updates coeffcient matrices (c_bo_A, c_bo_B) and densities (p_bo_A, p_bo_B)
	
	auto& t_diag = TIME.sub("Fock Diagonalization");
	
	t_diag.start();
	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	
	dbcsr::pgrid<2> grid({.comm = m_comm});
	
	bool log = false;
	int u = 0;
		
	if (LOG.global_plev() >= 2) {
		log = true;
		u = 6;
	}
	
	auto diagonalize = [&](dbcsr::tensor<2>& f_bb, dbcsr::tensor<2>& c_bm, std::vector<double>& eps, std::string x) {
		
		vec<int> blk_sizes;
		int nele;
		
		LOG.os<2>("Orthogonalizing Fock Matrix.");
		
		dbcsr::tensor<2> FX({.name = "FX", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = m_s_bb->blk_size()});
		dbcsr::tensor<2> XFX({.name = "XFX", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = m_s_bb->blk_size()});
		
		dbcsr::einsum<2,2,2>({"IJ, JK -> IK", f_bb, *m_x_bb, FX, .unit_nr = u, .log = log});
		
		dbcsr::einsum<2,2,2>({"JI, JK -> IK", *m_x_bb, FX, XFX, .unit_nr = u, .log = log});
		
		FX.destroy();
		
		if (LOG.global_plev() >= 1) 
			dbcsr::print(XFX);
		
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
		
		auto eigen_f_bb_x = dbcsr::tensor_to_eigen(XFX);
		
		es.compute(eigen_f_bb_x);
		
		auto eigval = es.eigenvalues();
		auto eigvec = es.eigenvectors();
		
		LOG.os<2>("Eigenvalues: \n");
		LOG.os<2>(eigval);
		
		eps.resize(eigval.size());
		std::copy(eigval.data(), eigval.data() + eigval.size(), eps.begin());
		
		LOG.os<2>("Eigenvectors: \n");
		LOG.os<2>(eigvec);
		
		Eigen::MatrixXd eigen_c_bm_x = es.eigenvectors();
	
		auto c_bm_x = dbcsr::eigen_to_tensor
			(eigen_c_bm_x, "c_bm_x_"+x, grid, {0}, {1}, c_bm.blk_size());
		
		c_bm_x.filter();
			
		if (LOG.global_plev() >= 2) 
			dbcsr::print(c_bm_x);
	
		//Transform back
		dbcsr::einsum<2,2,2>({"IJ, Ji -> Ii", *m_x_bb, c_bm_x, c_bm, .unit_nr = u, .log = log});
		
		//debug: eigen
		//auto xe = dbcsr::tensor_to_eigen(*m_x_bb);
		
		//auto c = xe * eigen_c_bm_x;
		
		//std::cout << "EIGEN: " << std::endl;
		//std::cout << c << std::endl;
		
		if (LOG.global_plev() >= 1) 
			dbcsr::print(c_bm);
		
		XFX.destroy();
		c_bm_x.destroy();
			
	};
	
	auto form_density = [&] (dbcsr::tensor<2>& p_bb, dbcsr::tensor<2>& c_bm, std::string x) {
		
		int limit = 0;
		
		if (x == "A") limit = m_mol->nocc_alpha() - 1;
		if (x == "B") limit = m_mol->nocc_beta() - 1;
		
		//std::cout << "LIMIT " << limit << std::endl;
		
		// make bounds
		std::vector<std::vector<int>> occ_bounds = {{0,limit}};
		
		dbcsr::einsum<2,2,2>({.x = "Mi, Ni -> MN", .t1 = c_bm, .t2 = c_bm, .t3 = p_bb, .b1 = occ_bounds});
		
		p_bb.filter();
		
		if (LOG.global_plev() >= 1) 
			dbcsr::print(p_bb);
		
	};
	
	diagonalize(*m_f_bb_A, *m_c_bm_A, *m_eps_A, "A");
	if (m_f_bb_B) {
		diagonalize(*m_f_bb_B, *m_c_bm_B, *m_eps_B, "B");
	}
	
	t_diag.finish();
	
	auto fraca = m_mol->frac_occ_alpha();
	if (fraca) {
		//std::cout << "Scaling!" << std::endl;
		std::vector<int> b = {0,fraca->size() - 1};
		math::scale({.t_in = *m_c_bm_A, .v_in = *fraca, .bounds = b});
	}
	
	auto fracb = m_mol->frac_occ_beta();
	if (fracb && m_c_bm_B) {
		std::vector<int> b = {0,fracb->size() - 1};
		math::scale({.t_in = *m_c_bm_B, .v_in = *fracb, .bounds = b});
	}
	
	auto& t_density = TIME.sub("Form Density Matrix.");
	
	t_density.start();
	
	form_density(*m_p_bb_A, *m_c_bm_A, "A");
	if (m_p_bb_B && !m_nobeta) {
		form_density(*m_p_bb_B, *m_c_bm_B, "B");
	} else if (!m_restricted && m_nobeta) {
		m_p_bb_B = dbcsr::make_stensor<2>({.name = "p_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = m_p_bb_A->blk_size()});
		m_p_bb_B->set(0.0);
		
		if (LOG.global_plev() >= 1) 
			dbcsr::print(*m_p_bb_B);
		
	}
	
	t_density.finish();

	grid.destroy();
	
}

} //end namespace
