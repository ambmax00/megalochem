#include "hf/hfmod.h"

namespace hf { 
	
void hfmod::diag_fock() {
	
	//updates coeffcient matrices (c_bo_A, c_bo_B) and densities (p_bo_A, p_bo_B)
	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	
	auto eigen_f_bb_A = dbcsr::tensor_to_eigen(m_f_bb_A);
	es.compute(eigen_f_bb_A);
	
	Eigen::MatrixXd eigen_c_bo_A = es.eigenvectors().leftCols(m_mol.nocc_alpha);
	
	dbcsr::pgrid<2> grid({.comm = m_comm});
	
	m_c_bo_A = std::make_shared<dbcsr::tensor<2,double>(
		dbcsr::eigen_to_tensor(result, "c_bo_A", grid, {0}, {1}, m_c_bo_A->blk_size())
	); 
	
	if (m_restricted || m_mol.nocc_beta == 0) {
	
		auto eigen_f_bb_B = dbcsr::tensor_to_eigen(m_f_bb_B);
		es.compute(eigen_f_bb_B);
	
		Eigen::MatrixXd eigen_c_bo_B = es.eigenvectors().leftCols(m_mol.nocc_beta);
	
		m_c_bo_B = std::make_shared<dbcsr::tensor<2,double>(
			dbcsr::eigen_to_tensor(result, "c_bo_B", grid, {0}, {1}, m_s_bb->blk_size())
		);
		
	}
		
		
	
}
