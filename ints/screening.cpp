#include "ints/screening.h"
#include <dbcsr_matrix_ops.hpp>

namespace ints {

bool screener::skip_block(int i, int j, int k) {
	return false;
}

bool screener::skip(int i, int j, int k) {
	return false;
}

void schwarz_screener::compute() {
	
	auto z_mn_dist = p_fac->ao_schwarz();
	auto z_x_dist = p_fac->ao_3cschwarz();
	
	//dbcsr::print(*z_mn_dist);
	//dbcsr::print(*z_x_dist); 
	
	m_blk_norms_mn = dbcsr::block_norms(*z_mn_dist);
	m_blk_norms_x = dbcsr::block_norms(*z_x_dist);
	
	//std::cout << m_blk_norms_mn << std::endl;
	//std::cout << m_blk_norms_x << std::endl;
	
	m_z_mn = dbcsr::matrix_to_eigen(*z_mn_dist);
	m_z_x = dbcsr::matrix_to_eigen(*z_x_dist);
	
	std::cout << "SCREENER COMPUTED." << std::endl;
	
	std::cout << m_z_mn << std::endl;
	std::cout << m_z_x << std::endl;
	
	z_mn_dist->release();
	z_x_dist->release();
	
}

bool schwarz_screener::skip_block(int i, int j, int k) {
	
	float f = m_blk_norms_mn(j,k) * m_blk_norms_x(i,0);
	
	if (f > m_blk_threshold) return false;
	return true;
	
}

bool schwarz_screener::skip(int i, int j, int k) {
	
	//std::cout << "MN X " << i << " " << j << " " << k << std::endl;
	//std::cout << m_z_mn(j,k) << " " << m_z_x(i,0) << " " << m_z_mn(j,k)*m_z_x(i,0) << std::endl;
	if (m_z_mn(j,k)*m_z_x(i,0) > m_int_threshold) {
		//std::cout << " ===== WILL NOT BE SCREENED ===== " << std::endl;
		return false;
	}
		
	//std::cout << " ===== WILL BE SCREENED ===== " << std::endl;
	
	return true;
	
}

} // end namespace
