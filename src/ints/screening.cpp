#include "ints/screening.h"
#include <dbcsr_matrix_ops.hpp>

namespace ints {

void schwarz_screener::compute() {
	
	auto z_mn_dist = m_fac.ao_schwarz();
	auto z_x_dist = m_fac.ao_3cschwarz();
	
	m_blk_norms_mn = dbcsr::block_norms(*z_mn_dist);
	m_blk_norms_x = dbcsr::block_norms(*z_x_dist);
	
	m_z_mn = dbcsr::matrix_to_eigen(z_mn_dist);
	m_z_x = dbcsr::matrix_to_eigen(z_x_dist);
	
	z_mn_dist->release();
	z_x_dist->release();
	
}

bool schwarz_screener::skip_block_xbb(int i, int j, int k) {
	
	float f = m_blk_norms_mn(j,k) * m_blk_norms_x(i,0);
	
	if (f > m_blk_threshold) return false;
	return true;
	
}

bool schwarz_screener::skip_block_bbbb(int i, int j, int k, int l) {
	
	float f = m_blk_norms_mn(i,j) * m_blk_norms_mn(k,l);
	
	if (f > m_blk_threshold) return false;
	return true;
	
}

bool schwarz_screener::skip_xbb(int i, int j, int k) {
	
	if (m_z_mn(j,k)*m_z_x(i,0) > m_int_threshold) {
		return false;
	}
			
	return true;
	
}

bool schwarz_screener::skip_bbbb(int i, int j, int k, int l) {
	
	if (m_z_mn(i,j)*m_z_mn(k,l) > m_int_threshold) {
		return false;
	}
			
	return true;
	
}

} // end namespace
