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

void atomic_screener::compute() {
	
	m_schwarz.compute();
	
	auto cbas = m_mol->c_basis();
	auto xbas = m_mol->c_dfbasis();
	auto atoms = m_mol->atoms();
	
	auto get_check = [&](auto bas) {
		
		auto blktoatom = bas->block_to_atom(atoms); 
		int nblk = blktoatom.size();
		
		std::vector<bool> check(nblk,false);
		
		for (int iblk = 0; iblk != nblk; ++iblk) {
		
			int iatom = blktoatom[iblk];
		
			auto iter = std::find(m_atom_list.begin(), m_atom_list.end(), iatom);
			auto iter_end = m_atom_list.end();
		
			if (iter != iter_end) check[iblk] = true;
			
		}
		
		return check;
		
	};
	
	m_blklist_b = get_check(cbas);
	m_blklist_x = get_check(xbas);
	
	for (int ix = 0; ix != m_blklist_x.size(); ++ix) m_blklist_x[ix] = true;
	 
}

bool atomic_screener::skip_block_xbb(int i, int j, int k) {
	
	bool on_atom = (m_blklist_x[i] && m_blklist_b[j] && m_blklist_b[k]);
	
	if (on_atom && !m_schwarz.skip_block_xbb(i,j,k)) return false;
	
	return true;
	
}

bool atomic_screener::skip_block_bbbb(int i, int j, int k, int l) {
	
	bool on_atom = (m_blklist_b[i] && m_blklist_b[j] && m_blklist_b[k]
		&& m_blklist_b[l]);
	
	if (on_atom && !m_schwarz.skip_block_bbbb(i,j,k,l)) return false;
	
	return true;
	
}

bool atomic_screener::skip_xbb(int i, int j, int k) {
		
	return false;
	
}

bool atomic_screener::skip_bbbb(int i, int j, int k, int l) {
			
	return false;
	
}

} // end namespace
