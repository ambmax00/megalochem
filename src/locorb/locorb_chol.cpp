#include "locorb/locorb.hpp"
#include "math/linalg/piv_cd.hpp"

namespace locorb {

std::pair<smat_d,smat_d>  
		mo_localizer::compute_cholesky(smat_d c_bm, smat_d s_bb) {
			
	auto b = c_bm->row_blk_sizes();
	auto m = c_bm->col_blk_sizes();
	
	int norb = c_bm->nfullcols_total();
	
	auto p_bb = dbcsr::matrix<>::create()
		.name("p_bb")
		.set_world(m_world)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::symmetric)
		.build();
		
	dbcsr::multiply('N', 'T', 1.0, *c_bm, *c_bm, 0.0, *p_bb).perform();
	
	math::pivinc_cd piv(p_bb, 0);
	
	piv.compute();
	
	if (piv.rank() != norb) {
		throw std::runtime_error("Locorb cholesky has failed.");
	}
	
	auto L_bm = piv.L(b,m);
	
	auto u_mm = this->compute_conversion(c_bm, s_bb, L_bm);
	
	return std::make_pair<smat_d,smat_d>(
		std::move(L_bm), std::move(u_mm)
	);
			
}

} // end namespace
