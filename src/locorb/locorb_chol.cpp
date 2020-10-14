#include "locorb/locorb.h"
#include "math/linalg/piv_cd.h"

namespace locorb {

dbcsr::shared_matrix<double> 
		mo_localizer::compute_cholesky(dbcsr::shared_matrix<double> c_bm) {
			
	auto b = c_bm->row_blk_sizes();
	auto m = c_bm->col_blk_sizes();
	
	int norb = c_bm->nfullcols_total();
	
	auto p_bb = dbcsr::create<double>()
		.name("p_bb")
		.set_world(m_world)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::symmetric)
		.get();
		
	dbcsr::multiply('N', 'T', *c_bm, *c_bm, *p_bb).perform();
	
	math::pivinc_cd piv(p_bb, LOG.global_plev());
	
	piv.compute();
	
	if (piv.rank() != norb) {
		throw std::runtime_error("Locorb cholesky has failed.");
	}
	
	auto L_bm = piv.L(b,m);
	return L_bm;
			
}

} // end namespace
