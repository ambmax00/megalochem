#include "locorb/locorb.h"

namespace locorb {

std::pair<smat_d,smat_d> mo_localizer::compute_pao(smat_d c_bm, smat_d s_bb) {
	
	auto l_bb = dbcsr::create_template<double>(s_bb)
		.name("u_bb")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto u_bm = dbcsr::create_template<double>(c_bm)
		.name("u_bm")
		.get();
		
	dbcsr::multiply('N', 'N', *s_bb, *c_bm, *u_bm).perform();
	dbcsr::multiply('N', 'T', *c_bm, *u_bm, *l_bb).perform();
	
	return std::make_pair<smat_d,smat_d>(
		std::move(l_bb), std::move(u_bm)
	);
	
} 

} // end namespace
