#include "adc/adc_mvp.hpp"

namespace adc {
	
/* transforms u */
smat u_transform(smat& u, char to, smat& c_bo, char tv, smat& c_bv) {
	
	auto w = u->get_world();
	
	// create new matrices
	auto rblks_in = u->row_blk_sizes();
	auto cblks_in = u->col_blk_sizes();
	
	auto rblks_out = (to == 'N') ? c_bo->row_blk_sizes() : c_bo->col_blk_sizes();
	auto cblks_out = (tv == 'T') ? c_bv->row_blk_sizes() : c_bv->col_blk_sizes();
	
	smat u_t1 = dbcsr::matrix<>::create()
		.set_world(w)
		.name("u_t1")
		.row_blk_sizes(rblks_out)
		.col_blk_sizes(cblks_in)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	smat u_t2 = dbcsr::matrix<>::create()
		.set_world(w)
		.name("u_transformed")
		.row_blk_sizes(rblks_out)
		.col_blk_sizes(cblks_out)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	dbcsr::multiply(to, 'N', *c_bo, *u, *u_t1).perform();
	dbcsr::multiply('N', tv, *u_t1, *c_bv, *u_t2).perform();
	
	return u_t2;
	
}
	
MVP::MVP(dbcsr::world w, desc::shared_molecule smol, int nprint, std::string name) :
	m_world(w), m_mol(smol), 
	LOG(w.comm(), nprint),
	TIME(w.comm(), name)
{}
	
smat MVP::compute_sigma_0(smat& u_ia, std::vector<double> epso, 
	std::vector<double> epsv) {
	
	// ADC0 : u_ia = - f_ij u_ja + f_ab u_ib
	
	smat s_ia_o = dbcsr::matrix<>::copy(*u_ia).build();
	smat s_ia_v = dbcsr::matrix<>::copy(*u_ia).build();
	
	//dbcsr::print(*s_ia_o);
	
	s_ia_o->scale(epso, "left");
	s_ia_v->scale(epsv, "right");
	
	//std::cout << "SIAO + V" << std::endl;
	//dbcsr::print(*s_ia_o);
	//dbcsr::print(*s_ia_v);
		
	s_ia_o->add(-1.0,1.0,*s_ia_v);
		
	return s_ia_o;
	
}

} // end namespace
