#include "adc/adc_mvp.h"

namespace adc {
	
MVP::MVP(dbcsr::world& w, desc::smolecule smol, 
	desc::options opt, util::registry& reg,
	svector<double> epso, svector<double> epsv) :
	m_world(w), m_mol(smol), m_opt(opt), 
	LOG(w.comm(), m_opt.get<int>("print", ADC_PRINT_LEVEL)),
	TIME(w.comm(), "MVP"),
	m_reg(reg), m_epso(epso), m_epsv(epsv) 
{
	m_s_bb = m_reg.get_matrix<double>("s_bb");
	m_c_bo = m_reg.get_matrix<double>("c_bo");
	m_c_bv = m_reg.get_matrix<double>("c_bv");
	
	m_sc_bo = dbcsr::create_template<double>(m_c_bo)
		.name("sc_bo").get();
		
	m_sc_bv = dbcsr::create_template<double>(m_c_bv)
		.name("sc_bv").get();
		
	dbcsr::multiply('N', 'N', *m_s_bb, *m_c_bo, *m_sc_bo).perform();
	dbcsr::multiply('N', 'N', *m_s_bb, *m_c_bv, *m_sc_bv).perform();
	
}
	
smat MVP::compute_sigma_0(smat& u_ia) {
	
	// ADC0 : u_ia = - f_ij u_ja + f_ab u_ib
	
	smat s_ia_o = dbcsr::copy(u_ia).get();
	smat s_ia_v = dbcsr::copy(u_ia).get();
	
	//dbcsr::print(*s_ia_o);
	
	s_ia_o->scale(*m_epso, "left");
	s_ia_v->scale(*m_epsv, "right");
	
	//std::cout << "SIAO + V" << std::endl;
	//dbcsr::print(*s_ia_o);
	//dbcsr::print(*s_ia_v);
		
	s_ia_o->add(-1.0,1.0,*s_ia_v);
		
	return s_ia_o;
	
}

/* transforms u */
smat MVP::u_transform(smat& u, char to, smat& c_bo, char tv, smat& c_bv) {
	
	auto w = u->get_world();
	
	// create new matrices
	auto rblks_in = u->row_blk_sizes();
	auto cblks_in = u->col_blk_sizes();
	
	auto rblks_out = (to == 'N') ? c_bo->row_blk_sizes() : c_bo->col_blk_sizes();
	auto cblks_out = (tv == 'T') ? c_bv->row_blk_sizes() : c_bv->col_blk_sizes();
	
	smat u_t1 = dbcsr::create<double>()
		.set_world(w)
		.name("u_t1")
		.row_blk_sizes(rblks_out)
		.col_blk_sizes(cblks_in)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	smat u_t2 = dbcsr::create<double>()
		.set_world(w)
		.name("u_transformed")
		.row_blk_sizes(rblks_out)
		.col_blk_sizes(cblks_out)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	dbcsr::multiply(to, 'N', *c_bo, *u, *u_t1).perform();
	dbcsr::multiply('N', tv, *u_t1, *c_bv, *u_t2).perform();
	
	return u_t2;
	
}
	

} // end namespace
