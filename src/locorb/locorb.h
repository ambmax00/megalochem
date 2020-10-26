#ifndef LOCORB_LOCORB_H
#define LOCORB_LOCORB_H

#include <dbcsr_matrix_ops.hpp>

#include "utils/mpi_log.h"
#include "desc/molecule.h"
#include "ints/aofactory.h"
#include <utility>

namespace locorb {
	
using smat_d = dbcsr::shared_matrix<double>;
	
class mo_localizer {
private:	

	dbcsr::world m_world;
	desc::smolecule m_mol;
	
	std::shared_ptr<ints::aofactory> m_aofac;
	
	util::mpi_log LOG;
	
public:

	mo_localizer(dbcsr::world w, desc::smolecule mol) :
		m_world(w),
		m_mol(mol),
		LOG(w.comm(), 0),
		m_aofac(std::make_shared<ints::aofactory>(mol,w))
	{}
	
	std::pair<smat_d,smat_d> compute_cholesky(smat_d c_bm, smat_d s_bb);
#if 0	
	std::pair<smat_d,smat_d> compute_boys(smat_d c_bm, smat_d s_bb);
#endif	
	std::pair<smat_d,smat_d> compute_pao(smat_d c_bm, smat_d s_bb); 
		
	dbcsr::shared_matrix<double>
		compute_conversion(dbcsr::shared_matrix<double> c_bm,
		dbcsr::shared_matrix<double> s_bb, dbcsr::shared_matrix<double> l_bm)
	{
		
		auto temp = dbcsr::create_template<double>(c_bm)
			.name("temp").get();
		
		auto w = c_bm->get_world();
		auto m = c_bm->col_blk_sizes();
		
		auto u_mm = dbcsr::create<double>()
			.name("u_mm")
			.set_world(w)
			.row_blk_sizes(m)
			.col_blk_sizes(m)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
		
		dbcsr::multiply('N', 'N', *s_bb, *c_bm, *temp).perform();
		dbcsr::multiply('T', 'N', *l_bm, *temp, *u_mm).perform();
		
		return u_mm;
		
	}	
	
};
	
} // end namespace

#endif
