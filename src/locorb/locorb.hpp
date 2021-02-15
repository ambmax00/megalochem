#ifndef LOCORB_LOCORB_H
#define LOCORB_LOCORB_H

#include <dbcsr_matrix_ops.hpp>

#include "utils/mpi_log.hpp"
#include "desc/molecule.hpp"
#include "ints/aofactory.hpp"
#include <utility>

namespace locorb {
	
using smat_d = dbcsr::shared_matrix<double>;
	
class mo_localizer {
private:	

	dbcsr::world m_world;
	desc::shared_molecule m_mol;
	
	std::shared_ptr<ints::aofactory> m_aofac;
	
	util::mpi_log LOG;
	
public:

	mo_localizer(dbcsr::world w, desc::shared_molecule mol) :
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
	
	std::tuple<smat_d, smat_d, std::vector<double>> compute_truncated_pao(
		smat_d c_bm, smat_d s_bb, std::vector<double> eps_m,
		std::vector<int> blkidx,
		smat_d u_km = nullptr);
		
	dbcsr::shared_matrix<double>
		compute_conversion(dbcsr::shared_matrix<double> c_bm,
		dbcsr::shared_matrix<double> s_bb, dbcsr::shared_matrix<double> l_bm)
	{
		
		auto temp = dbcsr::create_template<double>(*c_bm)
			.name("temp").get();
		
		auto w = c_bm->get_world();
		auto m = c_bm->col_blk_sizes();
		
		auto u_mm = dbcsr::matrix<>::create()
			.name("u_mm")
			.set_world(w)
			.row_blk_sizes(m)
			.col_blk_sizes(m)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();
		
		dbcsr::multiply('N', 'N', *s_bb, *c_bm, *temp).perform();
		dbcsr::multiply('T', 'N', *l_bm, *temp, *u_mm).perform();
		
		return u_mm;
		
	}	
	
	//double pop_mulliken(dbcsr::matrix<double>& c_bo, dbcsr::matrix<double>& s_bb,
	//	dbcsr::matrix<double>& s_sqrt_bb);
	
};
	
} // end namespace

#endif
