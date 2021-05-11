#ifndef LOCORB_LOCORB_H
#define LOCORB_LOCORB_H

#include <dbcsr_matrix_ops.hpp>
#include "megalochem.hpp"
#include "utils/mpi_log.hpp"
#include "desc/molecule.hpp"
#include "ints/aofactory.hpp"
#include <utility>

namespace megalochem {

namespace locorb {
	
using smat_d = dbcsr::shared_matrix<double>;
	
class mo_localizer {
private:	

	world m_world;
	desc::shared_molecule m_mol;
	
	std::shared_ptr<ints::aofactory> m_aofac;
	
	util::mpi_log LOG;
	
public:

	mo_localizer(world w, desc::shared_molecule mol) :
		m_world(w),
		m_mol(mol),
		LOG(w.comm(), 0),
		m_aofac(std::make_shared<ints::aofactory>(mol,w))
	{}
	
	std::tuple<smat_d,smat_d> compute_cholesky(smat_d c_bm, smat_d s_bb);
	
	std::tuple<smat_d,smat_d> compute_boys(smat_d c_bm, smat_d s_bb);
	
	std::tuple<smat_d,smat_d> compute_pao(smat_d c_bm, smat_d s_bb); 
	
	std::tuple<smat_d, smat_d, std::vector<double>> compute_truncated_pao(
		smat_d c_bm, smat_d s_bb, std::vector<double> eps_m,
		std::vector<int> blkidx,
		smat_d u_km = nullptr);
		
	dbcsr::shared_matrix<double>
		compute_conversion(dbcsr::shared_matrix<double> c_bm,
		dbcsr::shared_matrix<double> s_bb, dbcsr::shared_matrix<double> l_bm)
	{
		
		auto temp = dbcsr::matrix<>::create_template(*c_bm)
			.name("temp").build();
		
		auto w = c_bm->get_cart();
		auto mcanon = c_bm->col_blk_sizes();
		auto mlocc = l_bm->col_blk_sizes();
		
		auto u_mm = dbcsr::matrix<>::create()
			.name("u_mm")
			.set_cart(w)
			.row_blk_sizes(mlocc)
			.col_blk_sizes(mcanon)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();
		
		dbcsr::multiply('N', 'N', 1.0, *s_bb, *c_bm, 0.0, *temp).perform();
		dbcsr::multiply('T', 'N', 1.0, *l_bm, *temp, 0.0, *u_mm).perform();
		
		return u_mm;
		
	}	
	
	//double pop_mulliken(dbcsr::matrix<double>& c_bo, dbcsr::matrix<double>& s_bb,
	//	dbcsr::matrix<double>& s_sqrt_bb);
	
};
	
} // end namespace

} // end namespace

#endif
