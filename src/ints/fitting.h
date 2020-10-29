#ifndef INTS_FITTING_H
#define INTS_FITTING_H

#include "utils/registry.h"
#include "desc/molecule.h"
#include "desc/options.h"
#include "ints/screening.h"
#include "ints/aofactory.h"
#include <dbcsr_btensor.hpp>
#include <dbcsr_tensor.hpp>
#include <dbcsr_matrix.hpp>
#include "utils/mpi_time.h"

namespace ints {

class dfitting {
private:

	dbcsr::world m_world;
	desc::smolecule m_mol;
	util::mpi_log LOG;
	util::mpi_time TIME;

public:

	dfitting(dbcsr::world w, desc::smolecule smol, int print = 0) :
		m_world(w),
		m_mol(smol),
		LOG(w.comm(), print),
		TIME(w.comm(), "Fitting Coefficients")
	{}
	
	dbcsr::sbtensor<3,double> compute(dbcsr::sbtensor<3,double> eris, 
		dbcsr::shared_matrix<double> s_inv, std::string cfit_btype);

	dbcsr::sbtensor<3,double> compute(dbcsr::sbtensor<3,double> eris, 
		dbcsr::shared_tensor<2,double> s_inv, std::string cfit_btype);
		
	dbcsr::shared_tensor<3,double> compute_pari(dbcsr::sbtensor<3,double> eris,
		dbcsr::shared_matrix<double> s_xx, shared_screener scr_s);
		
	dbcsr::sbtensor<3,double> compute_qr(dbcsr::shared_matrix<double> s_xx_inv, 
		dbcsr::shared_matrix<double> m_xx, 
		dbcsr::shared_pgrid<3> spgrid3_xbb,
		shared_screener scr_s, 
		std::array<int,3> bdims,
		dbcsr::btype mytype);
		
	//void compute_qr(dbcsr::sbtensor<3,double> eris, dbcsr::shared_matrix<double> s_xx);
	
	void print_info() { TIME.print_info(); }

};
	
} // end namespace

#endif
