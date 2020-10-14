#ifndef LOCORB_LOCORB_H
#define LOCORB_LOCORB_H

#include <dbcsr_matrix_ops.hpp>

#include "utils/mpi_log.h"
#include "desc/molecule.h"
#include "ints/aofactory.h"

namespace locorb {
	
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
	
	dbcsr::shared_matrix<double> 
		compute_cholesky(dbcsr::shared_matrix<double> c_bm);
		
	dbcsr::shared_matrix<double>
		compute_boys(dbcsr::shared_matrix<double> c_bm);
	
};
	
} // end namespace

#endif
