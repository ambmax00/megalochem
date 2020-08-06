#ifndef FOCKMOD_H
#define FOCKMOD_H

#include "desc/molecule.h"
#include "desc/options.h"
#include "utils/mpi_time.h"
#include "fock/jkbuilder.h"

#include <dbcsr_matrix.hpp>

namespace fock {

class fockmod {
private:

	desc::smolecule m_mol;
	desc::options m_opt;
	dbcsr::world m_world;
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	dbcsr::shared_matrix<double> m_c_A;
	dbcsr::shared_matrix<double> m_c_B;
	dbcsr::shared_matrix<double> m_p_A;
	dbcsr::shared_matrix<double> m_p_B;
	dbcsr::shared_matrix<double> m_core;
	
	dbcsr::shared_matrix<double> m_f_bb_A;
	dbcsr::shared_matrix<double> m_f_bb_B;
	
	std::shared_ptr<J> m_J_builder;
	std::shared_ptr<K> m_K_builder;
	
	dbcsr::shared_pgrid<2> spgrid2;
	dbcsr::shared_pgrid<3> spgrid3_xbb;
	dbcsr::shared_pgrid<4> spgrid4;
	
	util::registry m_reg;

public:

	fockmod (dbcsr::world iworld, desc::smolecule imol, desc::options iopt);
	~fockmod() {}
	
	fockmod(fockmod& f) = delete;
	
	fockmod& set_density_alpha(dbcsr::shared_matrix<double>& p_A) { m_p_A = p_A; return *this; }
	fockmod& set_density_beta(dbcsr::shared_matrix<double>& p_B) { m_p_B = p_B; return *this; }
	fockmod& set_coeff_alpha(dbcsr::shared_matrix<double>& c_A) { m_c_A = c_A; return *this; }
	fockmod& set_coeff_beta(dbcsr::shared_matrix<double>& c_B) { m_c_B = c_B; return *this; }
	fockmod& set_core(dbcsr::shared_matrix<double>& core) { m_core = core; return *this; }
	
	void init();
	
	void compute(bool SAD_iter = false, int rank = 0);
	
	dbcsr::shared_matrix<double> get_f_A() { return m_f_bb_A; }
	dbcsr::shared_matrix<double> get_f_B() { return m_f_bb_B; }
	
	void print_info() { 
		TIME.print_info();
		if (m_J_builder) m_J_builder->print_info();
		if (m_K_builder) m_K_builder->print_info();
	}
	
};
	
} // end namespace

#endif
