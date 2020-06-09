#ifndef FOCK_JK_BUILDER_H
#define FOCK_JK_BUILDER_H

#include "ints/aofactory.h"
#include "desc/molecule.h"
#include "desc/options.h"
#include "utils/mpi_time.h"
#include <dbcsr_matrix.hpp>

namespace fock {

class JK_common {
protected:
	
	desc::options m_opt;
	dbcsr::world m_world;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	dbcsr::smat_d m_p_A;
	dbcsr::smat_d m_p_B;
	dbcsr::smat_d m_c_A;
	dbcsr::smat_d m_c_B;
	
	std::shared_ptr<ints::aofactory> m_factory;
	
public:
	
	JK_common(dbcsr::world& w, desc::options opt);
	void set_density_alpha(dbcsr::smat_d& ipA) { m_p_A = ipA; }
	void set_density_beta(dbcsr::smat_d& ipB) { m_p_B = ipB; }
	void set_coeff_alpha(dbcsr::smat_d& icA) { m_c_A = icA; }
	void set_coeff_beta(dbcsr::smat_d& icB) { m_c_B = icB; }
	void set_factory(std::shared_ptr<ints::aofactory>& ifac) { m_factory = ifac; }
	
	~JK_common() {}
	
};

class J : public JK_common {
protected: 

	dbcsr::stensor2_d m_J;
	std::shared_ptr<J> m_builder;
	
public:
	
	J(dbcsr::world& w, desc::options& opt) : JK_common(w,opt) {}
	~J() {}
	virtual void compute_J() = 0;
	
	void init();
	
	dbcsr::stensor2_d get_J() { return m_J; }	
		
};

class K : public JK_common {
protected:

	dbcsr::stensor2_d m_K_A;
	dbcsr::stensor2_d m_K_B;
	
public:
	
	K(dbcsr::world& w, desc::options& opt) : JK_common(w,opt) {}
	~K() {}
	virtual void compute_K() = 0;
	void init();
	
	dbcsr::stensor2_d get_K_A() { return m_K_A; }
	dbcsr::stensor2_d get_K_B() { return m_K_B; }
	
};


class EXACT_J : public J {
private:

	dbcsr::stensor3_d m_J_bbd;
	dbcsr::stensor3_d m_ptot_bbd;

public:
	
	EXACT_J(dbcsr::world& w, desc::options& opt);
	void compute_J() override;
	
};

class EXACT_K : public K {
private:

	dbcsr::stensor3_d m_K_bbd;
	dbcsr::stensor3_d m_p_bbd;
	
public:

	EXACT_K(dbcsr::world& w, desc::options& opt);
	void compute_K() override;
	
};

} // end namespace

#endif 
