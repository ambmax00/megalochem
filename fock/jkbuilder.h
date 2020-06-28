#ifndef FOCK_JK_BUILDER_H
#define FOCK_JK_BUILDER_H

#include "ints/aofactory.h"
#include "ints/registry.h"
#include "desc/molecule.h"
#include "desc/options.h"
#include "utils/mpi_time.h"
#include "tensor/batchtensor.h"
#include <dbcsr_matrix.hpp>

namespace fock {

class JK_common {
protected:
	
	desc::smolecule m_mol;
	desc::options m_opt;
	dbcsr::world m_world;
	
	ints::registry m_reg;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	dbcsr::smat_d m_p_A;
	dbcsr::smat_d m_p_B;
	dbcsr::smat_d m_c_A;
	dbcsr::smat_d m_c_B;
	
	bool m_SAD_iter;
	
	std::shared_ptr<ints::aofactory> m_factory;
	
public:
	
	JK_common(dbcsr::world& w, desc::options opt);
	void set_density_alpha(dbcsr::smat_d& ipA) { m_p_A = ipA; }
	void set_density_beta(dbcsr::smat_d& ipB) { m_p_B = ipB; }
	void set_coeff_alpha(dbcsr::smat_d& icA) { m_c_A = icA; }
	void set_coeff_beta(dbcsr::smat_d& icB) { m_c_B = icB; }
	void set_factory(std::shared_ptr<ints::aofactory>& ifac) { 
		m_factory = ifac; 
		m_mol = m_factory->mol();
	}
	void set_SAD(bool SAD) { m_SAD_iter = SAD; }
	
	~JK_common() {}
	
	void print_info() { TIME.print_info(); }
	
};

class J : public JK_common {
protected: 

	dbcsr::smat_d m_J;
	std::shared_ptr<J> m_builder;
	
public:
	
	J(dbcsr::world& w, desc::options& opt) : JK_common(w,opt) {}
	~J() {}
	virtual void compute_J() = 0;
	virtual void init_tensors() = 0;
	
	void init();
	
	dbcsr::smat_d get_J() { return m_J; }	
		
};

class K : public JK_common {
protected:

	dbcsr::smat_d m_K_A;
	dbcsr::smat_d m_K_B;
	
public:
	
	K(dbcsr::world& w, desc::options& opt) : JK_common(w,opt) {}
	~K() {}
	virtual void compute_K() = 0;
	virtual void init_tensors() = 0;
	
	void init();
	
	dbcsr::smat_d get_K_A() { return m_K_A; }
	dbcsr::smat_d get_K_B() { return m_K_B; }
	
};


class EXACT_J : public J {
private:

	dbcsr::stensor3_d m_J_bbd;
	dbcsr::stensor3_d m_ptot_bbd;

public:
	
	EXACT_J(dbcsr::world& w, desc::options& opt);
	void compute_J() override;
	void init_tensors() override;
	
};

class EXACT_K : public K {
private:

	dbcsr::stensor3_d m_K_bbd;
	dbcsr::stensor3_d m_p_bbd;
	
public:

	EXACT_K(dbcsr::world& w, desc::options& opt);
	void compute_K() override;
	void init_tensors() override;
	
};

class DF_J : public J {
private:
	
	dbcsr::stensor3_d m_J_bbd;
	dbcsr::stensor3_d m_ptot_bbd;
	dbcsr::stensor2_d m_c_xd;
	dbcsr::stensor2_d m_c2_xd;
	dbcsr::stensor2_d m_inv;

public:

	DF_J(dbcsr::world& w, desc::options& opt);
	void compute_J() override;
	void init_tensors() override;
	
};

class DF_K : public K {
private:
	
	dbcsr::stensor2_d m_K_01;
	dbcsr::stensor3_d m_HT_01_2;
	dbcsr::stensor3_d m_HT_0_12;
	dbcsr::stensor3_d m_HT_02_1;
	dbcsr::stensor3_d m_D_0_12;
	dbcsr::stensor3_d m_D_02_1;
	dbcsr::stensor3_d m_INTS_01_2;
	dbcsr::stensor2_d m_inv;
	dbcsr::stensor2_d m_c_bm;

public:

	DF_K(dbcsr::world& w, desc::options& opt);
	void compute_K() override;
	void init_tensors() override;
	
};

class BATCHED_DF_J : public J {
private:
	
	dbcsr::stensor3_d m_J_bbd;
	dbcsr::stensor3_d m_ptot_bbd;
	dbcsr::stensor2_d m_gp_xd;
	dbcsr::stensor2_d m_gq_xd;
	dbcsr::stensor2_d m_inv;
	bool m_direct = false;;

public:

	BATCHED_DF_J(dbcsr::world& w, desc::options& opt);
	void set_direct(bool i) { m_direct = i; }
	void compute_J() override;
	void init_tensors() override;
	void fetch_integrals(tensor::sbatchtensor<3,double>& btensor, int ibatch);
	
};

} // end namespace

#endif 
