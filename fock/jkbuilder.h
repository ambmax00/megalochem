#ifndef FOCK_JK_BUILDER_H
#define FOCK_JK_BUILDER_H

#include "ints/aofactory.h"
#include "utils/registry.h"
#include "desc/molecule.h"
#include "desc/options.h"
#include "utils/mpi_time.h"
#include <dbcsr_btensor.hpp>
#include <dbcsr_matrix.hpp>

namespace fock {
	
class JK_common {
protected:
	
	desc::smolecule m_mol;
	desc::options m_opt;
	dbcsr::world m_world;
	
	util::registry m_reg;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	dbcsr::smat_d m_p_A;
	dbcsr::smat_d m_p_B;
	dbcsr::smat_d m_c_A;
	dbcsr::smat_d m_c_B;
	
	bool m_SAD_iter;
	int m_SAD_rank;
	
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
	void set_SAD(bool SAD, int rank) { 
		m_SAD_iter = SAD; 
		m_SAD_rank = rank;
	}
	void set_reg(util::registry& reg) {
		m_reg = reg;
	}
	
	virtual ~JK_common() {}
	
	void print_info() { TIME.print_info(); }
	
};

class J : public JK_common {
protected: 

	dbcsr::smat_d m_J;
	std::shared_ptr<J> m_builder;
	
public:
	
	J(dbcsr::world& w, desc::options& opt) : JK_common(w,opt) {}
	virtual ~J() {}
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
	virtual ~K() {}
	virtual void compute_K() = 0;
	virtual void init_tensors() = 0;
	
	void init();
	
	dbcsr::smat_d get_K_A() { return m_K_A; }
	dbcsr::smat_d get_K_B() { return m_K_B; }
	
};


class EXACT_J : public J {
private:

	dbcsr::sbtensor<4,double> m_eri_batched;
	dbcsr::shared_tensor<3,double> m_J_bbd;
	dbcsr::shared_tensor<3,double> m_ptot_bbd;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;

public:
	
	EXACT_J(dbcsr::world& w, desc::options& opt);
	void compute_J() override;
	void init_tensors() override;
	
};

class EXACT_K : public K {
private:

	dbcsr::sbtensor<4,double> m_eri_batched;
	dbcsr::shared_tensor<3,double> m_K_bbd;
	dbcsr::shared_tensor<3,double> m_p_bbd;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;
	
public:

	EXACT_K(dbcsr::world& w, desc::options& opt);
	void compute_K() override;
	void init_tensors() override;
	
};

class BATCHED_DF_J : public J {
private:

	dbcsr::sbtensor<3,double> m_eri_batched;
	
	dbcsr::shared_tensor<3,double> m_J_bbd;
	dbcsr::shared_tensor<3,double> m_ptot_bbd;
	dbcsr::shared_tensor<2,double> m_gp_xd;
	dbcsr::shared_tensor<2,double> m_gq_xd;
	dbcsr::shared_tensor<2,double> m_inv;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;
	dbcsr::shared_pgrid<2> m_spgrid_xd;

public:

	BATCHED_DF_J(dbcsr::world& w, desc::options& opt);
	void compute_J() override;
	void init_tensors() override;
	
	~BATCHED_DF_J() {}
	
};


class BATCHED_DFMO_K : public K {
private:
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	
	dbcsr::shared_tensor<3,double> m_HT1_xmb_02_1;
	dbcsr::shared_tensor<3,double> m_HT1_xmb_0_12;
	dbcsr::shared_tensor<3,double> m_HT2_xmb_0_12;
	dbcsr::shared_tensor<3,double> m_HT2_xmb_01_2;
	
	dbcsr::shared_tensor<2,double> m_c_bm;
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_invsqrt;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
public:

	BATCHED_DFMO_K(dbcsr::world& w, desc::options& opt);
	void compute_K() override;
	void init_tensors() override;
	
	~BATCHED_DFMO_K() {}
	
};

class BATCHED_DFAO_K : public K {
private:
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	dbcsr::sbtensor<3,double> m_c_xbb_batched;
		
	dbcsr::shared_tensor<3,double> m_c_xbb_1_02;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_01_2;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_02_1;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_p_bb;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
public:

	BATCHED_DFAO_K(dbcsr::world& w, desc::options& opt);
	void compute_K() override;
	void init_tensors() override;
	
	~BATCHED_DFAO_K() {}
	
};
/*
class CADF_K : public K {
private:
	
	dbcsr::stensor2_d m_s_xx_inv;
	dbcsr::smat_d m_s_xx;
	
	tensor::sbatchtensor<3,double> m_fit_batched;
	tensor::sbatchtensor<3,double> m_eri_batched;
	
	std::shared_ptr<ints::screener> m_scr;
	
	arrvec<int,3> m_L3;
	arrvec<int,3> m_LB;
	
	vec<int> m_blk_to_atom_x;
	vec<int> m_blk_to_atom_b;
	vec<vec<int>> m_atom_to_blk_x;
	vec<vec<int>> m_atom_to_blk_b;
	
	void compute_fit();
	void compute_L3();
	void compute_LB();
	
public:

	CADF_K(dbcsr::world& w, desc::options& opt);
	void compute_K() override;
	void init_tensors() override;

};*/

} // end namespace

#endif 
