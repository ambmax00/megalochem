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
	
	dbcsr::shared_matrix<double> m_p_A;
	dbcsr::shared_matrix<double> m_p_B;
	dbcsr::shared_matrix<double> m_c_A;
	dbcsr::shared_matrix<double> m_c_B;
	
	bool m_SAD_iter;
	int m_SAD_rank;
	
	bool m_sym = true;
		
public:
	
	JK_common(dbcsr::world& w, desc::options opt, std::string name);
	void set_density_alpha(dbcsr::shared_matrix<double>& ipA) { m_p_A = ipA; }
	void set_density_beta(dbcsr::shared_matrix<double>& ipB) { m_p_B = ipB; }
	void set_coeff_alpha(dbcsr::shared_matrix<double>& icA) { m_c_A = icA; }
	void set_coeff_beta(dbcsr::shared_matrix<double>& icB) { m_c_B = icB; }
	void set_sym(bool sym) { m_sym = sym; }
	void set_mol(desc::smolecule& smol) { 
		m_mol = smol;
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

	dbcsr::shared_matrix<double> m_J;
	std::shared_ptr<J> m_builder;
	
	void init_base();
	
public:
	
	J(dbcsr::world& w, desc::options& opt, std::string name) : JK_common(w,opt,name) {}
	virtual ~J() {}
	virtual void compute_J() = 0;
	
	virtual void init() = 0;
	
	dbcsr::shared_matrix<double> get_J() { return m_J; }	
		
};

class K : public JK_common {
protected:

	dbcsr::shared_matrix<double> m_K_A;
	dbcsr::shared_matrix<double> m_K_B;
	
	void init_base();
	
public:
	
	K(dbcsr::world& w, desc::options& opt, std::string name) : JK_common(w,opt,name) {}
	virtual ~K() {}
	virtual void compute_K() = 0;
	
	virtual void init() = 0;
	
	dbcsr::shared_matrix<double> get_K_A() { return m_K_A; }
	dbcsr::shared_matrix<double> get_K_B() { return m_K_B; }
	
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
	void init() override;
	
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
	void init() override;
	
};

class BATCHED_DF_J : public J {
private:

	dbcsr::sbtensor<3,double> m_eri_batched;
	
	dbcsr::shared_tensor<3,double> m_J_bbd;
	dbcsr::shared_tensor<3,double> m_ptot_bbd;
	dbcsr::shared_tensor<2,double> m_gp_xd;
	dbcsr::shared_tensor<2,double> m_gq_xd;
	dbcsr::shared_tensor<2,double> m_inv;
	dbcsr::shared_matrix<double> m_inv_mat;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;
	dbcsr::shared_pgrid<2> m_spgrid_xd, m_spgrid2;

public:

	BATCHED_DF_J(dbcsr::world& w, desc::options& opt);
	void compute_J() override;
	void init() override;
	
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
	void init() override;
	
	~BATCHED_DFMO_K() {}
	
};

class BATCHED_DFAO_K : public K {
private:
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	dbcsr::sbtensor<3,double> m_c_xbb_batched;
		
	dbcsr::shared_tensor<3,double> m_cbar_xbb_01_2;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_1_02;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_p_bb;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<3> m_spgrid3_xbb;
	
public:

	BATCHED_DFAO_K(dbcsr::world& w, desc::options& opt);
	void compute_K() override;
	void init() override;
	
	~BATCHED_DFAO_K() {}
	
};

class BATCHED_PARI_K : public K {
private:

	dbcsr::sbtensor<3,double> m_eri_batched;
	dbcsr::shared_tensor<3,double> m_cfit_xbb;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_p_bb;
	dbcsr::shared_tensor<2,double> m_s_xx_01;
	dbcsr::shared_matrix<double> m_s_xx;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<3> m_spgrid3_xbb;
	
public:

	BATCHED_PARI_K(dbcsr::world& w, desc::options& opt);
	void compute_K() override;
	void init() override;
	
	~BATCHED_PARI_K() {}
	
};

inline std::shared_ptr<J> get_J(
	std::string name, dbcsr::world& w, desc::options opt) {
	
	std::shared_ptr<J> out;
	J* ptr = nullptr;
	
	if (name == "exact") {
		ptr = new EXACT_J(w,opt);
	} else if (name == "batchdf") {
		ptr = new BATCHED_DF_J(w,opt);
	}
	
	if (!ptr) {
		throw std::runtime_error("INVALID J BUILDER SPECIFIED");
	}
	
	out.reset(ptr);
	return out;

}

inline std::shared_ptr<K> get_K(
	std::string name, dbcsr::world& w, desc::options opt) {
	
	std::shared_ptr<K> out;
	K* ptr = nullptr;
	
	if (name == "exact") {
		ptr = new EXACT_K(w,opt);
	} else if (name == "batchdfao") {
		ptr = new BATCHED_DFAO_K(w,opt);
	} else if (name == "batchdfmo") {
		ptr = new BATCHED_DFMO_K(w,opt);
	} else if (name == "batchpari") {
		ptr = new BATCHED_PARI_K(w,opt);
	}
	
	if (!ptr) {
		throw std::runtime_error("INVALID K BUILDER SPECIFIED");
	}
	
	out.reset(ptr);
	return out;

}

} // end namespace

#endif 
