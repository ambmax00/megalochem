#ifndef FOCK_JK_BUILDER_H
#define FOCK_JK_BUILDER_H

#include "ints/aofactory.h"
#include "utils/registry.h"
#include "utils/ppdirs.h"
#include "utils/params.hpp"
#include "desc/molecule.h"
#include "desc/options.h"
#include "utils/mpi_time.h"
#include <dbcsr_btensor.hpp>
#include <dbcsr_matrix.hpp>

namespace fock {

enum class jmethod {
	invalid,
	exact,
	dfao
};

enum class kmethod {
	invalid,
	exact,
	dfao,
	dfmo,
	dfmem,
	pari
};

inline jmethod str_to_jmethod(std::string s) {
	if (s == "exact") return jmethod::exact;
	if (s == "dfao") return jmethod::dfao;
	return jmethod::invalid;
}

inline kmethod str_to_kmethod(std::string s) {
	if (s == "exact") return kmethod::exact;
	if (s == "dfao") return kmethod::dfao;
	if (s == "dfmo") return kmethod::dfmo;
	if (s == "dfmem") return kmethod::dfmem;
	if (s == "pari") return kmethod::pari;
	return kmethod::invalid;
}
	
class JK_common {
protected:
	
	desc::smolecule m_mol;
	dbcsr::world m_world;
		
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
	
	JK_common(dbcsr::world w, desc::smolecule smol, int print, std::string name);
	void set_density_alpha(dbcsr::shared_matrix<double>& ipA) { m_p_A = ipA; }
	void set_density_beta(dbcsr::shared_matrix<double>& ipB) { m_p_B = ipB; }
	void set_coeff_alpha(dbcsr::shared_matrix<double>& icA) { m_c_A = icA; }
	void set_coeff_beta(dbcsr::shared_matrix<double>& icB) { m_c_B = icB; }
	void set_sym(bool sym) { m_sym = sym; }
	
	void set_SAD(bool SAD, int rank) { 
		m_SAD_iter = SAD; 
		m_SAD_rank = rank;
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
	
	J(dbcsr::world w, desc::smolecule smol, int print, std::string name) 
		: JK_common(w,smol,print,name) {}
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
	
	K(dbcsr::world w, desc::smolecule smol, int print, std::string name) 
		: JK_common(w,smol,print,name) {}
	virtual ~K() {}
	virtual void compute_K() = 0;
	
	virtual void init() = 0;
	
	dbcsr::shared_matrix<double> get_K_A() { return m_K_A; }
	dbcsr::shared_matrix<double> get_K_B() { return m_K_B; }
	
};

class create_EXACT_J_base;
class create_EXACT_K_base;

class EXACT_J : public J {
private:

	dbcsr::sbtensor<4,double> m_eri4c2e_batched;
	dbcsr::shared_tensor<3,double> m_J_bbd;
	dbcsr::shared_tensor<3,double> m_ptot_bbd;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;
	
	friend class create_EXACT_J_base;

public:
	
	EXACT_J(dbcsr::world w, desc::smolecule smol, int print);
	void compute_J() override;
	void init() override;
	
};

MAKE_STRUCT(
	EXACT_J, J,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri4c2e_batched, (dbcsr::sbtensor<4,double>), required, val)
	)
)

class EXACT_K : public K {
private:

	dbcsr::sbtensor<4,double> m_eri4c2e_batched;
	dbcsr::shared_tensor<3,double> m_K_bbd;
	dbcsr::shared_tensor<3,double> m_p_bbd;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;
	
	friend class create_EXACT_K_base;
	
public:

	EXACT_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
};

MAKE_STRUCT(
	EXACT_K, K,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri4c2e_batched, (dbcsr::sbtensor<4,double>), required, val)
	)
)

class create_BATCHED_DF_J_base;
class BATCHED_DF_J : public J {
private:

	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::shared_matrix<double> m_v_inv;
	
	dbcsr::shared_tensor<3,double> m_J_bbd;
	dbcsr::shared_tensor<3,double> m_ptot_bbd;
	dbcsr::shared_tensor<2,double> m_gp_xd;
	dbcsr::shared_tensor<2,double> m_gq_xd;
	dbcsr::shared_tensor<2,double> m_v_inv_01;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;
	dbcsr::shared_pgrid<2> m_spgrid_xd, m_spgrid2;
	
	friend class create_BATCHED_DF_J_base;

public:

	BATCHED_DF_J(dbcsr::world w, desc::smolecule smol, int print);
	void compute_J() override;
	void init() override;
	
	~BATCHED_DF_J() {}
	
};

MAKE_STRUCT(
	BATCHED_DF_J, J,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(v_inv, (dbcsr::shared_matrix<double>), required, val)
	)
)

class create_BATCHED_DFMO_K_base;
class BATCHED_DFMO_K : public K {
private:
	
	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::shared_matrix<double> m_v_invsqrt;
	int m_occ_nbatches;
	
	dbcsr::shared_tensor<3,double> m_HT1_xmb_02_1;
	dbcsr::shared_tensor<3,double> m_HT1_xmb_0_12;
	dbcsr::shared_tensor<3,double> m_HT2_xmb_0_12;
	dbcsr::shared_tensor<3,double> m_HT2_xmb_01_2;
	
	dbcsr::shared_tensor<2,double> m_c_bm;
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_v_invsqrt_01;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
	friend class create_BATCHED_DFMO_K_base;
	
public:

	BATCHED_DFMO_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
	~BATCHED_DFMO_K() {}
	
};

MAKE_STRUCT(
	BATCHED_DFMO_K, K,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(v_invsqrt, (dbcsr::shared_matrix<double>), required, val),
		(occ_nbatches, (int), optional, val, 5)
	)
)

class create_BATCHED_DFAO_K_base;
class BATCHED_DFAO_K : public K {
private:
	
	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::sbtensor<3,double> m_fitting_batched;
		
	dbcsr::shared_tensor<3,double> m_cbar_xbb_01_2;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_1_02;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_p_bb;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<3> m_spgrid3_xbb;
	
	friend class create_BATCHED_DFAO_K_base;
	
public:

	BATCHED_DFAO_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
	~BATCHED_DFAO_K() {}
	
};

MAKE_STRUCT(
	BATCHED_DFAO_K, K,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(fitting_batched, (dbcsr::sbtensor<3,double>), required, val)
	)
)

class create_BATCHED_PARI_K_base;
class BATCHED_PARI_K : public K {
private:

	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::shared_tensor<3,double> m_fitting;
	dbcsr::shared_matrix<double> m_v_xx;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_p_bb;
	dbcsr::shared_tensor<2,double> m_s_xx_01;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<3> m_spgrid3_xbb;
	
	friend class create_BATCHED_PARI_K_base;
	
public:

	BATCHED_PARI_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
	~BATCHED_PARI_K() {}
	
};

MAKE_STRUCT(
	BATCHED_PARI_K, K,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(fitting, (dbcsr::shared_tensor<3,double>), required, val),
		(v_xx, (dbcsr::shared_matrix<double>), required, val)
	)
)

class create_BATCHED_DFMEM_K_base;
class BATCHED_DFMEM_K : public K {
private:

	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::shared_matrix<double> m_v_xx;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_p_bb;
	dbcsr::shared_tensor<2,double> m_v_xx_01;
	
	dbcsr::shared_tensor<3,double> m_c_xbb_02_1;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_01_2;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_02_1;
	
	dbcsr::shared_tensor<3,double> m_cpq_xbb_0_12;
	dbcsr::shared_tensor<3,double> m_cpq_xbb_01_2;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<3> m_spgrid3_xbb;
	
	Eigen::MatrixXi m_idx_list;
	
	friend class create_BATCHED_DFMEM_K_base;
	
public:

	BATCHED_DFMEM_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
	~BATCHED_DFMEM_K() {}
	
};

MAKE_STRUCT(
	BATCHED_DFMEM_K, K,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(v_xx, (dbcsr::shared_matrix<double>), required, val)
	)
)
/*
class create_SPARSE_K_base;
class SPARSE_K : public K {
private:

	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::shared_matrix<double> m_v_xx;
		
	friend class create_SPARSE_K_base;
	
public:

	SPARSE_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
	~SPARSE_K() {}
	
};

MAKE_STRUCT(
	SPARSE_K, K,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(v_xx, (dbcsr::shared_matrix<double>), required, val)
	)
)

*/
} // end namespace

#endif 
