#ifndef FOCK_JK_BUILDER_H
#define FOCK_JK_BUILDER_H

#include "ints/aoloader.hpp"
#include "utils/registry.hpp"
#include "utils/ppdirs.hpp"
#include "utils/params.hpp"
#include "desc/molecule.hpp"
#include "desc/options.hpp"
#include "utils/mpi_time.hpp"
#include <dbcsr_btensor.hpp>
#include <dbcsr_matrix.hpp>

namespace fock {

enum class jmethod {
	exact,
	dfao
};

enum class kmethod {
	exact,
	dfao,
	dfmo,
	dfmem,
	dfrobust,
	dflmo
};

inline jmethod str_to_jmethod(std::string s) {
	if (s == "exact") {
		return jmethod::exact;
	} else if (s == "dfao") {
		return jmethod::dfao;
	} else {
		throw std::runtime_error("Invalid jmethod");
	}
}

inline kmethod str_to_kmethod(std::string s) {
	if (s == "exact") {
		return kmethod::exact;
	} else if (s == "dfao") {
		return kmethod::dfao;
	} else if (s == "dfmo") {
		return kmethod::dfmo;
	} else if (s == "dfmem") {
		return kmethod::dfmem;
	} else if (s == "dfrobust") {
		return kmethod::dfrobust;
	} else if (s == "dflmo") {
		return kmethod::dflmo;
	} else {
		throw std::runtime_error("Invalid kmethod");
	}
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
	
	util::mpi_time get_time() {
		return TIME;
	}
	
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

class create_DF_J_base;
class DF_J : public J {
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
	
	friend class create_DF_J_base;

public:

	DF_J(dbcsr::world w, desc::smolecule smol, int print);
	void compute_J() override;
	void init() override;
	
	~DF_J() {}
	
};

MAKE_STRUCT(
	DF_J, J,
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

class create_DFMO_K_base;
class DFMO_K : public K {
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
	
	friend class create_DFMO_K_base;
	
public:

	DFMO_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
	~DFMO_K() {}
	
};

MAKE_STRUCT(
	DFMO_K, K,
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

class create_DFAO_K_base;
class DFAO_K : public K {
private:
	
	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::sbtensor<3,double> m_fitting_batched;
		
	dbcsr::shared_tensor<3,double> m_cbar_xbb_01_2;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_1_02;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_p_bb;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	dbcsr::shared_pgrid<3> m_spgrid3_xbb;
	
	friend class create_DFAO_K_base;
	
public:

	DFAO_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
	~DFAO_K() {}
	
};

MAKE_STRUCT(
	DFAO_K, K,
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

class create_DFROBUST_K_base;
class DFROBUST_K : public K {
private:

	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::sbtensor<3,double> m_fitting_batched;
	dbcsr::shared_matrix<double> m_v_xx;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_p_bb;
	dbcsr::shared_tensor<2,double> m_v_xx_01;
	
	dbcsr::shared_tensor<3,double> m_cbar_xbb_01_2;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_02_1;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_0_12;
	dbcsr::shared_tensor<3,double> m_cfit_xbb_01_2;
	dbcsr::shared_tensor<3,double> m_cpq_xbb_0_12;
	dbcsr::shared_tensor<3,double> m_cpq_xbb_02_1;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
	friend class create_DFROBUST_K_base;
	
public:

	DFROBUST_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
	~DFROBUST_K() {}
	
};

MAKE_STRUCT(
	DFROBUST_K, K,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(fitting_batched, (dbcsr::sbtensor<3,double>), required, val),
		(v_xx, (dbcsr::shared_matrix<double>), required, val)
	)
)

class create_DFMEM_K_base;
class DFMEM_K : public K {
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
	
	friend class create_DFMEM_K_base;
	
public:

	DFMEM_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
	~DFMEM_K() {}
	
};

MAKE_STRUCT(
	DFMEM_K, K,
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

class create_DFLMO_K_base;
class DFLMO_K : public K {
private:

	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::shared_matrix<double> m_v_xx;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_v_xx_01;
	
	dbcsr::shared_pgrid<2> m_spgrid2;

	friend class create_DFLMO_K_base;
	
	int m_occ_nbatches;
	
public:

	DFLMO_K(dbcsr::world w, desc::smolecule smol, int print);
	void compute_K() override;
	void init() override;
	
	~DFLMO_K() {}
	
};

MAKE_STRUCT(
	DFLMO_K, K,
	(
		(world, (dbcsr::world)),
		(mol, (desc::smolecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(v_xx, (dbcsr::shared_matrix<double>), required, val),
		(occ_nbatches, (int), optional, val, 1)
	)
)

inline void load_jints(jmethod jmet, ints::metric metr, ints::aoloader& ao) {
	
	// set J
	if (jmet == jmethod::exact) {
		
		ao.request(ints::key::coul_bbbb, true);
		
	} else if (jmet == jmethod::dfao) {
		
		ao.request(ints::key::scr_xbb,true);
		
		if (metr == ints::metric::coulomb) {
			ao.request(ints::key::coul_xx, false);
			ao.request(ints::key::coul_xx_inv, true);
			ao.request(ints::key::coul_xbb, true);
		} else if (metr == ints::metric::erfc_coulomb) {
			ao.request(ints::key::erfc_xx, false);
			ao.request(ints::key::erfc_xx_inv, true);
			ao.request(ints::key::erfc_xbb, true);
		} else if (metr == ints::metric::qr_fit) {
			ao.request(ints::key::coul_xx, true);
			ao.request(ints::key::ovlp_xx, false);
			ao.request(ints::key::ovlp_xx_inv, false);
			ao.request(ints::key::qr_xbb, true);
		} else if (metr == ints::metric::pari) {
			ao.request(ints::key::coul_xx, true);
			ao.request(ints::key::pari_xbb, true);
		}
		
	}
	
}
	
inline void load_kints(kmethod kmet, ints::metric metr, ints::aoloader& ao) {
	
	if (kmet == kmethod::exact) {
		
		ao.request(ints::key::coul_bbbb,true);
		
	} else if (kmet == kmethod::dfao) {
		
		ao.request(ints::key::scr_xbb,true);
		
		if (metr == ints::metric::coulomb) {
			ao.request(ints::key::coul_xx,false);
			ao.request(ints::key::coul_xx_inv,false);
			ao.request(ints::key::coul_xbb,true);
			ao.request(ints::key::dfit_coul_xbb,true);
		} else if (metr == ints::metric::erfc_coulomb) {
			ao.request(ints::key::erfc_xx,false);
			ao.request(ints::key::erfc_xx_inv,false);
			ao.request(ints::key::erfc_xbb,true);
			ao.request(ints::key::dfit_erfc_xbb,true);
		} else if (metr == ints::metric::qr_fit) {
			ao.request(ints::key::coul_xx, true);
			ao.request(ints::key::ovlp_xx, false);
			ao.request(ints::key::ovlp_xx_inv, false);
			ao.request(ints::key::qr_xbb, true);
			ao.request(ints::key::dfit_qr_xbb, true);
		} else if (metr == ints::metric::pari) {
			ao.request(ints::key::coul_xx, true);
			ao.request(ints::key::pari_xbb, true);
			ao.request(ints::key::dfit_pari_xbb, true);
		}
		
	} else if (kmet == kmethod::dfmo) {
		
		ao.request(ints::key::scr_xbb,true);
				
		if (metr == ints::metric::coulomb) {
			ao.request(ints::key::coul_xx,false);
			ao.request(ints::key::coul_xx_invsqrt,true);
			ao.request(ints::key::coul_xbb,true);
		} else {
			throw std::runtime_error("DFMO with non coulomb metric disabled.");
		}
		
	} else if (kmet == kmethod::dfmem || kmet == kmethod::dflmo) {
		
		ao.request(ints::key::scr_xbb,true);
		
		if (metr == ints::metric::coulomb) {
			ao.request(ints::key::coul_xx,false);
			ao.request(ints::key::coul_xx_inv,true);
			ao.request(ints::key::coul_xbb,true);
		} else if (metr == ints::metric::erfc_coulomb) {
			ao.request(ints::key::erfc_xx,false);
			ao.request(ints::key::erfc_xx_inv,true);
			ao.request(ints::key::erfc_xbb,true);
		} else if (metr == ints::metric::qr_fit) {
			ao.request(ints::key::coul_xx, true);
			ao.request(ints::key::ovlp_xx, false);
			ao.request(ints::key::ovlp_xx_inv, false);
			ao.request(ints::key::qr_xbb, true);
		} else if (metr == ints::metric::pari) {
			ao.request(ints::key::coul_xx, true);
			ao.request(ints::key::pari_xbb, true);
		}
		
	} else if (kmet == kmethod::dfrobust) {
		
		ao.request(ints::key::scr_xbb,true);
		
		if (metr != ints::metric::pari) {
			throw std::runtime_error(
				"Cannot use robust k for method other than PARI");
		} else if (metr == ints::metric::pari) {
			ao.request(ints::key::coul_xx, false);
			ao.request(ints::key::coul_xbb, true);
			ao.request(ints::key::pari_xbb, true);
		}
		
	}
}

class create_j_base {

	make_param(create_j_base, world, dbcsr::world, required, val)
	make_param(create_j_base, mol, desc::smolecule, required, val)
	make_param(create_j_base, method, jmethod, required, val)
	make_param(create_j_base, aoloader, ints::aoloader, required, ref)
	make_param(create_j_base, print, int, optional, val)
	make_param(create_j_base, metric, ints::metric, required, val)

public:

	create_j_base() {}
	
	std::shared_ptr<J> get() {

		std::shared_ptr<J> jbuilder;
		auto aoreg = c_aoloader->get_registry();
		
		int nprint = (c_print) ? *c_print : 0;
	
		if (*c_method == jmethod::exact) {
			
			auto eris = aoreg.get<dbcsr::sbtensor<4,double>>(ints::key::coul_bbbb);
			
			jbuilder = create_EXACT_J(*c_world, *c_mol, nprint)
				.eri4c2e_batched(eris)
				.get();
			
		} else if (*c_method == jmethod::dfao) {
			
			dbcsr::sbtensor<3,double> eris;
			dbcsr::shared_matrix<double> v_inv;
			
			if (*c_metric == ints::metric::coulomb) {
				
				eris = aoreg.get<decltype(eris)>(ints::key::coul_xbb);
				v_inv = aoreg.get<decltype(v_inv)>(ints::key::coul_xx_inv);
				
			} else if (*c_metric == ints::metric::erfc_coulomb) {
				
				eris = aoreg.get<decltype(eris)>(ints::key::erfc_xbb);
				v_inv = aoreg.get<decltype(v_inv)>(ints::key::erfc_xx_inv);
				
			} else if (*c_metric == ints::metric::qr_fit) {
				
				eris = aoreg.get<decltype(eris)>(ints::key::qr_xbb);
				v_inv = aoreg.get<decltype(v_inv)>(ints::key::coul_xx);
				
			} else if (*c_metric == ints::metric::pari) {
				
				eris = aoreg.get<decltype(eris)>(ints::key::pari_xbb);
				v_inv = aoreg.get<decltype(v_inv)>(ints::key::coul_xx);
				
			}
			
			jbuilder = create_DF_J(*c_world, *c_mol, nprint)
				.eri3c2e_batched(eris)
				.v_inv(v_inv)
				.get();
			
		}
		
		return jbuilder;
		
	}
	
};

inline create_j_base create_j() { return create_j_base(); }

class create_k_base {

	make_param(create_k_base, world, dbcsr::world, required, val)
	make_param(create_k_base, mol, desc::smolecule, required, val)
	make_param(create_k_base, method, kmethod, required, val)
	make_param(create_k_base, aoloader, ints::aoloader, required, ref)
	make_param(create_k_base, print, int, optional, val)
	make_param(create_k_base, metric, ints::metric, required, val)
	make_param(create_k_base, occ_nbatches, int, optional, val)

public:

	create_k_base() {}
	
	std::shared_ptr<K> get() {

		std::shared_ptr<K> kbuilder;
		auto aoreg = c_aoloader->get_registry();
		
		int nprint = (c_print) ? *c_print : 0;
		int nobatches = (c_occ_nbatches) ? *c_occ_nbatches : 1;
	
		// set K
		if (*c_method == kmethod::exact) {
			
			auto eris = aoreg.get<dbcsr::sbtensor<4,double>>(ints::key::coul_bbbb);
			
			kbuilder = create_EXACT_K(*c_world, *c_mol, nprint)
				.eri4c2e_batched(eris)
				.get();
			
		} else if (*c_method == kmethod::dfao) {
			
			dbcsr::sbtensor<3,double> eris;
			dbcsr::sbtensor<3,double> cfit;
			
			switch (*c_metric) {
				case ints::metric::coulomb:
					eris = aoreg.get<decltype(eris)>(ints::key::coul_xbb);
					cfit = aoreg.get<decltype(eris)>(ints::key::dfit_coul_xbb);
					break;
				case ints::metric::erfc_coulomb:
					eris = aoreg.get<decltype(eris)>(ints::key::erfc_xbb);
					cfit = aoreg.get<decltype(eris)>(ints::key::dfit_erfc_xbb);
					break;
				case ints::metric::qr_fit:
					eris = aoreg.get<decltype(eris)>(ints::key::qr_xbb);
					cfit = aoreg.get<decltype(eris)>(ints::key::dfit_qr_xbb);
					break;
				case ints::metric::pari:
					eris = aoreg.get<decltype(eris)>(ints::key::pari_xbb);
					cfit = aoreg.get<decltype(eris)>(ints::key::dfit_pari_xbb);
					break;
			}
			
			kbuilder = create_DFAO_K(*c_world, *c_mol, nprint)
				.eri3c2e_batched(eris)
				.fitting_batched(cfit)
				.get();
			
		} else if (*c_method == kmethod::dfmo) {
			
			auto eris = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::coul_xbb);
			auto invsqrt = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::coul_xx_invsqrt);
			
			kbuilder = create_DFMO_K(*c_world, *c_mol, nprint)
				.eri3c2e_batched(eris)
				.v_invsqrt(invsqrt)
				.occ_nbatches(nobatches)
				.get();
		
		} else if (*c_method == kmethod::dfmem || *c_method == kmethod::dflmo) {
			
			dbcsr::sbtensor<3,double> eris;
			dbcsr::shared_matrix<double> v_xx;
			
			switch (*c_metric) {
				case ints::metric::coulomb:
					eris = aoreg.get<decltype(eris)>(ints::key::coul_xbb);
					v_xx = aoreg.get<decltype(v_xx)>(ints::key::coul_xx_inv);
					break;
				case ints::metric::erfc_coulomb:
					eris = aoreg.get<decltype(eris)>(ints::key::erfc_xbb);
					v_xx = aoreg.get<decltype(v_xx)>(ints::key::erfc_xx_inv);
					break;
				case ints::metric::qr_fit:
					eris = aoreg.get<decltype(eris)>(ints::key::qr_xbb);
					v_xx = aoreg.get<decltype(v_xx)>(ints::key::coul_xx);
					break;
				case ints::metric::pari:
					eris = aoreg.get<decltype(eris)>(ints::key::pari_xbb);
					v_xx = aoreg.get<decltype(v_xx)>(ints::key::coul_xx);
					break;
			}
			
			if (*c_method == kmethod::dfmem) {
				kbuilder = create_DFMEM_K(*c_world, *c_mol, nprint)
					.eri3c2e_batched(eris)
					.v_xx(v_xx)
					.get();
			} else {
				kbuilder = create_DFLMO_K(*c_world, *c_mol, nprint)
					.eri3c2e_batched(eris)
					.v_xx(v_xx)
					.occ_nbatches(nobatches)
					.get();
			}
			
		} else if (*c_method == kmethod::dfrobust) {
			
			auto eris = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::coul_xbb);
			auto cfit = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::pari_xbb);
			auto v_xx = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::coul_xx);
			
			kbuilder = create_DFROBUST_K(*c_world, *c_mol, nprint)
				.eri3c2e_batched(eris)
				.fitting_batched(cfit)
				.v_xx(v_xx)
				.get();
				
		}
		
		return kbuilder;
		
	}
	
};

inline create_k_base create_k() { return create_k_base(); }

} // end namespace

#endif 
