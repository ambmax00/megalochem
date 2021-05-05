#ifndef FOCK_JK_BUILDER_H
#define FOCK_JK_BUILDER_H

#ifndef TEST_MACRO
#include "megalochem.hpp"
#include "ints/aoloader.hpp"
#include "ints/registry.hpp"
#include "desc/molecule.hpp"
#include "utils/mpi_time.hpp"
#include <dbcsr_btensor.hpp>
#include <dbcsr_matrix.hpp>
#endif

#include "utils/ppdirs.hpp"

namespace megalochem {

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
	
	desc::shared_molecule m_mol;
	megalochem::world m_world;
	
	dbcsr::cart m_cart;
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
	
	JK_common(megalochem::world w, desc::shared_molecule smol, int print, std::string name);
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
	
	J(megalochem::world w, desc::shared_molecule smol, int print, std::string name) 
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
	
	K(megalochem::world w, desc::shared_molecule smol, int print, std::string name) 
		: JK_common(w,smol,print,name) {}
	virtual ~K() {}
	virtual void compute_K() = 0;
	
	virtual void init() = 0;
	
	dbcsr::shared_matrix<double> get_K_A() { return m_K_A; }
	dbcsr::shared_matrix<double> get_K_B() { return m_K_B; }
	
};

#define BASE_INIT(jk, jkname) \
	jk(p.p_set_world, p.p_molecule, (p.p_print) ? *p.p_print : 0, #jkname)

#define BASE_LIST (\
	((megalochem::world), set_world),\
	((desc::shared_molecule), molecule),\
	((util::optional<int>), print))

class EXACT_J : public J {
private:

	dbcsr::sbtensor<4,double> m_eri4c2e_batched;
	dbcsr::shared_tensor<3,double> m_J_bbd;
	dbcsr::shared_tensor<3,double> m_ptot_bbd;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;
	
public:

#define EXACT_J_LIST (\
	((dbcsr::sbtensor<4,double>), eri4c2e_batched))

	MAKE_PARAM_STRUCT(create, CONCAT(BASE_LIST, EXACT_J_LIST), ())
	MAKE_BUILDER_CLASS(EXACT_J, create, CONCAT(BASE_LIST, EXACT_J_LIST), ())

	EXACT_J(create_pack&& p) : 
		m_eri4c2e_batched(p.p_eri4c2e_batched),
		BASE_INIT(J, EXACT_J)
	{}
	void compute_J() override;
	void init() override;
	
};

class EXACT_K : public K {
private:

	dbcsr::sbtensor<4,double> m_eri4c2e_batched;
	dbcsr::shared_tensor<3,double> m_K_bbd;
	dbcsr::shared_tensor<3,double> m_p_bbd;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;
		
public:

#define EXACT_K_LIST (\
	((dbcsr::sbtensor<4,double>), eri4c2e_batched))

	MAKE_PARAM_STRUCT(create, CONCAT(BASE_LIST, EXACT_J_LIST), ())
	MAKE_BUILDER_CLASS(EXACT_K, create, CONCAT(BASE_LIST, EXACT_J_LIST), ())

	EXACT_K(create_pack&& p) :
		m_eri4c2e_batched(p.p_eri4c2e_batched),
		BASE_INIT(K, EXACT_K)
	{}
	void compute_K() override;
	void init() override;
	
};

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
	
public:

#define DF_J_LIST (\
	((dbcsr::sbtensor<3,double>), eri3c2e_batched),\
	((dbcsr::shared_matrix<double>), metric_inv))
	
	MAKE_PARAM_STRUCT(create, CONCAT(BASE_LIST, DF_J_LIST), ())
	MAKE_BUILDER_CLASS(DF_J, create, CONCAT(BASE_LIST, DF_J_LIST), ())
	
	DF_J(create_pack&& p) : 
		m_eri3c2e_batched(p.p_eri3c2e_batched),
		m_v_inv(p.p_metric_inv),
		BASE_INIT(J, DF_J)
	{}
	void compute_J() override;
	void init() override;
	
	~DF_J() {}
	
};

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
		
public:
	
#define DFMO_K_LIST (\
	((dbcsr::sbtensor<3,double>), eri3c2e_batched),\
	((dbcsr::shared_matrix<double>), metric_invsqrt),\
	((util::optional<int>), occ_nbatches))
	
	MAKE_PARAM_STRUCT(create, CONCAT(BASE_LIST, DFMO_K_LIST), ())
	MAKE_BUILDER_CLASS(DFMO_K, create, CONCAT(BASE_LIST, DFMO_K_LIST), ())

	DFMO_K(create_pack&& p) : 
		m_eri3c2e_batched(p.p_eri3c2e_batched),
		m_v_invsqrt(p.p_metric_invsqrt),
		m_occ_nbatches((p.p_occ_nbatches) ? *p.p_occ_nbatches : 5),
		BASE_INIT(K, DFMO_K)
	{}
	void compute_K() override;
	void init() override;
	
	~DFMO_K() {}
	
};

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
		
public:

#define DFAO_K_LIST (\
	((dbcsr::sbtensor<3,double>), eri3c2e_batched),\
	((dbcsr::sbtensor<3,double>), fitting_batched))

	MAKE_PARAM_STRUCT(create, CONCAT(BASE_LIST, DFAO_K_LIST), ())
	MAKE_BUILDER_CLASS(DFAO_K, create, CONCAT(BASE_LIST, DFAO_K_LIST), ())

	DFAO_K(create_pack&& p) : 
		m_eri3c2e_batched(p.p_eri3c2e_batched),
		m_fitting_batched(p.p_fitting_batched),
		BASE_INIT(K, DFAO_K)
	{}
	
	void compute_K() override;
	void init() override;
	
	~DFAO_K() {}
	
};

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
		
public:

#define DFROBUST_K_LIST (\
	((dbcsr::sbtensor<3,double>), eri3c2e_batched),\
	((dbcsr::sbtensor<3,double>), fitting_batched),\
	((dbcsr::shared_matrix<double>), metric))

	MAKE_PARAM_STRUCT(create, CONCAT(BASE_LIST, DFROBUST_K_LIST), ())
	MAKE_BUILDER_CLASS(DFROBUST_K, create, CONCAT(BASE_LIST, DFROBUST_K_LIST), ())

	DFROBUST_K(create_pack&& p) : 
		m_eri3c2e_batched(p.p_eri3c2e_batched),
		m_fitting_batched(p.p_fitting_batched),
		m_v_xx(p.p_metric),
		BASE_INIT(K, DFROBUST_K)
	{}
	
	void compute_K() override;
	void init() override;
	
	~DFROBUST_K() {}
	
};

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
		
public:

#define DFMEM_K_LIST (\
	((dbcsr::sbtensor<3,double>), eri3c2e_batched),\
	((dbcsr::shared_matrix<double>), metric_inv))
	
	MAKE_PARAM_STRUCT(create, CONCAT(BASE_LIST, DFMEM_K_LIST), ())
	MAKE_BUILDER_CLASS(DFMEM_K, create, CONCAT(BASE_LIST, DFMEM_K_LIST), ())

	DFMEM_K(create_pack&& p) : 
		m_eri3c2e_batched(p.p_eri3c2e_batched),
		m_v_xx(p.p_metric_inv),
		BASE_INIT(K, DFMEM_K)
	{}
	
	void compute_K() override;
	void init() override;
	
	~DFMEM_K() {}
	
};

class create_DFLMO_K_base;
class DFLMO_K : public K {
private:

	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::shared_matrix<double> m_v_xx;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_v_xx_01;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
	int m_occ_nbatches;
	
public:

#define DFLMO_K_LIST (\
	((dbcsr::sbtensor<3,double>), eri3c2e_batched),\
	((dbcsr::shared_matrix<double>),metric_inv),\
	((util::optional<int>), occ_nbatches))
	
	MAKE_PARAM_STRUCT(create, CONCAT(BASE_LIST, DFLMO_K_LIST), ())
	MAKE_BUILDER_CLASS(DFLMO_K, create, CONCAT(BASE_LIST, DFLMO_K_LIST), ())

	DFLMO_K(create_pack&& p) : 
		m_eri3c2e_batched(p.p_eri3c2e_batched),
		m_v_xx(p.p_metric_inv),
		m_occ_nbatches((p.p_occ_nbatches) ? *p.p_occ_nbatches : 5),
		BASE_INIT(K, DFLMO_K)
	{}
	
	void compute_K() override;
	void init() override;
	
	~DFLMO_K() {}
	
};

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
			ao.request(ints::key::ovlp_bb, false);
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
			ao.request(ints::key::ovlp_bb, false);
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
			ao.request(ints::key::ovlp_bb, false);
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
	
#define CREATE_J_LIST (\
	((megalochem::world), set_world),\
	((desc::shared_molecule), molecule),\
	((jmethod), method),\
	((ints::aoloader), aoloader),\
	((util::optional<int>), print),\
	((ints::metric), metric))
	
class create_j_base {
private:

	typedef create_j_base _create_base;
	
	MAKE_BUILDER_MEMBERS(create, CREATE_J_LIST)

public:

	MAKE_BUILDER_SETS(create, CREATE_J_LIST)

	create_j_base() {}
	
	std::shared_ptr<J> build() {

		CHECK_REQUIRED(CREATE_J_LIST)

		std::shared_ptr<J> jbuilder;
		auto aoreg = c_aoloader->get_registry();
		
		int nprint = (c_print) ? *c_print : 0;
	
		if (*c_method == jmethod::exact) {
			
			auto eris = aoreg.get<dbcsr::sbtensor<4,double>>(ints::key::coul_bbbb);
			
			jbuilder = EXACT_J::create()
				.set_world(*c_set_world)
				.molecule(*c_molecule)
				.print(nprint)
				.eri4c2e_batched(eris)
				.build();
			
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
			
			jbuilder = DF_J::create()
				.set_world(*c_set_world)
				.molecule(*c_molecule)
				.print(nprint)
				.eri3c2e_batched(eris)
				.metric_inv(v_inv)
				.build();
			
		}
		
		return jbuilder;
		
	}
	
};

inline create_j_base create_j() { return create_j_base(); }

#define CREATE_K_LIST (\
	((megalochem::world), set_world),\
	((desc::shared_molecule), molecule),\
	((kmethod), method),\
	((ints::aoloader), aoloader),\
	((util::optional<int>), print),\
	((ints::metric), metric),\
	((util::optional<int>), occ_nbatches))

class create_k_base {
private:
	
	typedef create_k_base _create_base;
	MAKE_BUILDER_MEMBERS(create, CREATE_K_LIST)
	
public:

	MAKE_BUILDER_SETS(create, CREATE_K_LIST)

	create_k_base() {}
	
	std::shared_ptr<K> build() {
	
		CHECK_REQUIRED(CREATE_K_LIST)

		std::shared_ptr<K> kbuilder;
		auto aoreg = c_aoloader->get_registry();
		
		int nprint = (c_print) ? *c_print : 0;
		int nobatches = (c_occ_nbatches) ? *c_occ_nbatches : 1;
	
		// set K
		if (*c_method == kmethod::exact) {
			
			auto eris = aoreg.get<dbcsr::sbtensor<4,double>>(ints::key::coul_bbbb);
			
			kbuilder = EXACT_K::create()
				.set_world(*c_set_world)
				.molecule(*c_molecule)
				.print(nprint)
				.eri4c2e_batched(eris)
				.build();
			
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
			
			kbuilder = DFAO_K::create()
				.set_world(*c_set_world)
				.molecule(*c_molecule)
				.print(nprint)
				.eri3c2e_batched(eris)
				.fitting_batched(cfit)
				.build();
			
		} else if (*c_method == kmethod::dfmo) {
			
			auto eris = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::coul_xbb);
			auto invsqrt = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::coul_xx_invsqrt);
			
			kbuilder = DFMO_K::create()
				.set_world(*c_set_world)
				.molecule(*c_molecule)
				.print(nprint)
				.eri3c2e_batched(eris)
				.metric_invsqrt(invsqrt)
				.occ_nbatches(nobatches)
				.build();
		
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
				kbuilder = DFMEM_K::create()
					.set_world(*c_set_world)
					.molecule(*c_molecule)
					.print(nprint)
					.eri3c2e_batched(eris)
					.metric_inv(v_xx)
					.build();
					
			} else {
				kbuilder = DFLMO_K::create()
					.set_world(*c_set_world)
					.molecule(*c_molecule)
					.print(nprint)
					.eri3c2e_batched(eris)
					.metric_inv(v_xx)
					.occ_nbatches(nobatches)
					.build();
					
			}
			
		} else if (*c_method == kmethod::dfrobust) {
			
			auto eris = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::coul_xbb);
			auto cfit = aoreg.get<dbcsr::sbtensor<3,double>>(ints::key::pari_xbb);
			auto v_xx = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::coul_xx);
			
			kbuilder = DFROBUST_K::create()
				.set_world(*c_set_world)
				.molecule(*c_molecule)
				.print(nprint)
				.eri3c2e_batched(eris)
				.fitting_batched(cfit)
				.metric(v_xx)
				.build();
				
		}
		
		return kbuilder;
		
	}
	
};

inline create_k_base create_k() { return create_k_base(); }

} // end namespace

} // end namespace mega

#endif 
