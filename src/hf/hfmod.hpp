#ifndef HF_MOD_H
#define HF_MOD_H

#ifndef TEST_MACRO
#include "megalochem.hpp"
#include "desc/molecule.hpp"
#include "desc/options.hpp"
#include "desc/wfn.hpp"
#include <dbcsr_conversions.hpp>
#include "utils/mpi_time.hpp"
#include "fock/jkbuilder.hpp"
#include "ints/aofactory.hpp"

#include <mpi.h>
#include <memory>
#include <iostream>
#endif

#include "utils/ppdirs.hpp"

using mat_d = dbcsr::matrix<double>;
using smat_d = dbcsr::shared_matrix<double>;

namespace megalochem {

namespace hf {

#define HFMOD_LIST (\
	((world), set_world),\
	((desc::shared_molecule), set_molecule),\
	((util::optional<desc::shared_cluster_basis>), df_basis),\
	((util::optional<desc::shared_cluster_basis>), df_basis2))
	
#define HFMOD_LIST_OPT (\
	((util::optional<std::string>), guess, "SAD"),\
	((util::optional<double>), scf_threshold, 1e-6),\
	((util::optional<int>), max_iter, 100),\
	((util::optional<bool>),do_diis, true),\
	((util::optional<int>), diis_max_vecs, 10),\
	((util::optional<int>), diis_min_vecs, 2),\
	((util::optional<int>), diis_start, 1),\
	((util::optional<bool>), do_diis_beta, true),\
	((util::optional<std::string>), build_J, "exact"),\
	((util::optional<std::string>), build_K, "exact"),\
	((util::optional<std::string>), eris, "core"),\
	((util::optional<std::string>), imeds, "core"),\
	((util::optional<std::string>), df_metric, "coulomb"),\
	((util::optional<int>), print, 0),\
	((util::optional<int>), nbatches_b, 5),\
	((util::optional<int>), nbatches_x, 5),\
	((util::optional<int>), nbatches_occ, 3),\
	((util::optional<bool)>, read, false),\
	((util::optional<std::string>), SAD_guess, "core"),\
	((util::optional<double>), SAD_scf_threshold, 1e-6),\
	((util::optional<bool>), SAD_do_diis, false),\
	((util::optional<bool>), SAD_spin_average, true))

class hfmod {
private:
	
	// descriptors
	world m_world;
	desc::shared_molecule m_mol;
	
	desc::shared_cluster_basis m_df_basis, m_df_basis2;
	
	MAKE_MEMBER_VARS(HFMOD_LIST_OPT)
	
	dbcsr::cart m_cart;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	// other
	int m_SAD_rank;
	bool m_restricted;
	bool m_nobetaorb;
	
	// results
	double m_nuc_energy;
	double m_scf_energy;
	
	dbcsr::shared_matrix<double> m_s_bb, //overlap
		  m_v_bb, // nuclear reulsion
		  m_t_bb, // kinetic
		  m_core_bb, // core hamiltonian
		  m_x_bb, // orthogonalizing matrix
		  m_f_bb_A, // alpha fock mat
		  m_p_bb_A, // alpha/beta density matrix
		  m_pv_bb_A,
		  m_c_bm_A; // alpha/beta coefficient matrix	  
				  
	dbcsr::shared_matrix<double> m_f_bb_B, m_p_bb_B, m_pv_bb_B, m_c_bm_B;
	
	svector<double> m_eps_A, m_eps_B;
	
	std::shared_ptr<ints::aoloader> m_aoloader;
	std::shared_ptr<fock::J> m_jbuilder;
	std::shared_ptr<fock::K> m_kbuilder;
	
	void init();
	
	void compute_nucrep();
	
	void one_electron();
	
	void two_electron();
	
	void form_fock(bool SAD_iter, int rank);
	
	void compute_guess();
	
	dbcsr::shared_matrix<double> compute_errmat(dbcsr::shared_matrix<double>& F, 
		dbcsr::shared_matrix<double>& P, dbcsr::shared_matrix<double>& S, std::string x);
	
	void diag_fock();
	
	void compute_scf_energy();
	
	void compute_virtual_density();

public:

	MAKE_PARAM_STRUCT(create, CONCAT(HFMOD_LIST, HFMOD_LIST_OPT), ())
	MAKE_BUILDER_CLASS(hfmod, create, CONCAT(HFMOD_LIST, HFMOD_LIST_OPT), ())

	hfmod(world wrd, desc::shared_molecule mol, desc::options opt);
	
	hfmod(create_pack&& p) : 
		m_world(p.p_set_world),
		m_mol(p.p_set_molecule),
		m_cart(p.p_set_world.dbcsr_grid()),
		m_df_basis(p.p_df_basis ? *p.p_df_basis : nullptr),
		m_df_basis2(p.p_df_basis2 ? *p.p_df_basis2 : nullptr),
		MAKE_INIT_LIST_OPT(HFMOD_LIST_OPT),
		LOG(m_world.comm(), m_print),
		TIME(m_world.comm(), "hfmod", m_print)
	{
		init();
	}
	
	hfmod() = delete;
	hfmod(hfmod& hfmod_in) = delete;
	
	~hfmod();
	
	desc::shared_wavefunction compute();	
	
};

} // end namespace hf

} // end namespace megalochem

#endif
