#ifndef MPMOD_H
#define MPMOD_H

#ifndef TEST_MACRO
#include "megalochem.hpp"
#include "mp/mp_wfn.hpp"
#include "desc/options.hpp"
#include "utils/mpi_time.hpp"
#include <dbcsr_common.hpp>
#endif

#include "utils/ppdirs.hpp"

namespace megalochem {

namespace mp {

#define MPMOD_LIST (\
	((world), set_world),\
	((hf::shared_hf_wfn), set_hf_wfn),\
	((util::optional<desc::shared_cluster_basis>), df_basis)) 

#define MPMOD_OPTLIST (\
	((util::optional<int>), print, 0),\
	((util::optional<std::string>), df_metric, "coulomb"),\
	((util::optional<int>), nlap, 5),\
	((util::optional<int>), nbatches_b, 5),\
	((util::optional<int>), nbatches_x, 5),\
	((util::optional<double>), c_os, 1.3),\
	((util::optional<std::string>), eris, "core"),\
	((util::optional<std::string>), imeds, "core"),\
	((util::optional<std::string>), build_Z, "llmp_full"))

class mpmod {
private:

	world m_world;
	hf::shared_hf_wfn m_hfwfn;
	desc::shared_cluster_basis m_df_basis;
	
	MAKE_MEMBER_VARS(MPMOD_OPTLIST)
	
	util::mpi_time TIME;
	util::mpi_log LOG;
	
	smp_wfn m_mpwfn;
	
	void init();
	
public:

	MAKE_PARAM_STRUCT(create, CONCAT(MPMOD_LIST, MPMOD_OPTLIST), ())
	MAKE_BUILDER_CLASS(mpmod, create, CONCAT(MPMOD_LIST, MPMOD_OPTLIST), ())
	
	mpmod(create_pack&& p) :
		m_world(p.p_set_world),
		m_hfwfn(p.p_set_hf_wfn),
		m_df_basis(p.p_df_basis ? *p.p_df_basis : nullptr),
		LOG(m_world.comm(), m_print),
		TIME(m_world.comm(), "mpmod", m_print),
		MAKE_INIT_LIST_OPT(MPMOD_OPTLIST)
	{ init(); }
	
	~mpmod() {}
	
	void compute();
	
	smp_wfn wfn() {
		return m_mpwfn;
	}
	
};

} // namespace mp

} // namespace megalochem

#endif
