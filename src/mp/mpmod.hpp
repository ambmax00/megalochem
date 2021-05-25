#ifndef MPMOD_H
#define MPMOD_H

#ifndef TEST_MACRO
#include "megalochem.hpp"
#include "desc/wfn.hpp"
#include "utils/mpi_time.hpp"
#include <dbcsr_common.hpp>
#endif

#include "utils/ppdirs.hpp"

namespace megalochem {

namespace mp {

#define MPMOD_LIST (\
	((world), set_world),\
	((desc::shared_wavefunction), set_wfn),\
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
	desc::shared_wavefunction m_wfn;
	desc::shared_cluster_basis m_df_basis;
	
	MAKE_MEMBER_VARS(MPMOD_OPTLIST)
	
	util::mpi_log LOG;
	util::mpi_time TIME;
		
	void init();
	
public:

	MAKE_PARAM_STRUCT(create, CONCAT(MPMOD_LIST, MPMOD_OPTLIST), ())
	MAKE_BUILDER_CLASS(mpmod, create, CONCAT(MPMOD_LIST, MPMOD_OPTLIST), ())
	
	mpmod(create_pack&& p) :
		m_world(p.p_set_world),
		m_wfn(p.p_set_wfn),
		m_df_basis(p.p_df_basis ? *p.p_df_basis : nullptr),
		MAKE_INIT_LIST_OPT(MPMOD_OPTLIST),
		LOG(m_world.comm(), m_print),
		TIME(m_world.comm(), "mpmod", m_print)
	{ init(); }
	
	~mpmod() {}
	
	desc::shared_wavefunction compute();
	
};

} // namespace mp

} // namespace megalochem

#endif
