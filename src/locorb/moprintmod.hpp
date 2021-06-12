#ifndef LOCORB_MOPRINT_HPP
#define LOCORB_MOPRINT_HPP

#ifndef TEST_MACRO
	#include "locorb/locorb.hpp"
	#include "desc/wfn.hpp"
#endif

#include "utils/ppdirs.hpp"

namespace megalochem {

namespace locorb {

using smat_d = dbcsr::shared_matrix<double>;

ENUM_STRINGS(lmo_type, (boys, pao, cholesky))

ENUM_STRINGS(job_type, (local, nto, cmo))

#define MOPRINT_LIST (\
	((world), set_world),\
	((desc::shared_wavefunction), set_wfn),\
	((std::string), filename),\
	((job_type), job_name),\
	((util::optional<lmo_type>), lmo_occ),\
	((util::optional<lmo_type>), lmo_vir),\
	((util::optional<int>), print)\
)

class moprintmod {
 private:
	
	world m_world;
	desc::shared_wavefunction m_wfn;
	std::string m_filename;
	job_type m_job_name;
	util::optional<lmo_type> m_lmo_occ;
	util::optional<lmo_type> m_lmo_vir;
	int m_print;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	
 public:

	MAKE_PARAM_STRUCT(create, MOPRINT_LIST, ())
	MAKE_BUILDER_CLASS(moprintmod, create, MOPRINT_LIST, ())
	
	moprintmod(create_pack&& p);
	
	void compute();
	
};

}

}

#endif
