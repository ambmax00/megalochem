#ifndef MPMOD_H
#define MPMOD_H

#include "desc/wfn.h"
#include "desc/options.h"
#include "utils/mpi_time.h"
#include <dbcsr_common.hpp>

namespace mp {

class mpmod {
private:

	desc::shf_wfn m_hfwfn;
	desc::options m_opt;
	dbcsr::world m_world;
	
	util::mpi_time TIME;
	util::mpi_log LOG;
	
public:

	mpmod(desc::shf_wfn& wfn_in, desc::options& opt_in, dbcsr::world& w);
	~mpmod() {}
	
	void compute();
	

};

}

#endif
