#ifndef MPMOD_H
#define MPMOD_H

#include "mp/mp_wfn.h"
#include "desc/options.h"
#include "utils/mpi_time.h"
#include <dbcsr_common.hpp>

namespace mp {

class mpmod {
private:

	hf::shared_hf_wfn m_hfwfn;
	desc::options m_opt;
	dbcsr::world m_world;
	
	util::mpi_time TIME;
	util::mpi_log LOG;
	
	smp_wfn m_mpwfn;
	
public:

	mpmod(hf::shared_hf_wfn& wfn_in, desc::options& opt_in, dbcsr::world& w);
	~mpmod() {}
	
	void compute();
	
	void compute_batch();
	
	smp_wfn wfn() {
		return m_mpwfn;
	}
	
};

}

#endif
