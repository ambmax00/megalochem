#ifndef IO_READER_H
#define IO_READER_H

#include <string>
#include <stdexcept>

#include "desc/options.h"
#include "desc/molecule.h"
#include "utils/mpi_log.h"

namespace filio {

class reader {
private:

	desc::smolecule m_mol;
	desc::options m_opt;
	util::mpi_log LOG;
	MPI_Comm m_comm;

public:	
	
	reader(MPI_Comm comm, std::string filename, int print = 0);
	
	desc::smolecule get_mol() {
		return m_mol;
	};
	
	desc::options get_opt() {
		return m_opt;
	};
	
};

}

#endif
