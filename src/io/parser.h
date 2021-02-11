#ifndef IO_READER_H
#define IO_READER_H

#include <string>
#include <stdexcept>

#include "desc/options.h"
#include "desc/molecule.h"
#include "utils/mpi_log.h"

#include "utils/json.hpp"

namespace filio {

desc::smolecule parse_molecule(nlohmann::json& j, MPI_Comm comm, int nprint);

desc::options parse_options(nlohmann::json& j, MPI_Comm comm, int nprint);

}

#endif
