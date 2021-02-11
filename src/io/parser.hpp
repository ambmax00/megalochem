#ifndef IO_READER_H
#define IO_READER_H

#include <string>
#include <stdexcept>

#include "desc/options.hpp"
#include "desc/molecule.hpp"
#include "utils/mpi_log.hpp"

#include "utils/json.hpp"

namespace filio {

desc::shared_molecule parse_molecule(nlohmann::json& j, MPI_Comm comm, int nprint);

desc::options parse_options(nlohmann::json& j, MPI_Comm comm, int nprint);

}

#endif
