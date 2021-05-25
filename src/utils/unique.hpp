#ifndef UTIL_UNIQUE_DIR_H
#define UTIL_UNIQUE_DIR_H

#include <filesystem>
#include <mpi.h>

// functions to obtain unique file names and directories

namespace util {
	
inline std::string unique(std::string prefix, std::string suffix, MPI_Comm comm) {
	
	int rank = -1;
	MPI_Comm_rank(comm, &rank);
	
	long long int id = 0;
	
	if (rank == 0) {
		for (size_t ii = 0; ii <= std::numeric_limits<size_t>::max(); ++ii) {
			id = ii;
			std::string fname = prefix + std::to_string(id) + suffix;
			if (!std::filesystem::exists(fname)) break;
		}
	}
	
	MPI_Bcast(&id, 1, MPI_LONG_LONG, 0, comm);
	
	std::string uname;
	uname = prefix + std::to_string(id) + suffix;
	
	return uname;
	
}
	
} // end namespace

#endif
