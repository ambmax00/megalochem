#ifndef FOCK_DEFAULTS_H
#define FOCK_DEFAULTS_H

#include <string>

namespace fock {
	
	inline const int FOCK_PRINT_LEVEL = 0;
	inline const int FOCK_NBATCHES_X = 5;
	inline const int FOCK_NBATCHES_B = 5;
	
	inline const std::string FOCK_BUILD_J = "exact";
	inline const std::string FOCK_BUILD_K = "exact";
	inline const std::string FOCK_METRIC = "coulomb";
	inline const std::string FOCK_ERIS = "core";
	inline const std::string FOCK_INTERMEDS = "core";
	
} // end namespace

#endif
