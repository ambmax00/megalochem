#ifndef MEGALOCHEM_HPP
#define MEGALOCHEM_HPP

#inlcude <mpi.h>
#include <dbcsr_common.hpp>

namespace megalochem {

// initilizes dbcsr, and sets logging in LOG class which is global	
void init(MPI_Comm, std::optional<std::string> log_filename) {
	
	dbcsr::init(Comm, handle);
	
}

void finalize() {
	
	//dbcsr::finalize();
	
}

class world {
private:

	MPI_Comm m_comm;
	int m_mpirank, m_mpisize;
	
	dbcsr::cart 
