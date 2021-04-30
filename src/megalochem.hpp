#ifndef MEGALOCHEM_HPP
#define MEGALOCHEM_HPP

#include <filesystem>

#include "extern/scalapack.hpp"
#include "utils/mpi_time.hpp"
#include <dbcsr_common.hpp>

namespace megalochem {

// initilizes dbcsr, and sets logging in LOG class which is global	
inline void init(MPI_Comm comm, std::string workdir, 
	std::optional<std::string> log_filename = std::nullopt) {
	
	if (!std::filesystem::exists(workdir))
		throw std::runtime_error("Working directory does not exist.");
	
	//set working directory
	std::filesystem::current_path(workdir);
	
	dbcsr::init(comm, nullptr);
	
}

inline void finalize() {
	
	//dbcsr::finalize();
	
}	

class world {
private:

	MPI_Comm m_comm;
	int m_mpirank, m_mpisize;
	
	dbcsr::cart m_dbcsr_cart;
	scalapack::grid m_scalapack_grid;
	
public:

	world(MPI_Comm comm) : 
		m_comm(comm), 
		m_dbcsr_cart(comm),
		m_scalapack_grid(comm, 'A', m_dbcsr_cart.nprow(), m_dbcsr_cart.npcol())  
	{
		MPI_Comm_size(comm, &m_mpisize);
		MPI_Comm_rank(comm, &m_mpirank);
	}
	
	world(const world& w) = default;
	
	dbcsr::cart dbcsr_grid() {
		return m_dbcsr_cart;
	}
	
	scalapack::grid scalapack_grid() {
		return m_scalapack_grid;
	}
	
	int rank() const { return m_mpirank; }
	int size() const { return m_mpisize; }
	MPI_Comm comm() const { return m_comm; }
	
	~world() {}
	
	void free() {
		m_dbcsr_cart.free();
		m_scalapack_grid.free();
	}
	
};

}

#endif
