#ifndef INTS_SCREENING_H
#define INTS_SCREENING_H

#include <mpi.h>
#include <string>

#include "desc/molecule.h"
#include "utils/pool.h"

#include <libint2.hpp>
#include <Eigen/Core>
#include <map>

namespace ints {

// Matrix which holds info about screening of blocks and shells
class Zmat {
private:

	//???
	MPI_Comm m_comm;
	desc::molecule& m_mol;
	util::ShrPool<libint2::Engine>& m_eng;
	std::string m_method;
	
	// first approach: array storing cluster block norms (row major format)
	std::unordered_map<int64_t, double> m_blkmap;
	
	// second: matrix storing values, too
	// array of blocks! Eigen::MatrixXd m_matvalues;
	
	// methods
	void compute_schwarz();
	// void compute_QQR
	
public:

	Zmat(MPI_Comm comm, desc::molecule& mol, util::ShrPool<libint2::Engine>& engine, std::string method)
		: m_comm(comm), m_mol(mol), m_eng(engine), m_method(method) {}
		

	void compute();
	
};


} // end namespace

#endif
