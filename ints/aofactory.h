#ifndef INTS_AOFACTORY_H
#define INTS_AOFACTORY_H

#include "math/tensor/dbcsr.hpp"
#include "desc/molecule.h"
#include <string>
#include <mpi.h>

/* Loads the AO integrals
 * Op: Operator (Coulom, kinetic, ...)
 * bis: tensor space, e.g. bb is a matrix in the AO basis, xbb are the 3c2e integrals etc...
*/

namespace ints {
	
class aofactory {
	
private:
	
	desc::molecule m_mol;
	MPI_Comm m_comm;
	
public:

	aofactory(desc::molecule mol, MPI_Comm c) : m_mol(mol), m_comm(c)  {};
	
	~aofactory() {};
	
	template <int N>
	dbcsr::tensor<N,double> compute(std::string Op, std::string bis);
		
}; // end class aofactory

} // end namespace ints
			
		
#endif
