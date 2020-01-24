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
	
	desc::molecule& m_mol;
	MPI_Comm m_comm;
	
	
	
public:

	aofactory(desc::molecule& mol, MPI_Comm c) : m_mol(mol), m_comm(c)  {};
	
	~aofactory() {};
	
	struct aofac_params {
		required<std::string,val> 	op;
		required<std::string,val> 	bas;
		required<std::string,val>	name;
		required<vec<int>,val>		map1;
		required<vec<int>,val>		map2;
	};
	
	template <int N>
	dbcsr::tensor<N,double> compute(aofac_params&& p);
		
}; // end class aofactory

} // end namespace ints
			
		
#endif
