#ifndef INTS_AOFACTORY_H
#define INTS_AOFACTORY_H

#include <dbcsr.hpp>
#include "desc/molecule.h"
#include "utils/params.hpp"
#include "utils/ppdirs.h"
#include <string>
#include <map>
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
	required<std::string,val> c_op;
	required<std::string,val> c_dim;
	required<vec<int>,val> c_map1;
	required<vec<int>,val> c_map2;
	
public:

	inline aofactory& op(required<std::string,val> i_op) { c_op = i_op; return *this; }
	inline aofactory& dim(required<std::string,val> i_dim) { c_dim = i_dim; return *this; }
	inline aofactory& map1(required<vec<int>,val> i_map1) { c_map1 = i_map1; return *this; }
	inline aofactory& map2(required<vec<int>,val> i_map2) { c_map2 = i_map2; return *this; }

	aofactory(desc::molecule& mol, MPI_Comm c) : m_mol(mol), m_comm(c)  {};
	
	~aofactory() {};
	
	template <int N>
	dbcsr::stensor<N,double> compute();

	dbcsr::stensor<2> invert(dbcsr::stensor<2>& in, int order);
		
}; // end class aofactory

} // end namespace ints
			
		
#endif
