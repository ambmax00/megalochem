#ifndef INTS_AOFACTORY_H
#define INTS_AOFACTORY_H

#include <dbcsr_tensor.hpp>
#include "desc/molecule.h"
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
	
	struct impl;
	impl* pimpl;
	
public:

	aofactory& op(std::string op);
	aofactory& dim(std::string dim); 
	aofactory(desc::molecule& mol, dbcsr::world& w);
	~aofactory();
	
	dbcsr::stensor<2,double> compute_2(std::vector<int> map1, std::vector<int> map2);
	dbcsr::stensor<3,double> compute_3(std::vector<int> map1, std::vector<int> map2);
	dbcsr::stensor<4,double> compute_4(std::vector<int> map1, std::vector<int> map2);
	dbcsr::smatrix<double> compute();
		
}; // end class aofactory

} // end namespace ints
			
		
#endif
