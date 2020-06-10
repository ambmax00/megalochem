#ifndef INTS_AOFACTORY_H
#define INTS_AOFACTORY_H

#include <dbcsr_tensor.hpp>
#include <dbcsr_conversions.hpp>
#include "desc/molecule.h"
#include <string>
#include <map>
#include <mpi.h>

/* Loads the AO integrals
 * Op: Operator (Coulom, kinetic, ...)
 * bis: tensor space, e.g. bb is a matrix in the AO basis, xbb are the 3c2e integrals etc...
*/

using eigen_smat_f = std::shared_ptr<Eigen::MatrixXf>;

namespace ints {

class aofactory {
private:
	
	struct impl;
	impl* pimpl;
	
public:

	desc::smolecule m_mol;

	aofactory(desc::smolecule mol, dbcsr::world& w);
	~aofactory();
	
	desc::smolecule mol();
	
	dbcsr::smatrix<double> ao_overlap();
	dbcsr::smatrix<double> ao_kinetic();
	dbcsr::smatrix<double> ao_nuclear();
	
	dbcsr::smatrix<double> ao_3coverlap();
	
	dbcsr::stensor<3,double> ao_3c2e(vec<int> map1, vec<int> map2, dbcsr::smatrix<double> scr = nullptr);
	dbcsr::stensor<4,double> ao_eri(vec<int> map1, vec<int> map2);
	
	dbcsr::smatrix<double> ao_schwarz();
	dbcsr::smatrix<double> ao_3cschwarz();
		
}; // end class aofactory

} // end namespace ints
			
		
#endif
