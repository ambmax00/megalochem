#ifndef INTS_AOFACTORY_H
#define INTS_AOFACTORY_H

#include <dbcsr_tensor.hpp>
#include <dbcsr_conversions.hpp>
#include "tensor/batchtensor.h"
#include "desc/molecule.h"
#include <string>
#include <limits>
#include <map>
#include <mpi.h>

/* Loads the AO integrals
 * Op: Operator (Coulom, kinetic, ...)
 * bis: tensor space, e.g. bb is a matrix in the AO basis, xbb are the 3c2e integrals etc...
*/

using eigen_smat_f = std::shared_ptr<Eigen::MatrixXf>;

namespace ints {
	
struct global {
	static inline double precision = std::numeric_limits<double>::epsilon();
	static inline double omega = 1;
};
	
class screener;

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
	
	dbcsr::smatrix<double> ao_3coverlap(std::string metric);
	
	dbcsr::stensor<3,double> ao_3c2e(vec<int> map1, vec<int> map2, std::string metric, screener* scr = nullptr);
	dbcsr::stensor<4,double> ao_eri(vec<int> map1, vec<int> map2);
	
	dbcsr::smatrix<double> ao_schwarz(std::string metric);
	dbcsr::smatrix<double> ao_3cschwarz(std::string metric);
	
	void ao_3c2e_setup(std::string metric);
	
	dbcsr::stensor<3,double> ao_3c2e_setup_tensor(vec<int> map1, vec<int> map2);
	
	void ao_3c2e_fill(dbcsr::stensor<3,double>& t_in, vec<vec<int>>& blkbounds, screener* scr);
	
	void ao_3c2e_fill(dbcsr::stensor<3,double>& t_in);
		
}; // end class aofactory

} // end namespace ints
			
		
#endif
