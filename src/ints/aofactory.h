#ifndef INTS_AOFACTORY_H
#define INTS_AOFACTORY_H

#include <dbcsr_tensor.hpp>
#include <dbcsr_conversions.hpp>
#include "desc/molecule.h"
#include <string>
#include <limits>
#include <map>
#include <functional>
#include <mpi.h>

/* Loads the AO integrals
 * Op: Operator (Coulom, kinetic, ...)
 * bis: tensor space, e.g. bb is a matrix in the AO basis, xbb are the 3c2e integrals etc...
*/

using eigen_smat_f = std::shared_ptr<Eigen::MatrixXf>;

namespace ints {
	
struct global {
	static inline double precision = std::numeric_limits<double>::epsilon();
	static inline double omega = 0.1;
};
	
class screener;

class aofactory {
private:
	
	struct impl;
	impl* pimpl;
	
public:

	aofactory(desc::smolecule mol, dbcsr::world& w);
	~aofactory();
		
	dbcsr::shared_matrix<double> ao_overlap();
	dbcsr::shared_matrix<double> ao_kinetic();
	dbcsr::shared_matrix<double> ao_nuclear();
	
	dbcsr::shared_matrix<double> ao_3coverlap(std::string metric);
	
	dbcsr::shared_matrix<double> ao_schwarz(std::string metric);
	dbcsr::shared_matrix<double> ao_3cschwarz(std::string metric);
	
	void ao_3c2e_setup(std::string metric);
	
	void ao_eri_setup(std::string metric);
	
	dbcsr::shared_tensor<3,double> ao_3c2e_setup_tensor(
		dbcsr::shared_pgrid<3> spgrid, vec<int> map1, vec<int> map2);
		
	dbcsr::shared_tensor<4,double> ao_eri_setup_tensor(
		dbcsr::shared_pgrid<4> spgrid, vec<int> map1, vec<int> map2);
	
	void ao_3c2e_fill(dbcsr::shared_tensor<3,double>& t_in, vec<vec<int>>& blkbounds, 
		std::shared_ptr<screener> scr, bool sym = false);
		
	void ao_3c2e_fill(dbcsr::shared_tensor<3,double>& t_in, arrvec<int,3>& idx, 
		std::shared_ptr<screener> scr);
		
	void ao_eri_fill(dbcsr::shared_tensor<4,double>& t_in, vec<vec<int>>& blkbounds, 
		std::shared_ptr<screener> scr, bool sym = false);
		
	std::function<void(dbcsr::shared_tensor<3,double>&,vec<vec<int>>&)>
	get_generator(std::shared_ptr<screener> s_scr);
	
	desc::smolecule mol();
	
		
}; // end class aofactory

} // end namespace ints
			
		
#endif
