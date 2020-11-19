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

enum class metric {
	invalid,
	coulomb,
	erfc_coulomb,
	qr_fit,
	pari
};

inline metric str_to_metric(std::string m) {
	if (m == "coulomb") return metric::coulomb;
	if (m == "erfc_coulomb") return metric::erfc_coulomb;
	if (m == "qr_fit") return metric::qr_fit;
	if (m == "pari") return metric::pari;
	return metric::invalid;
}
	
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
	
	//dbcsr::shared_matrix<double> ao_diag_mnmn();
	//dbcsr::shared_matrix<double> ao_diag_mmnn();
	
	dbcsr::shared_matrix<double> ao_2c2e(metric m);
	dbcsr::shared_matrix<double> ao_auxoverlap();
	
	dbcsr::shared_matrix<double> ao_schwarz();
	dbcsr::shared_matrix<double> ao_3cschwarz();
#if 0
	std::array<dbcsr::shared_matrix<double>,3> 
		ao_emultipole(std::array<int,3> O = {0,0,0});
#endif	

	void ao_3c1e_ovlp_setup();

	void ao_3c2e_setup(metric m);

	void ao_eri_setup(metric m);
	
	void ao_3c_fill(dbcsr::shared_tensor<3,double>& t_in);
	
	void ao_3c_fill(dbcsr::shared_tensor<3,double>& t_in, vec<vec<int>>& blkbounds, 
		std::shared_ptr<screener> scr);
	
	void ao_3c_fill_idx(dbcsr::shared_tensor<3,double>& t_in, arrvec<int,3>& idx, 
		std::shared_ptr<screener> scr);
		
	void ao_4c_fill(dbcsr::shared_tensor<4,double>& t_in, vec<vec<int>>& blkbounds, 
		std::shared_ptr<screener> scr);
		
	std::function<void(dbcsr::shared_tensor<3,double>&,vec<vec<int>>&)>
	get_generator(std::shared_ptr<screener> s_scr);
	
	desc::smolecule mol();
	
		
}; // end class aofactory

} // end namespace ints
			
		
#endif
