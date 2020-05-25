#ifndef PIV_CD_H
#define PIV_CD_H

#include <dbcsr_conversions.hpp>

namespace math {
	
class piv_cholesky_decomposition {
private:

	dbcsr::smat_d m_mat_in;

public:

	piv_cholesky_decomposition(dbcsr::smat_d mat_in) : m_mat_in(mat_in) {}
	~piv_cholesky_decomposition() {}
	
	void compute(int nb = 5);
	
	
};
	
} // end namespace

#endif
