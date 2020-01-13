#include "hf/hfmod.h"
#include "math/tensor/dbcsr_conversions.hpp"

#include <Eigen/Core>

namespace hf {
	
void hfmod::compute_guess() {
	
	// form density and coefficients by diagonalizing the core matrix
	*m_f_bb_A = *m_core_bb;
	if (!m_restricted) *m_f_bb_B = *m_core_bb;
	
	diag_fock();
	
}

}
