#include "hf/hfmod.h"
#include "math/tensor/dbcsr_conversions.hpp"
#include "math/linalg/symmetrize.h"

#include <Eigen/Core>

namespace hf {
	
void hfmod::compute_guess() {
	
	if (m_guess == "core") {
		// form density and coefficients by diagonalizing the core matrix
		m_f_bb_A = m_core_bb;
		
		if (!m_restricted) m_f_bb_B = m_core_bb;
		
		std::cout << "Diagonalizing..." << std::endl;
		diag_fock();
	
	} else if (m_guess == "SAD") {
		
		// divide up comm 
	
	} else {
		
		throw std::runtime_error("Unknown option for guess: "+m_guess);
		
	}
	
	
	
}

}
