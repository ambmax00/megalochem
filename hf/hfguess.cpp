#include "hf/hfmod.h"
#include "math/tensor/dbcsr_conversions.hpp"
#include "math/linalg/symmetrize.h"

#include <Eigen/Core>
#include <limits>

namespace hf {

// Taken from PSI4
static const std::vector<int> reference_S = {  0,
											   1,                                                                                           0,
											   1, 0,                                                                         1, 2, 3, 2, 1, 0,
											   1, 0,                                                                         1, 2, 3, 2, 1, 0,
											   1, 0,                                           1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,
											   1, 0,                                           1, 2, 5, 6, 5, 4, 3, 0, 1, 0, 1, 2, 3, 2, 1, 0,
											   1, 0, 1, 0, 3, 4, 5, 6, 7, 8, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0 };
                                           
// Also Taken from PSI4
static const std::vector<int> conf_orb = {0, 2, 10, 18, 36, 54, 86, 118};

void hfmod::compute_guess() {
	
	if (m_guess == "core") {
		
		LOG.os<>("Forming guess from core...\n");
		
		if (m_restricted) std::cout << "ITS RESTRICTED." << std::endl;
		
		// form density and coefficients by diagonalizing the core matrix
		*m_f_bb_A = *m_core_bb;
		
		std::cout << "HERE" << std::endl;
		
		if (!m_restricted && !m_nobeta) {
			*m_f_bb_B = *m_core_bb;
		}
		
		std::cout << "Entering." << std::endl;
		
		diag_fock();
	
	} else if (m_guess == "SAD") {
		
		// divide up comm <- for later
		
		// get mol info, check for atom types...
		std::vector<libint2::Atom> atypes;
		
		for (auto atom1 : m_mol.atoms()) {
			
			auto found = std::find_if(atypes.begin(), atypes.end(), 
				[&atom1](const libint2::Atom& atom2) {
					return atom2.atomic_number == atom1.atomic_number;
				});
				
			if (found == atypes.end()) {
				atypes.push_back(atom1);
			}
			
		}
		
		std::cout << "TYPES: " << std::endl;
		for (auto a : atypes) {
			std::cout << a.atomic_number << std::endl;
		}
		
		auto are_oncentre = [&](libint2::Atom& atom, libint2::Shell& shell) {
			double lim = std::numeric_limits<double>::epsilon();
			if ( fabs(atom.x - shell.O[0]) < lim &&
				 fabs(atom.y - shell.O[1]) < lim &&
				 fabs(atom.z - shell.O[2]) < lim ) return true;
				 
			return false;
		};
		
		for (int I = 0; I != atypes.size(); ++I) {
			
			auto atom = atypes[I];
			int Z = atom.atomic_number;
			
			//set up options
			desc::options at_opt(m_opt);
			at_opt.set<std::string>("guess", "core");
			
			int charge = 0;
			int mult = 0; // will be overwritten
			
			std::vector<libint2::Atom> atvec = {atom};
			std::vector<libint2::Shell> at_basis;
			optional<std::vector<libint2::Shell>,val> at_dfbasis;
			
			// find basis functions
			for (auto shell : m_mol.c_basis().libint_basis()) {
				if (are_oncentre(atom, shell)) at_basis.push_back(shell);
			}
			
			if (m_mol.c_dfbasis()) {
				std::cout << "INSIDE HERE." << std::endl;
				std::vector<libint2::Shell> temp;
				at_dfbasis = optional<std::vector<libint2::Shell>,val>(temp);
				for (auto shell : m_mol.c_dfbasis()->libint_basis()) {
					if (are_oncentre(atom, shell)) at_dfbasis->push_back(shell);
				}
			}
			
			desc::molecule at_mol({.atoms = atvec, .charge = charge,
				.mult = mult, .split = 20, .basis = at_basis, .dfbasis = at_dfbasis, .fractional = true});
				
			at_mol.print_info(LOG.global_plev());
			
			hf::hfmod atomic_hf(at_mol,at_opt,m_comm);
			
			atomic_hf.compute();
			
		}
				
		// set up HF calculations
	
	} else {
		
		throw std::runtime_error("Unknown option for guess: "+m_guess);
		
	}
	
	
	
}

}
