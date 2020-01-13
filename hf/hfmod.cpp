#include "hf/hfmod.h"
#include "ints/aofactory.h"
#include "math/linalg/orthogonalizer.h"
#include "math/tensor/dbcsr_conversions.hpp"

namespace hf {
	
hfmod::hfmod(desc::molecule& mol, desc::options& opt, MPI_Comm comm) 
	: m_mol(mol), 
	  m_opt(opt), 
	  m_comm(comm)
{
	if (m_mol.nocc_alpha() == m_mol.nocc_beta()) m_restricted = true;
}

void hfmod::compute_nucrep() {
	
	m_nuc_energy = 0.0;
	
	auto atoms = m_mol.atoms();
	
	for (int i = 0; i != atoms.size(); ++i) {
		for (int j = i+1; j < atoms.size(); ++j) {
			
			int Zi = atoms[i].atomic_number;
			int Zj = atoms[j].atomic_number;
			
			double dx = atoms[i].x - atoms[j].x;
			double dy = atoms[i].y - atoms[j].y;
			double dz = atoms[i].z - atoms[j].z;
			
			double R = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));
			
			m_nuc_energy += (Zi*Zj)/R;
			
		}
	}
	
	//LOG.os<>(0,"Nuclear Repulsion Energy: ", nuc_energy, '\n');
	
}

void hfmod::one_electron() {
	
	ints::aofactory int_engine(m_mol, m_comm);
	
	// overlap			 
	m_s_bb = std::make_shared<dbcsr::tensor<2,double>>(
		int_engine.compute<2>(
			{.op = "overlap", .bas = "bb",
			 .name = "S_bb", .map1 = {0}, .map2 = {1}
			 }
	));		 
	
	//kinetic
	m_k_bb = std::make_shared<dbcsr::tensor<2,double>>(
		int_engine.compute<2>(
		{.op = "kinetic", .bas = "bb", 
		 .name = "k_bb", .map1 = {0}, .map2 = {1}
		}
	));
	
	// nuclear
	m_v_bb = std::make_shared<dbcsr::tensor<2,double>>(
		int_engine.compute<2>(
		{.op = "nuclear", .bas = "bb",
		 .name = "v_bb", .map1 = {0}, .map2 = {1}
		}
	));
	
	// get X
	math::orthgon og(*m_s_bb);
	og.compute();
	auto result = og.result();
	
	dbcsr::pgrid<2> grid({.comm = MPI_COMM_WORLD});
	
	m_x_bb = std::make_shared<dbcsr::tensor<2,double>>
	(
		dbcsr::eigen_to_tensor
		(result, "x_bb", grid, {0}, {1}, m_s_bb->blk_size()) // <== BLOCK PRECISION!!
	);
	
	grid.destroy();
	
}
	

void hfmod::compute() {
	
	// first, get one-electron integrals...
	one_electron();
	
	*m_core_bb = *m_v_bb + *m_k_bb;
	
	// form the guess
	compute_guess();
	
		
}

} // end namespace
	
