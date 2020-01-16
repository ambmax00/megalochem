#include "hf/hfmod.h"
#include "ints/aofactory.h"
#include "math/linalg/orthogonalizer.h"
#include "math/tensor/dbcsr_conversions.hpp"

namespace hf {
	
hfmod::hfmod(desc::molecule& mol, desc::options& opt, MPI_Comm comm) 
	: m_mol(mol), 
	  m_opt(opt), 
	  m_comm(comm),
	  m_restricted(false),
	  m_nobeta(false)
{
	if (m_mol.nocc_alpha() == m_mol.nocc_beta()) m_restricted = true;
	if (m_mol.nocc_beta() == 0) m_nobeta = true;
	
	dbcsr::pgrid<2> grid({.comm = m_comm});
	
	auto b = m_mol.dims().b();
	auto oA = m_mol.dims().oa();
	auto oB = m_mol.dims().ob();
	auto vA = m_mol.dims().va();
	auto vB = m_mol.dims().vb();
	
	vec<int> mA = oA;
	mA.insert(mA.end(), vA.begin(), vA.end());
	
	vec<int> mB = oB;
	mB.insert(mB.end(), vB.begin(), vB.end());
	
	vec<vec<int>> bb = {b,b};
	vec<vec<int>> bm_A = {b, mA};
	vec<vec<int>> bm_B = {b, mB};
	
	m_s_bb = dbcsr::tensor<2>({.name = "s_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	m_x_bb = dbcsr::tensor<2>({.name = "x_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	m_v_bb = dbcsr::tensor<2>({.name = "v_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	m_k_bb = dbcsr::tensor<2>({.name = "k_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	
	m_p_bb_A = dbcsr::tensor<2>({.name = "p_bb_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	m_p_bb_B = dbcsr::tensor<2>({.name = "p_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	
	m_c_bm_A = dbcsr::tensor<2>({.name = "c_bm_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bm_A});
	m_c_bm_B = dbcsr::tensor<2>({.name = "c_bm_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bm_B});
	
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
	m_s_bb = int_engine.compute<2>(
			{.op = "overlap", .bas = "bb",
			 .name = "S_bb", .map1 = {0}, .map2 = {1}
			 }
	);		 
	
	//kinetic
	m_k_bb = int_engine.compute<2>(
		{.op = "kinetic", .bas = "bb", 
		 .name = "k_bb", .map1 = {0}, .map2 = {1}
		}
	);
	
	// nuclear
	m_v_bb = int_engine.compute<2>(
		{.op = "nuclear", .bas = "bb",
		 .name = "v_bb", .map1 = {0}, .map2 = {1}
		}
	);
	
	// get X
	math::orthgon og(m_s_bb);
	og.compute();
	auto result = og.result();
	
	dbcsr::pgrid<2> grid({.comm = m_comm});
	
	m_x_bb = dbcsr::eigen_to_tensor(result, "x_bb", grid, {0}, {1}, m_s_bb.blk_size()); // <== BLOCK PRECISION!!
	
	std::cout << "TRANS MATRIX: " << std::endl;
	dbcsr::print(m_x_bb);
	
	std::cout << "Adding..." << std::endl;
	m_core_bb = m_v_bb + m_k_bb;
	
	std::cout << "Overlap: " << std::endl;
	dbcsr::print(m_s_bb);
	std::cout << "Kinetic: " << std::endl;
	dbcsr::print(m_k_bb);
	std::cout << "Nuclear: " << std::endl;
	dbcsr::print(m_v_bb);
	std::cout << "Core: " << std::endl;
	dbcsr::print(m_core_bb);
	
	grid.destroy();
	
}
	

void hfmod::compute() {
	
	// first, get one-electron integrals...
	std::cout << "One electron..." << std::endl;
	one_electron();
	
	// form the guess
	std::cout << "Guessing..." << std::endl;
	compute_guess();
	
	
	
		
}

} // end namespace
	
