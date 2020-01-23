#include "hf/hfmod.h"
#include "hf/fock_builder.h"
#include "hf/hfdefaults.h"
#include "ints/aofactory.h"
#include "math/linalg/orthogonalizer.h"
#include "math/tensor/dbcsr_conversions.hpp"

namespace hf {
	
hfmod::hfmod(desc::molecule& mol, desc::options& opt, MPI_Comm comm) 
	: m_mol(mol), 
	  m_opt(opt), 
	  LOG(m_opt.get<int>("print_level", HF_PRINT_LEVEL)),
	  m_guess(m_opt.get<std::string>("guess", HF_GUESS)),
	  m_max_iter(m_opt.get<int>("max_iter", HF_MAX_ITER)),
	  m_scf_threshold(m_opt.get<double>("scf_thresh", HF_SCF_THRESH)),
	  m_restricted(m_opt.get<bool>("restricted", true)),
	  m_comm(comm),
	  m_nobeta(false)
{
	if ((m_mol.nocc_alpha() != m_mol.nocc_beta()) && m_restricted) 
		throw std::runtime_error("Cannot do restricted calculations for this multiplicity.");
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
	m_t_bb = dbcsr::tensor<2>({.name = "k_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	
	m_p_bb_A = dbcsr::tensor<2>({.name = "p_bb_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	m_p_bb_B = dbcsr::tensor<2>({.name = "p_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	
	m_c_bm_A = dbcsr::tensor<2>({.name = "c_bm_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bm_A});
	m_c_bm_B = dbcsr::tensor<2>({.name = "c_bm_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bm_B});
	
	m_f_bb_A = dbcsr::tensor<2>({.name = "f_bb_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	m_f_bb_B = dbcsr::tensor<2>({.name = "f_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	
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
	m_t_bb = int_engine.compute<2>(
		{.op = "kinetic", .bas = "bb", 
		 .name = "t_bb", .map1 = {0}, .map2 = {1}
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
	m_core_bb = m_v_bb + m_t_bb;
	
	std::cout << "Overlap: " << std::endl;
	dbcsr::print(m_s_bb);
	std::cout << "Kinetic: " << std::endl;
	dbcsr::print(m_t_bb);
	std::cout << "Nuclear: " << std::endl;
	dbcsr::print(m_v_bb);
	std::cout << "Core: " << std::endl;
	dbcsr::print(m_core_bb);
	
	grid.destroy();
	
}
	
void hfmod::calc_scf_energy() {
	
	double e1 = dbcsr::dot<2>(m_core_bb, m_p_bb_A);
	double e2 = dbcsr::dot<2>(m_f_bb_A, m_p_bb_A);
	
	std::cout << "CORE: " << std::endl;
	dbcsr::print(m_core_bb);
	
	std::cout << "FOCK:" << std::endl;
	dbcsr::print(m_f_bb_A);
	
	std::cout << "Density" << std::endl;
	dbcsr::print(m_p_bb_A);
	
	std::cout << "E1 " << e1 << std::endl;
	std::cout << "E2 " << e2 << std::endl;
	
	m_scf_energy = 0.5 * (2.0 * (e1 + e2));
	
}

void hfmod::compute() {
	
	// first, get one-electron integrals...
	std::cout << "One electron..." << std::endl;
	one_electron();
	
	// form the guess
	std::cout << "Guessing..." << std::endl;
	compute_guess();
	
	// Now enter loop
	int iter = 0;
	bool converged = false;
	
	fockbuilder fbuilder(m_mol, m_opt, m_comm);
	
	/*
	fbuilder.compute({.core = m_core_bb, .c_A = m_c_bm_A, .p_A = m_p_bb_A});
	
	m_f_bb_A = std::move(fbuilder.m_f_bb_A);
	
	std::cout << "FOCK MATRIX: " << std::endl;
	dbcsr::print(m_f_bb_A);

	double en = dbcsr::dot(m_core_bb, m_f_bb_A);

	std::cout << "EN: " << en << std::endl;
	*/

	m_max_iter = 10;

	while (!converged && iter < m_max_iter ) {
		
		LOG.os<>("Iteration: ", iter, '\n');
		
		// form fock matrix
		fbuilder.compute({.core = m_core_bb, .c_A = m_c_bm_A, .p_A = m_p_bb_A});
		std::swap(m_f_bb_A, fbuilder.fock_alpha());
		
		calc_scf_energy();
		
		std::cout << "ENERGY: " << m_scf_energy << std::endl;
		
		// compute error, do diis, compute energy
		
		// diag fock
		diag_fock();
		
		// loop
		 ++iter;
		
		
	} // end while
	
	std::cout << "FINAL ENERGY: " << m_scf_energy << std::endl;
		
}

} // end namespace
	
