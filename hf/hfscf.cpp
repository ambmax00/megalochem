#include "hf/hfmod.h"
#include "hf/fock_builder.h"
#include "hf/hfdefaults.h"
#include "ints/aofactory.h"
#include "math/linalg/orthogonalizer.h"
#include "math/linalg/symmetrize.h"
#include "math/solvers/diis.h"
#include "math/tensor/dbcsr_conversions.hpp"

namespace hf {
	
hfmod::hfmod(desc::molecule& mol, desc::options& opt, MPI_Comm comm) 
	: m_mol(mol), 
	  m_opt(opt), 
	  m_comm(comm),
	  LOG(m_comm,m_opt.get<int>("print", HF_PRINT_LEVEL)),
	  TIME(comm, "Hartree Fock", LOG.global_plev()),
	  m_guess(m_opt.get<std::string>("guess", HF_GUESS)),
	  m_max_iter(m_opt.get<int>("max_iter", HF_MAX_ITER)),
	  m_scf_threshold(m_opt.get<double>("scf_thresh", HF_SCF_THRESH)),
	  m_diis(m_opt.get<bool>("diis", HF_SCF_DIIS)),
	  m_diis_beta(m_opt.get<bool>("diis_beta", HF_DIIS_BETA)),
	  m_scf_energy(0.0),
	  m_nobeta(false)
{
	
	m_restricted = (m_mol.nele_alpha() == m_mol.nele_beta()) ? true : false; 
	
	if (m_mol.nele_beta() == 0) m_nobeta = true;
	
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
	
	m_s_bb = dbcsr::make_stensor<2>({.name = "s_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	m_x_bb = dbcsr::make_stensor<2>({.name = "x_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	m_v_bb = dbcsr::make_stensor<2>({.name = "v_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	m_t_bb = dbcsr::make_stensor<2>({.name = "k_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	m_core_bb = dbcsr::make_stensor<2>({.name = "core_bb", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	
	m_p_bb_A = dbcsr::make_stensor<2>({.name = "p_bb_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	if (!m_restricted && !m_nobeta) 
		m_p_bb_B = dbcsr::make_stensor<2>({.name = "p_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	
	m_c_bm_A = dbcsr::make_stensor<2>({.name = "c_bm_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bm_A});
	if (m_p_bb_B) 
		m_c_bm_B = dbcsr::make_stensor<2>({.name = "c_bm_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bm_B});
	
	m_f_bb_A = dbcsr::make_stensor<2>({.name = "f_bb_A", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
	if (m_p_bb_B) 
		m_f_bb_B = dbcsr::make_stensor<2>({.name = "f_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = bb});
		
	LOG.os<>("Options: \n");
	LOG.os<>("print ", LOG.global_plev(), '\n');
	LOG.os<>("m_guess ", m_guess, '\n');
	
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
	
	LOG.os<>("Forming one-electron integrals...");
	auto& TIME_1e = TIME.sub("One-Electron Integrals");
	
	TIME_1e.start();
	
	ints::aofactory int_engine(m_mol, m_comm);
	
	// overlap			 
	m_s_bb = int_engine.compute<2>(
			{.op = "overlap", .bas = "bb",
			 .name = "S_bb", .map1 = {0}, .map2 = {1}
			 });		 
	
	//m_s_bb = math::symmetrize(m_s_bb, "m_s_bb");
	
	//kinetic
	m_t_bb = int_engine.compute<2>(
		{.op = "kinetic", .bas = "bb", 
		 .name = "t_bb", .map1 = {0}, .map2 = {1}
		});
	
	//m_t_bb = math::symmetrize(m_t_bb, "m_t_bb");
	
	// nuclear
	m_v_bb = int_engine.compute<2>(
		{.op = "nuclear", .bas = "bb",
		 .name = "v_bb", .map1 = {0}, .map2 = {1}
		});
		
	//m_v_bb = math::symmetrize(m_v_bb, "m_v_bb");
	
	// get X
	math::orthgon og(m_s_bb);
	og.compute();
	m_x_bb = og.result("x_bb");
	
	//std::cout << "H1" << std::endl;
	dbcsr::copy<2>({.t_in = *m_v_bb, .t_out = *m_core_bb});
	
	//std::cout << "H2" << std::endl;
	dbcsr::copy<2>({.t_in = *m_t_bb, .t_out = *m_core_bb, .sum = true});
	
	//std::cout << "H3" << std::endl;
	
	TIME_1e.finish();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_x_bb);
		LOG.os<>('\n');
		dbcsr::print(*m_s_bb);
		LOG.os<>('\n');
		dbcsr::print(*m_t_bb);
		LOG.os<>('\n');
		dbcsr::print(*m_v_bb);
		LOG.os<>('\n');
		dbcsr::print(*m_core_bb);
	}
	
	LOG.os<>("Done with 1 electron integrals.");
	
}
	
void hfmod::compute_scf_energy() {		
	
	double e1A, e2A, e1B = 0.0, e2B = 0.0;
	
	e1A = dbcsr::dot(*m_core_bb, *m_p_bb_A);
	e2A = dbcsr::dot(*m_f_bb_A, *m_p_bb_A);
	
	if (!m_restricted && !m_nobeta) {
		e1B = dbcsr::dot(*m_core_bb, *m_p_bb_B);
		e2B = dbcsr::dot(*m_f_bb_B, *m_p_bb_B);
	} 
	
	if (m_restricted) {
		m_scf_energy = 0.5 * (2.0 * (e1A + e2A));
	} else {
		m_scf_energy = 0.5 * ((e1A + e2A) + (e1B + e2B));
	}
	
}

dbcsr::tensor<2> hfmod::compute_errmat(dbcsr::tensor<2>& F_x, dbcsr::tensor<2>& P_x, dbcsr::tensor<2>& S, std::string x) {
	
	//create
	dbcsr::pgrid<2> grid({.comm = m_comm});
	dbcsr::tensor<2> e_1({.name = "e_1_"+x, .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = F_x.blk_size()});
	dbcsr::tensor<2> e_2({.name = "e_2_"+x, .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = F_x.blk_size()});
	
	//DO E = FPS - SPF 
	dbcsr::einsum<2,2,2>({.x = "ij, jk -> ik", .t1 = F_x, .t2 = P_x, .t3 = e_1}); // e1 = F * P
	dbcsr::einsum<2,2,2>({.x = "ij, jk -> ik", .t1 = e_1, .t2 = S, .t3 = e_1}); // e1 = e1 * S
	dbcsr::einsum<2,2,2>({.x = "ij, jk -> ik", .t1 = S, .t2 = P_x, .t3 = e_2, .alpha = -1.0}); // e2 =  - S *P
	dbcsr::einsum<2,2,2>({.x = "ij, jk -> ik", .t1 = e_2, .t2 = F_x, .t3 = e_1, .beta = 1.0}); // e1 = e1 + e2 * F
	
	e_2.destroy();
	grid.destroy();
	
	return std::move(e_1);
	
}

void hfmod::compute() {
	
	TIME.start();
	
	if (LOG.global_plev() >= 0) 
		LOG.banner<>("HARTREE FOCK", 50, '*');
	
	// first, get one-electron integrals...
	one_electron();
	
	// form the guess
	compute_guess();
	
	compute_nucrep();
	
	// Now enter loop
	int iter = 0;
	bool converged = false;
	
	int dmax = m_opt.get<int>("diis_max_vecs", HF_DIIS_MAX_VECS);
	int dmin = m_opt.get<int>("diis_min_vecs", HF_DIIS_MIN_VECS);
	int dstart = m_opt.get<int>("diis_start", HF_DIIS_START);
	
	fockbuilder fbuilder(m_mol, m_opt, m_comm, LOG.global_plev(), TIME);
	
	math::diis_helper<2> diis_A(m_comm,dstart, dmin, dmax, (LOG.global_plev() >= 2) ? true : false );
	math::diis_helper<2> diis_B(m_comm,dstart, dmin, dmax, (LOG.global_plev() >= 2) ? true : false );

	m_max_iter = 10;
	
	// ERROR MATRICES
	dbcsr::tensor<2> e_A;
	dbcsr::tensor<2> e_B;
	
	double rms_A = 10;
	double rms_B = 10;
	
	// ---------> print info here <-------
	
	LOG.os<>("Iteration Nr\t", "Energy (Ht)\t", "Error (Ht)\t", "RMS alpha(Ht)\t", "RMS beta(Ht)\n");

	while (true) {
		
		// form fock matrix
		
		bool SAD_iter = ((iter == 0) && (m_guess == "SAD")) ? true : false;
		
		fbuilder.compute({.core = m_core_bb, .c_A = m_c_bm_A, .p_A = m_p_bb_A, 
				.c_B = m_c_bm_B, .p_B = m_p_bb_B, .SAD_iter = SAD_iter});
		
		m_f_bb_A = fbuilder.fock_alpha();
		m_f_bb_B = fbuilder.fock_beta();
		
		// compute error, do diis, compute energy
		
		e_A = compute_errmat(*m_f_bb_A, *m_p_bb_A, *m_s_bb, "A");
		if (!m_restricted && !m_nobeta)
			e_B = compute_errmat(*m_f_bb_B, *m_p_bb_B, *m_s_bb, "B");
		
		double old_energy = m_scf_energy;
		compute_scf_energy();
		
		rms_A = dbcsr::RMS(e_A);
		if (m_restricted || m_nobeta) {
			rms_B = rms_A;
		} else {
			rms_B = dbcsr::RMS(e_B);
		}
		
		LOG.os<>("UHF@", iter, '\t', m_scf_energy + m_nuc_energy, '\t', old_energy - m_scf_energy, '\t', rms_A, '\t', rms_B, '\n');
		
		if (rms_A < m_scf_threshold && rms_B < m_scf_threshold) break;
		if (iter > HF_MAX_ITER) break;
		
		if (m_diis) {
			diis_A.compute_extrapolation_parameters(*m_f_bb_A, e_A, iter);
			diis_A.extrapolate(*m_f_bb_A, iter);
			if (!m_restricted && !m_nobeta) {
				
				if (m_diis_beta) {	
					// separate diis optimization for beta
					diis_B.compute_extrapolation_parameters(*m_f_bb_B, e_B, iter);
					diis_B.extrapolate(*m_f_bb_B, iter);
					
				} else {
					
					// impose the same coefficients for both alpha and beta
					auto coeffA = diis_A.coeffs();
					diis_B.compute_extrapolation_parameters(*m_f_bb_B, e_B, iter);
					diis_B.extrapolate(*m_f_bb_B,coeffA,iter);
					
				}
			}
		}
		
		// diag fock
		diag_fock();
		
		// loop
		 ++iter;
		
		
	} // end while
	
	e_A.destroy();
	e_B.destroy();
	
	if (m_nobeta) {
		// take care of density
		dbcsr::pgrid<2> grid({.comm = m_comm});
		m_p_bb_B = dbcsr::make_stensor<2>({.name = "p_bb_B", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = m_p_bb_A->blk_size()});
		m_p_bb_B->set(0.0);
	}
	
	LOG.os<>("Done with SCF cycle. Took ", iter, " iterations.\n");
	LOG.os<>("Final SCF energy: ", m_scf_energy, '\n');
	LOG.os<>("Total energy: ", m_scf_energy + m_nuc_energy, '\n');
	
	TIME.finish();
	TIME.print_info();
	
		
}

} // end namespace
	
