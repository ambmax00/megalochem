#include "hf/hfmod.h"
#include "ints/registry.h"
#include "hf/hfdefaults.h"
#include "math/tensor/dbcsr_conversions.hpp"
#include "math/linalg/symmetrize.h"
#include "math/other/scale.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
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

/*
void scale_huckel(dbcsr::tensor<2>& t, std::vector<double>& v) {
	
	//does t[ij] = (v[i] + v[j])*t[ij]
	
	dbcsr::iterator<2> iter(t);
	
	auto blkoff = t.blk_offset();
	
	while (iter.blocks_left()) {
		
		iter.next();
		
		auto idx = iter.idx();
		auto sizes = iter.sizes();
		bool found = false;
		
		auto blk = t.get_block({.idx = idx, .blk_size = sizes, .found = found});
		
		int off1 = blkoff[0][idx[0]];
		int off2 = blkoff[1][idx[1]];
		
		double prefac = 1.0;
		
		for (int j = 0; j != sizes[1]; ++j) {
			for (int i = 0; i != sizes[0]; ++i) {
				
				prefac = (off1 + i == off2 + j) ? 1.0 : HF_GWH_K;
				
				blk(i,j) *= 0.5 * prefac * (v[i + off1] + v[j + off2]);
		}}
		
		t.put_block({.idx = idx, .blk = blk});
		
	}
	
}	
*/				

void hfmod::compute_guess() {
	
	auto& t_guess = TIME.sub("Forming Guess");
	
	t_guess.start();
	
	if (m_guess == "core") {
		
		LOG.os<>("Forming guess from core...\n");
		
		//if (m_restricted) std::cout << "ITS RESTRICTED." << std::endl;
		
		// form density and coefficients by diagonalizing the core matrix
		dbcsr::copy<2>({.t_in = *m_core_bb, .t_out = *m_f_bb_A}); 
		
		
		// HEREEEEE !!!!!!!!!!!!!!
		
		
		//std::cout << "HERE" << std::endl;
		
		if (!m_restricted && !m_nobeta) {
			dbcsr::copy<2>({.t_in = *m_core_bb, .t_out = *m_f_bb_B}); 
		}
		
		//std::cout << "Entering." << std::endl;
		
		diag_fock();
	
	} else if (m_guess == "SAD") {
		
		LOG.os<>("Forming guess from SAD :'( ...\n");
		// divide up comm <- for later
		
		// get mol info, check for atom types...
		std::vector<libint2::Atom> atypes;
		
		for (auto atom1 : m_mol->atoms()) {
			
			auto found = std::find_if(atypes.begin(), atypes.end(), 
				[&atom1](const libint2::Atom& atom2) {
					return atom2.atomic_number == atom1.atomic_number;
				});
				
			if (found == atypes.end()) {
				atypes.push_back(atom1);
			}
			
		}
		
		//std::cout << "TYPES: " << std::endl;
		//for (auto a : atypes) {
		//	std::cout << a.atomic_number << std::endl;
		//}
		
		auto are_oncentre = [&](libint2::Atom& atom, libint2::Shell& shell) {
			double lim = std::numeric_limits<double>::epsilon();
			if ( fabs(atom.x - shell.O[0]) < lim &&
				 fabs(atom.y - shell.O[1]) < lim &&
				 fabs(atom.z - shell.O[2]) < lim ) return true;
				 
			return false;
		};
		
		std::map<int,Eigen::MatrixXd> locdensitymap;
		std::map<int,Eigen::MatrixXd> densitymap;
		
		int myrank = -1;
		int commsize = -1;
		
		MPI_Comm_rank(m_comm, &myrank);
		MPI_Comm_size(m_comm, &commsize);
		
		// divide it up
		std::vector<libint2::Atom> my_atypes;
		
		for (int I = 0; I != atypes.size(); ++I) {
			if (myrank == I % commsize) my_atypes.push_back(atypes[I]);
		}
		
		for (int i = 0; i != commsize; ++i) {
			if (myrank == i) {
				std::cout << "Rank " << myrank << std::endl;
				for (auto e : my_atypes) {
					std::cout << e.atomic_number << " ";
				} std::cout << std::endl;
			}
			MPI_Barrier(m_comm);
		}
		
		MPI_Comm mycomm;
		
		MPI_Comm_split(m_comm, myrank, myrank, &mycomm);
		
		for (int I = 0; I != my_atypes.size(); ++I) {
			
			auto atom = my_atypes[I];
			int Z = atom.atomic_number;
			
			//set up options
			desc::options at_opt(m_opt);
			
			int atprint = LOG.global_plev() - 1;
			at_opt.set<int>("print", atprint);
			at_opt.set<std::string>("guess", "core");
			
			int charge = 0;
			int mult = 0; // will be overwritten
			
			std::vector<libint2::Atom> atvec = {atom};
			std::vector<libint2::Shell> at_basis;
			optional<std::vector<libint2::Shell>,val> at_dfbasis;
			
			// find basis functions
			for (auto shell : m_mol->c_basis().libint_basis()) {
				if (are_oncentre(atom, shell)) at_basis.push_back(shell);
			}
			
			if (m_mol->c_dfbasis()) {
				//std::cout << "INSIDE HERE." << std::endl;
				std::vector<libint2::Shell> temp;
				at_dfbasis = optional<std::vector<libint2::Shell>,val>(temp);
				for (auto shell : m_mol->c_dfbasis()->libint_basis()) {
					if (are_oncentre(atom, shell)) at_dfbasis->push_back(shell);
				}
			}
			
			std::string name = "ATOM_rank" + std::to_string(myrank) + "_" + std::to_string(Z);
			
			desc::molecule at_mol({.name = name, .atoms = atvec, .charge = charge,
				.mult = mult, .split = 20, .basis = at_basis, .dfbasis = at_dfbasis, .fractional = true});
				
			auto at_smol = std::make_shared<desc::molecule>(std::move(at_mol));
				
			at_smol->print_info(mycomm,LOG.global_plev());
			
			hf::hfmod atomic_hf(at_smol,at_opt,mycomm);
			
			LOG(myrank).os<1>("Starting Atomic UHF for atom nr. ", I, " on rank ", myrank, '\n');
			atomic_hf.compute();
			LOG(myrank).os<1>("Done with Atomic UHF on rank ", myrank, "\n");
			
			auto pA = atomic_hf.p_bb_A();
			auto pB = atomic_hf.p_bb_B();
			
			//std::cout << "PA on rank: " << myrank << std::endl;
			//dbcsr::print(*pA);
			
			dbcsr::copy<2>({.t_in = *pB, .t_out = *pA, .sum = true, .move_data = true});
			pA->scale(0.5);
			
			locdensitymap[Z] = dbcsr::tensor_to_eigen(*pA);
			
			ints::registry INTS_REGISTRY;
			INTS_REGISTRY.clear(name);
			
		}
		
		MPI_Barrier(m_comm);
		
		//std::cout << "DISTRIBUTING" << std::endl;
		
		// distribute to other nodes
		for (int i = 0; i != commsize; ++i) {
			
			std::cout << "LOOP: " << i << std::endl;
			
			if (i == myrank) {
				
				int n = locdensitymap.size();
				
				//std::cout << "N: " << n << std::endl;
				
				MPI_Bcast(&n,1,MPI_INT,i,m_comm);
				
				for (auto& den : locdensitymap) {
					
					int Z = den.first;
					size_t size = den.second.size();
					
					//std::cout << "B1 " << Z << std::endl;
					MPI_Bcast(&Z,1,MPI_INT,i,m_comm);
					//std::cout << "B2 " << size << std::endl;
					MPI_Bcast(&size,1,MPI_UNSIGNED,i,m_comm);
					
					auto& mat = den.second;
					
					//std::cout << "B3" << std::endl;
					MPI_Bcast(mat.data(),size,MPI_DOUBLE,i,m_comm);
					
					densitymap[Z] = mat;
					
					MPI_Barrier(m_comm);
					
				 }
				
			} else {
				
				int n = -1;
				
				MPI_Bcast(&n,1,MPI_INT,i,m_comm);
				
				for (int ni = 0; ni != n; ++ni) {
					
					size_t size = 0;
					int Z = 0;
					
					MPI_Bcast(&Z,1,MPI_INT,i,m_comm);
					MPI_Bcast(&size,1,MPI_UNSIGNED,i,m_comm);
					
					//std::cout << "Other rank: " << Z << " " << size << std::endl;
					
					Eigen::MatrixXd mat((int)sqrt(size),(int)sqrt(size));
					
					MPI_Bcast(mat.data(),size,MPI_DOUBLE,i,m_comm);
					
					densitymap[Z] = mat;
					
					MPI_Barrier(m_comm);
					
				}
				
			}
			
			MPI_Barrier(m_comm);
			
		}
			
		
		size_t nbas = m_mol->c_basis().nbf();
		
		Eigen::MatrixXd ptot = Eigen::MatrixXd::Zero(nbas,nbas);
		auto csizes = m_mol->dims().b();
		int off = 0;
		int size = 0;
		
		for (int i = 0; i != m_mol->atoms().size(); ++i) {
			
			int Z = m_mol->atoms()[i].atomic_number;
			size = csizes[i];
			
			ptot.block(off,off,size,size) = densitymap[Z];
			
			off += size;
			
		}
		
		//std::cout << "PTOT: " << nbas << std::endl;
		//std::cout << ptot << std::endl;
		
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
		es.compute(ptot);
		
		if (es.info() != Eigen::Success) 
			throw std::runtime_error("Eigen hermitian eigensolver failed.");
		
		Eigen::VectorXd eigval = es.eigenvalues();
		Eigen::MatrixXd eigvec = es.eigenvectors();
		
		//std::cout << "Eigenvalues: " << std::endl;
		//std::cout << eigval << std::endl;
		
		//std::cout << "Eigenvectors: " << std::endl;
		//std::cout << eigvec << std::endl;
		
		std::vector<double> v(eigval.data(), eigval.data() + eigval.size());
		
		for (auto& x : v) {
			x = (x < std::numeric_limits<double>::epsilon()) ? 0 : sqrt(x);
		}
		
		dbcsr::pgrid<2> grid({.comm = m_comm});
		
		m_c_bm_A = (dbcsr::eigen_to_tensor(eigvec, "c_bm_A", grid, vec<int>{0}, vec<int>{1}, m_c_bm_A->blk_size())).get_stensor();
		math::scale({.t_in = *m_c_bm_A, .v_in = v});
		
		if (!m_restricted && !m_nobeta) {
			m_c_bm_B = (dbcsr::eigen_to_tensor(eigvec, "c_bm_B", grid, vec<int>{0}, vec<int>{1}, m_c_bm_B->blk_size())).get_stensor();
			math::scale({.t_in = *m_c_bm_B, .v_in = v});
		}
		
		LOG.os<2>("SAD Coefficient matrices.\n");
		if (LOG.global_plev() >= 2) {
			dbcsr::print(*m_c_bm_A);
			if (m_c_bm_B) dbcsr::print(*m_c_bm_B);
		}
			
		m_p_bb_A = (dbcsr::eigen_to_tensor(ptot, "p_bb_A", grid, 
			vec<int>{0}, vec<int>{1}, m_p_bb_A->blk_size())).get_stensor();
		
		m_p_bb_A->filter({.use_absolute = true});
		dbcsr::print(*m_p_bb_A);
			
		if (!m_restricted && !m_nobeta) m_p_bb_B = (dbcsr::eigen_to_tensor(ptot, 
			"p_bb_B", grid, vec<int>{0}, vec<int>{1}, m_p_bb_B->blk_size())).get_stensor();
		
		
		LOG.os<2>("SAD density matrices.\n");
		if (LOG.global_plev() >= 2) {
			dbcsr::print(*m_p_bb_A);
			if (m_p_bb_B) dbcsr::print(*m_p_bb_B);
		}
		
		LOG.os<>("Finished with SAD.\n");
		
		//end SAD :(
		
	} else {
		
		throw std::runtime_error("Unknown option for guess: "+m_guess);
		
	}
	
	t_guess.finish();
	
}

}
