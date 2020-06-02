#include "hf/hfmod.h"
#include "ints/registry.h"
#include "hf/hfdefaults.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include "math/linalg/piv_cd.h"
#include <dbcsr_conversions.hpp>
#include <limits>

#include <dbcsr_matrix_ops.hpp>


#ifdef USE_SCALAPACK
#include "extern/scalapack.h"
#endif

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
	
	dbcsr::iterator_t<2> iter(t);
	
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
		
		// form density and coefficients by diagonalizing the core matrix
		//dbcsr::copy<2>({.t_in = *m_core_bb, .t_out = *m_f_bb_A}); 
		m_f_bb_A->copy_in(*m_core_bb);
		
		//dbcsr::print(*m_core_bb);
		//dbcsr::print(*m_f_bb_A);
		
		if (!m_restricted) m_f_bb_B->copy_in(*m_core_bb);
		
		diag_fock();
	
	} else if (m_guess == "SADNO" || m_guess == "SAD") {
		
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
		
		// divide it up
		std::vector<libint2::Atom> my_atypes;
		
		for (int I = 0; I != atypes.size(); ++I) {
			if (m_world.rank() == I % m_world.size()) my_atypes.push_back(atypes[I]);
		}
		
		if (LOG.global_plev() >= 2) {
			LOG.os<>("Distribution of atom types along processors:\n");
			for (int i = 0; i != m_world.size(); ++i) {
				if (m_world.rank() == i) {
					std::cout << "Rank " << m_world.rank() << std::endl;
					for (auto e : my_atypes) {
						std::cout << e.atomic_number << " ";
					} std::cout << std::endl;
				}
				MPI_Barrier(m_world.comm());
			}
		}
		
		MPI_Comm mycomm;
		
		MPI_Comm_split(m_world.comm(), m_world.rank(), m_world.rank(), &mycomm);
		
		// set up new scalapack grid
#ifdef USE_SCALAPACK
		int sysctxt = -1;
		c_blacs_get(0, 0, &sysctxt);
		
		std::vector<int> contexts(m_world.size(),0);
		int prevctxt = scalapack::global_grid.ctx();
		
		for (int r = 0; r != m_world.size(); ++r) {
			contexts[r] = sysctxt;
			int usermap[1] = {r};
			c_blacs_gridmap(&contexts[r], &usermap[0], 1, 1, 1);
		}
		
		scalapack::global_grid.set(contexts[m_world.rank()]);
		
#endif
		
		//for (int r = 0; r != m_world.size(); ++r) {
			
		//if (r == m_world.rank()) {
		
		for (int I = 0; I != my_atypes.size(); ++I) {
			
			auto atom = my_atypes[I];
			int Z = atom.atomic_number;
			
			//set up options
			desc::options at_opt(m_opt);
			
			int atprint = LOG.global_plev() - 2;
			at_opt.set<int>("print", atprint);
			at_opt.set<std::string>("guess", m_opt.get<std::string>("SAD_guess", HF_SAD_GUESS));
			at_opt.set<bool>("diis", m_opt.get<bool>("SAD_diis", HF_SAD_SCF_DIIS));
			at_opt.set<bool>("use_df", m_opt.get<bool>("SAD_use_df", HF_SAD_USE_DF));
			at_opt.set<double>("scf_thresh",m_opt.get<double>("SAD_scf_thresh", HF_SAD_SCF_THRESH));
			
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
			
			std::string name = "ATOM_rank" + std::to_string(m_world.rank()) + "_" + std::to_string(Z);
			
			bool spinav = m_opt.get<bool>("SAD_spin_average",HF_SAD_SPIN_AVERAGE);
			
			desc::molecule at_mol = desc::molecule::create().name(name).atoms(atvec).charge(charge)
				.mo_split(10).atom_split(1)
				.mult(mult).basis(at_basis).dfbasis(at_dfbasis).fractional(true).spin_average(spinav);
				
			auto at_smol = std::make_shared<desc::molecule>(std::move(at_mol));
				
			if (LOG.global_plev() >= 2) {
				at_smol->print_info(mycomm,LOG.global_plev());
			}
			
			dbcsr::world at_world(mycomm);
			
			hf::hfmod atomic_hf(at_smol,at_opt,at_world);
			
			LOG(m_world.rank()).os<1>("Starting Atomic UHF for atom nr. ", I, " on rank ", m_world.rank(), '\n');
			atomic_hf.compute();
			LOG(m_world.rank()).os<1>("Done with Atomic UHF for atom nr. ", I, " on rank ", m_world.rank(), "\n");
			
			auto at_wfn = atomic_hf.wfn();
			
			auto pA = at_wfn->po_bb_A();
			auto pB = at_wfn->po_bb_B();
			
			//std::cout << "PA on rank: " << myrank << std::endl;
			//dbcsr::print(*pA);
			
			if (pB) {
			
				mat_d pscaled = mat_d::copy<double>(*pA).name(at_smol->name() + "_density");
				pscaled.add(0.5, 0.5, *pB);
				locdensitymap[Z] = dbcsr::matrix_to_eigen(pscaled);
				
			} else {
				
				locdensitymap[Z] = dbcsr::matrix_to_eigen(*pA);
				
			}
			
			ints::registry INTS_REGISTRY;
			INTS_REGISTRY.clear(name);
			
		}
		
		//}
		
		//MPI_Barrier(m_world.comm());
		
		//}
		
		MPI_Barrier(m_world.comm());
		
#ifdef USE_SCALAPACK
		scalapack::global_grid.free();
		scalapack::global_grid.set(prevctxt);
#endif
		
		MPI_Barrier(m_world.comm());
		
		//std::cout << "DISTRIBUTING" << std::endl;
		
		// distribute to other nodes
		for (int i = 0; i != m_world.size(); ++i) {
			
			//std::cout << "LOOP: " << i << std::endl;
			
			if (i == m_world.rank()) {
				
				int n = locdensitymap.size();
				
				//std::cout << "N: " << n << std::endl;
				
				MPI_Bcast(&n,1,MPI_INT,i,m_world.comm());
				
				for (auto& den : locdensitymap) {
					
					int Z = den.first;
					size_t size = den.second.size();
					
					//std::cout << "B1 " << Z << std::endl;
					MPI_Bcast(&Z,1,MPI_INT,i,m_world.comm());
					//std::cout << "B2 " << size << std::endl;
					MPI_Bcast(&size,1,MPI_UNSIGNED,i,m_world.comm());
					
					auto& mat = den.second;
					
					//std::cout << "B3" << std::endl;
					MPI_Bcast(mat.data(),size,MPI_DOUBLE,i,m_world.comm());
					
					densitymap[Z] = mat;
					
					MPI_Barrier(m_world.comm());
					
				 }
				
			} else {
				
				int n = -1;
				
				MPI_Bcast(&n,1,MPI_INT,i,m_world.comm());
				
				for (int ni = 0; ni != n; ++ni) {
					
					size_t size = 0;
					int Z = 0;
					
					MPI_Bcast(&Z,1,MPI_INT,i,m_world.comm());
					MPI_Bcast(&size,1,MPI_UNSIGNED,i,m_world.comm());
					
					//std::cout << "Other rank: " << Z << " " << size << std::endl;
					
					Eigen::MatrixXd mat((int)sqrt(size),(int)sqrt(size));
					
					MPI_Bcast(mat.data(),size,MPI_DOUBLE,i,m_world.comm());
					
					densitymap[Z] = mat;
					
					MPI_Barrier(m_world.comm());
					
				}
				
			}
			
			MPI_Barrier(m_world.comm());
			
		}
			
		
		size_t nbas = m_mol->c_basis().nbf();
		
		Eigen::MatrixXd ptot_eigen = Eigen::MatrixXd::Zero(nbas,nbas);
		auto csizes = m_mol->dims().b();
		int off = 0;
		int size = 0;
		
		for (int i = 0; i != m_mol->atoms().size(); ++i) {
			
			int Z = m_mol->atoms()[i].atomic_number;
			auto& at_density = densitymap[Z];
			int at_nbas = at_density.rows();
			
			ptot_eigen.block(off,off,at_nbas,at_nbas) = densitymap[Z];
			
			off += at_nbas;
			
		}
		
		
		auto b = m_mol->dims().b();
		mat_d ptot = dbcsr::eigen_to_matrix(ptot_eigen, m_world, "p_bb_A", b, b, dbcsr_type_symmetric);
		
		m_p_bb_A = ptot.get_smatrix();
		m_p_bb_A->filter();
		
		if (m_guess == "SADNO") {
			
			LOG.os<>("Forming natural orbitals from SAD guess density.\n");
		
			math::hermitian_eigen_solver solver(m_p_bb_A, 'V', (LOG.global_plev() >= 2) ? true : false);
			
			solver.compute();
			
			auto eigvals = solver.eigvals();
			m_c_bm_A = solver.eigvecs();
			
			std::for_each(eigvals.begin(),eigvals.end(),
				[](double& d) 
				{ 
					d = (d < std::numeric_limits<double>::epsilon())
						? 0 : sqrt(d); 
			});
			
			m_c_bm_A->scale(eigvals, "right");
			
		} else {
			
			LOG.os<>("Forming cholesky orbitals from SAD guess.");
			
			math::pivinc_cd cd(m_p_bb_A, LOG.global_plev());
			
			cd.compute();
			
			int rank = cd.rank();
			
			auto o_sad = m_mol->dims().split_range(rank, m_mol->mo_split());
			auto b = m_mol->dims().b(); 
			
			m_c_bm_A = cd.L(b,o_sad);
			
			m_c_bm_A->setname("c_bm_A");
			m_c_bm_A->filter();
			
			//dbcsr::print(*m_c_bm_A);
			
			//dbcsr::multiply('N', 'T', *m_c_bm_A, *m_c_bm_A, *m_p_bb_A).beta(-1.0).perform();
			
			//dbcsr::print(*m_p_bb_A);
			
		}
			
		if (!m_restricted) {
			if (!m_nobetaorb) m_c_bm_B->copy_in(*m_c_bm_A);
			m_p_bb_B->copy_in(*m_p_bb_A);
		}
		
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
