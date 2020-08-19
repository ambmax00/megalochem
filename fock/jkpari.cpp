#include "fock/jkbuilder.h"
#include "ints/aofactory.h"
#include "ints/screening.h"
#include "math/linalg/SVD.h"
#include "math/other/blockmap.h"
#include "extern/lapack.h"
#include <Eigen/Core>
#include <Eigen/SVD>

namespace fock {

BATCHED_PARI_K::BATCHED_PARI_K(dbcsr::world& w, desc::options& opt) 
	: K(w,opt) {}

void BATCHED_PARI_K::init_tensors() {
	
	// get s_xx
	
	// get mapping
	
	// construct aofactory, initialize
	// compute schwarz screening
	
	// set up column world
	
	// set up blacs grid
	
	// loop over (batches) of atoms A
	
		// get mu,nu pairs
	   // reserve c_xab_mu,nu
	   // reserve temp blocks
	
		// loop over munu block pairs
	
		   // generate eris 
		 
		   // get mu,nu,x_ab indices
		   // get entries from s_xx
		   // do SVD
		   
		   // collect stacks
		   
		  // end loop
		 // end loop
		 
		// process matrix stacks
		// copy c_xab_mu,nu into c_xmn
		
	// end batch loop
		 
	// =================== get tensors/matrices ========================
	auto s_xx = m_reg.get_matrix<double>("s_xx");
	
	dbcsr::print(*s_xx);
	
	auto s_xx_desym = s_xx->desymmetrize();
	s_xx_desym->replicate_all();
	
	auto eri_batched = m_reg.get_btensor<3,double>("i_xbb_batched");
	
	std::shared_ptr<ints::aofactory> aofac =
		std::make_shared<ints::aofactory>(m_mol, m_world);
	
	std::shared_ptr<ints::screener> scr_s(
		new ints::schwarz_screener(aofac,"coulomb"));
	scr_s->compute();
	
	aofac->ao_3c2e_setup("coulomb");	
		
	// ================== get mapping ==================================
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,3> xbb = {x,b,b};
	arrvec<int,2> xx = {x,x};
	
	auto cbas = m_mol->c_basis();
	auto xbas = *m_mol->c_dfbasis();
	
	auto atoms = m_mol->atoms();
	int natoms = atoms.size();
	
	vec<int> blk_to_atom_b(b.size()), blk_to_atom_x(x.size());
	
	auto get_centre = [&atoms](libint2::Shell& s) {
		for (int i = 0; i != atoms.size(); ++i) {
			auto& a = atoms[i];
			double d = sqrt(pow(s.O[0] - a.x, 2)
				+ pow(s.O[1] - a.y,2)
				+ pow(s.O[2] - a.z,2));
			if (d < 1e-12) return i;
		}
		return -1;
	};
	
	for (int iv = 0; iv != cbas.size(); ++iv) {
		auto s = cbas[iv][0];
		blk_to_atom_b[iv] = get_centre(s);
	}
	
	for (int iv = 0; iv != xbas.size(); ++iv) {
		auto s = xbas[iv][0];
		blk_to_atom_x[iv] = get_centre(s);
	}
	
	LOG.os<1>("Block to atom mapping (b): \n");	
	if (LOG.global_plev() >= 1) {
		for (auto e : blk_to_atom_b) {
			LOG.os<1>(e, " ");
		} LOG.os<1>('\n');
	}
	
	LOG.os<1>("Block to atom mapping (x): \n");	
	if (LOG.global_plev() >= 1) {
		for (auto e : blk_to_atom_x) {
			LOG.os<1>(e, " ");
		} LOG.os<1>('\n');
	}
		
	// === END ===
	
	// ==================== new atom centered dists ====================
	
	int nbf = m_mol->c_basis().nbf();
	int dfnbf = m_mol->c_dfbasis()->nbf();
	
	std::array<int,3> xbbsizes = {dfnbf,nbf,nbf};
	
	auto spgrid3 = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(xbbsizes).map1({0}).map2({1,2}).get();
		
	auto dims = spgrid3->dims();
	
	for (auto p : dims) {
		LOG.os<1>(p, " ");
	} LOG.os<1>('\n');
	
	LOG.os<>("Grid size: ", m_world.nprow(), " ", m_world.npcol(), '\n');
		
	vec<int> d0(x.size());
	vec<int> d1(b.size());
	vec<int> d2(b.size());
	
	for (int i = 0; i != d0.size(); ++i) {
		d0[i] = blk_to_atom_x[i] % dims[0];
	}
	
	for (int i = 0; i != d1.size(); ++i) {
		d1[i] = blk_to_atom_b[i] % dims[1];
		d2[i] = blk_to_atom_b[i] % dims[2];
	}
	
	arrvec<int,3> distsizes = {d0,d1,d2};
	
	dbcsr::dist_t<3> cdist(spgrid3, distsizes);
	
	// === END

	// ===================== setup tensors =============================
	
	auto eri = eri_batched->get_stensor();
	
	dbcsr::print(*eri);
	
	auto c_xab_mn = dbcsr::tensor_create_template<3,double>(eri)
		.name("c_xab_mn_batch").get();
	
	arrvec<int,3> blkidx = c_xab_mn->blks_local();
	arrvec<int,3> blkoffsets = c_xab_mn->blk_offsets();
	
	auto& x_offsets = blkoffsets[0];
	auto& b_offsets = blkoffsets[1];
	
	auto& loc_x_idx = blkidx[0];
	auto& loc_m_idx = blkidx[1];
	auto& loc_n_idx = blkidx[2];
	
	auto print = [](vec<int>& v) {
		for (auto e : v) {
			std::cout << e << " ";
		} std::cout << std::endl;
	};
	
	print(loc_x_idx);
	print(loc_m_idx);
	print(loc_n_idx);	
	
	// Divide up atom pairs over procs
	
	vec<std::pair<int,int>> atom_pairs;
	int nproc = std::min(natoms*natoms,m_world.size());
	
	for (int i = 0; i != natoms; ++i) {
		for (int j = 0; j != natoms; ++j) {
			
			int a = i + j * natoms;
			if (a % nproc == m_world.rank()) {
				
				std::pair<int,int> ab = {i,j};
				atom_pairs.push_back(ab);
				
			}
			
		}
	}
	
	// get number of batches
	
	int nbatches = 5;
	int natom_pairs = atom_pairs.size();
	
	LOG(-1).os<>("NATOM PAIRS: ", natom_pairs, '\n');
	
	math::blockmap<3> i_xbb_ab(xbb);
	math::blockmap<2> inv_xx(xx);
	math::blockmap<3> c_xbb_ab(xbb);
	math::blockmap<3> c_xbb(xbb);
	
	double npairs_per_batch_d = (double)natom_pairs / (double)nbatches;
			
	if (npairs_per_batch_d < 1.0) {
		nbatches = natom_pairs;
		std::cout << "Readjusting to " << natom_pairs << " batches." << std::endl;  
	}
	
	int npairs_per_batch = std::ceil((double)natom_pairs / (double)nbatches);
	
	for (int ibatch = 0; ibatch != nbatches; ++ibatch) {
		
		std::cout << "PROCESSING BATCH NR: " << ibatch << " out of " << nbatches 
			<< " on proc " << m_world.rank() << std::endl;
		
		int low = ibatch * npairs_per_batch;
		int high = std::min(natom_pairs,(ibatch+1) * npairs_per_batch);
		
		std::cout << "LOW/HIGH: " << low << " " << high << std::endl;
				
		for (int ipair = low; ipair != high; ++ipair) {
			
			int iAtomA = atom_pairs[ipair].first;
			int iAtomB = atom_pairs[ipair].second;
			
			std::cout << "Processing atoms " << iAtomA << " " << iAtomB 
				<< " on proc " << m_world.rank() << std::endl;

			// Get mu/nu indices centered on alpha/beta
			
			vec<int> ma_idx, nb_idx, xab_idx;
			
			for (int m = 0; m != b.size(); ++m) {
				if (blk_to_atom_b[m] == iAtomA) ma_idx.push_back(m);
			}
				
			for (int n = 0; n != b.size(); ++n) {
				if (blk_to_atom_b[n] == iAtomB) nb_idx.push_back(n);
			}
								
			// get x indices centered on alpha or beta
				
			for (int i = 0; i != x.size(); ++i) {
				if (blk_to_atom_x[i] == iAtomA || 
					blk_to_atom_x[i] == iAtomB) 
					xab_idx.push_back(i);
			}
			
			int mnblks = ma_idx.size();
			int nnblks = nb_idx.size();		
			int xblks = xab_idx.size();
			
			// compute ints
			
			arrvec<int,3> blks = {xab_idx, ma_idx, nb_idx};
			
			aofac->ao_3c2e_fill_blockmap(i_xbb_ab, blks, scr_s);	
			
			std::cout << "BEFORE FILTER" << std::endl;
			i_xbb_ab.print();
			
			i_xbb_ab.filter(dbcsr::global::filter_eps);
			
			std::cout << "AFTER FILTER" << std::endl;
			i_xbb_ab.print();
			
			if (i_xbb_ab.num_blocks() == 0) continue;
			
			// get matrix parts of s_xx
				
			// problem size
			int m = 0;
				
			vec<int> xab_sizes(xab_idx.size()), xab_offs(xab_idx.size());
			std::map<int,int> xab_mapping;
			int off = 0;
				
			for (int i = 0; i != xab_idx.size(); ++i) {
				int ixab = xab_idx[i];
				m += x[ixab];
				xab_sizes[i] = x[ixab];
				xab_offs[i] = off;
				xab_mapping[ixab] = i;
				off += xab_sizes[i];
			}
			
			// get (alpha|beta) matrix 
			Eigen::MatrixXd alphaBeta = Eigen::MatrixXd::Zero(m,m);
			
			// loop over xab blocks
			for (auto ix : xab_idx) {
				for (auto iy : xab_idx) {
					
					int xsize = x[ix];
					int ysize = x[iy];
					
					bool found = false;
					double* ptr = s_xx_desym->get_block_data(ix,iy,found);
			
					if (found) {
						
						std::cout << "FOUND: " << ix << " " << iy << std::endl;
						Eigen::Map<Eigen::MatrixXd> blk(ptr,xsize,ysize);
						
						std::cout << blk << std::endl;
						
						int xoff = xab_offs[xab_mapping[ix]];
						int yoff = xab_offs[xab_mapping[iy]];
						
						alphaBeta.block(xoff,yoff,xsize,ysize) = blk;
						
					}	
					
				}
			}
			
			std::cout << "Problem size: " << m << " x " << m << std::endl;
			//std::cout << "MY ALPHABETA: " << std::endl;
			//for (int i = 0; i != m*m; ++i) {
			//	std::cout << alphaBeta.data()[i] << std::endl;
			//}
				
			Eigen::MatrixXd U = Eigen::MatrixXd::Zero(m,m);
			Eigen::MatrixXd Vt = Eigen::MatrixXd::Zero(m,m);
			Eigen::VectorXd s = Eigen::VectorXd::Zero(m);
			
			int info = 1;
			double worksize = 0;
			
			Eigen::MatrixXd alphaBeta_c = alphaBeta;
			
			c_dgesvd('A', 'A', m, m, alphaBeta.data(), m, s.data(), U.data(), 
				m, Vt.data(), m, &worksize, -1, &info);
	
			int lwork = (int)worksize; 
			double* work = new double [lwork];
			
			c_dgesvd('A', 'A', m, m, alphaBeta.data(), m, s.data(), U.data(), 
				m, Vt.data(), m, work, lwork, &info);
			
			for (int i = 0; i != m; ++i) {
				s(i) = 1.0/s(i);
			}
			
			Eigen::MatrixXd Inv = Eigen::MatrixXd::Zero(m,m);
			Inv = Vt.transpose() * s.asDiagonal() * U.transpose();
			
			//std::cout << "MY INV: " << std::endl;
			//for (int i = 0; i != m*m; ++i) {
			//	std::cout << Inv.data()[i] << std::endl;
			//}
			
			Eigen::MatrixXd E = Eigen::MatrixXd::Identity(m,m);
			
			E -= Inv * alphaBeta_c;
			std::cout << "ERROR1: " << E.norm() << std::endl;
			
			std::cout << Inv << std::endl;
			
			U.resize(0,0);
			Vt.resize(0,0);
			s.resize(0);
			delete [] work;
			
			// transfer inverse to blocks
			
			arrvec<int,2> res_x;
			
			for (auto ix : xab_idx) {
				for (auto iy : xab_idx) {
					res_x[0].push_back(ix);
					res_x[1].push_back(iy);
				}
			}
			
			inv_xx.reserve(res_x);
			
			for (auto ix : xab_idx) {
				for (auto iy : xab_idx) {
					
					int xoff = xab_offs[xab_mapping[ix]];
					int yoff = xab_offs[xab_mapping[iy]];
					int xsize = x[ix];
					int ysize = x[iy];
					
					Eigen::MatrixXd blk = Inv.block(xoff,yoff,xsize,ysize);
					std::array<int,2> idx = {ix,iy};
					double* data = inv_xx.get_block(idx);
					
					std::copy(blk.data(), blk.data() + xsize*ysize, data);
					
				}
			}
							 
			inv_xx.filter(dbcsr::global::filter_eps);
			inv_xx.print();
						
			// allocate cxbb
			
			arrvec<int,3> res_c;
			
			for (auto x : xab_idx) {
				for (auto m : ma_idx) {
					for (auto n : nb_idx) {
						res_c[0].push_back(x);
						res_c[1].push_back(m);
						res_c[2].push_back(n);
					}
				}
			}
			
			c_xbb_ab.reserve(res_c);
			
			// insert into stacks
			
			for (auto inv_pair = inv_xx.begin(); inv_pair != inv_xx.end(); ++inv_pair) {
				
				int ix0 = inv_pair->first[0];
				int ix1 = inv_pair->first[1];
				
				double* inv_data = inv_xx.get_data(inv_pair->second);
				
				for (auto im : ma_idx) {
					for (auto in : nb_idx) {
						
						std::array<int,3> idxc = {ix0,im,in};
						std::array<int,3> idxi = {ix1,im,in};
						
						double* cxbb_data = c_xbb_ab.get_block(idxc);
						double* ixbb_data = i_xbb_ab.get_block(idxi);
						
						if (cxbb_data && ixbb_data) {
							
							std::cout << "MULT: " << ix0 << "," << ix1 << " * " << 
								ix1 << "," << im << "," << in << " = " << ix0 << 
									"," << im << "," << in << std::endl;
							
							int m = x[ix0];
							int n = b[im] * b[in];
							int k = x[ix1];
							
							double alpha = 1.0;
							double beta = 1.0;
							
							libxsmm_gemm(NULL, NULL, m, n, k, &alpha, inv_data, NULL, ixbb_data, 
								NULL, &beta, cxbb_data, NULL);				
							
						} // endif
					} // end for in 
				} // end for im	
			} // end for inv_pair
			
			i_xbb_ab.clear();
			inv_xx.clear();
			
			c_xbb_ab.print();
			c_xbb_ab.filter(dbcsr::global::filter_eps);
			
			c_xbb.insert(c_xbb_ab);
			
				
		}
		
		std::cout << "CXBB" << std::endl;
		c_xbb.print();
		
	}	
	
	
	/*
	// ================== START ATOM BATCH LOOP  =======================
	
	int split = 5;
	int n_abatches = std::ceil( (double)natoms / (double)split);  
	
	for (int ia_batch = 0; ia_batch != n_abatches; ++ia_batch) {
		
		// get low and high
		int a_low = ia_batch * split;
		int a_high = (ia_batch != n_abatches - 1) ? (ia_batch + 1) * split : natoms;
	
		LOG.os<>("Processing batch of atoms ", a_low, " to ", a_high - 1, '\n');
	
		// =============== reserve c_xab ======================
		arrvec<int,3> c_res; 
		
		for (auto m : loc_m_idx) {
			int ia = blk_to_atom_b[m];
			
			if (ia < a_low || ia > a_high) continue;
			
			for (auto n : loc_n_idx) {
				int ib = blk_to_atom_b[n];
				
				for (auto x : loc_x_idx) {
					int iab = blk_to_atom_x[x];
					
					if (iab == ia || iab == ib) {
						
						c_res[0].push_back(x);
						c_res[1].push_back(m);
						c_res[2].push_back(n);
					}
				}
				
				
				
			}
		}
		
		c_xab_mn->reserve(c_res);
		
		dbcsr::print(*c_xab_mn);
		
		for (auto& a : c_res) {
			a.clear();
			a.shrink_to_fit();
		}
	
	
		// =============== START LOOP over atom pairs ==================
		for (int ia = a_low; ia < a_high; ++ia) {
			for (int ib = 0; ib < natoms ; ++ib) {
				
				 // get pointers
	
				MPI_Barrier(m_world.comm());

				// Get mu/nu indices centered on alpha/beta
			
				vec<int> ma_idx, nb_idx, mnab_idx, xab_idx;
			
				for (auto m : loc_m_idx) {
					if (blk_to_atom_b[m] == ia) ma_idx.push_back(m);
				}
				
				for (auto n : loc_n_idx) {
					if (blk_to_atom_b[n] == ib) nb_idx.push_back(n);
				}
				
				int mnblks = ma_idx.size();
				int nnblks = nb_idx.size();
				
				if (ma_idx.size() == 0 || nb_idx.size() == 0) continue;
				
				// get sizes 
				
				vec<int> ma_blksizes(mnblks), nb_blksizes(nnblks);
				
				for (int i = 0; i != mnblks; ++i) {
					ma_blksizes[i] = b[ma_idx[i]];
				}
				
				for (int i = 0; i != nnblks; ++i) {
					nb_blksizes[i] = b[nb_idx[i]];
				}
				
				// get x indices centered on alpha or beta
				
				for (int i = 0; i != x.size(); ++i) {
					if (blk_to_atom_x[i] == ia || 
						blk_to_atom_x[i] == ib) 
						xab_idx.push_back(i);
				}
				
				int xblks = xab_idx.size();
				
				COLLOG(-1).os<>("Hello from col process ", _grid.myprow(), " out of ",
					_grid.nprow(), ", my global rank is ", m_world.rank(), '\n');
					
				COLLOG.os<>("Processing atoms ", ia, " ", ib, '\n');
				
				// get matrix parts of s_xx
				
				// problem size
				int n = dfnbf;
				int m = 0;
				
				vec<int> xab_sizes(xab_idx.size());
				std::map<int,int> xab_mapping;
				
				for (int i = 0; i != xab_idx.size(); ++i) {
					int ixab = xab_idx[i];
					m += x[ixab];
					xab_sizes[i] = x[ixab];
					xab_mapping[ixab] = i;
				}
			
				COLLOG.os<>("Problem size: ", n, " x ", m, '\n');
			
				Eigen::MatrixXd s_xx_01(n,m);
				int ioff = 0;
				
				if (colworld.rank() == 0) {
					std::cout << s_xx_eigen << std::endl;
				}
				
				auto s_xx_01_mat = dbcsr::create<double>()
					.set_world(colworld)
					.name("column_s_xx")
					.row_blk_sizes(xab_sizes)
					.col_blk_sizes(xab_sizes)
					.matrix_type(dbcsr::type::no_symmetry)
					.get();
					
				s_xx_01_mat->reserve_all();
				
				dbcsr::iterator<double> it_sxx(*s_xx_01_mat);
				it_sxx.start();
				
				while (it_sxx.blocks_left()) {
					
					it_sxx.next_block();
					
					int ix_glob = xab_idx[it_sxx.row()];
					int ixab_glob = xab_idx[it_sxx.col()];
					
					int ix_off = blkoffsets[0][ix_glob];
					int ixab_off = blkoffsets[0][ixab_glob];
					
					int ix_size = it_sxx.row_size();
					int ixab_size = it_sxx.col_size();
					
					auto eigen_blk =  
						s_xx_eigen.block(ix_off, ixab_off, ix_size, ixab_size);
						
					for (int i = 0; i != ix_size; ++i) {
						for (int j = 0; j != ixab_size; ++j) {
							it_sxx(i,j) = eigen_blk(i,j);
						}
					}
					
				}
				
				it_sxx.stop();
				
				//for (int ic = 0; ic != m_world.nprow(); ++ic) {
				//	if (colworld.rank() == ic) {
				//		std::cout << "S_XX ? " << s_xx_01 << std::endl;
				//	}
				//	MPI_Barrier(col_comm);
				//}
				
				//Eigen::JacobiSVD<Eigen::MatrixXd> svd(s_xx_01, Eigen::ComputeThinU | Eigen::ComputeThinV);
				//std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
				//std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
				//std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
			
				// do an SVD
				
			
				/*
				auto c = dbcsr::matrix_to_eigen(s_xx_01_mat);
			
				Eigen::MatrixXd diff = c - s_xx_01;
				
				MPI_Barrier(col_comm);
				
				for (int ic = 0; ic != m_world.nprow(); ++ic) {
					if (colworld.rank() == ic) {
						std::cout << "ZEROS ? " << diff << std::endl;
					}
					MPI_Barrier(col_comm);
				}
			
				dbcsr::print(*s_xx_01_mat);
			
				MPI_Barrier(col_comm);
			
				math::SVD do_svd(s_xx_01_mat, 'V', 'V', 1);
			
				do_svd.compute();
			
				auto inv = do_svd.inverse();
				
				dbcsr::print(*inv);
			
				inv->replicate_all();
			
				// form mm stacks
			
				std::array<int,3> idx;
				std::array<int,3> blksize;
			
				std::map<std::array<int,3>,double*> c_xmn_blocks;
			
				stack mm_stack;
						
				for (auto im : ma_idx) {
					for (auto in : nb_idx) {
					
					for (auto ix : loc_x_idx) {
						
						std::cout << "FETCHING BLOCK NR: " << im << " "
							<< in << " " << ix << std::endl;
						
						idx[0] = ix;
						idx[1] = im;
						idx[2] = in;
						
						blksize[0] = x[ix];
						blksize[1] = b[im];
						blksize[2] = b[in];
						
						bool found = false;
						
						double* ptr_xmn = eri_centered->get_block_p(idx, found);
						if (!found) continue;
							
						for (auto ixab : xab_idx) {
							
							std::cout << "ixab glob: " << ixab << " ->  " 
								<< "ixab loc: " << xab_mapping[ixab] << std::endl; 
							
							std::cout << "CONTRACTING WITH: " << ix << " " << ixab
								<< std::endl;
								
							int blksizeab = x[ixab];
							
							double* ptr_xabx = inv->get_block_data(xab_mapping[ixab], ix, found);
							if (!found) continue;
							
							std::array<int,3> mnk = {blksizeab, blksize[1] * blksize[2], blksize[0]};
							
							std::array<int,3> c_idx = {ixab,im,in};
							
							double* ptr_c;
							
							auto iter = c_xmn_blocks.find(c_idx);
							if (iter == c_xmn_blocks.end()) {
								ptr_c = new double[blksizeab*blksize[1]*blksize[2]]();
								c_xmn_blocks[c_idx] = ptr_c;
							} else {
								ptr_c = iter->second;
							}
							
							std::array<double*,3> ptrs = {ptr_xabx,ptr_xmn,ptr_c};
							
							mm_stack.insert(mnk, ptrs);
															
						} // end xab
						
					} // end x
					
				} //end nu
				
			} // end mu
			
			// ========= PROCESS STACKS ==============
			
			
			
			// === now gather blocks =====
			
			MPI_Barrier(col_comm);
			
			COLLOG.os<>("REDUCING\n");
			for (auto& idxblk : c_xmn_blocks) {
				
				std::array<int,3> idx = idxblk.first;
				double* send = idxblk.second;
				
				const int s0 = x[idx[0]];
				const int s1 = b[idx[1]];
				const int s2 = b[idx[2]];
				
				const int size = s0*s1*s2;
				
				COLLOG.os<>("BLOCK: ", idx[0], " ", idx[1], " ", idx[2], '\n');
				
				for (int i = 0; i != colworld.size(); ++i) {
					if (i == colworld.rank()) {
						std::cout << "PROC: " << i << std::endl;
						for (int j = 0; j != size; ++j) {
							std::cout << send[j] << " ";
						} std::cout << std::endl;
					}
					std::cout << std::endl;
					MPI_Barrier(col_comm);
				}
				
				// to which proc does xab belong to? 
				int dest = c_xab_mn->proc(idx);
				
				COLLOG(-1).os<>("Destination: ", dest, '\n');
				
				double* recv = nullptr;
				if (dest == colworld.rank()) {
					bool found = true;
					recv = c_xab_mn->get_block_p(idx,found); 
					if (!found) std::cout << "NOT THERE!!!" << std::endl;
				}
	
				MPI_Reduce(send, recv, size, MPI_DOUBLE, MPI_SUM, dest, col_comm);
				
				COLLOG.os<>("After reduction: \n");
				if (dest == colworld.rank()) {
					std::cout << "PROC: " << dest << std::endl;
					for (int j = 0; j != size; ++j) {
						std::cout << recv[j] << " ";
					} std::cout << std::endl;
				}
				
				delete [] send;
				
			}
	
	
		 } // end atom b loop
		} // end atom a loop 
		
	} // end atom a batch loop
	
	dbcsr::print(*c_xab_mn);
	
	LOG.os<>("DONE");
	
	MPI_Barrier(m_world.comm());
	
	exit(0);
	*/
	
}

void BATCHED_PARI_K::compute_K() {
	
	TIME.start();
	
	TIME.finish();
			
}

} // end namespace
