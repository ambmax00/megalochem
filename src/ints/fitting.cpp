#include "ints/fitting.h"
#include "ints/aofactory.h"
#include "extern/lapack.h"
#include <Eigen/Core>
#include <Eigen/SVD>

namespace ints {
	
dbcsr::sbtensor<3,double> dfitting::compute(dbcsr::sbtensor<3,double> eri_batched, 
	dbcsr::shared_matrix<double> inv, std::string cfit_btype) {
		
	auto spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	auto x = m_mol->dims().x();
	
	arrvec<int,2> xx = {x,x};

	auto s_xx_inv = dbcsr::tensor_create<2>()
		.name("s_xx_inv")
		.pgrid(spgrid2)
		.blk_sizes(xx)
		.map1({0}).map2({1})
		.get();
		
	dbcsr::copy_matrix_to_tensor(*inv, *s_xx_inv);
	inv->clear();
	
	auto cfit = this->compute(eri_batched, s_xx_inv, cfit_btype);
	dbcsr::copy_tensor_to_matrix(*s_xx_inv, *inv);
	
	return cfit;
	
}
	
dbcsr::sbtensor<3,double> dfitting::compute(dbcsr::sbtensor<3,double> eri_batched, 
	dbcsr::shared_tensor<2,double> inv, std::string cfit_btype) {
	
	auto spgrid3_xbb = eri_batched->spgrid();
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,3> xbb = {x,b,b};
	
	// ======== Compute inv_xx * i_xxb ==============
	
	auto c_xbb_0_12 = dbcsr::tensor_create<3>()
		.name("c_xbb_0_12")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
	
	auto c_xbb_1_02 = dbcsr::tensor_create_template<3>(c_xbb_0_12)
		.name("c_xbb_1_02")
		.map1({1}).map2({0,2})
		.get();
			
	auto mytype = dbcsr::get_btype(cfit_btype);
	
	int nbatches_x = eri_batched->nbatches(0);
	int nbatches_b = eri_batched->nbatches(2);

	std::array<int,3> bdims = {nbatches_x,nbatches_b,nbatches_b};
	
	auto c_xbb_batched = dbcsr::btensor_create<3>()
		.name(m_mol->name() + "_c_xbb_batched")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.batch_dims(bdims)
		.btensor_type(mytype)
		.print(LOG.global_plev())
		.get();
	
	auto& con = TIME.sub("Contraction");
	auto& reo = TIME.sub("Reordering");
	auto& write = TIME.sub("Writing");
	auto& fetch = TIME.sub("Fetching ints");
	
	LOG.os<1>("Computing C_xbb.\n");
	
	TIME.start();
	
	eri_batched->decompress_init({2},vec<int>{0},vec<int>{1,2});
	c_xbb_batched->compress_init({2,0}, vec<int>{1}, vec<int>{0,2});
	
	for (int inu = 0; inu != c_xbb_batched->nbatches(2); ++inu) {
		
		fetch.start();
		eri_batched->decompress({inu});
		auto eri_0_12 = eri_batched->get_work_tensor();
		fetch.finish();
			
		for (int ix = 0; ix != c_xbb_batched->nbatches(0); ++ix) {
			
				vec<vec<int>> b2 = {
					c_xbb_batched->bounds(0,ix)
				};
				
				vec<vec<int>> b3 = {
					c_xbb_batched->full_bounds(1),
					c_xbb_batched->bounds(2,inu)
				};

				con.start();
				dbcsr::contract(*inv, *eri_0_12, *c_xbb_0_12)
					.bounds2(b2).bounds3(b3)
					.filter(dbcsr::global::filter_eps)
					.perform("XY, YMN -> XMN");
				con.finish();
				
				reo.start();
				dbcsr::copy(*c_xbb_0_12, *c_xbb_1_02)
					.move_data(true)
					.perform();
				reo.finish();
				
				//dbcsr::print(*c_xbb_1_02);
			
				c_xbb_batched->compress({inu,ix}, c_xbb_1_02);
				
		}
	}
	
	c_xbb_batched->compress_finalize();
	eri_batched->decompress_finalize();
	
	double c_occ = c_xbb_batched->occupation() * 100;
	LOG.os<1>("Occupancy of c_xbb: ", c_occ, "%\n");

	TIME.finish();
	
	LOG.os<1>("Done.\n");
	
	// ========== END ==========	
	
	return c_xbb_batched;
	
}

dbcsr::shared_tensor<3,double> dfitting::compute_pari(dbcsr::sbtensor<3,double> eri_batched,
	dbcsr::shared_matrix<double> s_xx, shared_screener scr_s) {
	
	auto aofac = std::make_shared<aofactory>(m_mol, m_world);
	aofac->ao_3c2e_setup("coulomb");
	
	auto& time_setup = TIME.sub("Setting up preliminary data");
	auto& time_form_cfit = TIME.sub("Forming cfit");
	
	TIME.start();
	
	time_setup.start();
	
	auto s_xx_desym = s_xx->desymmetrize();
	s_xx_desym->replicate_all();
	
	// ================== get mapping ==================================
	
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,2> bb = {b,b};
	arrvec<int,3> xbb = {x,b,b};
	arrvec<int,2> xx = {x,x};
	
	auto cbas = m_mol->c_basis();
	auto xbas = m_mol->c_dfbasis();
	
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
	
	for (int iv = 0; iv != cbas->size(); ++iv) {
		auto s = cbas->at(iv)[0];
		blk_to_atom_b[iv] = get_centre(s);
	}
	
	for (int iv = 0; iv != xbas->size(); ++iv) {
		auto s = xbas->at(iv)[0];
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
	
	int nbf = m_mol->c_basis()->nbf();
	int dfnbf = m_mol->c_dfbasis()->nbf();
	
	std::array<int,3> xbbsizes = {1,nbf,nbf};
	
	auto spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	auto spgrid3_xbb = eri_batched->spgrid();
	
	auto spgrid2_self = dbcsr::create_pgrid<2>(MPI_COMM_SELF).get();
	
	auto spgrid3_self = dbcsr::create_pgrid<3>(MPI_COMM_SELF).get();
		
	auto dims = spgrid3_xbb->dims();
	
	for (auto p : dims) {
		LOG.os<1>(p, " ");
	} LOG.os<1>('\n');
	
	LOG.os<>("Grid size: ", m_world.nprow(), " ", m_world.npcol(), '\n');
		
	vec<int> d0(x.size(),0.0);
	vec<int> d1(b.size());
	vec<int> d2(b.size());
	
	for (int i = 0; i != d1.size(); ++i) {
		d1[i] = blk_to_atom_b[i] % dims[1];
		d2[i] = blk_to_atom_b[i] % dims[2];
	}
	
	arrvec<int,3> distsizes = {d0,d1,d2};
	
	dbcsr::dist_t<3> cdist(spgrid3_xbb, distsizes);
	
	// === END

	// ===================== setup tensors =============================
	
	auto c_xbb_centered = dbcsr::tensor_create<3,double>()
		.name("c_xbb_centered")
		.ndist(cdist)
		.map1({0}).map2({1,2})
		.blk_sizes(xbb)
		.get();
		
	auto c_xbb_dist = dbcsr::tensor_create<3,double>()
		.name("fitting coefficients")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0,1}).map2({2})
		.get();
		
	auto c_xbb_self = dbcsr::tensor_create<3,double>()
		.name("c_xbb_self")
		.pgrid(spgrid3_self)
		.map1({0}).map2({1,2})
		.blk_sizes(xbb)
		.get();
		
	auto eri_self = 
		dbcsr::tensor_create_template<3,double>(c_xbb_self)
		.name("eri_self").get();
	
	auto inv_self = dbcsr::tensor_create<2,double>()
		.name("inv_self")
		.pgrid(spgrid2_self)
		.map1({0}).map2({1})
		.blk_sizes(xx)
		.get();
	
	arrvec<int,3> blkidx = c_xbb_dist->blks_local();
	arrvec<int,3> blkoffsets = c_xbb_dist->blk_offsets();
	
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
	
	//print(loc_x_idx);
	//print(loc_m_idx);
	//print(loc_n_idx);	
	
	// Divide up atom pairs over procs
	
	vec<std::pair<int,int>> atom_pairs;
	
	for (int i = 0; i != natoms; ++i) {
		for (int j = 0; j != natoms; ++j) {
			
			int rproc = i % dims[1];
			int cproc = j % dims[2];
		
			if (rproc == m_world.myprow() && cproc == m_world.mypcol()) {
				
				std::pair<int,int> ab = {i,j};
				atom_pairs.push_back(ab);
				
			}
			
		}
	}
	
	time_setup.finish();
	
	time_form_cfit.start();
	
	// get number of batches
	
	int nbatches = 25;
	int natom_pairs = atom_pairs.size();
	
	LOG(-1).os<>("NATOM PAIRS: ", natom_pairs, '\n');
	
	int npairs_per_batch = std::ceil((double)natom_pairs / (double)nbatches);
	
	for (int ibatch = 0; ibatch != nbatches; ++ibatch) {
		
#if 0	
		for (int ip = 0; ip != m_world.size(); ++ip) {
		
		if (ip == m_world.rank()) {
#endif
	
		//std::cout << "PROCESSING BATCH NR: " << ibatch << " out of " << nbatches 
			//<< " on proc " << m_world.rank() << std::endl;
		
		int low = ibatch * npairs_per_batch;
		int high = std::min(natom_pairs,(ibatch+1) * npairs_per_batch);
		
		//std::cout << "LOW/HIGH: " << low << " " << high << std::endl;
				
		for (int ipair = low; ipair < high; ++ipair) {
			
			int iAtomA = atom_pairs[ipair].first;
			int iAtomB = atom_pairs[ipair].second;
#if 0
			std::cout << "Processing atoms " << iAtomA << " " << iAtomB 
				<< " on proc " << m_world.rank() << std::endl;
#endif			
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
#if 0
			std::cout << "INDICES: " << std::endl;
			
			for (auto m : ma_idx) {
				std::cout << m << " ";
			} std::cout << std::endl;
			
			for (auto n : nb_idx) {
				std::cout << n << " ";
			} std::cout << std::endl;
			
			for (auto ix : xab_idx) {
				std::cout << ix << " ";
			} std::cout << std::endl;
#endif			
			int mnblks = ma_idx.size();
			int nnblks = nb_idx.size();		
			int xblks = xab_idx.size();
			
			// compute ints
			
			arrvec<int,3> blks = {xab_idx, ma_idx, nb_idx};
			
			aofac->ao_3c2e_fill(eri_self, blks, scr_s);	
			
			eri_self->filter(dbcsr::global::filter_eps);
			
			if (eri_self->num_blocks_total() == 0) continue;
			
			//dbcsr::print(*eri_self);			
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

#if 0
			auto met = dbcsr::tensor_create_template<2,double>(inv_self)
				.name("metric").get();
				
			met->reserve_all();
#endif		
			// loop over xab blocks
			for (auto ix : xab_idx) {
				for (auto iy : xab_idx) {
					
					int xsize = x[ix];
					int ysize = x[iy];
					
					bool found = false;
					double* ptr = s_xx_desym->get_block_data(ix,iy,found);
			
					if (found) {
						
						std::array<int,2> sizes = {xsize,ysize};
						
#if 0		
						std::array<int,2> idx = {ix,iy};
						met->put_block(idx, ptr, sizes);
#endif
						
						Eigen::Map<Eigen::MatrixXd> blk(ptr,xsize,ysize);
												
						int xoff = xab_offs[xab_mapping[ix]];
						int yoff = xab_offs[xab_mapping[iy]];
						
						alphaBeta.block(xoff,yoff,xsize,ysize) = blk;
						
					}	
					
				}
			}

#if 0	
			std::cout << "Problem size: " << m << " x " << m << std::endl;
			std::cout << "MY ALPHABETA: " << std::endl;
			
			std::cout << alphaBeta << std::endl;
			
			std::cout << "ALPHA TENSOR: " << std::endl;
			dbcsr::print(*met);

#endif

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

#if 0		
			Eigen::MatrixXd E = Eigen::MatrixXd::Identity(m,m);
			
			E -= Inv * alphaBeta_c;
			std::cout << "ERROR1: " << E.norm() << std::endl;
			
			std::cout << Inv << std::endl;
#endif
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
			
			inv_self->reserve(res_x);
			
			dbcsr::iterator_t<2,double> iter_inv(*inv_self);
			iter_inv.start();
			
			while (iter_inv.blocks_left()) {
				iter_inv.next();
				
				int ix0 = iter_inv.idx()[0];
				int ix1 = iter_inv.idx()[1];
			
				int xoff0 = xab_offs[xab_mapping[ix0]];
				int xoff1 = xab_offs[xab_mapping[ix1]];
			
				int xsize0 = x[ix0];
				int xsize1 = x[ix1];
					
				Eigen::MatrixXd blk = Inv.block(xoff0,xoff1,xsize0,xsize1);
				
				bool found = true;
				double* blkptr = inv_self->get_block_p(iter_inv.idx(), found);
					
				std::copy(blk.data(), blk.data() + xsize0*xsize1, blkptr);
				
			}
			
			iter_inv.stop();
							 
			inv_self->filter(dbcsr::global::filter_eps);
			//dbcsr::print(*inv_self);
						
			dbcsr::contract(*inv_self, *eri_self, *c_xbb_self)
				.alpha(1.0).beta(1.0)
				.filter(dbcsr::global::filter_eps)
				.perform("XY, YMN -> XMN");
		
#if 0		
			dbcsr::contract(*met, *c_xbb_self, *eri_self)
				.alpha(-1.0).beta(1.0)
				.filter(dbcsr::global::filter_eps)
				.perform("XY, YMN -> XMN");
				
			std::cout << "SUM" << std::endl;
			dbcsr::print(*eri_self);
#endif
		
			eri_self->clear();
			inv_self->clear();
			
				
		} // end loop over atom pairs
		
		//dbcsr::print(*c_xbb_self);
		
		arrvec<int,3> c_res;
		
		dbcsr::iterator_t<3> iter_c_self(*c_xbb_self);
		iter_c_self.start();
		
		while (iter_c_self.blocks_left()) {
			iter_c_self.next();
			
			auto& idx = iter_c_self.idx();
			
			c_res[0].push_back(idx[0]);
			c_res[1].push_back(idx[1]);
			c_res[2].push_back(idx[2]);
			
		}
		
		iter_c_self.stop();
		
		//std::cout << "RESERVING" << std::endl;
		c_xbb_centered->reserve(c_res);
				
		for (size_t iblk = 0; iblk != c_res[0].size(); ++iblk) {
			
			std::array<int,3> idx = {c_res[0][iblk], 
					c_res[1][iblk], c_res[2][iblk]};
			
			int ntot = x[idx[0]] * b[idx[1]] * b[idx[2]];
			
			bool found = true;
					
			double* ptr_self = c_xbb_self->get_block_p(idx, found);
			double* ptr_all = c_xbb_centered->get_block_p(idx, found);
			
			std::copy(ptr_self, ptr_self + ntot, ptr_all);
			
		}
		
		//dbcsr::print(*c_xbb_centered);
		
		c_xbb_self->clear();

#if 0
		} // end if
		
		MPI_Barrier(m_world.comm());
		
		} // end proc loop
#endif
	
		dbcsr::copy(*c_xbb_centered, *c_xbb_dist).move_data(true).sum(true).perform();
		
	} // end loop over atom pair batches
		
	//dbcsr::print(*c_xbb_dist);
	
	time_form_cfit.finish();
	
	TIME.finish();
	
	return c_xbb_dist;
	
}
	
} // end namespace
