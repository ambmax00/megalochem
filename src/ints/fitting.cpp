#include "ints/fitting.h"
#include "ints/aofactory.h"
#include "extern/lapack.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include "utils/constants.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>

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
	
	auto print = [](auto v) {
		for (auto e : v) {
			std::cout << e << " ";
		} std::cout << std::endl;
	};
	
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

	std::cout << "NBATCHES: " << nbatches_x << " " << nbatches_b << std::endl;

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
	//c_xbb_0_12->batched_contract_init();
	//c_xbb_1_02->batched_contract_init();
	
	//auto e = eri_batched->get_work_tensor();
	//std::cout << "ERI OCC: " << e->occupation() * 100 << std::endl;
	
	for (int inu = 0; inu != c_xbb_batched->nbatches(2); ++inu) {
		
		fetch.start();
		eri_batched->decompress({inu});
		auto eri_0_12 = eri_batched->get_work_tensor();
		fetch.finish();
		
		//inv->batched_contract_init();
			
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
					
				c_xbb_batched->compress({inu,ix}, c_xbb_1_02);
				
		}
		
		//inv->batched_contract_finalize();
		
	}
	
	c_xbb_batched->compress_finalize();
	eri_batched->decompress_finalize();
	//c_xbb_0_12->batched_contract_finalize();
	//c_xbb_1_02->batched_contract_finalize();
	
	//auto cw = c_xbb_batched->get_work_tensor();
	//dbcsr::print(*cw);
	
	double cfit_occupation = c_xbb_batched->occupation() * 100;
	LOG.os<1>("Occupancy of c_xbb: ", cfit_occupation, "%\n");
	
	assert(cfit_occupation <= 100);

	TIME.finish();
	
	LOG.os<1>("Done.\n");
	
	// ========== END ==========	
	
	return c_xbb_batched;
	
}

dbcsr::shared_tensor<3,double> dfitting::compute_pari(dbcsr::sbtensor<3,double> eri_batched,
	dbcsr::shared_matrix<double> s_xx, shared_screener scr_s) {
	
	auto aofac = std::make_shared<aofactory>(m_mol, m_world);
	aofac->ao_3c2e_setup(metric::coulomb);
	
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
	
	auto get_centre = [&atoms](desc::Shell& s) {
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
	
	auto spgrid3_pair = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(xbbsizes)
		.get();
	
	auto spgrid2_self = dbcsr::create_pgrid<2>(MPI_COMM_SELF).get();
	
	auto spgrid3_self = dbcsr::create_pgrid<3>(MPI_COMM_SELF).get();
		
	auto dims = spgrid3_pair->dims();
	
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
	
	dbcsr::dist_t<3> cdist(spgrid3_pair, distsizes);
	
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
		
	auto c_xbb_self_mn = dbcsr::tensor_create<3,double>()
		.name("c_xbb_self_mn")
		.pgrid(spgrid3_self)
		.map1({0}).map2({1,2})
		.blk_sizes(xbb)
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
			
			aofac->ao_3c_fill_idx(eri_self, blks, scr_s);	
			
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
			//std::cout << "MY ALPHABETA: " << std::endl;
			
			//std::cout << alphaBeta << std::endl;
			
			std::cout << "ALPHA TENSOR: " << std::endl;
			met->filter(dbcsr::global::filter_eps);
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
			
			//std::cout << Inv << std::endl;
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
			//std::cout << "INV" << std::endl;
			//dbcsr::print(*inv_self);
			
			//std::cout << "INTS" << std::endl;
			//dbcsr::print(*eri_self);
			
			dbcsr::contract(*inv_self, *eri_self, *c_xbb_self_mn)
				.alpha(1.0)
				.filter(dbcsr::global::filter_eps)
				.perform("XY, YMN -> XMN");
		
#if 0		
			dbcsr::contract(*met, *c_xbb_self_mn, *eri_self)
				.alpha(-1.0).beta(1.0)
				.filter(dbcsr::global::filter_eps)
				.perform("XY, YMN -> XMN");
				
			std::cout << "SUM" << std::endl;
			dbcsr::print(*eri_self);
			auto nblk = eri_self->num_blocks_total();
			std::cout << nblk << std::endl;
#endif
		
			dbcsr::copy(*c_xbb_self_mn, *c_xbb_self)
				.move_data(true)
				.sum(true)
				.perform();
		
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

struct exp_pos {
	double exp;
	std::array<double,3> pos;
};

dbcsr::sbtensor<3,double> dfitting::compute_qr(dbcsr::shared_matrix<double> s_xx_inv, 
		dbcsr::shared_matrix<double> m_xx, dbcsr::shared_pgrid<3> spgrid3_xbb,
		shared_screener scr_s, std::array<int,3> bdims, dbcsr::btype mytype)
{
	
	auto aofac = std::make_shared<aofactory>(m_mol, m_world);
	
	double T = 1e-7;
	double R = 40 * pow(BOHR_RADIUS,2);
	
	auto x = m_mol->dims().x();
	auto b = m_mol->dims().b();
	
	auto xoff = m_xx->row_blk_offsets();
	
	arrvec<int,2> xx = {x,x};
	arrvec<int,3> xbb = {x,b,b};
	
	int nbf = m_mol->c_basis()->nbf();
	std::array<int,3> xbbsizes = {1,nbf,nbf};
	
	auto spgrid3_global_1bb = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(xbbsizes)
		.get();
		
	auto spgrid3_local_1bb = dbcsr::create_pgrid<3>(MPI_COMM_SELF)
		.tensor_dims(xbbsizes)
		.get();
	
	auto spgrid2_self = dbcsr::create_pgrid<2>(MPI_COMM_SELF).get();
	
	auto dims = spgrid3_global_1bb->dims();
	
	for (auto p : dims) {
		LOG.os<1>(p, " ");
	} LOG.os<1>('\n');
	
	LOG.os<>("Grid size: ", m_world.nprow(), " ", m_world.npcol(), '\n');
		
	vec<int> d0(x.size(),0.0);
	vec<int> d1(b.size());
	vec<int> d2(b.size());
	
	for (int i = 0; i != d1.size(); ++i) {
		d1[i] = i % dims[1];
		d2[i] = i % dims[2];
	}
	
	arrvec<int,3> distsizes = {d0,d1,d2};
	
	// a distribution where a process owns ALL (X) blocks for a certain (μν) pair
	dbcsr::dist_t<3> dist3_global_1bb(spgrid3_global_1bb, distsizes);
	
	// form tensors
	
	auto c_xbb_global_1bb = dbcsr::tensor_create<3,double>()
		.name("c_xbb_global_1bb")
		.ndist(dist3_global_1bb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	auto c_xbb_global_xbb = dbcsr::tensor_create<3,double>()
		.name("c_xbb_global_xbb")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	auto c_xbb_local_1bb = dbcsr::tensor_create<3,double>()
		.name("c_xbb_local_xbb")
		.pgrid(spgrid3_local_1bb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	auto eri_local = dbcsr::tensor_create<3,double>()
		.name("eri_local")
		.pgrid(spgrid3_local_1bb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	auto c_xbb_batched = dbcsr::btensor_create<3,double>()
		.name("cqr_xbb_batched")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.batch_dims(bdims)
		.btensor_type(mytype)
		.print(0)
		.get();
		
	auto s_xx_inv_local = dbcsr::tensor_create<2,double>()
		.name("s_xx_inv_local")
		.pgrid(spgrid2_self)
		.blk_sizes(xx)
		.map1({0}).map2({1})
		.get();
	
	s_xx_inv_local->reserve_all();
	
	// copy s_xx_inv
	auto s_xx_inv_eigen = dbcsr::matrix_to_eigen(s_xx_inv);
	auto m_xx_eigen = dbcsr::matrix_to_eigen(m_xx);
	
	dbcsr::iterator_t<2,double> iter(*s_xx_inv_local);
	
	iter.start();
	while (iter.blocks_left()) {
		iter.next();
		
		auto& idx = iter.idx();
		auto& off = iter.offset();
		auto& size = iter.size();
		
		dbcsr::block<2,double> blk(size);
		
		for (int j = 0; j != size[1]; ++j) {
			for (int i = 0; i != size[0]; ++i) {
				blk(i,j) = s_xx_inv_eigen(i + off[0], j + off[1]);
			}
		}
		
		s_xx_inv_local->put_block(idx, blk);
		
	}
	
	s_xx_inv_local->filter(dbcsr::global::filter_eps);
	
	iter.stop();
	s_xx_inv_eigen.resize(0,0);
	
	arrvec<int,3> blk_idx;
	blk_idx[0].resize(x.size());
	blk_idx[1].resize(1);
	blk_idx[2].resize(1);
	
	auto x_basis = m_mol->c_dfbasis();
	int nxblks = x.size();
	
	// get minimum exponent and position for each block
	std::vector<exp_pos> min_exp_x(nxblks);
	for (int i = 0; i != nxblks; ++i) {
		exp_pos set;
		
		double min_exp = std::numeric_limits<double>::max();
		auto& cluster = x_basis->at(i);
		
		for (auto& con_gauss : cluster) {
			for (auto& exp : con_gauss.alpha) {
				min_exp = std::min(min_exp, exp);
			}
		}
		
		set.exp = min_exp;
		set.pos = cluster[0].O;
		
		min_exp_x[i] = set;
		
	}

#if 0
	if (m_world.rank() == 0) {
	std::cout << "X BASIS INFO: " << std::endl;
	for (auto s : min_exp_x) {
		std::cout << "EXP: " << s.exp << std::endl;
		std::cout << "POS: " << s.pos[0] << " " << s.pos[1]
			<< " " << s.pos[2] << std::endl;
	}
	}
#endif	
		
	// ====== LOOP OVER NU BATCHES ==============
	
	c_xbb_batched->compress_init({2}, vec<int>{0}, vec<int>{1,2});
	
	for (int ibatch_nu = 0; ibatch_nu != c_xbb_batched->nbatches(2); ++ibatch_nu) {
		
		LOG.os<>("BATCH: ", ibatch_nu, '\n');
		
		auto blk_nu = c_xbb_batched->blk_bounds(2,ibatch_nu);
		
		std::vector<std::pair<int,int>> mn_blk_list;
	
		for (int m = 0; m != b.size(); ++m) {
			for (int n = blk_nu[0]; n != blk_nu[1]+1; ++n) {
				
				int rproc = m % dims[1];
				int cproc = n % dims[2];
							
				if (rproc == m_world.myprow() && cproc == m_world.mypcol()) {
					std::pair<int,int> p = {m,n};
					mn_blk_list.push_back(p);
				}
				
			}
		}
		
		LOG(-1).os<>("Number of pairs on rank ", m_world.rank(), ": ", 
			mn_blk_list.size(), '\n');
		
		size_t npairs = mn_blk_list.size();
		
	auto& time1 = TIME.sub("1");
	auto& time2 = TIME.sub("2");
	auto& time3 = TIME.sub("3");
	auto& time4 = TIME.sub("4");
	auto& time5 = TIME.sub("5");
	auto& time6 = TIME.sub("6");
	
#if 0
	for (int iproc = 0; iproc != m_world.size(); ++iproc) {
	if (iproc == m_world.rank()) {
#endif
	
	for (auto& mn_pair : mn_blk_list) {
	
		//std::cout << "PROC: " << m_world.rank() << std::endl;
		std::cout << "PAIR: " << mn_pair.first << " " << mn_pair.second << std::endl;
	
		time1.start();
	
		int mu = mn_pair.first;
		int nu = mn_pair.second;
	
		blk_idx[0] = vec<int>(x.size());
		std::iota(blk_idx[0].begin(), blk_idx[0].end(), 0);
		blk_idx[1][0] = mu;
		blk_idx[2][0] = nu;
	
		aofac->ao_3c1e_ovlp_setup();
		aofac->ao_3c_fill_idx(eri_local, blk_idx, scr_s);
		
		auto c_idx = dbcsr::contract(*s_xx_inv_local, *eri_local,
			*c_xbb_local_1bb)
			.filter(T)
			.get_index("XY, Ymn -> Xmn");
		
		std::vector<bool> blk_P_bool(x.size(),false);
		for (auto i : c_idx[0]) {
			blk_P_bool[i] = true;
		}
		
		std::vector<int> blk_P;
		for (int i = 0; i != x.size(); ++i) {
			if (blk_P_bool[i]) blk_P.push_back(i);
		}
		
		int npblks = blk_P.size();
		
		//std::cout << "OVERLAP" << std::endl;
		//dbcsr::print(*eri_local);
		eri_local->clear();
		
		time1.finish();

		if (npblks == 0) {
			std::cout << "SKIPPING" << std::endl;
			continue;
		}
#if 0
		std::cout << "Primary fitting set" << std::endl;
		for (auto p : blk_P) {
			std::cout << p << " ";
		} std::cout << std::endl;
#endif
		
#if 0
		std::cout << "Now generating another set..." << std::endl;
#endif		
		
		time2.start();
		
		std::vector<bool> blk_Q_bool(nxblks,false); // whether block x is involved 
		
		for (int ix = 0; ix != nxblks; ++ix) {
			for (auto ip : blk_P) {
				
				auto& pos_x = min_exp_x[ix].pos;
				auto& pos_p = min_exp_x[ip].pos;
				double alpha_x = min_exp_x[ix].exp;
				double alpha_p = min_exp_x[ip].exp;

				double f = (alpha_x * alpha_p) / (alpha_x + alpha_p)
					* (pow(pos_x[0] - pos_p[0],2)
					+ pow(pos_x[1] - pos_p[1],2)
					+ pow(pos_x[2] - pos_p[2],2));
					
				if (f < R) blk_Q_bool[ix] = true;
#if 0			
				std::cout << "F: " << f << std::endl;
				if (blk_Q_bool[ix]) {
					std::cout << "IX/IP: " << ix << " " << ip << std::endl;
					std::cout << "POS: " << pos_x[0] << " " << pos_x[1] <<
					" " << pos_x[2] << " " << pos_p[0] << " " << pos_p[1] <<
					" " << pos_p[2] << std::endl;
					std::cout << "ADDING" << std::endl;
				}
#endif				
			}
		}
		
		std::vector<int> blk_Q;
		for (int ix = 0; ix != nxblks; ++ix) {
			if (blk_Q_bool[ix]) blk_Q.push_back(ix);
		}
		
#if 0
		std::cout << "NEW SET: " << std::endl;
		for (auto q : blk_Q) {
			std::cout << q << " ";
		} std::cout << std::endl;
#endif		
		
		time2.finish();
	
		
		time3.start();
		// generate integrals
		aofac->ao_3c2e_setup(metric::coulomb);
		blk_idx[0] = blk_Q;
		aofac->ao_3c_fill_idx(eri_local, blk_idx, scr_s);
		time3.finish();
		
		// how many x functions?
		int nq = 0;
		int np = 0;
		
		for (auto iq : blk_Q) {
			nq += x[iq];
		}
		
		for (auto ip : blk_P) {
			np += x[ip];
		}
		
		time4.start();
		
		// how many b functions?
		int nb = b[mu] * b[nu];
		
		// copy to eigen matrix
		Eigen::MatrixXd eris_eigen = Eigen::MatrixXd::Zero(nq,nb);
		Eigen::MatrixXd cxbb_eigen = Eigen::MatrixXd::Zero(np,nb);
		Eigen::MatrixXd m_qp_eigen = Eigen::MatrixXd::Zero(nq,np);
		
		int poff = 0;
		int qoff = 0;
		int moff = b[mu]; 
		
		for (auto iq : blk_Q) {
			
			std::array<int,3> idx = {iq,mu,nu};
			std::array<int,3> size = {x[iq],b[mu],b[nu]};
			bool found = true;
			auto blk = eri_local->get_block(idx,size,found);
			
			if (!found) continue;
			
			for (int qq = 0; qq != size[0]; ++qq) {
				for (int mm = 0; mm != size[1]; ++mm) {
					for (int nn = 0; nn != size[2]; ++nn) { 
						eris_eigen(qq + qoff, mm + nn*moff)
							= blk(qq,mm,nn);
					}
				}
			}
			
			qoff += size[0];
		}
		
		//dbcsr::print(*eri_local);
		
		//std::cout << "ERIS: " << std::endl;
		//std::cout << eris_eigen << std::endl;
				
		// copy metric to eigen
		for (auto ip : blk_P) {
			
			int sizep = x[ip];
			int poff_m = xoff[ip];
			
			qoff = 0;
			
			for (auto iq : blk_Q) {
				
				int sizeq = x[iq];
				int qoff_m = xoff[iq];
				
				for (int qq = 0; qq != sizeq; ++qq) {
					for (int pp = 0; pp != sizep; ++pp) {
						m_qp_eigen(qq + qoff,pp + poff) = 
							m_xx_eigen(qq + qoff_m, pp + poff_m);
					}
				}
				
				qoff += sizeq;
			}
			poff += sizep;
		}
		
		time4.finish();
		
		//std::cout << "FULL:" << std::endl;
		//std::cout << m_xx_eigen << std::endl;
		
		//std::cout << "M" << std::endl;
		//std::cout << m_qp_eigen << std::endl;
		
		time5.start();
		Eigen::MatrixXd c_eigen = m_qp_eigen.householderQr().solve(eris_eigen);
		time5.finish();
		
		std::cout << "X: " << mu << " " << nu << " " 
			<< blk_P.size() << "/" << x.size() << " "
			<< blk_Q.size() << "/" << x.size() << std::endl;
		
		//std::cout << "C_EIGEN: " << std::endl;
		//std::cout << c_eigen << std::endl;

#if 0
		assert(eris_eigen.isApprox(m_qp_eigen*c_eigen,1e-8));
#endif
		eris_eigen.resize(0,0);
		eri_local->clear();
		
		time6.start();
		// transfer it to c_xbb
		arrvec<int,3> cfit_idx;
		cfit_idx[0] = blk_P;
		cfit_idx[1] = vec<int>(blk_P.size(), mu);
		cfit_idx[2] = vec<int>(blk_P.size(), nu);
		
		c_xbb_global_1bb->reserve(cfit_idx);
		
		poff = 0;
		for (auto ip : blk_P) {
			
			std::array<int,3> idx = {ip, mu, nu};
			std::array<int,3> size = {x[ip], b[mu], b[nu]};
			
			dbcsr::block<3,double> blk(size);
			for (int pp = 0; pp != size[0]; ++pp) {
				for (int mm = 0; mm != size[1]; ++mm) {
					for (int nn = 0; nn != size[2]; ++nn) {
						blk(pp,mm,nn) = c_eigen(pp + poff, mm + nn*moff);
					}
				}
			}
			
			c_xbb_global_1bb->put_block(idx, blk);
			poff += size[0];
			
		}
		time6.finish();
		
	}
	
	//dbcsr::print(*c_xbb_global_1bb);

#if 0
	}
	MPI_Barrier(m_world.comm());
	}
#endif	

	c_xbb_batched->compress({ibatch_nu}, c_xbb_global_1bb);

	}
	
	c_xbb_batched->compress_finalize();
	
	double occupation = c_xbb_batched->occupation() * 100;
	LOG.os<>("Occupation of QR fitting coefficients: ", occupation, "%\n");
	
	assert(occupation <= 100.0);
	
	TIME.print_info();
	//exit(0);
	
	return c_xbb_batched;
	
}
	
} // end namespace
