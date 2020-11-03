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

	auto blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
	auto blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
		
	arrvec<int,3> blkmaps = {blkmap_x, blkmap_b, blkmap_b};
	
	auto c_xbb_batched = dbcsr::btensor_create<3>()
		.name(m_mol->name() + "_c_xbb_batched")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.batch_dims(bdims)
		.blk_map(blkmaps)
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
	
	if (cfit_occupation > 100) throw std::runtime_error(
		"Fitting coefficients occupation more than 100%");

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

struct block_info {
	double alpha;
	double radius;
	std::array<double,3> pos;
};

std::vector<block_info> get_block_info(desc::cluster_basis& cbas) {
	
	std::vector<block_info> blkinfo(cbas.size());
	auto radii = cbas.radii();
	auto min_alphas = cbas.min_alpha();
	
	for (int ic = 0; ic != cbas.size(); ++ic) {
		blkinfo[ic].pos = cbas[ic][0].O;
		blkinfo[ic].radius = radii[ic];
		blkinfo[ic].alpha = min_alphas[ic];
	}
	
	return blkinfo;
	
}

dbcsr::sbtensor<3,double> dfitting::compute_qr(dbcsr::shared_matrix<double> s_xx_inv, 
		dbcsr::shared_matrix<double> m_xx, dbcsr::shared_pgrid<3> spgrid3_xbb,
		shared_screener scr_s, std::array<int,3> bdims, dbcsr::btype mytype, 
		bool atomic)
{
	
	TIME.start();
	
	auto aofac = std::make_shared<aofactory>(m_mol, m_world);
	
	double T = 1e-5;
	double R2 = 40;
	
	auto x = m_mol->dims().x();
	auto b = m_mol->dims().b();
	
	auto xoff = m_xx->row_blk_offsets();
	auto boff = b;
	int off = 0;
	for (int i = 0; i != b.size(); ++i) {
		boff[i] = off;
		off += b[i];
	}
	
	arrvec<int,2> xx = {x,x};
	arrvec<int,3> xbb = {x,b,b};
	
	int nbf = m_mol->c_basis()->nbf();
	int nxbf = m_mol->c_dfbasis()->nbf();
	std::array<int,3> xbbsizes = {1,nbf,nbf};
	
	auto spgrid3_global_1bb = dbcsr::create_pgrid<3>(m_world.comm())
		.tensor_dims(xbbsizes)
		.get();
		
	auto spgrid3_local_1bb = dbcsr::create_pgrid<3>(MPI_COMM_SELF)
		.tensor_dims(xbbsizes)
		.get();
	
	auto spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	auto dims = spgrid3_global_1bb->dims();
	
	for (auto p : dims) {
		LOG.os<1>(p, " ");
	} LOG.os<1>('\n');
	
	LOG.os<>("Grid size: ", m_world.nprow(), " ", m_world.npcol(), '\n');
		
	vec<int> d0(x.size(),0);
	vec<int> d1(b.size());
	vec<int> d2(b.size());
	
	std::vector<int> blkmap_b;
	std::vector<int> blkmap_x;
	
	if (atomic) {
		blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
		blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
	} else {
		blkmap_b.resize(b.size());
		blkmap_x.resize(x.size());
		std::iota(blkmap_b.begin(), blkmap_b.end(), 0);
		std::iota(blkmap_x.begin(), blkmap_x.end(), 0);
	}
	
	auto is_diff = m_mol->c_basis()->diffuse();
	
	const int natoms = m_mol->atoms().size();
	// make sure that each process has one atom block, but diffuse
	// and tight blocks are separated
	int offset = 0;
	auto dist = blkmap_b;
	for (int i = 0; i != blkmap_b.size(); ++i) {
		dist[i] = (!is_diff[i]) ? dist[i] : dist[i] + natoms;
	}
	
	for (int i = 0; i != d1.size(); ++i) {
		d1[i] = dist[i] % dims[1];
		d2[i] = dist[i] % dims[2];
	}
	
	int atom_nblks = *(std::max_element(dist.begin(), dist.end())) + 1;
	
	std::vector<std::vector<int>> atom_blocks;
	std::vector<int> sub_block;
	int prev_centre = 0;
	
	for (int i = 0; i != b.size(); ++i) {
		int current_centre = dist[i];
		if (current_centre != prev_centre) {
			atom_blocks.push_back(sub_block);
			sub_block.clear();
		}
		sub_block.push_back(i);
		if (i == b.size() - 1) {
			atom_blocks.push_back(sub_block);
		}
		prev_centre = current_centre;
	}
	
	if (m_world.rank() == 0) {
		std::cout << "ATOM BLOCKS: " << std::endl;
		for (auto p : atom_blocks) {
			for (auto i : p) {
				std::cout << i << " ";
			} std::cout << std::endl;
		}

		std::cout << "DISTS: " << std::endl;
		for (auto i : d1) std::cout << i << " ";
		std::cout << std::endl;
		for (auto i : d2) std::cout << i << " ";
		std::cout << std::endl;
	}
	
	arrvec<int,3> distsizes = {d0,d1,d2};
	
	// a distribution where a process owns ALL (X) blocks for a certain (μν) atomic pair
	dbcsr::dist_t<3> dist3_global_1bb(spgrid3_global_1bb, distsizes);
	
	// form tensors
	
	auto prs_xbb_global_1bb = dbcsr::tensor_create<3,double>()
		.name("prs_xbb_global_1bb")
		.ndist(dist3_global_1bb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	auto prs_xbb_global_xbb = dbcsr::tensor_create<3,double>()
		.name("prs_xbb_global_xbb")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	auto c_xbb_global_1bb = dbcsr::tensor_create<3,double>()
		.name("c_xbb_global_1bb")
		.ndist(dist3_global_1bb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	auto ovlp_xbb_global = dbcsr::tensor_create<3,double>()
		.name("eri_local")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	auto eri_local = dbcsr::tensor_create<3,double>()
		.name("eri_local")
		.pgrid(spgrid3_local_1bb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	arrvec<int,3> blkmaps = {blkmap_x, blkmap_b, blkmap_b};
		
	auto c_xbb_batched = dbcsr::btensor_create<3,double>()
		.name("cqr_xbb_batched")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.blk_map(blkmaps)
		.batch_dims(bdims)
		.btensor_type(mytype)
		.print(0)
		.get();
		
	auto s_xx_inv_global = dbcsr::tensor_create<2,double>()
		.name("s_xx_inv_global")
		.pgrid(spgrid2)
		.blk_sizes(xx)
		.map1({0}).map2({1})
		.get();
	
	dbcsr::copy_matrix_to_tensor(*s_xx_inv, *s_xx_inv_global);
	
	auto b_basis = m_mol->c_basis();
	auto x_basis = m_mol->c_dfbasis();
	int nxblks = x.size();
	
	// get minimum exponent and position for each block
	auto blkinfo_x = get_block_info(*x_basis);
	auto blkinfo_b = get_block_info(*b_basis);

#if 1
	if (m_world.rank() == 0) {
	std::cout << "X BASIS INFO: " << std::endl;
	for (auto s : blkinfo_x) {
		std::cout << "EXP: " << s.alpha << std::endl;
		std::cout << "RADIUS: " << s.radius << std::endl;
		std::cout << "POS: " << s.pos[0] << " " << s.pos[1]
			<< " " << s.pos[2] << std::endl;
	}
	}
#endif	

	auto m_xx_eigen = dbcsr::matrix_to_eigen(m_xx);
	
	Eigen::HouseholderQR<Eigen::MatrixXd> full_qr, blk_qr;
	bool full_is_computed = false;
	
	auto ovlp_blklocidx = ovlp_xbb_global->blks_local();
		
	// ====== LOOP OVER NU BATCHES ==============
	
	c_xbb_batched->compress_init({2}, vec<int>{0}, vec<int>{1,2});
	
	for (int ibatch_nu = 0; ibatch_nu != c_xbb_batched->nbatches(2); ++ibatch_nu) {
		
		LOG.os<>("BATCH: ", ibatch_nu, '\n');
		
		auto blk_nu = c_xbb_batched->blk_bounds(2,ibatch_nu);
		
		std::vector<arrvec<int,2>> atom_blk_list;
		int npairs = 0;
	
		for (auto iblock : atom_blocks) {
			for (auto jblock : atom_blocks) {
				
				// find out processes for atom blocks
				// due to the dsitribution we created, we are guarenteed
				// that each process has a whole block
				int rproc = d1[iblock[0]];
				int cproc = d2[jblock[0]];
				
				if (m_world.myprow() == rproc && m_world.mypcol() == cproc) 
				{
					arrvec<int,2> sublist;
					
					for (auto m : iblock) {
						sublist[0].push_back(m);
					}
					
					for (auto n : jblock) {
						if (n < blk_nu[0] || n > blk_nu[1]) continue;
						sublist[1].push_back(n);
					}
					
					npairs += sublist[1].size() * sublist[0].size();
					if (!sublist[1].empty()) atom_blk_list.push_back(sublist);
				}
			}
		}
				
		LOG(-1).os<>("Number of pairs on rank ", m_world.rank(), ": ", 
			npairs, '\n');
			
		for (int iproc = 0; iproc != m_world.size(); ++iproc) {
			if (iproc == m_world.rank()) {
				std::cout << "PROC " << iproc << std::endl;
				for (auto& subset : atom_blk_list) {
					for (auto& p : subset[0]) {
						std::cout << p << " ";
					} std::cout << std::endl;
					for (auto& p : subset[1]) {
						std::cout << p << " ";
					} std::cout << std::endl;
				}
			}
			MPI_Barrier(m_world.comm());
		}
				
		auto& time1 = TIME.sub("1");
		auto& time2 = TIME.sub("2");
		auto& time3 = TIME.sub("3");
		auto& time4 = TIME.sub("4");
		auto& time5 = TIME.sub("5");
		auto& time6 = TIME.sub("6");
		
		// compute overlap
		
		auto xblks = c_xbb_batched->full_blk_bounds(0);
		auto mblks = c_xbb_batched->full_blk_bounds(1);
		auto nblks = c_xbb_batched->blk_bounds(2, ibatch_nu);
		
		arrvec<int,3> ovlp_blkidx;
		auto dist = [](std::array<double,3>& p1, std::array<double,3>& p2) {
			return sqrt(
					pow(p1[0] - p2[0],2.0) +
					pow(p1[1] - p2[1],2.0) +
					pow(p1[2] - p2[2],2.0));
		};				
		
		// reserve blocks 
		
		int nu_blks = nblks[1] - nblks[0] + 1;
		
		long long int nblktot = x.size() * b.size() * nu_blks;
		long long int nblkloc = 0;
		long long int nblk = 0;
		
		for (auto ix : ovlp_blklocidx[0]) {
			for (auto imu : ovlp_blklocidx[1]) {
				
				auto& pos_x = blkinfo_x[ix].pos;
				auto& pos_m = blkinfo_b[imu].pos;
				double r_x = blkinfo_x[ix].radius;
				double r_mu = blkinfo_b[imu].radius;
				
				double ab_xm = dist(pos_x, pos_m);
				if (ab_xm > r_x + r_mu) continue; 
				
				for (auto inu : ovlp_blklocidx[2]) {
					
					if (inu < nblks[0] || inu > nblks[1]) continue;
					
					auto& pos_n = blkinfo_b[inu].pos;
					double r_nu = blkinfo_b[inu].radius;
					
					double ab_xn = dist(pos_x, pos_n);
					if (ab_xn > r_x + r_mu) continue;
					
					ovlp_blkidx[0].push_back(ix);
					ovlp_blkidx[1].push_back(imu);
					ovlp_blkidx[2].push_back(inu);
					
					++nblkloc;
					
				}
			}
		}
		
		MPI_Allreduce(&nblkloc, &nblk, 1, MPI_LONG_LONG_INT, MPI_SUM, m_world.comm());
		LOG.os<>("FILTERED BLOCKS: ", nblk, "/", nblktot, '\n');
				
		ovlp_xbb_global->reserve(ovlp_blkidx);
		
		aofac->ao_3c1e_ovlp_setup();
		aofac->ao_3c_fill(ovlp_xbb_global);
		
		//ovlp_xbb_global->filter(dbcsr::global::filter_eps);
		
		//dbcsr::print(*ovlp_xbb_global);
		
		// compute c_xbb = s^1/2 Xmn
		vec<vec<int>> mn_bounds = {
			c_xbb_batched->full_bounds(1),
			c_xbb_batched->bounds(2,ibatch_nu)
		};
		
		dbcsr::contract(*s_xx_inv_global, *ovlp_xbb_global, *prs_xbb_global_xbb)
			.filter(T)
			.bounds3(mn_bounds)
			.perform("XY, Ymn -> Xmn");
			
		ovlp_xbb_global->clear();
		
		//dbcsr::print(*prs_xbb_global_xbb);
		
		dbcsr::copy(*prs_xbb_global_xbb, *prs_xbb_global_1bb)
			.move_data(true)
			.perform();
	
		//dbcsr::print(*prs_xbb_global_1bb);
	
#if 0
	for (int iproc = 0; iproc != m_world.size(); ++iproc) {
	if (iproc == m_world.rank()) {
#endif
	
		int nsubset = 0;
	
		// now loop over munu block pairs
		for (auto& subset : atom_blk_list) {
			
			std::cout << "Subset " << nsubset++ << std::endl;
			
			std::vector<int> mu_blks = subset[0];
			std::vector<int> nu_blks = subset[1];
			
			for (auto& p : subset[0]) {
				std::cout << p << " ";
			} std::cout << std::endl;
			for (auto& p : subset[1]) {
				std::cout << p << " ";
			} std::cout << std::endl;
			
			std::cout << "DIMS: " << mu_blks.size() << " x " << nu_blks.size() << std::endl;
			
			
			vec<bool> blk_P_bool(x.size(),false);
			
			std::array<int,3> idx = {0,0,0};
			std::array<int,3> size = {0,0,0};
						
			// get all relevant P functions
			for (auto imu : mu_blks) {
				for (auto inu : nu_blks) {
				
					idx[1] = imu;
					idx[2] = inu;
					
					size[1] = b[imu];
					size[2] = b[inu];
				
					bool keep = false;
					
					for (int ix = 0; ix != x.size(); ++ix) {
						bool found = false;
						idx[0] = ix;
						size[0] = x[ix];
						auto blk = prs_xbb_global_1bb->get_block(idx, size, found);
						if (!found) continue;
						
						auto max_iter = std::max_element(blk.data(), blk.data() + blk.ntot(),
							[](double a, double b) {
								return fabs(a) < fabs(b);
							}
						);
						
						if (fabs(*max_iter) > T) blk_P_bool[ix] = true;
						
						/*double tot = 0.0;
						for (int i = 0; i != blk.ntot(); ++i) {
							tot += fabs(blk.data()[i]);
						}
						
						tot /= blk.ntot();
						if (tot > T) blk_P_bool[ix] = true;*/
										
					}
			}}
			
			vec<int> blk_P;
			blk_P.reserve(x.size());
			for (int ix = 0; ix != x.size(); ++ix) {
				if (blk_P_bool[ix]) blk_P.push_back(ix);
			}
			
			int nblkp = blk_P.size();
			std::cout << "FUNCS: " << nblkp << "/" << x.size() << std::endl;
			
			if (nblkp == 0) continue;
			
			time2.start();
			
			std::vector<bool> blk_Q_bool(x.size(),false); // whether block x is involved 
			
			for (int ix = 0; ix != nxblks; ++ix) {
				for (auto ip : blk_P) {
					
					auto& pos_x = blkinfo_x[ix].pos;
					auto& pos_p = blkinfo_x[ip].pos;
					double alpha_x = blkinfo_x[ix].alpha;
					double alpha_p = blkinfo_x[ip].alpha;
					
					double f = (alpha_x * alpha_p) / (alpha_x + alpha_p)
						* dist(pos_x, pos_p);
						
					if (f < R2) blk_Q_bool[ix] = true;
	
				}
			}
			
			std::vector<int> blk_Q;
			blk_Q.reserve(x.size());
			for (int ix = 0; ix != x.size(); ++ix) {
				if (blk_Q_bool[ix]) blk_Q.push_back(ix);
			}
			
			int nblkq = blk_Q.size();
			std::cout << "NEW SET: " << nblkq << "/" << x.size() << std::endl;

			time2.finish();
			
			time3.start();
			
			arrvec<int,3> blk_idx;
			blk_idx[0] = blk_Q;
			blk_idx[1] = mu_blks;
			blk_idx[2] = nu_blks;
			
			int nb = 0; // number of total nb functions
			int mstride = 0;
			int nstride = 0;
					
			for (auto imu : mu_blks) {
				mstride += b[imu];
			}
			
			for (auto inu : nu_blks) {
				nstride += b[inu];
			}
			
			nb = nstride * mstride;
			
			// generate integrals
			aofac->ao_3c2e_setup(metric::coulomb);
			aofac->ao_3c_fill_idx(eri_local, blk_idx, nullptr);
			time3.finish();
			
			//dbcsr::print(*eri_local);
			
			// how many x functions?
			int nq = 0;
			int np = 0;
			
			for (auto iq : blk_Q) {
				nq += x[iq];
			}
			
			for (auto ip : blk_P) {
				np += x[ip];
			}
			
			std::cout << "NP,NQ,NB: " << np << "/" << nq << "/" << nb << std::endl;
			
			time4.start();
			
			// copy to eigen matrix
			Eigen::MatrixXd eris_eigen = Eigen::MatrixXd::Zero(nq,nb);
			Eigen::MatrixXd m_qp_eigen = Eigen::MatrixXd::Zero(nq,np);
			
			int poff = 0;
			int qoff = 0;
			int moff = 0;
			int noff = 0;
			
			for (auto iq : blk_Q) {
				for (auto imu : mu_blks) {
					for (auto inu : nu_blks) {
				
						std::array<int,3> idx = {iq,imu,inu};
						std::array<int,3> size = {x[iq],b[imu],b[inu]};
						bool found = true;
						auto blk = eri_local->get_block(idx,size,found);
						if (!found) continue;
						
						for (int qq = 0; qq != size[0]; ++qq) {
							for (int mm = 0; mm != size[1]; ++mm) {
								for (int nn = 0; nn != size[2]; ++nn) {
									eris_eigen(qq + qoff, mm + moff + (nn+noff)*mstride)
										= blk(qq,mm,nn);
						}}}
						noff += b[inu];
					}
					noff = 0;
					moff += b[imu];
				}
				moff = 0;
				qoff += x[iq];
			}
					
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
			
			eri_local->clear();	
			
			time5.start();
			Eigen::MatrixXd c_eigen;
			
			if (nblkq == x.size() && nblkp == x.size() && !full_is_computed) {
				full_qr.compute(m_qp_eigen);
				full_is_computed = true;
				c_eigen = full_qr.solve(eris_eigen);
			} else if (nblkq == x.size() && nblkp == x.size() && full_is_computed) {
				c_eigen = full_qr.solve(eris_eigen);
			} else if (nq != x.size() || np != x.size()) {
				blk_qr.compute(m_qp_eigen);
				c_eigen = blk_qr.solve(eris_eigen);
			}
			
			time5.finish();
			
			eris_eigen.resize(0,0);
			
			time6.start();
			// transfer it to c_xbb
			arrvec<int,3> cfit_idx;
			
			for (auto ip : blk_P) {
				for (auto imu : mu_blks) {
					for (auto inu : nu_blks) {
						cfit_idx[0].push_back(ip);
						cfit_idx[1].push_back(imu);
						cfit_idx[2].push_back(inu);
					}
				}
			}
			
			c_xbb_global_1bb->reserve(cfit_idx);
			
			poff = 0;
			moff = 0;
			noff = 0;
			
			for (auto ip : blk_P) {
				for (auto imu : mu_blks) {
					for (auto inu : nu_blks) {
				
						std::array<int,3> idx = {ip, imu, inu};
						std::array<int,3> size = {x[ip], b[imu], b[inu]};
						
						dbcsr::block<3,double> blk(size);
						for (int pp = 0; pp != size[0]; ++pp) {
							for (int mm = 0; mm != size[1]; ++mm) {
								for (int nn = 0; nn != size[2]; ++nn) {
									blk(pp,mm,nn) = c_eigen(pp + poff, 
										mm + moff + (nn+noff)*mstride);
						}}}
						c_xbb_global_1bb->put_block(idx, blk);
						noff += b[inu];
					}
					noff = 0;
					moff += b[imu];
				}			
				moff = 0;	
				poff += x[ip];
			}
			time6.finish();
			
		} // end loop over atom block pairs
#if 0	
		}
		MPI_Barrier(m_world.comm());
		}
#endif
		
		c_xbb_global_1bb->filter(dbcsr::global::filter_eps);
		
		c_xbb_batched->compress({ibatch_nu}, c_xbb_global_1bb);
		prs_xbb_global_1bb->clear();
		c_xbb_global_1bb->clear();

	} // end loop over batches
	
	c_xbb_batched->compress_finalize();
	
	double occupation = c_xbb_batched->occupation() * 100;
	LOG.os<>("Occupation of QR fitting coefficients: ", occupation, "%\n");
	
	if (occupation > 100) throw std::runtime_error(
		"Fitting coefficients occupation more than 100%");
	
	TIME.finish();
	
	TIME.print_info();
	//exit(0);
	
	return c_xbb_batched;
	
}

// for C_Xμν computes sparsity between X and ν 
std::shared_ptr<Eigen::MatrixXi> dfitting::compute_idx(
	dbcsr::sbtensor<3,double> cfit_xbb)
{
	auto x = m_mol->dims().x();
	auto b = m_mol->dims().b();
	
	Eigen::MatrixXi idx_local = Eigen::MatrixXi::Zero(x.size(),b.size());
	Eigen::MatrixXi idx_global = Eigen::MatrixXi::Zero(x.size(),b.size());
	
	cfit_xbb->decompress_init({2}, vec<int>{0}, vec<int>{1,2});
	int nbatches = (cfit_xbb->get_type() == dbcsr::btype::core) ?
		1 : cfit_xbb->nbatches(2);
		
	for (int inu = 0; inu != nbatches; ++inu) {
		cfit_xbb->decompress({inu});
		auto cfit = cfit_xbb->get_work_tensor();
		
		dbcsr::iterator_t<3> iter(*cfit);
		iter.start();
		
		while (iter.blocks_left()) {
			iter.next();
			auto& idx = iter.idx();
			
			int ix = idx[0];
			int inu = idx[2];
			
			idx_local(ix,inu) = 1;
		}
		
		iter.stop();
		
	}
	
	cfit_xbb->decompress_finalize();
	
	MPI_Allreduce(idx_local.data(), idx_global.data(), x.size() * b.size(),
		MPI_INT, MPI_LOR, m_world.comm());
		
	int nblks = 0;
		
	for (int ix = 0; ix != x.size(); ++ix) {
		for (int inu = 0; inu != b.size(); ++inu) {
			if (idx_global(ix,inu)) ++nblks;
		}
	}
	
	LOG.os<>("BLOCKS: ", nblks, "/", x.size()*b.size(), '\n');
	
	auto out_ptr = std::make_shared<Eigen::MatrixXi>(std::move(idx_global));
	return out_ptr;

}
	
} // end namespace
