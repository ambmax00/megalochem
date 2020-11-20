#include "ints/fitting.h"
#include "extern/lapack.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include "utils/constants.h"
#include "utils/scheduler.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>

namespace ints {

dbcsr::sbtensor<3,double> dfitting::compute_pari(dbcsr::shared_matrix<double> s_xx, 
	shared_screener scr_s, std::array<int,3> bdims, dbcsr::btype mytype) {
	
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
	
	auto blkmap_b = cbas->block_to_atom(atoms);
	auto blkmap_x = xbas->block_to_atom(atoms);

	LOG.os<1>("Block to atom mapping (b): \n");	
	if (LOG.global_plev() >= 1) {
		for (auto e : blkmap_b) {
			LOG.os<1>(e, " ");
		} LOG.os<1>('\n');
	}
	
	LOG.os<1>("Block to atom mapping (x): \n");	
	if (LOG.global_plev() >= 1) {
		for (auto e : blkmap_x) {
			LOG.os<1>(e, " ");
		} LOG.os<1>('\n');
	}
		
	// === END ===
	
	// ==================== new atom centered dists ====================
	
	int nbf = m_mol->c_basis()->nbf();
	int dfnbf = m_mol->c_dfbasis()->nbf();
	
	std::array<int,3> xbbsizes = {1,nbf,nbf};
	
	auto spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	
	auto spgrid3_xbb =dbcsr::create_pgrid<3>(m_world.comm()).get();
	
	auto spgrid2_self = dbcsr::create_pgrid<2>(MPI_COMM_SELF).get();
	
	auto spgrid3_self = dbcsr::create_pgrid<3>(MPI_COMM_SELF).get();
		
	LOG.os<>("Grid size: ", m_world.nprow(), " ", m_world.npcol(), '\n');
	
	// === END

	// ===================== setup tensors =============================
	
	arrvec<int,3> blkmaps = {blkmap_x, blkmap_b, blkmap_b};
	
	auto c_xbb_batched = dbcsr::btensor_create<3,double>()
		.name("cpari_xbb_batched")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.blk_map(blkmaps)
		.batch_dims(bdims)
		.btensor_type(mytype)
		.print(1)
		.get();
		
	auto c_xbb_global = dbcsr::tensor_create<3,double>()
		.name("fitting coefficients")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0,1}).map2({2})
		.get();
		
	auto c_xbb_local = dbcsr::tensor_create<3,double>()
		.name("c_xbb_local")
		.pgrid(spgrid3_self)
		.map1({0}).map2({1,2})
		.blk_sizes(xbb)
		.get();
		
	auto c_xbb_AB = dbcsr::tensor_create<3,double>()
		.name("c_xbb_AB")
		.pgrid(spgrid3_self)
		.map1({0}).map2({1,2})
		.blk_sizes(xbb)
		.get();
		
	auto eri_local = 
		dbcsr::tensor_create_template<3,double>(c_xbb_local)
		.name("eri_local").get();
	
	auto inv_local = dbcsr::tensor_create<2,double>()
		.name("inv_local")
		.pgrid(spgrid2_self)
		.map1({0}).map2({1})
		.blk_sizes(xx)
		.get();
	
	arrvec<int,3> blkidx = c_xbb_local->blks_local();
	arrvec<int,3> blkoffsets = c_xbb_local->blk_offsets();
	
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
	
	c_xbb_batched->compress_init({2}, vec<int>{0}, vec<int>{1,2});
	
	int nbatches = c_xbb_batched->nbatches(2); 
	
	// Loop over batches N - they are guaranteed to always include all
	// shells of an atom, i.e. blocks belonging to the same atom are not
	// separated
	
	int64_t task_off = 0;
	
	for (int inu = 0; inu != nbatches; ++inu) {
	
		auto nu_bds = c_xbb_batched->blk_bounds(2,inu);
	
		// get atom range
		int atom_lb = natoms-1;
		int atom_ub = 0;
		
		for (int iblk = nu_bds[0]; iblk != nu_bds[1]+1; ++iblk) {
			atom_lb = std::min(atom_lb, blkmap_b[iblk]);
			atom_ub = std::max(atom_ub, blkmap_b[iblk]);
		}
		
		LOG.os<1>("BATCH: ", inu, " with atoms [", atom_lb, ",", atom_ub, "]\n");
		
		int64_t ntasks = (atom_ub - atom_lb + 1) * natoms;
		int rank = m_world.rank();
		
		std::function<void(int64_t)> workfunc = 
			[rank,natoms,&blkmap_b,&blkmap_x,&aofac,&inv_local,&eri_local,
				&c_xbb_AB,&c_xbb_local,&s_xx_desym,&scr_s,&x,&b,&task_off]
			(int64_t itask) 
		{
			
			int atomA = (itask + task_off) % (int64_t)natoms;
			int atomB = (itask + task_off) / (int64_t)natoms;
		
			std::cout << rank << ": ATOMS " << atomA << " " << atomB << std::endl;
					
			// Get mu/nu indices centered on alpha/beta
			vec<int> mA_idx, nB_idx, xAB_idx;
			
			for (int m = 0; m != b.size(); ++m) {
				if (blkmap_b[m] == atomA) mA_idx.push_back(m);
				if (blkmap_b[m] == atomB) nB_idx.push_back(m);
			}
						
			// get x indices centered on alpha or beta
				
			for (int i = 0; i != x.size(); ++i) {
				if (blkmap_x[i] == atomA || blkmap_x[i] == atomB) 
					xAB_idx.push_back(i);
			}
			
			#if 1
			std::cout << "INDICES: " << std::endl;
			
			for (auto m : mA_idx) {
				std::cout << m << " ";
			} std::cout << std::endl;
			
			for (auto n : nB_idx) {
				std::cout << n << " ";
			} std::cout << std::endl;
			
			for (auto ix : xAB_idx) {
				std::cout << ix << " ";
			} std::cout << std::endl;
			#endif	
			
					
			int mu_nblks = mA_idx.size();
			int nu_nblks = nB_idx.size();		
			int xAB_nblks = xAB_idx.size();
			
			// compute 3c2e ints
		
			arrvec<int,3> blks = {xAB_idx, mA_idx, nB_idx};
			
			aofac->ao_3c_fill_idx(eri_local, blks, scr_s);	
			eri_local->filter(dbcsr::global::filter_eps);
		
			if (eri_local->num_blocks_total() == 0) return;
			
			int m = 0;
			
			// make mapping for compressed metric matrix
			/* | x 0 0 |      | x 0 |
			 * | 0 y 0 |  ->  | 0 y |
			 * | 0 0 z | 
			 */
			 
			vec<int> xAB_sizes(xAB_idx.size());
			vec<int> xAB_offs(xAB_idx.size());
			std::map<int,int> xAB_mapping;
		
			int off = 0;
			for (int k = 0; k != xAB_idx.size(); ++k) {
				int kAB = xAB_idx[k];
				m += x[kAB];
				xAB_sizes[k] = x[kAB];
				xAB_offs[k] = off;
				xAB_mapping[kAB] = k;
				off += xAB_sizes[k];
			}
			
			// get (alpha|beta) matrix 
			Eigen::MatrixXd alphaBeta = Eigen::MatrixXd::Zero(m,m);

#if 1
			auto met = dbcsr::tensor_create_template<2,double>(inv_local)
				.name("metric").get();
				
			met->reserve_all();
#endif		
			// loop over xab blocks
			for (auto ix : xAB_idx) {
				for (auto iy : xAB_idx) {
					
					int xsize = x[ix];
					int ysize = x[iy];
					
					bool found = false;
					double* ptr = s_xx_desym->get_block_data(ix,iy,found);
			
					if (found) {
						
						std::array<int,2> sizes = {xsize,ysize};
						
#if 1		
						std::array<int,2> idx = {ix,iy};
						met->put_block(idx, ptr, sizes);
#endif
						
						Eigen::Map<Eigen::MatrixXd> blk(ptr,xsize,ysize);
												
						int xoff = xAB_offs[xAB_mapping[ix]];
						int yoff = xAB_offs[xAB_mapping[iy]];
						
						alphaBeta.block(xoff,yoff,xsize,ysize) = blk;
						
					}	
					
				}
			}

#if 1	
			std::cout << "Problem size: " << m << " x " << m << std::endl;
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
			
#if 1
			Eigen::MatrixXd E = Eigen::MatrixXd::Identity(m,m);
			
			E -= Inv * alphaBeta_c;
			std::cout << "ERROR1: " << E.norm() << std::endl;
			
#endif
			U.resize(0,0);
			Vt.resize(0,0);
			s.resize(0);
			delete [] work;
			
			// transfer inverse to blocks
			
			arrvec<int,2> res_x;
			
			for (auto ix : xAB_idx) {
				for (auto iy : xAB_idx) {
					res_x[0].push_back(ix);
					res_x[1].push_back(iy);
				}
			}
			
			inv_local->reserve(res_x);
			
			dbcsr::iterator_t<2,double> iter_inv(*inv_local);
			iter_inv.start();
			
			while (iter_inv.blocks_left()) {
				iter_inv.next();
				
				int ix0 = iter_inv.idx()[0];
				int ix1 = iter_inv.idx()[1];
			
				int xoff0 = xAB_offs[xAB_mapping[ix0]];
				int xoff1 = xAB_offs[xAB_mapping[ix1]];
			
				int xsize0 = x[ix0];
				int xsize1 = x[ix1];
					
				Eigen::MatrixXd blk = Inv.block(xoff0,xoff1,xsize0,xsize1);
				
				bool found = true;
				double* blkptr = inv_local->get_block_p(iter_inv.idx(), found);
					
				std::copy(blk.data(), blk.data() + xsize0*xsize1, blkptr);
				
			}
			
			iter_inv.stop();
							 
			inv_local->filter(dbcsr::global::filter_eps);
			
			dbcsr::contract(*inv_local, *eri_local, *c_xbb_AB)
				.filter(dbcsr::global::filter_eps)
				.perform("XY, YMN -> XMN");
		
#if 1		
			dbcsr::contract(*met, *c_xbb_AB, *eri_local)
				.alpha(-1.0).beta(1.0)
				.filter(dbcsr::global::filter_eps)
				.perform("XY, YMN -> XMN");
				
			std::cout << "SUM" << std::endl;
			auto nblk = eri_local->num_blocks_total();
			std::cout << nblk << std::endl;
#endif
		
			dbcsr::copy(*c_xbb_AB, *c_xbb_local)
				.move_data(true)
				.sum(true)
				.perform();
		
			eri_local->clear();
			inv_local->clear();
			
				
		}; // end worker function
	
		util::scheduler worker(m_world.comm(), ntasks, workfunc);
		worker.run();
				
		dbcsr::copy_local_to_global(*c_xbb_local, *c_xbb_global);
				
		c_xbb_batched->compress({inu}, c_xbb_global);
		c_xbb_local->clear();
		c_xbb_global->clear();
		
		task_off += ntasks;
	
	} // end loop over batches
	
	c_xbb_batched->compress_finalize();
	
	TIME.finish();
	
	double occupation = c_xbb_batched->occupation() * 100;
	LOG.os<>("Occupation of PARI fitting coefficients: ", occupation, "%\n");
	
	if (occupation > 100) throw std::runtime_error(
		"Fitting coefficients occupation more than 100%");
	
	return c_xbb_batched;
	
}
	
} // end namespace ints
