#include "adc/adcmod.h"
#include "ints/aofactory.h"
#include <dbcsr_matrix_ops.hpp>

namespace adc {

/*
void get_diag_4(dbcsr::tensor<4>& t4, dbcsr::tensor<2>& t2, vec<int>& isizes, 
	vec<int>& asizes, int order) {
	
	// if order = 0 -> iiaa
	// else -> iaia
	
	int myrank = -1;
	MPI_Comm comm = t4.comm();
	
	MPI_Comm_rank(comm, &myrank);
	
	for (int iblk = 0; iblk != isizes.size(); ++iblk) {
			for (int ablk = 0; ablk != asizes.size(); ++ablk) {
			
				int proc_2 = -1;
				int proc_4 = -1;
				
				int isize = isizes[iblk];
				int asize = asizes[ablk];
				
				dbcsr::idx2 idx_2 = {iblk,ablk};
				dbcsr::idx4 idx_4 = (order == 0) ? dbcsr::idx4{iblk,iblk,ablk,ablk} : dbcsr::idx4{iblk,ablk,iblk,ablk};
				
				proc_4 = t4.proc(idx_4);
				proc_2 = t2.proc(idx_2);
				
				std::cout << "IA " << iblk << " " << ablk << std::endl;
				
				dbcsr::idx2 size_2 = {isize,asize};
				dbcsr::block<2> blk2(size_2);
				
				bool found4 = false;
				
				// get block from t4
				if (myrank == proc_4) {
					
					auto blk_size = (order == 0) ? arr<int,4>{isize,isize,asize,asize} 
						: arr<int,4>{isize,asize,isize,asize};
					auto blk4 = t4.get_block(idx_4, blk_size, found4);
					
					//extract diagonal from block
					if (found4) {
						if (order == 0) {
							std::cout << "HERE" << std::endl;
							for (int i = 0; i != isize; ++i) {
								for (int a = 0; a != asize; ++a) {		
									blk2(i,a) = blk4(i,i,a,a);
							}}
						} else {
							for (int i = 0; i != isize; ++i) {
								for (int a = 0; a != asize; ++a) {		
									blk2(i,a) = blk4(i,a,i,a);
							}}
						}
					}
					
				}
				
				MPI_Bcast(&found4,1,MPI_C_BOOL,proc_4,comm);
					
				if (found4) {
					
					// Send block to correct processor
					if (myrank == proc_4 && proc_2 != proc_4) {
						MPI_Send(blk2.data(),blk2.ntot(),MPI_DOUBLE,proc_2,1,comm);
					} else if (myrank == proc_2 && proc_2 != proc_4) {
						MPI_Recv(blk2.data(),blk2.ntot(),MPI_DOUBLE,proc_4,1,comm, MPI_STATUS_IGNORE);
					}
					
					if (myrank == proc_2) {
						std::cout << "PUT" << std::endl; 
						t2.put_block(idx_2, blk2);
					}
				}
				
				MPI_Barrier(comm);
					
			} // end ablk
		} // end iblk
		
}*/

dbcsr::shared_matrix<double> adcmod::compute_diag_0() {
	
	LOG.os<>("Computing zeroth order diagonal.\n");
	
	auto epso = m_hfwfn->eps_occ_A();
	auto epsv = m_hfwfn->eps_vir_A();
	
	auto o = m_hfwfn->mol()->dims().oa();
	auto v = m_hfwfn->mol()->dims().va();
	
	auto d_ov_0 = dbcsr::create<double>()
		.name("diag_ov_0")
		.set_world(m_world)
		.row_blk_sizes(o)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	d_ov_0->reserve_all();
	
	dbcsr::iterator<double> iter(*d_ov_0);
	
	iter.start();
	
	while (iter.blocks_left()) {
		
		iter.next_block();
		
		int roff = iter.row_offset();
		int coff = iter.col_offset();
		
		int rsize = iter.row_size();
		int csize = iter.col_size();
		
		for (int i = 0; i != rsize; ++i) {
			for (int j = 0; j != csize; ++j) {
				iter(i,j) = - epso->at(i + roff) 
					+ epsv->at(j + coff);
			}
		}
		
	}
	
	iter.stop();
	
	if (LOG.global_plev() >= 2) dbcsr::print(*d_ov_0);
	
	LOG.os<>("Done with diagonal.\n");
	
	return d_ov_0;
	
}

dbcsr::shared_matrix<double> adcmod::compute_diag_1() {
	
	ints::aofactory aofac(m_hfwfn->mol(), m_world);
	ints::screener* scr = new ints::schwarz_screener(m_world, m_hfwfn->mol());
	ints::shared_screener s_scr(scr);
	
	s_scr->compute();
		
	// setup tensors
	auto mol = m_hfwfn->mol();
	
	auto b = mol->dims().b();
	auto o = mol->dims().oa();
	auto v = mol->dims().va();
	
	arrvec<int,2> bo = {b,o};
	arrvec<int,2> bv = {b,v};
	
	arrvec<int,4> bbbb = {b,b,b,b};
	arrvec<int,4> obbb = {o,b,b,b};
	arrvec<int,4> obob = {o,b,o,b};
	arrvec<int,4> ovob = {o,v,o,b};
	arrvec<int,4> ovov = {o,v,o,v};
	arrvec<int,4> oobb = {o,o,b,b};
	arrvec<int,4> oovb = {o,o,v,b};
	arrvec<int,4> oovv = {o,o,v,v};
	
	dbcsr::shared_pgrid<2> spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	dbcsr::shared_pgrid<4> spgrid4 = dbcsr::create_pgrid<4>(m_world.comm()).get();
	
	int nbatches = m_opt.get<int>("nbatches_b", ADC_NBATCHES_B);
	std::array<int,4> bdims = {nbatches,nbatches,nbatches,nbatches};
	auto blkmap_b = mol->c_basis()->block_to_atom(mol->atoms());
	arrvec<int,4> blkmap = {blkmap_b, blkmap_b, blkmap_b, blkmap_b};
	
	auto eri4c2e_direct = dbcsr::btensor_create<4>()
			.name("eri4c2e_direct")
			.pgrid(spgrid4)
			.blk_sizes(bbbb)
			.batch_dims(bdims)
			.btensor_type(dbcsr::btype::direct)
			.blk_map(blkmap)
			.print(LOG.global_plev())
			.get();
			
	auto eri_bbbb = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_bbbb")
		.blk_sizes(bbbb)
		.map1({0}).map2({1,2,3})
		.get();
			
	auto T_obbb = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_obbb")
		.blk_sizes(obbb)
		.map1({0}).map2({1,2,3})
		.get();
		
	auto T_obbb_2 = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_obbb_2")
		.blk_sizes(obbb)
		.map1({0}).map2({1,2,3})
		.get();
		
	auto T_obob = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_obob")
		.blk_sizes(obob)
		.map1({2}).map2({0,1,3})
		.get();
		
	auto T_oobb = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_oobb")
		.blk_sizes(oobb)
		.map1({1}).map2({0,2,3})
		.get();
		
	auto T_obob_tot = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_obob")
		.blk_sizes(obob)
		.map1({1}).map2({0,2,3})
		.get();
	
	auto T_oobb_tot = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_oobb")
		.blk_sizes(oobb)
		.map1({2}).map2({0,1,3})
		.get();
	
	auto T_ovob = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_ovob")
		.blk_sizes(ovob)
		.map1({0,1,2}).map2({3})
		.get();
		
	auto T_oovb = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_oovb")
		.blk_sizes(oovb)
		.map1({0,1,3}).map2({2})
		.get();
		
	auto T_ovov = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_ovov")
		.blk_sizes(ovov)
		.map1({0,1,2}).map2({3})
		.get();
		
	auto T_oovv = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_oovv")
		.blk_sizes(oovv)
		.map1({0,1,2}).map2({3})
		.get();
		
	auto T_ovov_tot = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_ovov")
		.blk_sizes(ovov)
		.map1({0,1,2}).map2({3})
		.get();
		
	auto T_oovv_tot = dbcsr::tensor_create<4>()
		.pgrid(spgrid4)
		.name("T_oovv")
		.blk_sizes(oovv)
		.map1({0,1,2}).map2({3})
		.get();
		
	auto d_iaia = dbcsr::create<double>()
		.set_world(m_world)
		.name("d_ia_1")
		.row_blk_sizes(o)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto d_iiaa = dbcsr::create<double>()
		.set_world(m_world)
		.name("d_ia_2")
		.row_blk_sizes(o)
		.col_blk_sizes(v)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto c_bo_01 = dbcsr::tensor_create<2>()
		.name("c_bo")
		.pgrid(spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bo)
		.get();
		
	auto c_bv_01 = dbcsr::tensor_create<2>()
		.name("c_bv")
		.pgrid(spgrid2)
		.map1({0}).map2({1})
		.blk_sizes(bv)
		.get();
		
	auto c_bo = m_hfwfn->c_bo_A();
	auto c_bv = m_hfwfn->c_bv_A();
	
	dbcsr::copy_matrix_to_tensor(*c_bo, *c_bo_01);
	dbcsr::copy_matrix_to_tensor(*c_bv, *c_bv_01);
	
	aofac.ao_eri_setup(ints::metric::coulomb);
	
	int nb = eri4c2e_direct->nbatches(0);
	
	// allocate only diags
	arrvec<int,4> blkidx_ovov, blkidx_oovv;
	std::array<int,4> idx;
	
	for (int io = 0; io != o.size(); ++io) {
		for (int iv = 0; iv != v.size(); ++iv) {
			
			idx[0] = io;
			idx[1] = iv;
			idx[2] = io;
			idx[3] = iv;
			
			if (T_ovov->proc(idx) == m_world.rank()) {
				blkidx_ovov[0].push_back(io);
				blkidx_ovov[1].push_back(iv);
				blkidx_ovov[2].push_back(io);
				blkidx_ovov[3].push_back(iv);
			}
			
			idx[1] = io;
			idx[2] = iv;
			
			if (T_oovv->proc(idx) == m_world.rank()) {
				blkidx_oovv[0].push_back(io);
				blkidx_oovv[1].push_back(io);
				blkidx_oovv[2].push_back(iv);
				blkidx_oovv[3].push_back(iv);
			}
			
		}
	}
	
	T_ovov->reserve(blkidx_ovov);
	T_oovv->reserve(blkidx_oovv);
	
	vec<int> full_obds = {0, mol->nocc_alpha()};
	vec<int> full_vbds = {0, mol->nvir_alpha()};
	auto full_blkbds = eri4c2e_direct->full_blk_bounds(0);
	auto full_bds = eri4c2e_direct->full_bounds(0);
		
	// MO transformation (ia|jb) = sum_μνσλ C_μi C_νa (μν|σλ) C_σj C_λb where a = b, i = j 
	
	// loop λ
	for (int ilam = 0; ilam != nb; ++ilam) {
		LOG.os<2>("BATCH LAM: ", ilam, '\n');
			
		auto lam_blkbds = eri4c2e_direct->blk_bounds(3,ilam);
		auto lam_bds = eri4c2e_direct->bounds(3,ilam);	
		
		// loop ν 
		for (int inu = 0; inu != nb; ++inu) {
			LOG.os<2>("BATCH NU: ", inu, '\n');
		
			auto nu_blkbds = eri4c2e_direct->blk_bounds(1,inu);
			auto nu_bds = eri4c2e_direct->bounds(1,inu);
			
			// loop σ
			for (int isig = 0; isig != nb; ++isig) {
				LOG.os<2>("BATCH SIG: ", isig, '\n');
			
				auto sig_blkbds = eri4c2e_direct->blk_bounds(2,isig);
				auto sig_bds = eri4c2e_direct->bounds(2,isig);
				
				// loop μ
				for (int imu = 0; imu != nb; ++imu) {
					LOG.os<2>("BATCH MU: ", imu, '\n');
					
					auto mu_blkbds = eri4c2e_direct->blk_bounds(0,imu);
					auto mu_bds = eri4c2e_direct->bounds(0,imu);
				
					vec<vec<int>> blkbds = {
						mu_blkbds, nu_blkbds,
						sig_blkbds, lam_blkbds 
					};
								
					aofac.ao_4c_fill(eri_bbbb, blkbds, s_scr);
					eri_bbbb->filter(dbcsr::global::filter_eps);
				
					//dbcsr::print(*eri_bbbb);
				
					vec<vec<int>> m_bds = {
						mu_bds
					};
				
					vec<vec<int>> nsl_bds = {
						nu_bds, sig_bds, lam_bds
					};
				
					dbcsr::contract(*eri_bbbb, *c_bo_01, *T_obbb)
						.bounds1(m_bds)
						.bounds2(nsl_bds)
						.beta(1.0)
						.filter(dbcsr::global::filter_eps)
						.perform("mnsl, mi -> insl");
					
					//dbcsr::print(*T_obbb);
					
					eri_bbbb->clear();
					
				} // end loop μ
					
				// allocate only diagonals
				arrvec<int,4> blkidx_obob, blkidx_oobb;
				
				for (int io = 0; io != o.size(); ++io) {
					for (int inu = nu_blkbds[0]; inu != nu_blkbds[1]+1; ++inu) {
						for (int ilam = lam_blkbds[0]; ilam != lam_blkbds[1]+1; ++ilam) {
							
							idx[0] = io;
							idx[1] = inu;
							idx[2] = io;
							idx[3] = ilam;
							
							if (T_obob->proc(idx) == m_world.rank()) {
								blkidx_obob[0].push_back(io);
								blkidx_obob[2].push_back(io);
								blkidx_obob[1].push_back(inu);
								blkidx_obob[3].push_back(ilam);
							}
							
							idx[0] = io;
							idx[1] = io;
							idx[2] = inu;
							idx[3] = ilam;
							
							if (T_oobb->proc(idx) == m_world.rank()) {
								blkidx_oobb[0].push_back(io);
								blkidx_oobb[1].push_back(io);
								blkidx_oobb[2].push_back(inu);
								blkidx_oobb[3].push_back(ilam);
							}
						}
					}
				}
				
				T_obob->reserve(blkidx_obob);
				T_oobb->reserve(blkidx_oobb);
				
				vec<vec<int>> s_bds = {
					sig_bds
				};
				
				vec<vec<int>> inl_bds = {
					full_obds,
					nu_bds,
					lam_bds
				};
				
				vec<int> order_idx = {0,2,1,3};
				dbcsr::copy(*T_obbb, *T_obbb_2)
					.order(order_idx)
					.perform();
				
				dbcsr::contract(*T_obbb, *c_bo_01, *T_obob)
					.filter(dbcsr::global::filter_eps)
					.bounds1(s_bds)
					.bounds2(inl_bds)
					.retain_sparsity(true)
					.perform("insl, sj -> injl");
					
				dbcsr::contract(*T_obbb_2, *c_bo_01, *T_oobb)
					.filter(dbcsr::global::filter_eps)
					.bounds1(s_bds)
					.bounds2(inl_bds)
					.retain_sparsity(true)
					.perform("isnl, sj -> ijnl");
					
				dbcsr::copy(*T_obob, *T_obob_tot)
					.sum(true)
					.move_data(true)
					.perform();
					
				dbcsr::copy(*T_oobb, *T_oobb_tot)
					.sum(true)
					.move_data(true)
					.perform();
					
				//dbcsr::print(*T_obob);
					
				T_obbb->clear();
				T_obbb_2->clear();
				
			} // end loop σ
				
			vec<vec<int>> n_bds = {
				nu_bds
			};
				
			vec<vec<int>> ijl_bds = {
				full_obds,
				full_obds,
				lam_bds
			};
					
			dbcsr::contract(*T_obob_tot, *c_bv_01, *T_ovob)
				.filter(dbcsr::global::filter_eps)
				.bounds1(n_bds)
				.bounds2(ijl_bds)
				.beta(1.0)
				.perform("injl, na -> iajl");
				
			dbcsr::contract(*T_oobb_tot, *c_bv_01, *T_oovb)
				.filter(dbcsr::global::filter_eps)
				.bounds1(n_bds)
				.bounds2(ijl_bds)
				.beta(1.0)
				.perform("ijnl, na -> ijal");
				
			T_obob_tot->clear();
			T_oobb_tot->clear();
			
		} // end loop ν
				
		vec<vec<int>> l_bds = {
			lam_bds
		};
		
		dbcsr::contract(*T_ovob, *c_bv_01, *T_ovov)
			.retain_sparsity(true)
			.bounds1(l_bds)
			.perform("iaju, ub -> iajb");
			
		dbcsr::copy(*T_ovov, *T_ovov_tot)
			.sum(true)
			//.move_data(true)
			.perform();
			
		dbcsr::contract(*T_oovb, *c_bv_01, *T_oovv)
			.retain_sparsity(true)
			.bounds1(l_bds)
			.perform("ijau, ub -> ijab");
			
		dbcsr::copy(*T_oovv, *T_oovv_tot)
			.sum(true)
			//.move_data(true)
			.perform();
	
		T_ovob->clear();
		T_oovb->clear();
				
	}
	
	T_ovov->clear();
	T_oovv->clear();
	
	d_iaia->reserve_all();
	
	// extract iaia diagonal
	dbcsr::iterator iter1(*d_iaia);
	iter1.start();
	
	std::array<int,4> blksize;
	
	while (iter1.blocks_left()) {
		iter1.next_block();
		int r = iter1.row();
		int c = iter1.col();
		int rsize = iter1.row_size();
		int csize = iter1.col_size();
		
		bool found = false;
		
		idx[0] = r;
		idx[1] = c;
		idx[2] = r;
		idx[3] = c;
		
		blksize[0] = rsize;
		blksize[1] = csize;
		blksize[2] = rsize;
		blksize[3] = csize;
		
		auto blk = T_ovov_tot->get_block(idx, blksize, found);
		if (!found) continue;
		
		for (int i = 0; i != rsize; ++i) {
			for (int j = 0; j != csize; ++j) {
				iter1(i,j) = blk(i,j,i,j);
			}
		}
	}
	
	iter1.stop();
		
	//dbcsr::print(*T_ovov_tot);	
	//dbcsr::print(*d_iaia);
	
	T_ovov_tot->clear();
	
	d_iiaa->reserve_all();
	dbcsr::iterator iter2(*d_iiaa);
	
	iter2.start();
	while (iter2.blocks_left()) {
		iter2.next_block();
		int r = iter2.row();
		int c = iter2.col();
		int rsize = iter2.row_size();
		int csize = iter2.col_size();
		
		bool found = false;
		
		idx[0] = r;
		idx[1] = r;
		idx[2] = c;
		idx[3] = c;
		
		blksize[0] = rsize;
		blksize[1] = rsize;
		blksize[2] = csize;
		blksize[3] = csize;
		
		auto blk = T_oovv_tot->get_block(idx, blksize, found);
		if (!found) continue;
		
		for (int i = 0; i != rsize; ++i) {
			for (int j = 0; j != csize; ++j) {
				iter2(i,j) = blk(i,i,j,j);
			}
		}
	}
	
	iter2.stop();
	
	//dbcsr::print(*T_oovv_tot);	
	//dbcsr::print(*d_iiaa);
	
	T_oovv_tot->clear();

	d_iaia->add(-1.0, -2.0, *d_iiaa);
	
	d_iaia->setname("diag_ov_1");
	
	return d_iaia;
	
}
	
void adcmod::compute_diag() {
	
	auto& diag_time = TIME.sub("Computing ADC diagonal elements");
	
	diag_time.start();
	int diag_order = m_opt.get<int>("diag_order", ADC_DIAG_ORDER);
	
	auto d_ov_0 = compute_diag_0();
	
	if (diag_order > 0) {
		//auto d_ov_1 = compute_diag_1();
		//d_ov_0->add(1.0, 1.0, *d_ov_1);
		m_d_ov = d_ov_0;
		m_d_ov->setname("diag_ov_1");
	} else {
		m_d_ov = d_ov_0;
	}
	diag_time.finish();
	
}

} // end namespace
		
