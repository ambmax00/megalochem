#include "adc/adcmod.h"

namespace adc {

/*
dbcsr::tensor<2> get_diag(dbcsr::stensor<3>& t_in, std::string name) {
			
		auto blksizes = t_in->blk_size();
		auto blkoff = t_in->blk_offset();
		auto nfull = t_in->nfull_tot();
		
		auto& xsizes = blksizes[0];
		auto& msizes = blksizes[1];
		
		int myrank = -1;
		int nproc = 0;
		
		Eigen::MatrixXd eigen_xm = Eigen::MatrixXd::Zero(nfull[0],nfull[1]);
		
		MPI_Comm_rank(t_in->comm(),&myrank);
		MPI_Comm_size(t_in->comm(),&nproc);
		
		for (int ix = 0; ix != xsizes.size(); ++ix) {
			for (int im = 0; im != msizes.size(); ++im) {
				
				dbcsr::idx3 idx_3 = {ix,im,im};
				
				int xs = xsizes[ix];
				int ms = msizes[im];
				
				int proc = -1;
				
				t_in->get_stored_coordinates({.idx = idx_3, .proc = proc});
				
				Eigen::MatrixXd eigen2(xs,ms);
				bool found = false;
				
				if (myrank == proc) {
					
					auto blk_3 = t_in->get_block({.idx = idx_3, .blk_size = {xs,ms,ms}, .found = found});
					for (int jx = 0; jx != xsizes[ix]; ++jx) {
						for (int jm = 0; jm != msizes[im]; ++jm) {
							eigen2(jx,jm) = blk_3(jx,jm,jm);
						}
					}
					
				}
				
				MPI_Bcast(&found,1,MPI_C_BOOL,proc,t_in->comm());
				
				if (found) {
					
					MPI_Bcast(eigen2.data(),xs*ms,MPI_DOUBLE,proc,t_in->comm());
					eigen_xm.block(blkoff[0][ix],blkoff[0][im],xs,ms) = eigen2;
					
				}
				
		}}
		
		dbcsr::pgrid<2> grid2({.comm = t_in->comm()});
		
		return dbcsr::eigen_to_tensor(eigen_xm, name, grid2, vec<int>{0}, vec<int>{1}, 
			vec<vec<int>>{xsizes,msizes});
		
}
*/

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
				
				t4.get_stored_coordinates({.idx = idx_4, .proc = proc_4});
				t2.get_stored_coordinates({.idx = idx_2, .proc = proc_2});
				
				std::cout << "IA " << iblk << " " << ablk << std::endl;
				
				dbcsr::block<2> blk2(vec<int>{isize,asize});
				
				bool found4 = false;
				
				// get block from t4
				if (myrank == proc_4) {
					
					vec<int> blk_size = (order == 0) ? vec<int>{isize,isize,asize,asize} : vec<int>{isize,asize,isize,asize};
					auto blk4 = t4.get_block({.idx = idx_4, .blk_size = blk_size, .found = found4});
					
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
						t2.put_block({.idx = idx_2, .blk = blk2});
					}
				}
				
				MPI_Barrier(comm);
					
			} // end ablk
		} // end iblk
		
}

void adcmod::mo_compute_diag() {
		
		/*			
		auto Dxo = get_diag(m_mo.b_xoo, "Dxo");
		auto Dxv = get_diag(m_mo.b_xvv, "Dxv");
		
		dbcsr::print(Dxo);
		dbcsr::print(Dxv);
		
		dbcsr::pgrid<2> grid2({.comm = m_comm});
		
		dbcsr::tensor<2> diag_ia_1({.name = "diag_ia_1", .pgridN = grid2, 
			.map1 = {0}, .map2 = {1}, .blk_sizes = {m_dims.o,m_dims.v}});
		
		dbcsr::einsum<2,2,2>({.x = "Xi, Xa -> ia", .t1 = Dxo, .t2 = Dxv, .t3 = diag_ia_1});
		
		dbcsr::print(diag_ia_1);
		
		Dxo.destroy();
		Dxv.destroy();
		*/
		int myproc = -1;
		MPI_Comm_rank(m_comm,&myproc);
		
		// Make diag_ia_1 = (ii|aa)
		dbcsr::pgrid<4> grid4({.comm = m_comm});
		dbcsr::pgrid<2> grid2({.comm = m_comm});
		
		dbcsr::tensor<4> d_iiaa({.name = "d_iiaa", .pgridN = grid4, .map1 = {0,1}, .map2 = {2,3},
			.blk_sizes = {m_dims.o,m_dims.o,m_dims.v,m_dims.v}});
			
		// Reserve diagonal blocks
		auto nblkstot = d_iiaa.nblks_tot();
		vec<vec<int>> res_iiaa(4);
		vec<vec<int>> res_iaia(4);
		
		for (int I = 0; I != nblkstot[0]; ++I) {
		 for (int A = 0; A != nblkstot[2]; ++A) {
			 int proc = -1;
			 dbcsr::idx4 idx = {I,I,A,A};
			 d_iiaa.get_stored_coordinates({.idx = idx, .proc = proc});
			 
			 if (proc == myproc) {
				 res_iiaa[0].push_back(I);
				 res_iaia[0].push_back(I);
				 
				 res_iiaa[1].push_back(I);
				 res_iaia[1].push_back(A);
				 
				 res_iiaa[2].push_back(A);
				 res_iaia[2].push_back(I);
				 
				 res_iiaa[3].push_back(A);
				 res_iaia[3].push_back(A);
			 }
			 
		 }
		}
		
		d_iiaa.reserve(res_iiaa);
		
		// now contract
		
		dbcsr::einsum<3,3,4>({.x = "Xij, Xab -> ijab", .t1 = *m_mo.b_xoo, .t2 = *m_mo.b_xvv, 
			.t3 = d_iiaa, .retain_sparsity = true});
			
		dbcsr::print(d_iiaa);
			
		// extract the diagonal
		
		dbcsr::tensor<2> diag_ia_1({.name = "diag_ia_1", .pgridN = grid2, .map1 = {0}, .map2 = {1},
			.blk_sizes = {m_dims.o, m_dims.v}});
			
		diag_ia_1.reserve_all();
		
		get_diag_4(d_iiaa,diag_ia_1,m_dims.o,m_dims.v,0);
		
		d_iiaa.destroy();
		res_iiaa.clear();
			
		dbcsr::print(diag_ia_1);
		
		
		// now make (ia|ia)
		dbcsr::tensor<4> d_iaia({.name = "d_iaia", .pgridN = grid4, 
			.map1 = {0,1}, .map2 = {2,3}, .blk_sizes = {m_dims.o,m_dims.v,m_dims.o,m_dims.v}});
		
		d_iaia.reserve(res_iaia);
		res_iaia.clear();
		
		auto blkoff4 = d_iaia.blk_offset();
		
		dbcsr::einsum<3,3,4>({.x = "Xia, Xjb -> iajb", .t1 = *m_mo.b_xov, .t2 = *m_mo.b_xov, .t3 = d_iaia, .retain_sparsity = true});
		
		dbcsr::print(d_iaia);
		
		// now extract them
		
		dbcsr::tensor<2> diag_ia_2({.tensor_in = diag_ia_1, .name = "diag_ia_2"});
		diag_ia_2.reserve_all();
		
		get_diag_4(d_iaia,diag_ia_2,m_dims.o,m_dims.v,1);
				
		d_iaia.destroy();
		dbcsr::print(diag_ia_2);	
		
		// form eps_ia
		
		dbcsr::tensor<2> eps_ia({.tensor_in = diag_ia_1, .name = "eps_ia"});
		
		eps_ia.reserve_all();
		
		dbcsr::iterator iter2(eps_ia);
		auto blkoff2 = eps_ia.blk_offset();
		
		auto epso = m_hfwfn->eps_occ_A();
		auto epsv = m_hfwfn->eps_vir_A();
		
		while (iter2.blocks_left()) {
			
			iter2.next();
			
			auto idx_2 = iter2.idx();
			auto blksize = iter2.sizes();
		
			int offo = blkoff2[0][idx_2[0]];
			int offv = blkoff2[1][idx_2[1]];
			
			dbcsr::block<2> blk2(blksize);
			
			for (int io = 0; io != blksize[0]; ++io) {
				for (int iv = 0; iv != blksize[1]; ++iv) {
					blk2(io,iv) = - epso->at(io + offo) + epsv->at(iv + offv);
				}
			}
			
			eps_ia.put_block({.idx = idx_2, .blk = blk2});
			
		}
		
		dbcsr::print(eps_ia);
		
		// Form d_ia = eps_ia - 2 * (ii|aa) - (ia|ia)
		
		m_mo.d_ov = dbcsr::make_stensor<2>({.name = "d_ov", .pgridN = grid2, 
			.map1 = {0}, .map2 = {1}, .blk_sizes = {m_dims.o,m_dims.v}});

		diag_ia_1.scale(-1.0);
		//diag_ia_2.scale(2.0);
		
		dbcsr::copy<2>({.t_in = eps_ia, .t_out = *m_mo.d_ov, .move_data = true});
		dbcsr::copy<2>({.t_in = diag_ia_1, .t_out = *m_mo.d_ov, .sum = true, .move_data = true});
		dbcsr::copy<2>({.t_in = diag_ia_2, .t_out = *m_mo.d_ov, .sum = true, .move_data = true});
		
		dbcsr::print(*m_mo.d_ov);
		
		diag_ia_1.destroy();
		diag_ia_2.destroy();
		eps_ia.destroy();
		
}

} // end namespace
		
