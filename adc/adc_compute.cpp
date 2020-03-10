#include "adc/adcmod.h"
#include "ints/registry.h"
#include "ints/aofactory.h"
#include "ints/gentran.h"
#include "math/tensor/dbcsr_conversions.hpp"

namespace adc {
	
void reserve_all(dbcsr::tensor<2>& t_in) {
	
	auto blks = t_in.blks_local();
	
	vec<vec<int>> resblks(2);
	
	for (auto i : blks[0]) {
		for (auto j : blks[1]) {
			resblks[0].push_back(i);
			resblks[1].push_back(j);
		}
	}
	
	t_in.reserve(resblks);
	
}
			
	
void adcmod::compute() {

		// load MO integrals (oo,ov,vv)
		
		ints::aofactory aofac(*m_hfwfn->mol(), m_comm);
		
		ints::registry INT_REGISTRY;
		
		//auto aoints = INT_REGISTRY.get<3>(m_hfwfn->mol()->name(),"i_xbb");
		auto aoints = aofac.compute<3>({.op = "coulomb", .bas = "xbb", .map1 = {0}, .map2 = {1,2}});
		auto metric = aofac.compute<2>({.op = "coulomb", .bas = "xx", .map1 = {0}, .map2 = {1}});
		
		//auto c_bm = std::make_shared<dbcsr::tensor<2>>(*m_hfwfn->c_bm_A());
		
		auto c_bo = m_hfwfn->c_bo_A();
		auto c_bv = m_hfwfn->c_bv_A();
		
		auto o = m_hfwfn->mol()->dims().oa();
		auto v = m_hfwfn->mol()->dims().va();
		auto x = m_hfwfn->mol()->dims().x();
		vec<int> d = {1};
		
		auto aosizes = aoints->blk_size();
		auto osizes = c_bo->blk_size();
		auto vsizes = c_bv->blk_size();
		//auto csizes = c_bm->blk_size();
		
		auto PQSQRT = aofac.invert(metric,2);
		
		//dbcsr::print(*aoints);
		//dbcsr::print(*PQSQRT);
		
		// contract X
		dbcsr::pgrid<3> grid3({.comm = m_comm});
		
		dbcsr::stensor<3> d_xbb = dbcsr::make_stensor<3>({.name = "i_xbb(xx)^-1/2", .pgridN = grid3, 
				.map1 = {0}, .map2 = {1,2}, .blk_sizes = {aosizes[0],aosizes[1],aosizes[2]}});
				
		dbcsr::einsum<3,2,3>({.x = "XMN, XY -> YMN", .t1 = *aoints, .t2 = *PQSQRT, .t3 = *d_xbb, .move = true});
		
		aoints.reset();
		
		auto d_xoo = ints::transform3({.t_in = *d_xbb, .c_1 = *c_bo, .c_2 = *c_bo, .name = "i_xoo(xx)^-1/2"});
		dbcsr::print(*d_xoo);
		auto d_xov = ints::transform3({.t_in = *d_xbb, .c_1 = *c_bo, .c_2 = *c_bv, .name = "i_xov(xx)^-1/2"});
		dbcsr::print(*d_xov);
		auto d_xvv = ints::transform3({.t_in = *d_xbb, .c_1 = *c_bv, .c_2 = *c_bv, .name = "i_xvv(xx)^-1/2"});
		dbcsr::print(*d_xvv);
		/*
		// 1st contract: no bounds
		dbcsr::stensor<3> d_xmb = dbcsr::make_stensor<3>({.name = "i_xbm(xx)^-1/2", .pgridN = grid3, 
				.map1 = {0,2}, .map2 = {1}, .blk_sizes =  {aosizes[0],csizes[1],aosizes[2]}});
				
		dbcsr::einsum<3,2,3>({.x = "XMN, Mi -> XiN", .t1 = *d_xbb, .t2 = *c_bm, .t3 = *d_xmb});
		
		// 2nd contract oo, ov, vv separately
		
		int nocc = m_hfwfn->mol()->nocc_alpha();
		int nvir = m_hfwfn->mol()->nvir_alpha();
		
		vec<vec<int>> obounds = {{0,nocc-1}};
		vec<vec<int>> vbounds = {{nocc, nocc + nvir - 1}};
				
		dbcsr::stensor<3> d_xmm = dbcsr::make_stensor<3>({.name = "i_xmm(xx)^-1/2", .pgridN = grid3, 
				.map1 = {0}, .map2 = {1,2}, .blk_sizes =  {aosizes[0],csizes[1],csizes[1]}});
				
		dbcsr::einsum<3,2,3>({.x = "XiN, Nj -> Xij", .t1 = *d_xmb, .t2 = *c_bm, .t3 = *d_xmm, .b2 = obounds/*, /*.b3 = obounds /*.move = true*///});
		//dbcsr::einsum<3,2,3>({.x = "XiN, Nj -> Xij", .t1 = *d_xmb, .t2 = *c_bm, .t3 = *d_xmm, .b2 = obounds, .b3 = vbounds /*.move = true*/});
		//dbcsr::einsum<3,2,3>({.x = "XiN, Nj -> Xij", .t1 = *d_xmb, .t2 = *c_bm, .t3 = *d_xmm, .b2 = vbounds, .b3 = vbounds /*.move = true*/});
		
		//std::cout << "d_xmm" << std::endl;
		//dbcsr::print(*d_xmm);
		
		// oo 
		//dbcsr::stensor<3> d_xbb = dbcsr::make_stensor<3>({.name = "i_xoo(xx)^-1/2", .pgridN = grid3, 
		//		.map1 = {0}, .map2 = {1,2}, .blk_sizes = {aosizes[0],aosizes[1],csizes[1]}});
		
		// test MO values
		
		
		dbcsr::pgrid<4> grid4a({.comm = m_comm});
		vec<vec<int>> m_sizes = {osizes[1],osizes[1],vsizes[1],vsizes[1]};
		dbcsr::stensor<4> mmmm = dbcsr::make_stensor<4>({.name = "mmmm", .pgridN = grid4a, .map1 = {0,1}, .map2 = {2,3}, .blk_sizes = m_sizes});
		
		dbcsr::einsum<3,3,4>({.x = "Xij, Xkl -> ijkl", .t1 = *d_xoo, .t2 = *d_xvv, .t3 = *mmmm});
		
		dbcsr::print(*mmmm);
		
		
		// compute amplitudes, if applicable
		
		
		// set up guess vectors
		std::vector<dbcsr::stensor<2>> guesses(m_nroots);
		dbcsr::pgrid<2> grid2({.comm = m_comm});
		vec<vec<int>> resblks(2);
		
		// make template
		dbcsr::stensor<2> gbase = dbcsr::make_stensor<2>({.name = "guess_", 
				.pgridN = grid2, .map1 = {0}, .map2 = {1}, .blk_sizes = {o,v}});
		
		reserve_all(*gbase);
		
		auto fillrand = [](dbcsr::tensor<2>& t_in) {
    
			dbcsr::iterator<2> iter(t_in);
			
			while (iter.blocks_left()) {
				  
				iter.next();
				
				dbcsr::block<2,double> blk(iter.sizes());
				
				blk.fill_rand(-1.0,1.0);
				
				auto idx = iter.idx();
				t_in.put_block({.idx = idx, .blk = blk});
					
			}
			
		};
		
		//gbase->reserve(resblks);
		
		for (int i = 0; i != m_nroots; ++i) {
			dbcsr::stensor<2> g = std::make_shared<dbcsr::tensor<2>>(*gbase,"guess_"+std::to_string(i));
			g->reserve(resblks);
			fillrand(*g);
			
			guesses[i] = g;
			
			std::cout << "GUESS: " <<  i << std::endl;
			dbcsr::print(*g);
			
		}
		
		// set up ADC0 expression
		
		auto epso = m_hfwfn->eps_occ_A();
		auto epsv = m_hfwfn->eps_vir_A();
		
		std::cout << "Energy: " << std::endl;
		for (auto e : *epsv) {
			std::cout << e << std::endl;
		}
		
		std::cout << "H1" << std::endl;
		
		// set up f_oo and f_vv
		Eigen::VectorXd eigen_epso = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(epso->data(),epso->size());
		Eigen::VectorXd eigen_epsv = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(epsv->data(),epsv->size());
	
		std::cout << "H2" << std::endl;
	
		Eigen::MatrixXd eigen_foo = eigen_epso.asDiagonal();
		Eigen::MatrixXd eigen_fvv = eigen_epsv.asDiagonal();
		
		std::cout << "H3" << std::endl;
		
		auto _f_oo = dbcsr::eigen_to_tensor(eigen_foo, "f_oo", grid2, vec<int>{0}, vec<int>{1}, vec<vec<int>>{o,o});
		auto _f_vv = dbcsr::eigen_to_tensor(eigen_fvv, "f_vv", grid2, vec<int>{0}, vec<int>{1}, vec<vec<int>>{v,v});
		
		auto f_oo = _f_oo.get_stensor();
		auto f_vv = _f_vv.get_stensor();
		
		std::cout << "f_oo" << std::endl;
		dbcsr::print(*f_oo);
		
		// set up davidson, pass it sigma construction function
		
		// make diagonal d_ia
		
			
		// make sum_P B_iiP B_aaP
		
		std::cout << "BEGIN" << std::endl;
		
		
		auto get_diag = [&](dbcsr::tensor<3>& t_in, dbcsr::tensor<2>& t_out, std::string name) {
			
			auto blksizes = t_in.blk_size();
			auto blkoff = t_in.blk_offset();
			auto nfull = t_in.nfull_tot();
			
			auto& xsizes = blksizes[0];
			auto& msizes = blksizes[1];
			
			int myrank = -1;
			int nproc = 0;
			
			Eigen::MatrixXd eigen_xm = Eigen::MatrixXd::Zero(nfull[0],nfull[1]);
			
			MPI_Comm_rank(m_comm,&myrank);
			MPI_Comm_size(m_comm,&nproc);
			
			for (int ix = 0; ix != xsizes.size(); ++ix) {
				for (int im = 0; im != msizes.size(); ++im) {
					
					dbcsr::idx3 idx_3 = {ix,im,im};
					
					int xs = xsizes[ix];
					int ms = msizes[im];
					
					int proc = -1;
					
					t_in.get_stored_coordinates({.idx = idx_3, .proc = proc});
					
					Eigen::MatrixXd eigen2(xs,ms);
					bool found = false;
					
					if (myrank == proc) {
						
						auto blk_3 = t_in.get_block({.idx = idx_3, .blk_size = {xs,ms,ms}, .found = found});
						for (int jx = 0; jx != xsizes[ix]; ++jx) {
							for (int jm = 0; jm != msizes[im]; ++jm) {
								eigen2(jx,jm) = blk_3(jx,jm,jm);
							}
						}
						
					}
					
					MPI_Bcast(&found,1,MPI_C_BOOL,proc,m_comm);
					
					if (found) {
						
						MPI_Bcast(eigen2.data(),xs*ms,MPI_DOUBLE,proc,m_comm);
						eigen_xm.block(blkoff[0][ix],blkoff[0][im],xs,ms) = eigen2;
						
					}
					
			}}
			
			t_out = dbcsr::eigen_to_tensor(eigen_xm, name, grid2, vec<int>{0}, vec<int>{1}, vec<vec<int>>{xsizes,msizes});
			
		};
						
		dbcsr::tensor<2> Dxo, Dxv;
						
		get_diag(*d_xoo, Dxo, "Dxo");
		get_diag(*d_xvv, Dxv, "Dxv");
		
		dbcsr::print(Dxo);
		dbcsr::print(Dxv);
		
		dbcsr::tensor<2> diag_ia_1({.name = "diag_ia_1", .pgridN = grid2, .map1 = {0}, .map2 = {1}, .blk_sizes = {o,v}});
		
		dbcsr::einsum<2,2,2>({.x = "Xi, Xa -> ia", .t1 = Dxo, .t2 = Dxv, .t3 = diag_ia_1});
		
		dbcsr::print(diag_ia_1);
		
		Dxo.destroy();
		Dxv.destroy();
		
		// now make (ia|ia)
		dbcsr::pgrid<4> grid4({.comm = m_comm});
		dbcsr::tensor<4> d_iaia({.name = "d_iaia", .pgridN = grid4, .map1 = {0,1}, .map2 = {2,3}, .blk_sizes = {o,v,o,v}});
		
		auto blkoff4 = d_iaia.blk_offset();
		
		for (int I = 0; I != o.size(); ++I) {
			for (int A = 0; A != v.size(); ++A) {
				
				int ioff = blkoff4[0][I];
				int aoff = blkoff4[1][A];
				int isize = o[I];
				int asize = v[A];
				
				std::cout << "ioff/aoff " << ioff << " " << aoff << std::endl;
				
				vec<vec<int>> bounds2 = {{ioff,ioff+isize-1},{aoff,aoff+asize-1}};
				vec<vec<int>> bounds3 = bounds2;
				
				dbcsr::einsum<3,3,4>({.x = "Xia, Xjb -> iajb", .t1 = *d_xov, .t2 = *d_xov, .t3 = d_iaia, .beta = 1.0, .b2 = bounds2, .b3 = bounds3});
				
			}
		} 
		
		dbcsr::print(d_iaia);
		
		// now extract them
		
		dbcsr::tensor<2> diag_ia_2({.name = "diag_ia_2", .pgridN = grid2, .map1 = {0}, .map2 = {1}, .blk_sizes = {o,v}});
		
		reserve_all(diag_ia_2);
		
		for (int iblk = 0; iblk != o.size(); ++iblk) {
			for (int jblk = 0; jblk != v.size(); ++jblk) {
			
				int proc_ia = -1;
				int proc_iaia = -1;
				int myrank = -1;
				
				MPI_Comm_rank(m_comm, &myrank);
				
				int osize = o[iblk];
				int vsize = v[jblk];
				
				dbcsr::idx2 idx_2 = {iblk,jblk};
				dbcsr::idx4 idx_4 = {iblk,jblk,iblk,jblk};
				
				d_iaia.get_stored_coordinates({.idx = idx_4, .proc = proc_iaia});
				diag_ia_2.get_stored_coordinates({.idx = idx_2, .proc = proc_ia});
				
				std::cout << "IAIA/IA " << proc_iaia << " " << proc_ia << std::endl;
				
				dbcsr::block<2> blk2(vec<int>{osize,vsize});
				
				bool found4 = false;
				
				// get block from D_iaia
				if (myrank == proc_iaia) {
					
					auto blk4 = d_iaia.get_block({.idx = idx_4, .blk_size = {osize,vsize,osize,vsize}, .found = found4});
					
					//extract diagonal from block
					if (found4) {
					
						for (int i = 0; i != osize; ++i) {
							for (int j = 0; j != vsize; ++j) {		
								blk2(i,j) = blk4(i,j,i,j);
						}}
						
					}
					
				}
				
				MPI_Bcast(&found4,1,MPI_C_BOOL,proc_iaia,m_comm);
					
				if (found4) {
					
					
					// Send block to correct processor
					if (myrank == proc_iaia && proc_ia != proc_iaia) {
						MPI_Send(blk2.data(),blk2.ntot(),MPI_DOUBLE,proc_ia,1,m_comm);
					} else if (myrank == proc_ia && proc_ia != proc_iaia) {
						MPI_Recv(blk2.data(),blk2.ntot(),MPI_DOUBLE,proc_iaia,1,m_comm, MPI_STATUS_IGNORE);
					}
					
					if (myrank == proc_ia) 
						diag_ia_2.put_block({.idx = idx_2, .blk = blk2});
					
				}
				
				MPI_Barrier(m_comm);
					
			} // end jblk
		} // end iblk	
		
		MPI_Barrier(m_comm);	
				
		d_iaia.destroy();
		dbcsr::print(diag_ia_2);	
		
		// form eps_ia
		
		dbcsr::tensor<2> eps_ia({.name = "eps_ia", .pgridN = grid2, .map1 = {0}, .map2 = {1}, .blk_sizes = {o,v}});
		
		reserve_all(eps_ia);
		
		dbcsr::iterator iter2(eps_ia);
		auto blkoff2 = eps_ia.blk_offset();
		
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
		
		dbcsr::tensor<2> diag_ia_tot({.name = "diag_ia_tot", .pgridN = grid2, .map1 = {0}, .map2 = {1}, .blk_sizes = {o,v}});
		
		diag_ia_1.scale(-2.0);
		diag_ia_2.scale(-1.0);
		
		dbcsr::copy<2>({.t_in = eps_ia, .t_out = diag_ia_tot, .move_data = true});
		dbcsr::copy<2>({.t_in = diag_ia_1, .t_out = diag_ia_tot, .sum = true, .move_data = true});
		dbcsr::copy<2>({.t_in = diag_ia_2, .t_out = diag_ia_tot, .sum = true, .move_data = true});
		
		dbcsr::print(diag_ia_tot);
		
		diag_ia_1.destroy();
		diag_ia_2.destroy();
		eps_ia.destroy();
		
		exit(0);
						
		auto make_sigma_singlet = [&](dbcsr::stensor<2>& r_ov) {
			
			dbcsr::stensor<2> u_ov = std::make_shared<dbcsr::tensor<2>>(*r_ov,"u_ov");
			
			// ADC0
			// form f_ii r_ia + f_aa r_ia
			
			dbcsr::tensor<2> sig_0_1(*r_ov,"sig_0_1");
			dbcsr::tensor<2> sig_0_2(*r_ov,"sig_0_2");
			
			dbcsr::copy<2>({.t_in = *r_ov, .t_out = sig_0_1});
			dbcsr::copy<2>({.t_in = *r_ov, .t_out = sig_0_2});
			
			std::cout << "E1" << std::endl;
			dbcsr::einsum<2,2,2>({.x = "ij, ja -> ia", .t1 = *f_oo, .t2 = *r_ov, .t3 = sig_0_1});
			std::cout << "E2" << std::endl;
			dbcsr::einsum<2,2,2>({.x = "ab, ib -> ia", .t1 = *f_vv, .t2 = *r_ov, .t3 = sig_0_2});
			
			dbcsr::copy<2>({.t_in = sig_0_2, .t_out = sig_0_1, .sum = true, .move_data = true});
			
			// ADC 1
			// == form [2*(ia|jb) - (ij|ab)] * r_ia
			
			// ==== c_x = (X|jb) * r_jb
			auto r_ovd = dbcsr::add_dummy(*r_ov);
			dbcsr::tensor<2> cx({.name = "cx", .pgridN = grid2, .map1 = {0}, .map2 = {1}, .blk_sizes = {x,d}});
			
			std::cout << "E3" << std::endl;
			dbcsr::einsum<3,3,2>({.x = "Xjb, jbD -> XD", .t1 = *d_xov, .t2 = r_ovd, .t3 = cx});
			
			// ==== sig_1_1 = 2 * (ia|X) * c_X
			dbcsr::tensor<2> sig_1_1(*r_ov, "sig_1_1");
			//dbcsr::tensor<2> sig_1_2(*r_ov, "sig_1_2");
			
			auto sig_1_1_d = dbcsr::add_dummy(sig_1_1);
			
			std::cout << "E4" << std::endl;
			dbcsr::einsum<3,2,3>({.x = "Xia, XD -> iaD", .t1 = *d_xov, .t2 = cx, .t3 = sig_1_1_d, .alpha = 2.0});
			
			sig_1_1 = dbcsr::remove_dummy(sig_1_1_d, vec<int>{0}, vec<int>{1});
			
			cx.destroy(); sig_1_1_d.destroy();
			
			// ==== (X|ab) r_jb = c_xja
			
			dbcsr::tensor<3> c_xja(*d_xov,"c_xja");
			
			std::cout << "E5" << std::endl;
			dbcsr::einsum<3,2,3>({.x = "Xab, jb -> Xja", .t1 = *d_xvv, .t2 = *r_ov, .t3 = c_xja});
			
			// ==== - (X|ij) c_xja = sig_1_2
			
			dbcsr::tensor<2> sig_1_2(*r_ov, "sig_1_2");
			
			std::cout << "E6" << std::endl;
			dbcsr::einsum<3,3,2>({.x = "Xij, Xja -> ia", .t1 = *d_xoo, .t2 = c_xja, .t3 = sig_1_2, .alpha = -1.0});
			
			c_xja.destroy();
			
			dbcsr::copy<2>({.t_in = sig_1_2, .t_out = sig_1_1, .sum = true, .move_data = true});
			
			// add ADC0 + ADC1
			dbcsr::copy<2>({.t_in = sig_0_1, .t_out = sig_1_1, .sum = true, .move_data = true});
			
			auto out = sig_1_1.get_stensor();
			
			return out;
			
		};
			
			
		auto s = make_sigma_singlet(guesses[0]);
		
		
		
		dbcsr::print(*s);	
			
			
		
		// compute
	
}

}
