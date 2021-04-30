#include "ints/integrals.hpp"
#include <stdexcept>
#include <map>
#include <mpi.h>
#include <omp.h>

#include <iostream>

namespace megalochem {

namespace ints {

// =====================================================================
//                       LIBCINT
// =====================================================================

void calc_ints(dbcsr::matrix<double>& m_out, std::vector<std::vector<int>>& shell_offsets, 
		std::vector<std::vector<int>>& nshells, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l) 
{
	
	//std::cout << "NATOMS/NBAS: " << natm << " " << nbas << std::endl;
	
	auto my_cart = m_out.get_cart();
	
	auto& nshells0 = nshells[0];
	auto& nshells1 = nshells[1];
	auto& shell_offsets0 = shell_offsets[0];
	auto& shell_offsets1 = shell_offsets[1]; 
	
	int max_buf_size = pow(2 * max_l + 1,2);
	
	#pragma omp parallel 
	{	
		
		dbcsr::iterator iter(m_out);
		
		iter.start();
		
		double* buf = new double[max_buf_size];
		int* shls = new int[2];
		
		while (iter.blocks_left()) {	
			
			iter.next_block();		
			
			int r = iter.row();
			int c = iter.col();
			
			//std::cout << "BLOCK: " << r << " " << c << std::endl;
			
			//block lower bounds
			int lb0 = iter.row_offset();
			int lb1 = iter.col_offset();
			
			// tensor offsets
			int toff0 = lb0;
			int toff1 = lb1;
			// local Block offsets
			int locblkoff0 = 0;
			int locblkoff1 = 0;
			
			int soff0 = shell_offsets0[r];
			int soff1 = shell_offsets1[c];
			int nshell0 = nshells0[r];
			int nshell1 = nshells1[c];
			
			//std::cout << "soffs: " << soff0 << " " << soff1 << std::endl;
			//std::cout << "nshells: " << nshell0 << " " << nshell1 << std::endl;
			
			for (int s0 = soff0; s0 != soff0+nshell0; ++s0) {
							
				toff0 += locblkoff0;
				int shellsize0 = CINTcgto_spheric(s0,bas);
				shls[0] = s0;
					
				for (int s1 = soff1; s1 != soff1+nshell1; ++s1) {
					
					toff1 += locblkoff1;
					int shellsize1 = CINTcgto_spheric(s1,bas);
					shls[1] = s1;
					
					//std::cout << "PROCESSING SHELL: " << s0 << " " << s1 << std::endl;
					
					int res = int_func(buf, shls, atm, natm, bas, nbas, env, nullptr);
											
					if (res != 0) {
						int idx = 0;
					
						for (int j = 0; j != shellsize1; ++j) {
							for (int i = 0; i != shellsize0; ++i) {
								//std::cout << buf[idx] << " ";
								iter(i + locblkoff0, j + locblkoff1) = buf[idx++];
						}}// std::cout << std::endl;
					}
					
					locblkoff1 += shellsize1;						
				}//endfor s2
				
				toff1 = lb1;
				locblkoff1 = 0;
				locblkoff0 += shellsize0;
			}//endfor s1
						
		}//end BLOCK LOOP
		
		iter.stop();
		m_out.finalize();
		
		delete [] buf;
		delete [] shls;
		
	}//end parallel omp	
	
}

void calc_ints(dbcsr::matrix<double>& m_x, dbcsr::matrix<double>& m_y, dbcsr::matrix<double>& m_z, 
		std::vector<std::vector<int>>& shell_offsets, 
		std::vector<std::vector<int>>& shell_sizes, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l) 
{
	
	auto& nshells0 = shell_sizes[0];
	auto& nshells1 = shell_sizes[1];
	auto& shell_offsets0 = shell_offsets[0];
	auto& shell_offsets1 = shell_offsets[1]; 
	
	int max_buf_size = pow(2 * max_l + 1,2);
	
	//#pragma omp parallel 
	//{	
	
	auto w = m_x.get_cart();
	auto rank = w.rank();
	
	auto row_offsets = m_x.row_blk_offsets();
	auto col_offsets = m_x.col_blk_offsets();
	
	double* buf = new double[max_buf_size * 3];
	int* shls = new int[2];
	
	int rtot = m_x.nblkrows_total();
	int ctot = m_x.nblkcols_total();
		
	for (int iblk0 = 0; iblk0 != rtot; ++iblk0) {
		for (int iblk1 = 0; iblk1 != ctot; ++iblk1) {
			
			if (m_x.proc(iblk0,iblk1) != rank) continue;
					
			bool found = true;
			
			auto blkx = m_x.get_block_p(iblk0, iblk1, found);
			if (!found) continue;
			auto blky = m_y.get_block_p(iblk0, iblk1, found);
			if (!found) continue;
			auto blkz = m_z.get_block_p(iblk0, iblk1, found);
			if (!found) continue;
						
			//block lower bounds
			int lb0 = row_offsets[iblk0];
			int lb1 = col_offsets[iblk1];
			
			// tensor offsets
			int toff0 = lb0;
			int toff1 = lb1;
			// local Block offsets
			int locblkoff0 = 0;
			int locblkoff1 = 0;
			
			int soff0 = shell_offsets0[iblk0];
			int soff1 = shell_offsets1[iblk1];
			int nshell0 = nshells0[iblk0];
			int nshell1 = nshells1[iblk1];
			
			//std::cout << "soffs: " << soff0 << " " << soff1 << std::endl;
			//std::cout << "nshells: " << nshell0 << " " << nshell1 << std::endl;
			
			for (int s0 = soff0; s0 != soff0+nshell0; ++s0) {
							
				toff0 += locblkoff0;
				int shellsize0 = CINTcgto_spheric(s0,bas);
				shls[0] = s0;
					
				for (int s1 = soff1; s1 != soff1+nshell1; ++s1) {
					
					toff1 += locblkoff1;
					int shellsize1 = CINTcgto_spheric(s1,bas);
					shls[1] = s1;
					
					//std::cout << "PROCESSING SHELL: " << s0 << " " << s1 << std::endl;
					
					int res = int_func(buf, shls, atm, natm, bas, nbas, env, nullptr);
											
					if (res != 0) {
						int idx = 0;
					
						for (int j = 0; j != shellsize1; ++j) {
							for (int i = 0; i != shellsize0; ++i) {
								//std::cout << buf[idx] << " ";
								blkx(i + locblkoff0, j + locblkoff1) = buf[idx++];
						}}
						
						for (int j = 0; j != shellsize1; ++j) {
							for (int i = 0; i != shellsize0; ++i) {
								//std::cout << buf[idx] << " ";
								blky(i + locblkoff0, j + locblkoff1) = buf[idx++];
						}}
						
						for (int j = 0; j != shellsize1; ++j) {
							for (int i = 0; i != shellsize0; ++i) {
								//std::cout << buf[idx] << " ";
								blkz(i + locblkoff0, j + locblkoff1) = buf[idx++];
						}}
						
					}
					
					locblkoff1 += shellsize1;						
				}//endfor s2
				
				toff1 = lb1;
				locblkoff1 = 0;
				locblkoff0 += shellsize0;
			}//endfor s1
						
		} // end for ic
	} // end for ir
	
	delete [] buf;
	delete [] shls;
		
	//}//end parallel omp	
	
}

void calc_ints(dbcsr::tensor<3,double>& m_out, std::vector<std::vector<int>>& shell_offsets, 
		std::vector<std::vector<int>>& nshells, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l)
{
	
	auto& nshells0 = nshells[0];
	auto& nshells1 = nshells[1];
	auto& nshells2 = nshells[2];
	auto& shell_offsets0 = shell_offsets[0];
	auto& shell_offsets1 = shell_offsets[1];
	auto& shell_offsets2 = shell_offsets[2];
	
	int max_buf_size = pow(2*max_l+1,3);
	
	#pragma omp parallel 
	{	
		
		dbcsr::iterator_t<3> iter(m_out);
		
		iter.start();
		
		double* buf = new double[max_buf_size];
		int* shls = new int[3];
		
		while (iter.blocks_left()) {	
			
			iter.next();	
			
			auto& idx = iter.idx();
			auto& size = iter.size();
			auto& off = iter.offset();	
			
			//block lower bounds
			int lb0 = off[0];
			int lb1 = off[1];
			int lb2 = off[2];
			
			// tensor offsets
			int toff0 = lb0;
			int toff1 = lb1;
			int toff2 = lb2;
			// local Block offsets
			int locblkoff0 = 0;
			int locblkoff1 = 0;
			int locblkoff2 = 0;
			
			int& soff0 = shell_offsets0[idx[0]];
			int& soff1 = shell_offsets1[idx[1]];
			int& soff2 = shell_offsets2[idx[2]];
			int& nshell0 = nshells0[idx[0]];
			int& nshell1 = nshells1[idx[1]];
			int& nshell2 = nshells2[idx[2]];
			
			//std::cout << "soffs: " << soff0 << " " << soff1 << std::endl;
			//std::cout << "nshells: " << nshell0 << " " << nshell1 << std::endl;
			
			dbcsr::block<3,double> blk(size);
			
			for (int s0 = soff0; s0 != soff0+nshell0; ++s0) {
							
				toff0 += locblkoff0;
				int shellsize0 = CINTcgto_spheric(s0,bas);
				shls[2] = s0;
					
				for (int s1 = soff1; s1 != soff1+nshell1; ++s1) {
					
					toff1 += locblkoff1;
					int shellsize1 = CINTcgto_spheric(s1,bas);
					shls[0] = s1;
					
					for (int s2 = soff2; s2 != soff2+nshell2; ++s2) {
						
						toff2 += locblkoff2;
						int shellsize2 = CINTcgto_spheric(s2,bas);
						shls[1] = s2;
						
						int res = int_func(buf, shls, atm, natm, bas, nbas, env, nullptr);
												
						if (res != 0) {
							int iidx = 0;
							
							for (int i = 0; i != shellsize0; ++i) {
								for (int k = 0; k != shellsize2; ++k) {
									for (int j = 0; j != shellsize1; ++j) {
											blk(i + locblkoff0, j + locblkoff1, k + locblkoff2) = buf[iidx++];
								
								}}}// std::cout << std::endl;
						}
						
						locblkoff2 += shellsize2;
						
					} // endfor s2
					
					toff2 = lb2;
					locblkoff2 = 0;
					locblkoff1 += shellsize1;						
				}//endfor s1
				
				toff1 = lb1;
				locblkoff1 = 0;
				locblkoff0 += shellsize0;
			}//endfor s0
			
			m_out.put_block(idx, blk);
						
		}//end BLOCK LOOP
		
		delete [] buf;
		delete [] shls;
		
		iter.stop();
		m_out.finalize();
		
	}//end parallel omp	
	
}

void calc_ints(dbcsr::tensor<4,double>& m_out, std::vector<std::vector<int>>& shell_offsets, 
		std::vector<std::vector<int>>& nshells, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l)
{
	const auto& nshells0 = nshells[0];
	const auto& nshells1 = nshells[1];
	const auto& nshells2 = nshells[2];
	const auto& nshells3 = nshells[3];
	const auto& shell_offsets0 = shell_offsets[0];
	const auto& shell_offsets1 = shell_offsets[1];
	const auto& shell_offsets2 = shell_offsets[2];
	const auto& shell_offsets3 = shell_offsets[3];
	
	int max_buf_size = pow(2*max_l+1,4);
	
	#pragma omp parallel 
	{	
		
		dbcsr::iterator_t<4> iter(m_out);
		
		iter.start();
		
		double* buf = new double[max_buf_size];
		int* shls = new int[4];
		
		while (iter.blocks_left()) {	
			
			iter.next();	
			
			auto& idx = iter.idx();
			auto& size = iter.size();
			auto& off = iter.offset();	
						
			//block lower bounds
			const int lb0 = off[0];
			const int lb1 = off[1];
			const int lb2 = off[2];
			const int lb3 = off[3];
			
			// tensor offsets
			int toff0 = lb0;
			int toff1 = lb1;
			int toff2 = lb2;
			int toff3 = lb3;
			// local Block offsets
			int locblkoff0 = 0;
			int locblkoff1 = 0;
			int locblkoff2 = 0;
			int locblkoff3 = 0;
			
			const int& soff0 = shell_offsets0[idx[0]];
			const int& soff1 = shell_offsets1[idx[1]];
			const int& soff2 = shell_offsets2[idx[2]];
			const int& soff3 = shell_offsets3[idx[3]];
			const int& nshell0 = nshells0[idx[0]];
			const int& nshell1 = nshells1[idx[1]];
			const int& nshell2 = nshells2[idx[2]];
			const int& nshell3 = nshells3[idx[3]];
			
			dbcsr::block<4,double> blk(size);
			
			for (int s0 = soff0; s0 != soff0+nshell0; ++s0) {
							
				toff0 += locblkoff0;
				int shellsize0 = CINTcgto_spheric(s0,bas);
				shls[0] = s0;
					
				for (int s1 = soff1; s1 != soff1+nshell1; ++s1) {
					
					toff1 += locblkoff1;
					const int shellsize1 = CINTcgto_spheric(s1,bas);
					shls[1] = s1;
					
					for (int s2 = soff2; s2 != soff2+nshell2; ++s2) {
						
						toff2 += locblkoff2;
						const int shellsize2 = CINTcgto_spheric(s2,bas);
						shls[2] = s2;
					
						for (int s3 = soff3; s3 != soff3+nshell3; ++s3) {
						
							toff3 += locblkoff3;
							const int shellsize3 = CINTcgto_spheric(s3,bas);
							shls[3] = s3;
					
							int res = int_func(buf, shls, atm, natm, bas, nbas, env, nullptr);
												
							if (res != 0) {
								int idx = 0;
							
								for (int l = 0; l != shellsize3; ++l) {
									for (int k = 0; k != shellsize2; ++k) {
										for (int j = 0; j != shellsize1; ++j) {
											for (int i = 0; i != shellsize0; ++i) {
																								
											blk(i + locblkoff0, j + locblkoff1, k + locblkoff2,
												l + locblkoff3) = buf[idx++];
								}}}}	
								
							}
							
							locblkoff3 += shellsize3;
							
						} // endfor s3					
						
						toff3 = lb3;
						locblkoff3 = 0;
						locblkoff2 += shellsize2;
						
					} // endfor s2
					
					toff2 = lb2;
					locblkoff2 = 0;
					locblkoff1 += shellsize1;						
				}//endfor s1
				
				toff1 = lb1;
				locblkoff1 = 0;
				locblkoff0 += shellsize0;
			}//endfor s0
			
			m_out.put_block(idx, blk);
						
		}//end BLOCK LOOP
		
		delete [] buf;
		delete [] shls;
		
		iter.stop();
		m_out.finalize();
		
	}//end parallel omp	
	
}

void calc_ints_schwarz_mn(dbcsr::matrix<double>& m_out, std::vector<std::vector<int>>& shell_offsets, 
		std::vector<std::vector<int>>& nshells, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l) 
{
		
	auto& nshells0 = nshells[0];
	auto& nshells1 = nshells[1];
	auto& shell_offsets0 = shell_offsets[0];
	auto& shell_offsets1 = shell_offsets[1]; 
	
	int max_buf_size = pow(2*max_l+1,4);
	
	#pragma omp parallel 
	{	
		
		dbcsr::iterator iter(m_out);
		
		iter.start();
		
		double* buf = new double[max_buf_size];
		int* shls = new int[4];
		
		while (iter.blocks_left()) {	
			
			iter.next_block();		
			
			int r = iter.row();
			int c = iter.col();
			
			//std::cout << "BLOCK: " << r << " " << c << std::endl;
			
			//block lower bounds
			int lb0 = iter.row_offset();
			int lb1 = iter.col_offset();
			
			// tensor offsets
			int toff0 = lb0;
			int toff1 = lb1;
			// local Block offsets
			int locblkoff0 = 0;
			int locblkoff1 = 0;
			
			int soff0 = shell_offsets0[r];
			int soff1 = shell_offsets1[c];
			int nshell0 = nshells0[r];
			int nshell1 = nshells1[c];
			
			//std::cout << "nshells: " << nshell0 << " " << nshell1 << std::endl;
			
			int s0_idx = 0;
			for (int s0 = soff0; s0 != soff0+nshell0; ++s0) {
							
				toff0 += locblkoff0;
				int shellsize0 = CINTcgto_spheric(s0,bas);
				shls[0] = s0;
				shls[2] = s0;
				
				int s1_idx = 0;
				for (int s1 = soff1; s1 != soff1+nshell1; ++s1) {
					
					toff1 += locblkoff1;
					int shellsize1 = CINTcgto_spheric(s1,bas);
					shls[1] = s1;
					shls[3] = s1;
					
					//std::cout << "PROCESSING SHELL: " << s0 << " " << s1 << std::endl;
					
					int res = int_func(buf, shls, atm, natm, bas, nbas, env, nullptr);
					
					double n = 0.0;
							
					if (res != 0) {
						int idx = 0;
					
						for (int i = 0; i != shellsize0; ++i) {
							for (int j = 0; j != shellsize1; ++j) {
								n += fabs(buf[i + j * shellsize0 + i * shellsize0 * shellsize1
									+ j * shellsize0 * shellsize1 * shellsize0]);
						}} //std::cout << std::endl;
					}
					
					iter(s0_idx,s1_idx) = sqrt(n);
					
					locblkoff1 += shellsize1;	
					++s1_idx;					
				}//endfor s2
				
				toff1 = lb1;
				locblkoff1 = 0;
				locblkoff0 += shellsize0;
				++s0_idx;
			}//endfor s1
				
		}//end BLOCK LOOP
		
		delete [] buf;
		delete [] shls;

		iter.stop();
		m_out.finalize();
		
	}//end parallel omp	
	
}

void calc_ints_schwarz_x(dbcsr::matrix<double>& m_out, std::vector<std::vector<int>>& shell_offsets, 
		std::vector<std::vector<int>>& nshells, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l) 
{
		
	auto& nshells0 = nshells[0];
	auto& shell_offsets0 = shell_offsets[0];
	int max_buf_size = pow(2*max_l+1,2);
	
	#pragma omp parallel 
	{	
		
		dbcsr::iterator iter(m_out);
		
		iter.start();
		
		double* buf = new double[max_buf_size];
		int* shls = new int[2];
		
		while (iter.blocks_left()) {	
			
			iter.next_block();		
			
			int r = iter.row();
			
			//std::cout << "BLOCK: " << r << std::endl;
			
			int soff0 = shell_offsets0[r];
			int nshell0 = nshells0[r];
			
			//std::cout << "soffs: " << soff0 << std::endl;
			//std::cout << "nshells: " << nshell0 << std::endl;
			
			int idx = 0;
			for (int s0 = soff0; s0 != soff0+nshell0; ++s0) {
							
				int shellsize0 = CINTcgto_spheric(s0,bas);
				shls[0] = s0;
				shls[1] = s0;
					
				//std::cout << "PROCESSING SHELL: " << s0 << std::endl;
				
				int res = int_func(buf, shls, atm, natm, bas, nbas, env, nullptr);
				
				double n = 0.0;
						
				if (res != 0) {
					int idx = 0;
				
					for (int i = 0; i != shellsize0; ++i) {
							n += fabs(buf[i + i * shellsize0]);
					} //std::cout << std::endl;
				}
					
				iter(idx++,0) = sqrt(n);
					
			}//endfor s1
						
		}//end BLOCK LOOP
		
		delete [] buf;
		delete [] shls;
		
		iter.stop();
		m_out.finalize();
		
	}//end parallel omp	
	
}

} 

} // end namespace megalochem
