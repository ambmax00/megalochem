#include "ints/integrals.h"
#include <stdexcept>
#include <map>
#include <mpi.h>
#include <omp.h>

#include <iostream>

namespace ints {

void calc_ints(dbcsr::mat_d& m_out, util::ShrPool<libint2::Engine>& engine,
	std::vector<const desc::cluster_basis*>& basvec) {

	auto my_world = m_out.get_world();
	
	const auto& cbas1 = *basvec[0];
	const auto& cbas2 = *basvec[1];
	
	#pragma omp parallel 
	{	
		
		dbcsr::iterator iter(m_out);
		
		auto& loc_eng = engine->local();
		const auto &results = loc_eng.results();
		
		iter.start();
		
		while (iter.blocks_left()) {	
			
			iter.next_block();		
						
			const auto& c1 = cbas1[iter.row()];
			const auto& c2 = cbas2[iter.col()];
			
			//block lower bounds
			int lb1 = iter.row_offset();
			int lb2 = iter.col_offset();
			
			// tensor offsets
			int toff1 = lb1;
			int toff2 = lb2;
			// local Block offsets
			int locblkoff1 = 0;
			int locblkoff2 = 0;
						
			for (int s1 = 0; s1!= c1.size(); ++s1) {
							
				const auto& sh1 = c1[s1];
				toff1 += locblkoff1;
					
				for (int s2 = 0; s2 != c2.size(); ++s2) {
					
					const auto& sh2 = c2[s2];
					toff2 += locblkoff2;
					
					//std::cout << "Tensor offsets: " << toff1 << " " << toff2 << std::endl;
					//std::cout << "Local offsets: " << locblkoff1 << " " << locblkoff2 << std::endl;
					//std::cout << "MULT: " << multiplicity(toff1,toff2) << std::endl;
					
					//std::cout << "Shells: " << std::endl;
					//std::cout << sh1 << std::endl;
					//std::cout << sh2 << std::endl;	
					//if (is_canonical(toff1,toff2)) {
						
						//int sfac = multiplicity(toff1,toff2);
						
						loc_eng.compute(sh1,sh2);										
						auto ints_shellsets = results[0];
											
						if (ints_shellsets != nullptr) {
							int idx = 0;
						
							for (int i = 0; i != sh1.size(); ++i) {
								for (int j = 0; j != sh2.size(); ++j) {
									iter(i + locblkoff1, j + locblkoff2) = ints_shellsets[idx++];
									//std::cout << i << " " << j << " " << blk(i + locblkoff1, j + locblkoff2) << std::endl;
							}}
						}
					//}
					
					locblkoff2 += sh2.size();						
				}//endfor s2
				
				toff2 = lb2;
				locblkoff2 = 0;
				locblkoff1 += sh1.size();
			}//endfor s1
						
		}//end BLOCK LOOP
		
		iter.stop();
		m_out.finalize();
		
	}//end parallel omp	
	
	//std::cout << "Done." << std::endl;
		
	//dbcsr::print(t);
		
	//std::cout << "Done reading." << std::endl;
}

void calc_ints(dbcsr::tensor<2>& t_out, util::ShrPool<libint2::Engine>& engine,
	std::vector<const desc::cluster_basis*>& basvec) {

	int myrank = 0;
	int mpi_size = 0;
	
	MPI_Comm_rank(t_out.comm(), &myrank); 
	MPI_Comm_size(t_out.comm(), &mpi_size);
	
	const auto& cbas1 = *basvec[0];
	const auto& cbas2 = *basvec[1];
	
	#pragma omp parallel 
	{	
		
		dbcsr::iterator_t<2> iter(t_out);
		
		auto& loc_eng = engine->local();
		const auto &results = loc_eng.results();
		
		iter.start();
		
		while (iter.blocks_left()) {	
						
			iter.next();
			
			auto idx = iter.idx();
			auto size = iter.size();
			auto off = iter.offset();		
						
			const auto& c1 = cbas1[idx[0]];
			const auto& c2 = cbas2[idx[1]];
			
			//std::cout << "WE ARE IN BLOCK: " << i1 << i2 << std::endl;
						
			dbcsr::block<2> blk(size); 
						
			//block lower bounds
			int lb1 = off[0];
			int lb2 = off[1];
			
			// tensor offsets
			int toff1 = lb1;
			int toff2 = lb2;
			// local Block offsets
			int locblkoff1 = 0;
			int locblkoff2 = 0;
						
			for (int s1 = 0; s1!= c1.size(); ++s1) {
							
				const auto& sh1 = c1[s1];
				toff1 += locblkoff1;
					
				for (int s2 = 0; s2 != c2.size(); ++s2) {
					
					const auto& sh2 = c2[s2];
					toff2 += locblkoff2;
					
					//std::cout << "Tensor offsets: " << toff1 << " " << toff2 << std::endl;
					//std::cout << "Local offsets: " << locblkoff1 << " " << locblkoff2 << std::endl;
					//std::cout << "MULT: " << multiplicity(toff1,toff2) << std::endl;
					
					//std::cout << "Shells: " << std::endl;
					//std::cout << sh1 << std::endl;
					//std::cout << sh2 << std::endl;	
					//if (is_canonical(toff1,toff2)) {
						
						//int sfac = multiplicity(toff1,toff2);
						
						loc_eng.compute(sh1,sh2);										
						auto ints_shellsets = results[0];
											
						if (ints_shellsets != nullptr) {
							int idx = 0;
						
							for (int i = 0; i != sh1.size(); ++i) {
								for (int j = 0; j != sh2.size(); ++j) {
									blk(i + locblkoff1, j + locblkoff2) = ints_shellsets[idx++];
									//std::cout << i << " " << j << " " << blk(i + locblkoff1, j + locblkoff2) << std::endl;
							}}
						}
					//}
					
					locblkoff2 += sh2.size();						
				}//endfor s2
				
				toff2 = lb2;
				locblkoff2 = 0;
				locblkoff1 += sh1.size();
			}//endfor s1
						
			t_out.put_block(idx, blk);
						
		}//end BLOCK LOOP
		
		t_out.finalize();
		
		iter.stop();
		
	}//end parallel omp	
	
	//std::cout << "Done." << std::endl;
		
	//dbcsr::print(t);
		
	//std::cout << "Done reading." << std::endl;
}

void calc_ints(dbcsr::tensor<3>& t_out, util::ShrPool<libint2::Engine>& engine,
	std::vector<const desc::cluster_basis*>& basvec, screener* scr) {
	
	int myrank = 0;
	int mpi_size = 0;
	
	MPI_Comm_rank(t_out.comm(), &myrank); 
	MPI_Comm_size(t_out.comm(), &mpi_size);

	const auto& cbas1 = *basvec[0];
	const auto& cbas2 = *basvec[1];
	const auto& cbas3 = *basvec[2];
	
	#pragma omp parallel 
	{	
	
		auto& loc_eng = engine->local();
		const auto &results = loc_eng.results();
		dbcsr::iterator_t<3> iter(t_out);
		
		iter.start();
		
		while (iter.blocks_left()) {
			
			iter.next();	
					
			auto& idx = iter.idx();
			auto& size = iter.size();
			auto& off = iter.offset();
			
			const auto& c1 = cbas1[idx[0]];
			const auto& c2 = cbas2[idx[1]];
			const auto& c3 = cbas3[idx[2]];
			
			//int sh1off = cbas1.shell_offset(idx[0]);
			//int sh2off = cbas2.shell_offset(idx[1]);
			//int sh3off = cbas3.shell_offset(idx[2]);
			
			dbcsr::block<3> blk(size); 
			
			//lower bounds
			int lb1 = off[0];
			int lb2 = off[1];
			int lb3 = off[2];
			
			// tensor offsets
			int toff1 = lb1;
			int toff2 = lb2;
			int toff3 = lb3;
			
			// local Block offsets
			int locblkoff1 = 0;
			int locblkoff2 = 0;
			int locblkoff3 = 0;
			
			for (int s1 = 0; s1!= c1.size(); ++s1) {
				
				const auto& sh1 = c1[s1];
				toff1 += locblkoff1;
		
				for (int s2 = 0; s2 != c2.size(); ++s2) {
					
					const auto& sh2 = c2[s2];
					toff2 += locblkoff2;
					
					for (int s3 = 0; s3 != c3.size(); ++s3) {
						
						//if (scr->skip(sh1off+s1,sh2off+s2,sh3off+s3)) {
						//	continue;
						//}
						
						const auto& sh3 = c3[s3];
						toff3 += locblkoff3;
								
						loc_eng.compute(sh1,sh2,sh3);
								
						auto ints_shellsets = results[0];
								
						if (ints_shellsets != nullptr) {
							size_t idx = 0;
							for (int i = 0; i != sh1.size(); ++i) {
							 for (int j = 0; j != sh2.size(); ++j) {
							  for (int k = 0; k != sh3.size(); ++k) {
							    blk(i + locblkoff1, j + locblkoff2, k + locblkoff3) 
												= ints_shellsets[idx++];
							}}}
						}//endif shellsets
						
						locblkoff3 += sh3.size();
					} //endfor s3
					toff3 = lb3;
					locblkoff3 = 0;
					locblkoff2 += sh2.size();
				}//endfor s2
				toff2 = lb2;
				locblkoff2 = 0;
				locblkoff1 += sh1.size();
			}//endfor s1
			
			t_out.put_block(idx, blk);
			
		}//end BLOCK LOOP
		
		t_out.finalize();
		iter.stop();
		
	}//end parallel omp	
	
}

void calc_ints(dbcsr::tensor<4>& t_out, util::ShrPool<libint2::Engine>& engine,
	std::vector<const desc::cluster_basis*>& basvec) {
	
	int myrank = 0;
	int mpi_size = 0;
	
	MPI_Comm_rank(t_out.comm(), &myrank); 
	MPI_Comm_size(t_out.comm(), &mpi_size);

	const auto& cbas1 = *basvec[0];
	const auto& cbas2 = *basvec[1];
	const auto& cbas3 = *basvec[2];
	const auto& cbas4 = *basvec[3];
	
	size_t nblks = 0;

	#pragma omp parallel 
	{	
		
		auto& loc_eng = engine->local();
		const auto &results = loc_eng.results();
		dbcsr::iterator_t<4> iter(t_out);
		
		iter.start();
		
		while (iter.blocks_left()) {
			
			iter.next();
						
			auto idx = iter.idx();
			auto size = iter.size();
			auto off = iter.offset();
			
			const auto& c1 = cbas1[idx[0]];
			const auto& c2 = cbas2[idx[1]];
			const auto& c3 = cbas3[idx[2]];
			const auto& c4 = cbas4[idx[3]];
			
			dbcsr::block<4> blk(size); //= t.get_block({.idx = IDX, .blk_size = blkdim, .found = found});
			
			//lower bounds
			int lb1 = off[0];
			int lb2 = off[1];
			int lb3 = off[2];
			int lb4 = off[3];
			
			// tensor offsets
			int toff1 = lb1;
			int toff2 = lb2;
			int toff3 = lb3;
			int toff4 = lb4;
			
			// local Block offsets
			int locblkoff1 = 0;
			int locblkoff2 = 0;
			int locblkoff3 = 0;
			int locblkoff4 = 0;
			
			//std::cout << "IDX: " << idx[0] << " " << idx[1] << " " << idx[2] << " " << idx[3] << " " << nblks++ << std::endl;
			//std::cout << "SIZE: " << size[0] << " " << size[1] << " " << size[2] << " " << size[3] << " " << nblks++ << std::endl;
			
			for (int s1 = 0; s1!= c1.size(); ++s1) {
		
				const auto& sh1 = c1[s1];
				toff1 += locblkoff1;

				for (int s2 = 0; s2 != c2.size(); ++s2) {
					
					const auto& sh2 = c2[s2];
					toff2 += locblkoff2;
					
					for (int s3 = 0; s3 != c3.size(); ++s3) {
						
						const auto& sh3 = c3[s3];
						toff3 += locblkoff3;
						
						for (int s4 = 0; s4 != c4.size(); ++s4) {
							
							const auto& sh4 = c4[s4];
							toff4 += locblkoff4;
							
							//if (is_canonical(toff1,toff2,toff3,toff4) && brablk(s1,s2)*ketblk(s3,s4) > screening_threshold) {
							//if (true) {
								//int sfac = multiplicity(toff1,toff2,toff3,toff4);
								
								//std::cout << "TENSOR AT " << toff1 << " " << toff2 << " " << toff3 << " " << toff4 << " : mult " << sfac << std::endl;
								
								//std::cout << "Shell: " << s1 << " " << s2 << " " << s3 << " " << s4 << std::endl;
								//std::cout << "Sizes: " << sh1.size() << " " << sh2.size() << " " << sh3.size() << " " << sh4.size() << std::endl;
								//std::cout << "Offset: " << off1 << " " << off2 << " " << off3 << " " << off4 << std::endl;
								
								loc_eng.compute(sh1,sh2,sh3,sh4);
								
								auto ints_shellsets = results[0];
								
								if (ints_shellsets != nullptr) {
									size_t idx = 0;
									for (int i = 0; i != sh1.size(); ++i) {
										for (int j = 0; j != sh2.size(); ++j) {
											for (int k = 0; k != sh3.size(); ++k) {
												for (int l = 0; l != sh4.size(); ++l) {
													blk(i + locblkoff1, j + locblkoff2, k + locblkoff3, l + locblkoff4) 
														= ints_shellsets[idx++];
													//std::cout << blk(i + off1, j + off2, k + off3, l + off4)
													//<< std::endl;
									}}}}
								}//endif shellsets
							//}//end if canonicaö		
							locblkoff4 += sh4.size();
						}//endfor s4
						toff4 = lb4;
						locblkoff4 = 0;
						locblkoff3 += sh3.size();
					} //endfor s3
					toff3 = lb3;
					locblkoff3 = 0;
					locblkoff2 += sh2.size();
				}//endfor s2
				toff2 = lb2;
				locblkoff2 = 0;
				locblkoff1 += sh1.size();
			}//endfor s1
			
			t_out.put_block(idx, blk);
			
		}//end BLOCK LOOP
		
		t_out.finalize();
		iter.stop();
		
	}//end parallel omp	
	
}

void calc_ints_schwarz_mn(dbcsr::mat_d& m_out, util::ShrPool<libint2::Engine>& engine,
	std::vector<const desc::cluster_basis*>& basvec) {

	auto my_world = m_out.get_world();
	
	const auto& cbas1 = *basvec[0];
	const auto& cbas2 = *basvec[1];
	
	#pragma omp parallel 
	{	
		
		dbcsr::iterator iter(m_out);
		
		auto& loc_eng = engine->local();
		const auto &results = loc_eng.results();
		
		iter.start();
		
		while (iter.blocks_left()) {	
			
			iter.next_block();		
						
			const auto& c1 = cbas1[iter.row()];
			const auto& c2 = cbas2[iter.col()];
			
			//block lower bounds
			int lb1 = iter.row_offset();
			int lb2 = iter.col_offset();
			
			// tensor offsets
			int toff1 = lb1;
			int toff2 = lb2;
			// local Block offsets
			int locblkoff1 = 0;
			int locblkoff2 = 0;
						
			for (int s1 = 0; s1!= c1.size(); ++s1) {
							
				const auto& sh1 = c1[s1];
				auto sh1size = sh1.size();
				toff1 += locblkoff1;
					
				for (int s2 = 0; s2 != c2.size(); ++s2) {
					
					const auto& sh2 = c2[s2];
					auto sh2size = sh2.size();
					toff2 += locblkoff2;
					
					//std::cout << "Tensor offsets: " << toff1 << " " << toff2 << std::endl;
					//std::cout << "Local offsets: " << locblkoff1 << " " << locblkoff2 << std::endl;
					//std::cout << "MULT: " << multiplicity(toff1,toff2) << std::endl;
					
					//std::cout << "Shells: " << std::endl;
					//std::cout << sh1 << std::endl;
					//std::cout << sh2 << std::endl;	
					//if (is_canonical(toff1,toff2)) {
						
						//int sfac = multiplicity(toff1,toff2);
						
						loc_eng.compute(sh1,sh2,sh1,sh2);										
						auto ints_shellsets = results[0];
						
						//std::cout << "IDX: " << toff1 << " " << toff2 << std::endl;
											
						if (ints_shellsets != nullptr) {
							
							double tot = 0.0;
						
							for (int i = 0; i != sh1.size(); ++i) {
								for (int j = 0; j != sh2.size(); ++j) {
									tot += fabs(ints_shellsets[i*sh2size*sh2size*sh1size
										+ j*sh1size*sh2size + i*sh2size + j]);
							}}
							
							iter(s1,s2) = sqrt(tot);
							
						}
					//}
					
					locblkoff2 += sh2.size();						
				}//endfor s2
				
				toff2 = lb2;
				locblkoff2 = 0;
				locblkoff1 += sh1.size();
			}//endfor s1
						
		}//end BLOCK LOOP
		
		iter.stop();
		m_out.finalize();
		
	}//end parallel omp	
	
	//std::cout << "Done." << std::endl;
		
	//dbcsr::print(t);
		
	//std::cout << "Done reading." << std::endl;
}

void calc_ints_schwarz_x(dbcsr::mat_d& m_out, util::ShrPool<libint2::Engine>& engine,
	std::vector<const desc::cluster_basis*>& basvec) {

	auto my_world = m_out.get_world();
	
	const auto& cbas1 = *basvec[0];
	
	#pragma omp parallel 
	{	
		
		dbcsr::iterator iter(m_out);
		
		auto& loc_eng = engine->local();
		const auto &results = loc_eng.results();
		
		iter.start();
		
		while (iter.blocks_left()) {	
			
			iter.next_block();		
						
			const auto& c1 = cbas1[iter.row()];
			std::cout << iter.row() << std::endl;
			
			//block lower bounds
			int lb1 = iter.row_offset();
			
			// tensor offsets
			int toff1 = lb1;
			// local Block offsets
			int locblkoff1 = 0;
						
			for (int s1 = 0; s1!= c1.size(); ++s1) {
							
				const auto& sh1 = c1[s1];
				auto sh1size = sh1.size();
				toff1 += locblkoff1;
						
					loc_eng.compute(sh1,sh1);										
					auto ints_shellsets = results[0];
										
					if (ints_shellsets != nullptr) {
					
						double tot = 0;
						int idx = 0;
						for (int i = 0; i != sh1.size(); ++i) {
							tot =+ fabs(ints_shellsets[i*sh1.size() + i]);
							//std::cout << i << " " << j << " " << blk(i + locblkoff1, j + locblkoff2) << std::endl;
						}
						
						iter(s1,0) = sqrt(tot);
						
					}
				//}
				
				locblkoff1 += sh1.size();
			}//endfor s1
						
		}//end BLOCK LOOP
		
		iter.stop();
		m_out.finalize();
		
	}//end parallel omp	
	
	//std::cout << "Done." << std::endl;
		
	//dbcsr::print(t);
		
	//std::cout << "Done reading." << std::endl;
}

}