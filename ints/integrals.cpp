#include "ints/integrals.h"
#include <stdexcept>
#include <map>
#include <mpi.h>
#include <omp.h>

#include <iostream>

namespace ints {
	
static double screening_threshold = 1e-9;

// return just block?
void calc_ints(dbcsr::tensor<2,double>& t, util::ShrPool<libint2::Engine>& engine, 
	std::vector<desc::cluster_basis>& basvec, Zmat* bra, Zmat* ket) {

	int myrank = 0;
	int mpi_size = 0;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	auto blksloc = t.blks_local();
	auto nblks = t.nblks_tot();
	auto blk_size = t.blk_size();
	auto blk_off = t.blk_offset();
	
	vec<vec<int>> reserve(2);
	
	auto& cbas1 = basvec[0];
	auto& cbas2 = basvec[1];
		
	for (int i1 = 0; i1 != blksloc[0].size(); ++i1) {
		for (int i2 = 0; i2 != blksloc[1].size(); ++i2) {
						
				int ix1 = blksloc[0][i1];
				int ix2 = blksloc[1][i2];
						
				if (ix1 < ix2) continue;
				
				reserve[0].push_back(ix1);
				reserve[1].push_back(ix2);
			
	}}
	
	t.reserve(reserve);

#pragma omp parallel 
{	
	
	auto& loc_eng = engine->local();
	const auto &results = loc_eng.results();
	
	#pragma omp for 
	for (size_t I = 0; I != reserve[0].size(); ++I) {	
					
		int i1 = reserve[0][I];
		int i2 = reserve[1][I];

					
		std::vector<libint2::Shell>& c1 = cbas1[i1];
		std::vector<libint2::Shell>& c2 = cbas2[i2];
		
		//std::cout << "WE ARE IN BLOCK: " << i1 << i2 << std::endl;
					
		dbcsr::idx2 IDX = {i1,i2};
		vec<int> blkdim = {blk_size[0][i1],blk_size[1][i2]};
		bool found = false;
		dbcsr::block<2> blk(blkdim); //= t.get_block({.idx = IDX, .blk_size = blkdim, .found = found});
					
		//block lower bounds
		int lb1 = blk_off[0][i1];
		int lb2 = blk_off[1][i2];
		
		// tensor offsets
		int toff1 = lb1;
		int toff2 = lb2;
		// local Block offsets
		int locblkoff1 = 0;
		int locblkoff2 = 0;
					
		for (int s1 = 0; s1!= c1.size(); ++s1) {
						
			auto& sh1 = c1[s1];
			toff1 += locblkoff1;
				
			for (int s2 = 0; s2 != c2.size(); ++s2) {
				
				auto& sh2 = c2[s2];
				toff2 += locblkoff2;
				
				//std::cout << "Tensor offsets: " << toff1 << " " << toff2 << std::endl;
				//std::cout << "Local offsets: " << locblkoff1 << " " << locblkoff2 << std::endl;
				//std::cout << "MULT: " << multiplicity(toff1,toff2) << std::endl;
										
				if (is_canonical(toff1,toff2)) {
					
					int sfac = multiplicity(toff1,toff2);
					
					loc_eng.compute(sh1,sh2);										
					auto ints_shellsets = results[0];
										
					if (ints_shellsets != nullptr) {
						int idx = 0;
					
						for (int i = 0; i != sh1.size(); ++i) {
							for (int j = 0; j != sh2.size(); ++j) {
								blk(i + locblkoff1, j + locblkoff2) = sfac * ints_shellsets[idx++];
						}}
					}
				}
				
				locblkoff2 += sh2.size();						
			}//endfor s2
			
			toff2 = lb2;
			locblkoff2 = 0;
			locblkoff1 += sh1.size();
		}//endfor s1
					
		#pragma omp critical
		{
			t.put_block({.idx = IDX, .blk = blk});
		}
					
	}//end BLOCK LOOP
}//end parallel omp	
	
	//std::cout << "Done." << std::endl;
	
	t.filter();
	
	//dbcsr::print(t);
		
	//std::cout << "Done reading." << std::endl;

}

void calc_ints(dbcsr::tensor<3,double>& t, util::ShrPool<libint2::Engine>& engine, 
	std::vector<desc::cluster_basis>& basvec, Zmat* bra, Zmat* ket) {
	
	int myrank = 0;
	int mpi_size = 0;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	auto blksloc = t.blks_local();
	auto nblks = t.nblks_tot();
	auto blk_size = t.blk_size();
	auto blk_off = t.blk_offset();
	
	vec<vec<int>> reserve(3);
	
	auto& cbas1 = basvec[0];
	auto& cbas2 = basvec[1];
	auto& cbas3 = basvec[2];
	
	//std::cout << "Reserving..." << std::endl;
	
	// reserve blocks
	//if (bra) {
	/*if (false) {
		
		for (auto& e1 : bra->m_blkmap) {
			for (auto& e2 : ket->m_blkmap) {
				
				auto i1 = e1.first;
				auto i2 = e2.first;
				
				//if (!is_canonical(i1[0],i1[1],i2[0],i2[1])) continue;
				
				auto n1 = e1.second.first;
				auto n2 = e2.second.first;
				
				std::cout << "NN " << n1 * n2 << std::endl;
				if (n1 * n2 < screening_threshold) {
					std::cout << "SCREENED!" << std::endl; 
					continue;
				}
				
				int p = -1;
				dbcsr::idx4 IDX = {i1[0],i1[1],i2[0],i2[1]};
				t.get_stored_coordinates({.idx = IDX, .proc = p});
				
				std::cout << "RESERVE: " << std::endl;
				std::cout << p << " " << i1[0] << " " << i1[1] << " " << i2[0] << " " << i2[1] << std::endl;
				
				if (myrank == p) {
					reserve[0].push_back(i1[0]);
					reserve[1].push_back(i1[1]);
					reserve[2].push_back(i2[0]);
					reserve[3].push_back(i2[1]);
				}
				
			}
		}
		
	} else {*/
		
		for (int i1 = 0; i1 != blksloc[0].size(); ++i1) {
			for (int i2 = 0; i2 != blksloc[1].size(); ++i2) {
				for (int i3 = 0; i3!= blksloc[2].size(); ++i3) {
						
						int ix1 = blksloc[0][i1];
						int ix2 = blksloc[1][i2];
						int ix3 = blksloc[2][i3];
						
						//if (!is_canonical(ix1,ix2,ix3)) continue;
						
						reserve[0].push_back(ix1);
						reserve[1].push_back(ix2);
						reserve[2].push_back(ix3);
			
		}}}
		
	//}
	
	t.reserve(reserve);	

	#pragma omp parallel 
	{	
	
		auto& loc_eng = engine->local();
		const auto &results = loc_eng.results();
		
		#pragma omp for 
		for (int I = 0; I != reserve[0].size(); ++I) {	
					
					int i1 = reserve[0][I];
					int i2 = reserve[1][I];
					int i3 = reserve[2][I];
					
					std::vector<libint2::Shell>& c1 = cbas1[i1];
					std::vector<libint2::Shell>& c2 = cbas2[i2];
					std::vector<libint2::Shell>& c3 = cbas3[i3];
					
					//std::cout << "WE ARE IN BLOCK: " << i1 << i2 << i3 << std::endl;
					
					dbcsr::idx3 IDX = {i1,i2,i3};
					vec<int> blkdim = {blk_size[0][i1],blk_size[1][i2],blk_size[2][i3]};
					
					//std::cout << "BLOCK DIMENSIONS: " << std::endl;
					//std::cout << blk_size[0][i1] << " " << blk_size[1][i2] << " " << blk_size[2][i3] << std::endl;
					
					
					bool found = false;
					dbcsr::block<3> blk(blkdim); //= t.get_block({.idx = IDX, .blk_size = blkdim, .found = found});
					
					// load bra and ket block for screening
					//auto& brablk = bra->m_blkmap[dbcsr::idx2{i1,i2}].second;
					//auto& ketblk = ket->m_blkmap[dbcsr::idx2{i3,i4}].second;
					
					//lower bounds
					int lb1 = blk_off[0][i1];
					int lb2 = blk_off[1][i2];
					int lb3 = blk_off[2][i3];
					
					// tensor offsets
					int toff1 = lb1;
					int toff2 = lb2;
					int toff3 = lb3;
					
					// local Block offsets
					int locblkoff1 = 0;
					int locblkoff2 = 0;
					int locblkoff3 = 0;
					
					for (int s1 = 0; s1!= c1.size(); ++s1) {
						
						auto& sh1 = c1[s1];
						toff1 += locblkoff1;
				
						for (int s2 = 0; s2 != c2.size(); ++s2) {
							
							auto& sh2 = c2[s2];
							toff2 += locblkoff2;
							
							for (int s3 = 0; s3 != c3.size(); ++s3) {
								
								auto& sh3 = c3[s3];
								toff3 += locblkoff3;
									
									//if (is_canonical(toff1,toff2,toff3)) {
									//if (true) {
										//int sfac = 1.0; //multiplicity(toff1,toff2,toff3);
										
										//std::cout << "TENSOR AT " << toff1 << " " << toff2 << " " << toff3 << " " << toff4 << " : mult " << sfac << std::endl;
										
										//std::cout << "Shells: " << s1 << " " << s2 << " " << s3 << " " << s4 << std::endl;
										//std::cout << "Sizes: " << sh1.size() << " " << sh2.size() << " " << sh3.size() << " " << sh4.size() << std::endl;
										//std::cout << "Offset: " << off1 << " " << off2 << " " << off3 << " " << off4 << std::endl;
										
										loc_eng.compute(sh1,sh2,sh3);
												
										auto ints_shellsets = results[0];
												
										if (ints_shellsets != nullptr) {
											size_t idx = 0;
											for (int i = 0; i != sh1.size(); ++i) {
												for (int j = 0; j != sh2.size(); ++j) {
													for (int k = 0; k != sh3.size(); ++k) {
															blk(i + locblkoff1, j + locblkoff2, k + locblkoff3) 
																= /*sfac */ ints_shellsets[idx++];
																	//std::cout << blk(i + off1, j + off2, k + off3, l + off4)
																	//<< std::endl;
											}}}
										}//endif shellsets
										
									//} // end if canonical
									
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
					
					#pragma omp critical
					{
						t.put_block({.idx = IDX, .blk = blk});
					}
					
				}//end BLOCK LOOP
	}//end parallel omp	
	
	//std::cout << "Done." << std::endl;
	
	t.filter();
	
	//dbcsr::print(t);
		
	//std::cout << "Done with 3c2e." << std::endl;

}

void calc_ints(dbcsr::tensor<4,double>& t, util::ShrPool<libint2::Engine>& engine, 
	std::vector<desc::cluster_basis>& basvec, Zmat* bra, Zmat* ket) {
	
	int myrank = 0;
	int mpi_size = 0;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	auto blksloc = t.blks_local();
	auto nblks = t.nblks_tot();
	auto blk_size = t.blk_size();
	auto blk_off = t.blk_offset();
	
	vec<vec<int>> reserve(4);
	
	auto& cbas1 = basvec[0];
	auto& cbas2 = basvec[1];
	auto& cbas3 = basvec[2];
	auto& cbas4 = basvec[3];
	
	// reserve blocks
	//if (bra) {
	if (false) {
		
		for (auto& e1 : bra->m_blkmap) {
			for (auto& e2 : ket->m_blkmap) {
				
				auto i1 = e1.first;
				auto i2 = e2.first;
				
				//if (!is_canonical(i1[0],i1[1],i2[0],i2[1])) continue;
				
				auto n1 = e1.second.first;
				auto n2 = e2.second.first;
				
				std::cout << "NN " << n1 * n2 << std::endl;
				if (n1 * n2 < screening_threshold) {
					std::cout << "SCREENED!" << std::endl; 
					continue;
				}
				
				int p = -1;
				dbcsr::idx4 IDX = {i1[0],i1[1],i2[0],i2[1]};
				t.get_stored_coordinates({.idx = IDX, .proc = p});
				
				std::cout << "RESERVE: " << std::endl;
				std::cout << p << " " << i1[0] << " " << i1[1] << " " << i2[0] << " " << i2[1] << std::endl;
				
				if (myrank == p) {
					reserve[0].push_back(i1[0]);
					reserve[1].push_back(i1[1]);
					reserve[2].push_back(i2[0]);
					reserve[3].push_back(i2[1]);
				}
				
			}
		}
		
	} else {
		
		for (int i1 = 0; i1 != blksloc[0].size(); ++i1) {
			for (int i2 = 0; i2 != blksloc[1].size(); ++i2) {
				for (int i3 = 0; i3!= blksloc[2].size(); ++i3) {
					for (int i4 = 0; i4 != blksloc[3].size(); ++i4) {
						
						int ix1 = blksloc[0][i1];
						int ix2 = blksloc[1][i2];
						int ix3 = blksloc[2][i3];
						int ix4 = blksloc[3][i4];
						
						reserve[0].push_back(ix1);
						reserve[1].push_back(ix2);
						reserve[2].push_back(ix3);
						reserve[3].push_back(ix4);
						
						std::cout << "RESERVE: " << std::endl;
						std::cout << ix1 << " " << ix2 << " " << ix3 << " " << ix4 << std::endl;
			
		}}}}
		
	}
	
	t.reserve(reserve);	



#pragma omp parallel 
{	
	
	auto& loc_eng = engine->local();
	const auto &results = loc_eng.results();
	
	#pragma omp for 
	for (int I = 0; I != reserve[0].size(); ++I) {	
					
					int i1 = reserve[0][I];
					int i2 = reserve[1][I];
					int i3 = reserve[2][I];
					int i4 = reserve[3][I];
					
					std::vector<libint2::Shell>& c1 = cbas1[i1];
					std::vector<libint2::Shell>& c2 = cbas2[i2];
					std::vector<libint2::Shell>& c3 = cbas3[i3];
					std::vector<libint2::Shell>& c4 = cbas4[i4];
					
					std::cout << "WE ARE IN BLOCK: " << i1 << i2 << i3 << i4 << std::endl;
					
					dbcsr::idx4 IDX = {i1,i2,i3,i4};
					vec<int> blkdim = {blk_size[0][i1],blk_size[1][i2],blk_size[2][i3],blk_size[3][i4]};
					
					std::cout << "BLOCK DIMENSIONS: " << std::endl;
					std::cout << blk_size[0][i1] << " " << blk_size[1][i2] << " " << blk_size[2][i3] << " " << blk_size[3][i4] << std::endl;
					
					
					bool found = false;
					dbcsr::block<4> blk(blkdim); //= t.get_block({.idx = IDX, .blk_size = blkdim, .found = found});
					
					// load bra and ket block for screening
					//auto& brablk = bra->m_blkmap[dbcsr::idx2{i1,i2}].second;
					//auto& ketblk = ket->m_blkmap[dbcsr::idx2{i3,i4}].second;
					
					//lower bounds
					int lb1 = blk_off[0][i1];
					int lb2 = blk_off[1][i2];
					int lb3 = blk_off[2][i3];
					int lb4 = blk_off[3][i4];
					
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
					
					for (int s1 = 0; s1!= c1.size(); ++s1) {
						
						auto& sh1 = c1[s1];
						toff1 += locblkoff1;
				
						for (int s2 = 0; s2 != c2.size(); ++s2) {
							
							auto& sh2 = c2[s2];
							toff2 += locblkoff2;
							
							for (int s3 = 0; s3 != c3.size(); ++s3) {
								
								auto& sh3 = c3[s3];
								toff3 += locblkoff3;
								
								for (int s4 = 0; s4 != c4.size(); ++s4) {
									
									auto& sh4 = c4[s4];
									toff4 += locblkoff4;
									
									//if (is_canonical(toff1,toff2,toff3,toff4) /*&& brablk(s1,s2)*ketblk(s3,s4) > screening_threshold*/) {
									//if (true) {
										//int sfac = multiplicity(toff1,toff2,toff3,toff4);
										
										//std::cout << "TENSOR AT " << toff1 << " " << toff2 << " " << toff3 << " " << toff4 << " : mult " << sfac << std::endl;
										
										//std::cout << "Shells: " << s1 << " " << s2 << " " << s3 << " " << s4 << std::endl;
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
																= /*sfac */ ints_shellsets[idx++];
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
					
					#pragma omp critical
					{
						t.put_block({.idx = IDX, .blk = blk});
					}
					
				}//end BLOCK LOOP
}//end parallel omp	
	
	//std::cout << "Done." << std::endl;
	
	t.filter();
	
	dbcsr::print(t);
		
	std::cout << "Done with 2e." << std::endl;

}
	
template <int N>
dbcsr::stensor<N,double> integrals(integral_parameters<N>&& p) {
		
	if (p.basvec->size() != N) throw std::runtime_error("Basis vector incompatible with tensor dimensions.");
	
	//create pgrid
	dbcsr::pgrid<N> pgridN({.comm = *p.comm});
	
	vec<vec<int>> blk_sizes;
	
	for (int i = 0; i != N; ++i) {
		blk_sizes.push_back(p.basvec->operator[](i).cluster_sizes());
	}
	
	/*
	int iter = 0;
	for (auto x : blk_sizes) {
		std::cout << iter++ << " : ";
		for (auto e : x) {
			std::cout << e << " ";
		} std::cout << std::endl;
	}
	*/
	//create tensor 
	auto out = dbcsr::make_stensor<N>({.name = *p.name, .pgridN = pgridN, .map1 = *p.map1, 
		.map2 = *p.map2, .blk_sizes = blk_sizes});
		
	//std::cout << "Q1" << std::endl;
	
	Zmat* bra_ptr = (p.bra) ? &*p.bra : nullptr;
	Zmat* ket_ptr = (p.ket) ? &*p.ket : nullptr;
	
	calc_ints(*out, *p.engine, *p.basvec, bra_ptr, ket_ptr);
	
	//std::cout << "Q2" << std::endl;
	
	return out;
		
}

template dbcsr::stensor<2,double> integrals(integral_parameters<2>&& p);
template dbcsr::stensor<3,double> integrals(integral_parameters<3>&& p);
template dbcsr::stensor<4,double> integrals(integral_parameters<4>&& p);


}
