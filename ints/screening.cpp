#include "ints/screening.h"
#include "ints/integrals.h"
#include "math/tensor/dbcsr.hpp"
#include "utils/mpi_log.h"

namespace ints {

// Z_mn = |(mn|mn)|^-1/2
void Zmat::compute_schwarz() {
	
	// it's symmetric!
	// make tensor
	dbcsr::pgrid<2> grid({.comm = m_comm});
	
	util::mpi_log LOG(0,m_comm);
	
	int comm_size, rank;
	MPI_Comm_size(m_comm, &comm_size);
	MPI_Comm_rank(m_comm, &rank);
	
	// make tensor
	
	auto s = m_mol.dims().s();
	m_blk_sizes = {s,s};
	dbcsr::tensor<2> Ztensor({.name = "Zmat_schwarz", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = m_blk_sizes});
	
	// reserve
	vec<vec<int>> reserved(2);
	
	for (int i = 0; i != s.size(); ++i) {
		for (int j = 0; j <= i; ++j) {
			
			dbcsr::idx2 idx = {i,j};
			int proc = -1;
			Ztensor.get_stored_coordinates({.idx = idx, .proc = proc});
			
			if (proc == rank) {
				reserved[0].push_back(i);
				reserved[1].push_back(j);
				std::cout << i << " " << j << std::endl;
			}
	}}
	
	Ztensor.reserve(reserved);
	
	
	auto cbas = m_mol.c_basis();
	
	//LOG.os<0,-1>(rank, " Entering Parallel Region...\n");
	
	#pragma omp parallel 
	{	
	
	auto& loc_eng = m_eng->local();
	const auto &results = loc_eng.results();
	
	#pragma omp for 
	for (int idx = 0; idx != reserved[0].size(); ++idx) {
		
		int idx0 = reserved[0][idx];
		int idx1 = reserved[1][idx];
		dbcsr::idx2 IDX = {idx0,idx1};
		
		dbcsr::block<2> blk(vec<int>{m_blk_sizes[0][idx0],m_blk_sizes[1][idx1]});
		
		std::cout << "HERE1" << std::endl;
		
		//LOG.os<0,-1>("Rank: ", rank, ", Thread: ", omp_get_thread_num(), " IDX: ", idx0, " ", idx1, '\n');		
		
		auto& cl0 = cbas[idx0];
		auto& cl1 = cbas[idx1];
		
		int off0 = 0;
		int off1 = 0;
		
		for (int s0 = 0; s0 != cl0.size(); ++s0) {
			auto& sh0 = cl0[s0];
			for (int s1 = 0; s1 != cl1.size(); ++s1) {
				auto& sh1 = cl1[s1];
				
				loc_eng.compute(sh0,sh1,sh0,sh1);
				auto ints_shellsets = results[0];
				
				double norm = 0.0;
				
				if (ints_shellsets != nullptr) {
					int sh0size = sh0.size();
					int sh1size = sh1.size();
					for (int i0 = 0; i0 != sh0.size(); ++i0) {
						for (int i1 = 0; i1 != sh1.size(); ++i1) {
							norm += pow(
								ints_shellsets[i0*sh1size*sh0size*sh1size + i1*sh1size*sh0size + i1*sh1size + i0],2);
						}
					}	
				}
				
				norm = sqrt(norm);
				
				blk(s0,s1) = sqrt(norm);
				
				off1 += sh1.size();
			}
			off1 = 0;
			off0 += sh0.size();
		}
		
		std::cout << "HERE2" << std::endl;
		
		#pragma omp critical 
		{
			Ztensor.put_block({.idx = IDX, .blk = blk});
		}
		
		std::cout << "HERE3" << std::endl;
				
	}
	
	} // end parallel region

	Ztensor.filter();
	
	std::cout << "HERE35" << std::endl;
	
	dbcsr::print(Ztensor);
	
	std::cout << "HERE4" << std::endl;
	
	LOG.os<0,-1>("HERE\n");
	// new para: get norm and index
	dbcsr::iterator<2> iter(Ztensor);
	int loc_nzblks = Ztensor.num_blocks();
	int off = cbas.size();
	
	dbcsr::idx2* loc_indices = new dbcsr::idx2[loc_nzblks];
	float* loc_norms = new float[loc_nzblks];
	dbcsr::block<2>* loc_blocks = new dbcsr::block<2>[loc_nzblks];
	
	// Loop again to collect blocks
	#pragma omp parallel for
	for (int inz = 0; inz != loc_nzblks; ++inz) {
		
		vec<int> blksize;
		dbcsr::idx2 idx;
		
		#pragma omp critical 
		{
			iter.next();
			blksize = iter.sizes();
			idx = iter.idx();
		}
		
		bool found = false;
		loc_blocks[inz] = Ztensor.get_block({.idx = idx, .blk_size = blksize, .found = found});
		
		loc_norms[inz] = loc_blocks[inz].norm(); 
		loc_indices[inz] = idx;
		
	}
	
	
	
	std::cout << rank << "OUT!" << std::endl;
	MPI_Barrier(m_comm);
	//LOG.os<0,-1>(rank, " DONE.\n");
	
	std::cout << rank << " OK??" << std::endl;
	// Nicely done! Now communicate that stuff
	std::cout << rank << "Getting blocks" << std::endl;
	size_t blkstot = Ztensor.num_blocks_total();
	Ztensor.destroy();
	std::cout << rank << "Ok. " << std::endl;
	
	
	LOG.os<0,0>("ALL BLOCKS: ", blkstot, '\n');
	LOG.flush();
	
	int* nzblks = new int[comm_size]();
	nzblks[rank] = loc_nzblks; 
	
	// collect all numbers of nonzero blocks on all nodes
	for (int r = 0; r != comm_size; ++r) {
		MPI_Bcast(&nzblks[r], 1, MPI_INT, r, m_comm);
	}

	
	if (rank == 0) {
		for (int i = 0; i != comm_size; ++i) {
			std::cout << nzblks[i] << " "; 
		} std::cout << std::endl;
	}
	
	// data structures
	dbcsr::idx2 *allidxs = new dbcsr::idx2[blkstot]();
	float *allnorms = new float[blkstot]();
	dbcsr::block<2> *allblocks = new dbcsr::block<2>[blkstot]();
	
	int offset = 0;
	
	for (int r = 0; r != comm_size; ++r) {
		
		if (r == rank) {
			for (int n = 0; n != loc_nzblks; ++n) {
				allidxs[n + offset] = loc_indices[n];
				allnorms[n + offset] = loc_norms[n];
			}
		}
		
		//norms
		MPI_Bcast(&allnorms[offset],nzblks[r],MPI_FLOAT,r,m_comm);
		
		for (int n = 0; n != nzblks[r]; ++n) {
			
			// indices
			MPI_Bcast(&allidxs[n + offset].data()[0],2,MPI_INT,r,m_comm);
			
			 // block sizes
			 int* bsizes = new int[2];
			
			if (r == rank) {
				bsizes[0] = loc_blocks[n].sizes()[0];
				bsizes[1] = loc_blocks[n].sizes()[1];
			} 
			
			MPI_Bcast(bsizes,2,MPI_INT,r,m_comm);
			
			dbcsr::block<2> blk = dbcsr::block<2>(vec<int>{bsizes[0],bsizes[1]});
			
			if (r == rank) {
				blk = std::move(loc_blocks[n]);
			}
			
			MPI_Bcast(blk.data(),bsizes[0]*bsizes[1],MPI_DOUBLE,r,m_comm);
			
			allblocks[offset + n] = std::move(blk);
			
			delete[] bsizes;
			
		}
	
		offset += nzblks[r];
		
	}

	
    if (rank == 0) {
    for (int i = 0; i != blkstot; ++i) {
		std::cout << allnorms[i] << " ";
	} std::cout << std::endl;
	for (int i = 0; i != blkstot; ++i) {
		std::cout << allidxs[i].data()[0] << " " << allidxs[i].data()[1] << " ";
	} std::cout << std::endl;
	for (int i = 0; i != blkstot; ++i) {
		allblocks[i].print();
	} std::cout << std::endl;
	}
	
	// ZIP IT UP SCOTTY!!!
	for (int i = 0; i != blkstot; ++i) {
		std::pair<float,dbcsr::block<2>> pair = std::make_pair(allnorms[i],std::move(allblocks[i]));
		std::pair<dbcsr::idx2,std::pair<float,dbcsr::block<2>>> pairtot =
			std::make_pair(allidxs[i],std::move(pair));
		m_blkmap.insert(std::move(pairtot));
	}
	
	delete[] allnorms;
	delete[] allidxs;
	delete[] allblocks;
	delete[] nzblks;
	delete[] loc_indices;
	delete[] loc_norms;
	delete[] loc_blocks;
}
			
void Zmat::compute() {
	
	if (m_method == "schwarz") {
		compute_schwarz();
	} else {
		throw std::runtime_error("ints/screening: unknown screening option.");
	}
	
}

} // end namespace
