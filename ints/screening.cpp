#include "ints/screening.h"
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
	auto b = m_mol.dims().b();
	vec<vec<int>> sizes = {b,b};
	dbcsr::tensor<2> Ztensor({.name = "Zmat_schwarz", .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = sizes});
	
	// reserve
	vec<vec<int>> reserved(2);
	
	for (int i = 0; i != sizes[0].size(); ++i) {
		for (int j = i; j != sizes[1].size(); ++j) {
			
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
		
		bool found = false;
		auto blk = Ztensor.get_block({.idx = IDX, .blk_size = {b[idx0],b[idx1]}, .found = found});
		
		if (!found) continue;
		
		//LOG.os<0,-1>("Rank: ", rank, ", Thread: ", omp_get_thread_num(), " IDX: ", idx0, " ", idx1, '\n');		
		
		auto& cl0 = cbas[idx0];
		auto& cl1 = cbas[idx1];
		
		int off0 = 0;
		int off1 = 0;
		
		// no perm here for now
		for (int s0 = 0; s0 != cl0.size(); ++s0) {
			auto& sh0 = cl0[s0];
			for (int s1 = 0; s1 != cl1.size(); ++s1) {
				auto& sh1 = cl1[s1];
				
				loc_eng.compute(sh0,sh1,sh0,sh1);
				auto ints_shellsets = results[0];
				
				if (ints_shellsets != nullptr) {
					int loc_idx = 0;
					int sh0size = sh0.size();
					int sh1size = sh1.size();
					for (int i0 = 0; i0 != sh0.size(); ++i0) {
						for (int i1 = 0; i1 != sh1.size(); ++i1) {
							blk(i0 + off0, i1 + off1) = sqrt(fabs(
								ints_shellsets[i0*sh1size*sh0size*sh1size + i1*sh1size*sh0size + i1*sh1size + i0]));
						}
					}
				}
				off1 += sh1.size();
			}
			off1 = 0;
			off0 += sh0.size();
		}
		
		#pragma omp critical 
		{
			Ztensor.put_block({.idx = IDX, .blk = blk});
		}
				
	}
	
	} // end parallel region

	Ztensor.filter();
	
	dbcsr::print(Ztensor);
	
	LOG.os<0,-1>("HERE\n");
	// new para: get norm and index
	dbcsr::iterator<2> iter(Ztensor);
	int nzblks = Ztensor.num_blocks();
	int off = cbas.size();
	
	int64_t* indices = new int64_t[nzblks];
	double* norms = new double[nzblks];
	
	//LOG.os<0,-1>("Entering Para\n");
	
	#pragma omp parallel for
	for (int inz = 0; inz != nzblks; ++inz) {
		
		vec<int> blksize;
		dbcsr::idx2 idx;
		
		#pragma omp critical 
		{
			iter.next();
			blksize = iter.sizes();
			idx = iter.idx();
		}
		
		bool found = false;
		auto blk = Ztensor.get_block({.idx = idx, .blk_size = blksize, .found = found});
		
		int64_t mapidx = (int64_t)idx[0] * off + (int64_t)idx[1];
		double norm = blk.norm();
		
		norms[inz] = norm; std::cout << norm << std::endl;
		indices[inz] = mapidx;
		
	}
	
	std::cout << rank << "OUT!" << std::endl;
	MPI_Barrier(m_comm);
	//LOG.os<0,-1>(rank, " DONE.\n");
	
	std::cout << rank << " OK??" << std::endl;
	// Nicely done! Now communicate that stuff
	std::cout << rank << "Getting blocks" << std::endl;
	size_t blkstot = Ztensor.num_blocks_total();
	std::cout << rank << "Ok. " << std::endl;
	
	
	LOG.os<0,0>("ALL BLOCKS: ", blkstot, '\n');
	LOG.flush();
	
	int* counts = new int[comm_size](); 
	
	// get numbers of non zero blocks
	MPI_Gather(&nzblks, 1, MPI_INT, counts, 1, MPI_INT, 0, m_comm);

	// Place to hold the gathered data
	// Allocate at root only
	
	if (rank == 0) {
		for (int i = 0; i != comm_size; ++i) {
			std::cout << counts[i] << " "; 
		} std::cout << std::endl;
	}
	
	std::cout << "HERE " << std::endl;
	int64_t *allidxs = new int64_t[blkstot];
	double *allnorms = new double[blkstot];
	
	off = 0;
	
	for (int i = 1; i != comm_size; ++i) {
		if (rank != 0) {
			MPI_Send(norms,nzblks,MPI_DOUBLE, 0, 10, m_comm);
			MPI_Send(indices,nzblks,MPI_INT64_T, 0, 11, m_comm);
		} else {
			off += counts[i-1];
			MPI_Recv(&allnorms[off],counts[i],MPI_DOUBLE,i,10,m_comm, MPI_STATUS_IGNORE);
			MPI_Recv(&allidxs[off],counts[i],MPI_INT64_T,i,11,m_comm, MPI_STATUS_IGNORE);
		}
	}
	
	if (rank == 0) {
		std::copy(norms, norms + nzblks, allnorms);
		std::copy(indices, indices + nzblks, allidxs);
	}
			
	
    if (rank == 0) {
    for (int i = 0; i != blkstot; ++i) {
		std::cout << allnorms[i] << " ";
	} std::cout << std::endl;
	for (int i = 0; i != blkstot; ++i) {
		std::cout << allidxs[i] << " ";
	} std::cout << std::endl;
	}
	
	MPI_Bcast(allnorms, blkstot, MPI_DOUBLE, 0, m_comm);
	MPI_Bcast(allidxs, blkstot, MPI_INT64_T, 0, m_comm);
	
	delete [] counts;
	delete [] allidxs;
	delete [] allnorms;
	
	MPI_Barrier(m_comm);
	
	for (size_t i = 0; i != blkstot; ++i) {
		m_blkmap.insert(std::make_pair(allidxs[i], allnorms[i]));
	}
	
	// have another map with individual shell info
		
	
}
			
void Zmat::compute() {
	
	if (m_method == "schwarz") {
		compute_schwarz();
	} else {
		throw std::runtime_error("ints/screening: unknown screening option.");
	}
	
}

} // end namespace
