#ifndef BATCHTENSOR_H
#define BATCHTENSOR_H

#include <dbcsr_tensor_ops.hpp>
#include "utils/mpi_time.h"

#include <map>
#include <mpi.h>
#include <cstdlib>
#include <memory>

namespace tensor {

struct global {
	inline static double default_batchsize = -1.0;
};

template <int N, typename T>
class batchtensor {
private:

	MPI_Comm m_comm;
	util::mpi_log LOG;
	int m_mpirank;
	int m_mpisize;
	
	dbcsr::stensor<N,T> m_stensor;
	// underlying shared tensor
	
	std::string m_filename;
	// root name for files
	
	bool m_onebatch;
	/* if everything can be held in core memory
	 * if true, we do not write or read files. */
	bool m_setup;
	/* wether batching has been set up */
	
	/* ========== static tensor variables ========== */
	
	arrvec<int,N> m_blksizes;
	arrvec<int,N> m_blkoffsets;
	std::array<int,N> m_blkstot;
	std::array<int,N> m_fulltot;
	
	/* ========== dynamic tensor variables ========== */
	
	int m_nblkloc = 0; 
	// number of local blocks
	int m_nblkloc_global = 0;
	// number og local blocks in non-sparse tensor
	int m_nzeloc = 0; 
	// number of local non-zero elements 
	int64_t m_nblktot_global = 0;
	// total number of blocks in non-sparse tensor
	int64_t m_nblktot = 0; 
	// total number of blocks
	int64_t m_nzetot = 0; 
	// total number of non-zero elements
	arrvec<int,N> m_locblkidx;  
	// local block-indices
	vec<MPI_Aint> m_locblkoff; 
	// local block-offsets in file (in multiples of T)
	
	/* ======== BATCHING INFO ========== */
	
	double m_batch_size = 0.0;
	int64_t m_maxblk_per_node = 0;
	// maximum number of blocks allowed per process (per batch)
	int64_t m_maxnblk_tot = 0;
	// maximum number of blocks allowed on all processes
	int m_nbatches = 0;
	// number of batches the tensor is subdivided into
	
	vec<int> m_current_batchdim; 
	// what dimensions are currently being used for batching
	
	vec<int> m_stored_batchdim; 
	// what dimensions have been used for writing the tensor
	
	vec<vec<std::array<int,2>>> m_boundsblk; 
	/* lower(0) and upper(1) bound for blocks per tensor dimension per batch
	 * upper bound is INCLUDED. */
	
	
	vec<vec<int>> m_nblksprocbatch; 
	// number of blocks per process per batch
	vec<vec<int>> m_nzeprocbatch; 
	// number of nonzero elements per process per batch
	
	const vec<int> m_map_nd;
	
public:

	/* batchsize: given in MB */
	batchtensor(dbcsr::stensor<N,T>& stensor_in, double batchsize, int print = -1) : 
		m_stensor(stensor_in),
		m_batch_size(batchsize),
		m_comm(stensor_in->comm()),
		m_blksizes(m_stensor->blk_sizes()),
		m_blkoffsets(m_stensor->blk_offsets()),
		m_fulltot(m_stensor->nfull_total()),
		m_blkstot(m_stensor->nblks_total()),
		m_filename(stensor_in->name()),
		m_mpirank(-1), m_mpisize(-1),
		LOG(stensor_in->comm(),print),
		m_nblktot(0), m_nzetot(0), m_onebatch(false),
		m_map_nd(m_stensor->map_nd())
	{
		
		MPI_Comm_rank(m_comm, &m_mpirank);
		MPI_Comm_size(m_comm, &m_mpisize);
		
		LOG.os<1>("Setting up tensor information.\n");
		
		LOG.os<1>("-- Number of blocks per dimension:\n");
		for (auto x : m_blkstot) {
			LOG.os<1>(x, " ");
		} LOG.os<1>('\n');
		
		LOG.os<1>("-- Number of elements per dimension:\n");
		for (auto x : m_fulltot) {
			LOG.os<1>(x, " ");
		} LOG.os<1>('\n');
		
	}
	
	/* sets up the preliminary batching information. */
	void setup_batch() {
		
		LOG.os<1>("Setting up preliminary batching info for ", m_stensor->name(), '\n');
		m_setup = true;
		
		if (m_batch_size < 0) {
			LOG.os<1>("-- Tensor is held in core memory. No I/O.");
			m_onebatch = true;
			return;
		}
		
		// determine how many blocks per batch are allowed
		// this makes the following assumptions:
		// -- Blocks are equally spread out over nodes
		// -- Blocks have the same dimensions (average is taken)
		
		// get average block size:
		std::array<double,N> meansizes;
		double mean = 1.0;
		m_nblktot_global = 1.0;
		
		for (int i = 0; i != N; ++i) {
			meansizes[i] = (double)m_fulltot[i]/(double)m_blkstot[i];
			mean *= meansizes[i];
			m_nblktot_global *= m_blkstot[i];
		}
		
		LOG.os<1>("-- Total number of blocks: ", m_nblktot_global, '\n');
		LOG.os<1>("-- Mean total block size: ", mean, '\n');
		LOG.os<1>("-- Mean block sizes: ");
		for (int i = 0; i != N; ++i) {
			LOG.os<1>(meansizes[i], " ");
		} LOG.os<1>('\n');
		
		// determine blocks held per rank per batch
		m_maxblk_per_node = m_batch_size * 1000000 / (mean * 8);
		
		LOG.os<1>("-- Holding at most ", m_maxblk_per_node, " blocks per node.\n");
		
		m_maxnblk_tot = m_mpisize * m_maxblk_per_node;
		
		LOG.os<1>("-- Holding ", m_maxnblk_tot, " at most on all nodes.\n");
		
		if (m_maxnblk_tot > m_nblktot_global) {
			LOG.os<1>("-- Tensor is held in core memory. No I/O.\n");
			m_onebatch = true;
		}
		
		LOG.os<1>("Finished setting up preliminary batching info.\n");
		
	}
	
	/* function to set up the batches along the dimension(s) specified in ndim 
	 * Only supports two dimensions at the moment */
	
	void set_batch_dim(std::vector<int> ndim) {
		
		if (!m_setup) throw std::runtime_error("Batching has not yet been setup!");
		
		if (m_onebatch) {
			m_nbatches = 1;
			return;
		}
		
		LOG.os<1>("Setting up batching dimensions for ", m_stensor->name(), "\n");
		
		for (auto n : ndim) {
			if (n >= N || n < 0) throw std::runtime_error("Invalid batch dimension.");
		}
		
		if (ndim.size() > 1) throw std::runtime_error("Batching for more than one dimension NYI.");
		
		m_current_batchdim = ndim;
				
		// set up bounds - again, we presume blocks are distributed homogeneously
		
		// number of blocks per block-slice
		int nblk_per_ndim = 1;
		
		for (int i = 0; i != N; ++i) {
			if (std::find(ndim.begin(),ndim.end(),i)
				!= ndim.end()) continue;
			nblk_per_ndim *= m_blkstot[i];
		}
		
		LOG.os<1>("-- Available RAM for tensor per node: ", m_batch_size);
		
		LOG.os<1>("-- Blocks per ndim slice: ", nblk_per_ndim, '\n');
		
		double nblk_ndim_per_batch = m_maxnblk_tot / nblk_per_ndim;
		int nblk_ndim_per_batch_last = m_maxnblk_tot % nblk_per_ndim;
		
		if (nblk_ndim_per_batch < 1.0) throw std::runtime_error("Insufficient memory."); 
		
		m_nbatches = std::ceil((double)m_nblktot_global / (nblk_ndim_per_batch * (double)nblk_per_ndim));
		
		LOG.os<1>("-- Dividing dimension(s)");
		for (auto n : ndim) {
			LOG.os<1>(" ", n);
		}
		LOG.os<1>(" into ", m_nbatches, " batches, with ", 
			nblk_ndim_per_batch, " blocks along that/those dimension(s).\n");
			
		int64_t idxblkoff = 0;
		
		// MAX blk super index
		int64_t max_idx_super = 1;
		
		for (int i = 0; i != ndim.size(); ++i) {
			max_idx_super *= m_blkstot[ndim[i]];
		}
		
		max_idx_super -= 1;
		
		LOG.os<1>("-- Maximum super index: ", max_idx_super, '\n');
		
		m_boundsblk.resize(m_nbatches);
		
		for (int ibatch = 0; ibatch != m_nbatches; ++ibatch) {
			
			m_boundsblk[ibatch].resize(ndim.size());
			
			int64_t low_super = idxblkoff;
			idxblkoff += nblk_ndim_per_batch;
			int64_t high_super = (idxblkoff <= max_idx_super) ? 
				idxblkoff - 1 : max_idx_super;
				
			//LOG.os<>("Batch ", ibatch, '\n');
			//LOG.os<>("Low/high superidx ", low_super, " ", high_super, '\n');
			
			// unroll super index
			std::vector<int> sizes(ndim.size());
			for (int i = 0; i != ndim.size(); ++i) {
				sizes[i] = m_blkstot[ndim[i]];
			}
			
			auto low = unroll_index(low_super,sizes);
			auto high = unroll_index(high_super,sizes);
			
			/*LOG.os<>("Unrolled: \n");
			for (auto l : low) LOG.os<>(l, " ");
			LOG.os<>('\n');
			for (auto h : high) LOG.os<>(h, " ");
			LOG.os<>('\n');*/
			
			for (int i = 0; i != ndim.size(); ++i) {
				m_boundsblk[ibatch][i][0] = low[i];
				m_boundsblk[ibatch][i][1] = high[i];
			}	
			
		}
		
		if (LOG.global_plev() >= 1) {
			LOG.os<1>("-- Bounds: \n");
			for (auto b : m_boundsblk) {
				for (int i = 0; i != b.size(); ++i) {
					LOG.os<>(b[i][0], " -> ", b[i][1], " ");
				}
			} LOG.os<1>('\n');
		}
		
		LOG.os<1>("Finished setting up all batching info.\n");
		
	}
	
	/* transforms a (flat) superindex into multiple tensor indices */ 
	std::vector<int> unroll_index(int64_t idx, std::vector<int> sizes) {
		
		int M = sizes.size();
		std::vector<int> out(M);
		
		if (M == 1) {
			out[0] = idx;
			return out;
		}
		
		int64_t dim = 1;
		
		for (int i = 0; i != M-1; ++i) {
			dim *= sizes[i];
		}
		
		out[M-1] = (int)(idx/dim);
		int64_t newidx = idx - out[M-1]*dim;
		
		if (M == 2) {
			out[0] = (int)newidx;
			return out;
		} else {
			std::vector<int> newsizes(M-1);
			std::copy(sizes.begin(),sizes.end()-1,newsizes.begin());
			auto newarr = unroll_index(newidx,newsizes);
			for (int i = 0; i != M-1; ++i) {
				out[i] = newarr[i];
			}
			return out;
		}
		
	}
	
	/* Create all necessary files. 
	 * !!! Deletes previous files and resets all variables !!! */
	void create_file() {
		
		if (m_onebatch) return;
		
		delete_file();
		
		MPI_File fh;
		
		LOG.os<1>("Creating files for ", m_filename, '\n');
		
		std::string data_fname = m_filename + ".dat";
		//std::string idx_fname = m_filename + ".info";
		
		int rc = MPI_File_open(m_comm,data_fname.c_str(),MPI_MODE_CREATE|MPI_MODE_WRONLY,
			MPI_INFO_NULL,&fh);
		
		MPI_File_close(&fh);
		
		
		
	}
	
	/* Deletes all files asscoiated with tensor. 
	 * !!! Resets variables !!! */
	void delete_file() {
		
		if (m_onebatch) return;
		
		std::string data_fname = m_filename + ".dat";
		//std::string idx_fname = m_filename + ".info";
		
		int rc = MPI_File_delete(data_fname.c_str(),MPI_INFO_NULL);
		//int rc = MPI_File_delete(idx_fname.c_str(),MPI_INFO_NULL);
		
		//reset_var();
		
	}
	
	void reset_var() {
		
		m_nblktot = 0;
		m_nblkloc = 0; 
		m_nzeloc = 0; 
		m_nblktot = 0; 
		m_nzetot = 0; 
		m_maxblk_per_node = 0;
		m_maxnblk_tot = 0;
		m_nbatches = 0;
		
		for (auto& v : m_locblkidx) {
			v.clear();
		}
		
		m_current_batchdim.clear();
		m_stored_batchdim.clear();
		
		m_boundsblk.clear();
		m_nblksprocbatch.clear();
		m_nzeprocbatch.clear();
	
	}	
	
	~batchtensor() { delete_file(); }
			
	/* ... */
	void write(int ibatch) {
		
		if (m_onebatch) return;
		
		m_stored_batchdim = m_current_batchdim;
				
		// writes the local blocks of batch ibatch to file
		// should only be called in order
		// blocks of a batch are stored as follows:
		
		// (batch1)                    (batch2)                .... 
		// (blks node1)(blks node2)....(blks node1)(blks node2)....
		
		// allocate data
		
		LOG.os<1>("Writing data of tensor ", m_filename, " to file.\n");
		LOG.os<1>("Batch ", ibatch, '\n');
		
		int nze = m_stensor->num_nze();
		int nblocks = m_stensor->num_blocks();
		
		m_nblksprocbatch.resize(m_nbatches);
		m_nzeprocbatch.resize(m_nbatches);
		
		m_nblksprocbatch[ibatch] = vec<int>(m_mpisize); 
		m_nzeprocbatch[ibatch] = vec<int>(m_mpisize);
				
		// communicate nzes + blks to other processes
		
		LOG.os<1>("Gathering nze and nblocks...\n");
		
		MPI_Allgather(&nze,1,MPI_INT,m_nzeprocbatch[ibatch].data(),1,MPI_INT,m_comm);
		MPI_Allgather(&nblocks,1,MPI_INT,m_nblksprocbatch[ibatch].data(),1,MPI_INT,m_comm);
		
		int64_t nblktotbatch = std::accumulate(m_nblksprocbatch[ibatch].begin(),
			m_nblksprocbatch[ibatch].end(),int64_t(0));
		
		int64_t nzetotbatch = std::accumulate(m_nzeprocbatch[ibatch].begin(),
			m_nzeprocbatch[ibatch].end(),int64_t(0));
		
		m_nblkloc += m_nblksprocbatch[ibatch][m_mpirank];
		m_nzeloc += m_nzeprocbatch[ibatch][m_mpirank];
		
		m_nblktot += nblktotbatch;
			
		m_nzetot += nzetotbatch;
		
		LOG(-1).os<1>("Local number of blocks: ", m_nblkloc, '\n');
		LOG.os<1>("Total number of blocks: ", nblktotbatch, " out of ", m_maxnblk_tot, '\n');
		LOG.os<1>("Total number of nze: ", nzetotbatch, '\n');
		
		// read blocks

		LOG.os<1>("Writing blocks...\n");

		dbcsr::iterator_t<N,T> iter(*m_stensor);
		
		iter.start();
		
		std::vector<MPI_Aint> blkoffbatch(nblocks); // block offsets in file for this batch
		arrvec<int,N> blkidxbatch; // block indices for this batch
		blkidxbatch.fill(vec<int>(nblocks));
		
		MPI_Aint offset = 0;
		int n = 0;
		
		int iblk = 0;
		
		while (iter.blocks_left()) {
			
			iter.next();
			
			auto& size = iter.size();
			auto& idx = iter.idx();
			
			for (int i = 0; i != N; ++i) {
				blkidxbatch[i][iblk] = idx[i];
			}
			
			int ntot = std::accumulate(size.begin(),size.end(),1,std::multiplies<int>());
			
			blkoffbatch[iblk++] = offset;
			
			offset += ntot;
			
		}
		
		iter.stop();
		
		// filenames
		
		std::string data_fname = m_filename + ".dat";
		
		LOG.os<1>("Computing offsets...\n");
	
		// offsets
		MPI_Offset data_batch_offset = 0;
		
		//int64_t nblkprev = 0;
		int64_t ndataprev = 0;
		
		// global batch offset
		for (int i = 0; i != ibatch; ++i) {
				
			ndataprev += std::accumulate(m_nzeprocbatch[i].begin(),
				m_nzeprocbatch[i].end(),int64_t(0));
			
		}
		
		// local processor dependent offset
		for (int i = 0; i < m_mpirank; ++i) {
			ndataprev += m_nzeprocbatch[ibatch][i];
		}
		
		data_batch_offset = ndataprev;
		
		// add it to blkoffsets
		for (auto& off : blkoffbatch) {
			off += data_batch_offset;
		}
		
		// concatenate indices and offsets
		LOG.os<1>("Adding offsets to vector...\n");
		
		for (int i = 0; i != N; ++i) {
			m_locblkidx[i].insert(m_locblkidx[i].end(),
				blkidxbatch[i].begin(),blkidxbatch[i].end());
			blkidxbatch[i].clear();
		}
		
		m_locblkoff.insert(m_locblkoff.end(),
			blkoffbatch.begin(),blkoffbatch.end());
		blkoffbatch.clear();
	
		// write data to file 
		
		LOG.os<1>("Writing tensor data...\n");
		
		long long int datasize;
		T* data = m_stensor->data(datasize);
		
		MPI_File fh_data;
		
		MPI_File_open(m_comm,data_fname.c_str(),MPI_MODE_WRONLY,
			MPI_INFO_NULL,&fh_data);
		
		MPI_File_write_at(fh_data,data_batch_offset*sizeof(T),data,
			nze,MPI_DOUBLE,MPI_STATUS_IGNORE);
			
		MPI_File_close(&fh_data);
		
		if (LOG.global_plev() >= 10) {
			std::cout << "LOC BLK IDX AND OFFSET." << std::endl;
			for (int i = 0; i != m_locblkidx[0].size(); ++i) {
				for (int j = 0; j != N; ++j) {
					std::cout << m_locblkidx[j][i] << " ";
				}
				std::cout << m_locblkoff[i] << std::endl;
			}
		}
		
		LOG.os<1>("Done with batch ", ibatch, '\n');
		
	}
	
	/* ... */
	void read(int ibatch) {
		
		if (m_onebatch) return;
		
		LOG.os<1>("Reading batch ", ibatch, '\n');
		
		// if reading is done in same way as writing
		if (m_stored_batchdim == m_current_batchdim) {
			
			LOG.os<1>("Same dimensions for writing and reading.\n");
		
			int nze = m_nzeprocbatch[ibatch][m_mpirank];
			int nblk = m_nblksprocbatch[ibatch][m_mpirank];
			
			int64_t nblktotbatch = std::accumulate(m_nblksprocbatch[ibatch].begin(),
				m_nblksprocbatch[ibatch].end(),int64_t(0));
		
			int64_t nzetotbatch = std::accumulate(m_nzeprocbatch[ibatch].begin(),
				m_nzeprocbatch[ibatch].end(),int64_t(0));
			
			// === Allocating blocks for tensor ===
			//// offsets
			
			int64_t blkoff = 0;
			
			for (int i = 0; i < ibatch; ++i) {
				blkoff += m_nblksprocbatch[i][m_mpirank];
			}
				
			arrvec<int,N> locblkidx;
			
			LOG(-1).os<1>("Offset: ", blkoff, '\n');
			LOG(-1).os<1>("NBLKTOTBATCH: ", nblktotbatch, '\n');
			//// setting
			for (int i = 0; i != N; ++i) {
				locblkidx[i].insert(locblkidx[i].end(),
					m_locblkidx[i].begin() + blkoff,
					m_locblkidx[i].begin() + blkoff + nblk);
			}
			
			if (LOG.global_plev() >= 10) {
			
				MPI_Barrier(m_comm);
				
				for (int i = 0; i != m_mpisize; ++i) {
				
					if (i == m_mpirank) {
					
						for (auto a : locblkidx) {
							for (auto l : a) {
								std::cout << l << " ";
							} std::cout << std::endl;
						}
						
					}
				
				MPI_Barrier(m_comm);
				
				}
				
			}
			
			//// reserving
			m_stensor->reserve(locblkidx);
			
			for (auto& v : locblkidx) v.resize(0);
			
			// === Reading from file ===
			//// Allocating
			
			//// Offset
			MPI_Offset data_batch_offset = m_locblkoff[blkoff];
			
			//// Opening File
			std::string fname = m_filename + ".dat";
			
			MPI_File fh_data;
			
			MPI_File_open(m_comm,fname.c_str(),MPI_MODE_RDONLY,
				MPI_INFO_NULL,&fh_data);
		
			//// Reading
			long long int datasize;
			T* data = m_stensor->data(datasize);
			
			MPI_File_read_at(fh_data,data_batch_offset*sizeof(T),data,
				nze,MPI_DOUBLE,MPI_STATUS_IGNORE);
			
			MPI_File_close(&fh_data);
		
		// needs special offsets
		} else {
			
			LOG.os<1>("Different dimension.\n");
			
			// check if dimension is slower
			if (m_stored_batchdim.size() == 1) {
				
				int idx_write = m_stored_batchdim[0];
				int idx_read = m_current_batchdim[0];
				
				auto iter_begin = m_map_nd.begin();
				auto iter_write = std::find(m_map_nd.begin(),m_map_nd.end(),idx_write);
				auto iter_read = std::find(m_map_nd.begin(),m_map_nd.end(),idx_read);
				
				int pos_write = iter_write - iter_begin;
				int pos_read = iter_read - iter_begin;
				
				if (pos_read < pos_write) {
					std::string message = "";
					message += "Batchtensor " + m_stensor->name() + ": \n";
					message += "Read index is faster than write index!\n";
					message += "This case is NYI!";
					throw std::runtime_error(message);
				}
				
			}
			
			int i = 0;
			
			// === First we need to figure out the block indices
			// get current block bounds
			
			auto& ndim = m_current_batchdim;
			int ndimsize = ndim.size();
			
			vec<vec<int>> bbounds(ndimsize);
			
			for (int i = 0; i != ndimsize; ++i) {
				bbounds[i] = this->bounds_blk(ibatch,ndim[i]);
				LOG.os<1>("Bounds: ", bbounds[i][0], " ", bbounds[i][1], '\n');
			}
			
			arrvec<int,N> res;
			vec<MPI_Aint> blkoff;
			
			LOG.os<1>("Maxblk per node: ", m_maxblk_per_node, '\n');
			
			int nzeloc = 0;
			
			for (int i = 0; i != N; ++i) {
				res[i].reserve(m_maxblk_per_node);
			}
			blkoff.reserve(m_maxblk_per_node);
			
			switch (ndimsize) {
				case 1 :
				{
					//std::cout << "CASE1" << std::endl;
					auto& idx = m_locblkidx[ndim[0]];
					int l = bbounds[0][0];
					int h = bbounds[0][1];
					for (size_t i = 0; i != m_nblkloc; ++i) {
						if (l <= idx[i] && idx[i] <= h) {
							for (int j = 0; j != N; ++j) {
								res[j].push_back(m_locblkidx[j][i]);
							}
							blkoff.push_back(m_locblkoff[i]*sizeof(T));
						}
					}
					break;
				}
				case 2 :
				{
					//std::cout << "CASE2" << std::endl;
					auto& idx0 = m_locblkidx[ndim[0]];
					auto& idx1 = m_locblkidx[ndim[1]];
					
					int l0 = bbounds[0][0];
					int h0 = bbounds[0][1];
					int l1 = bbounds[1][0];
					int h1 = bbounds[1][1];
					
					//std::cout << l0 << " " << h0 << "/" << l1 << " " << h1 << std::endl;
					
					for (size_t i = 0; i != m_nblkloc; ++i) {
						if (l0 <= idx0[i] && idx0[i] <= h0
							&& l1 <= idx1[i] && idx1[i] <= h1) {
							
							for (int j = 0; j != N; ++j) {
								res[j].push_back(m_locblkidx[j][i]);
							}
							blkoff.push_back(m_locblkoff[i]*sizeof(T));
						}
					}
					break;
				}
			}
			
			if (LOG.global_plev() >= 10) {
			
				std::cout << "READ LOC BLK IDX AND OFFSET." << std::endl;
				for (int i = 0; i != res[0].size(); ++i) {
					for (int j = 0; j != N; ++j) {
						std::cout << res[j][i] << " ";
					}
					std::cout << blkoff[i] << std::endl;
				}
				
			}
			
			int nblk = res[0].size();
			vec<int> blksizes(nblk);
			
			LOG.os<1>("Computing block sizes.\n");
			
			for (int i = 0; i != nblk; ++i) {
				int size = 1;
				for (int n = 0; n != N; ++n) {
					size *= m_blksizes[n][res[n][i]];
				}
				blksizes[i] = size;
			}
			
			LOG.os<1>("Reserving.\n");
			
			m_stensor->reserve(res);
			
			for (auto& a : res) a.clear();
			
			// now adjust the file view
			int nze = m_stensor->num_nze();
			
			LOG.os<1>("Creating MPI TYPE.\n");
			
			if (LOG.global_plev() >= 10) {
			
				std::cout << "BLOCK SIZES + OFF: " << std::endl;
				for (int i = 0; i != nblk; ++i) {
					std::cout << blksizes[i] << " " << blkoff[i] << std::endl;
				}
				
			}
			
			MPI_Datatype MPI_HINDEXED;
			
			MPI_Type_create_hindexed(nblk,blksizes.data(),blkoff.data(),MPI_DOUBLE,&MPI_HINDEXED);
			MPI_Type_commit(&MPI_HINDEXED);
			
			std::string fname = m_filename + ".dat";
			MPI_File fh_data;
			
			MPI_File_open(m_comm,fname.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&fh_data);
			
			LOG.os<1>("Setting file view.\n");
			
			MPI_File_set_view(fh_data, 0, MPI_DOUBLE, MPI_HINDEXED, "native", MPI_INFO_NULL);
			
			long long int datasize;
			T* data = m_stensor->data(datasize);
			
			LOG.os<1>("Reading from file...\n");
			
			MPI_File_read_at(fh_data,0,data,nze,MPI_DOUBLE,MPI_STATUS_IGNORE);
			
			MPI_File_close(&fh_data);
			
			LOG.os<1>("Done!!\n");
			
		} // end if
	}
		
	dbcsr::stensor<N,T> get_stensor() { return m_stensor; }
		
	int nbatches() { return m_nbatches; }
	
	/* returns bounds for batch ibatch and tensor dimension idim
	 * for use in dbcsr_copy, dbcsr_contract etc. ... */
	vec<int> bounds(int ibatch, int idim) { 
		
		vec<int> bblk = this->bounds_blk(ibatch,idim);
		vec<int> out(2);
		
		out[0] = m_blkoffsets[idim][bblk[0]];
		out[1] = m_blkoffsets[idim][bblk[1]] + m_blksizes[idim][bblk[1]] - 1;

		return out;
			
	}
	
	/* returns the BLOCK bounds */
	vec<int> bounds_blk(int ibatch, int idim) { 
		
		vec<int> out(2);
		
		if (m_onebatch) {
			out[0] = 0;
			out[1] = m_blkstot[idim] - 1;
			return out;
		}
		
		auto iter = std::find(m_current_batchdim.begin(),
			m_current_batchdim.end(),idim);
			
		if (iter != m_current_batchdim.end()) {
			int pos = iter - m_current_batchdim.begin();
			out[0] = m_boundsblk[ibatch][pos][0];
			out[1] = m_boundsblk[ibatch][pos][1];
		} else {
			out[0] = 0;
			out[1] = m_blkstot[idim] - 1;
		}
		
		return out;
			
	}
	
	void clear_batch() {
		
		if (m_onebatch) return;
		m_stensor->clear();
		
	}
	
}; //end class batchtensor

template <int N, typename T>
using sbatchtensor = std::shared_ptr<batchtensor<N,T>>;
	
} //end namespace

#endif
