#ifndef DISK_TENSOR_H
#define DISK_TENSOR_H

#include <dbcsr_tensor_ops.hpp>
#include "utils/mpi_time.h"

#include <map>
#include <mpi.h>
#include <cstdlib>

namespace dbcsr {

template <int N, typename T>
class disktensor {
private:

	MPI_Comm m_comm;
	util::mpi_log LOG;
	int m_mpirank;
	int m_mpisize;
	stensor<N,T> m_stensor;
	
	std::string m_filename;
	
	// static tensor char.
	
	arrvec<int,N> m_blksizes;
	arrvec<int,N> m_blkoffsets;
	std::array<int,N> m_blkstot;
	std::array<int,N> m_fulltot;
	
	// dynamic tensor char.
	
	int m_nblkloc; // number of local blocks
	int m_nzeloc; // number of local non-zero elements 
	int64_t m_nblktot; // total number of blocks
	int64_t m_nzetot; // total number of non-zero elements
	arrvec<int,N> m_locblkidx;  // local block-indices
	vec<MPI_Aint> m_locblkoff; // local block-offsets in file
	
	// batching info
	
	int64_t m_maxblk_per_node;
	int64_t m_maxnblk_tot;
	int m_nbatches;
	
	vec<int> m_current_batchdim;
	vec<int> m_stored_batchdim;
	
	vec<vec<std::array<int,2>>> m_boundsblk; 
	
	// write
	vec<vec<int>> m_nblksprocbatch; // number of blocks per process per batch
	vec<vec<int>> m_nzeprocbatch; // number of nonzero elements per process per batch
	
public:

	/* batchsize in MB */
	disktensor(stensor<N,T>& stensor_in, double batchsize) : 
		m_stensor(stensor_in),
		m_comm(stensor_in->comm()),
		m_mpirank(-1), m_mpisize(-1),
		LOG(stensor_in->comm(),0),
		m_nblktot(0), m_nzetot(0)
	{
		
		MPI_Comm_rank(m_comm, &m_mpirank);
		MPI_Comm_size(m_comm, &m_mpisize);
		
		m_blksizes = m_stensor->blk_sizes();
		m_blkoffsets = m_stensor->blk_offsets();
		m_fulltot = m_stensor->nfull_total();
		m_blkstot = m_stensor->nblks_total();
		
		
		LOG.os<>("Setting up tensor information.\n");
		
		LOG.os<>("-- Number of blocks per dimension:\n");
		for (auto x : m_blkstot) {
			LOG.os<>(x, " ");
		} LOG.os<>('\n');
		
		LOG.os<>("-- Number of elements per dimension:\n");
		for (auto x : m_fulltot) {
			LOG.os<>(x, " ");
		} LOG.os<>('\n');
		
		m_filename = stensor_in->name();
		
		// determine how many blocks per batch are allowed
		// this makes the following assumptions:
		// -- Blocks are equally spread out over nodes
		// -- Blocks have the same dimensions (average is taken)
		
		// get average block size:
		std::array<double,N> meansizes;
		double mean = 1.0;
		m_nblktot = 1.0;
		
		for (int i = 0; i != N; ++i) {
			meansizes[i] = (double)m_fulltot[i]/(double)m_blkstot[i];
			mean *= meansizes[i];
			m_nblktot *= m_blkstot[i];
		}
		
		LOG.os<>("-- Total number of blocks: ", m_nblktot, '\n');
		LOG.os<>("-- MEAN SIZE: ", mean, '\n');
		LOG.os<>("-- BLOCK SIZES: ");
		for (int i = 0; i != N; ++i) {
			LOG.os<>(meansizes[i], " ");
		} LOG.os<>('\n');
		
		// determine blocks held per rank per batch
		m_maxblk_per_node = batchsize * 1000 / (mean * 8);
		
		LOG.os<>("-- Holding at most ", m_maxblk_per_node, " blocks per node.\n");
		
		m_maxnblk_tot = m_mpisize * m_maxblk_per_node;
		
		LOG.os<>("-- Holding ", m_maxnblk_tot, " at most on all nodes.\n");
		
		//m_nbatches = m_nblktot / m_maxnblk_tot + (m_nblktot % m_maxnblk_tot != 0);
		
		//LOG.os<>("Tensor divided up into ", m_nbatches, " batches.\n");
		
	}
	
	void set_batch_dim(std::vector<int> ndim) {
		
		LOG.os<>("Setting up batching information.\n");
		
		for (auto n : ndim) {
			if (n >= N || n < 0) throw std::runtime_error("Invalid batch dimension.");
		}
		
		m_current_batchdim = ndim;
		
		// if we have blk indices available, reorder them
		//if (m_locblkidx.size() != 0) reorder_blk_idx(ndim);
				
		// set up bounds - again, we presume blocks are distributed homogeneously
		
		// number of blocks per block-slice
		int nblk_per_ndim = 1;
		
		for (int i = 0; i != N; ++i) {
			if (std::find(ndim.begin(),ndim.end(),i)
				!= ndim.end()) continue;
			nblk_per_ndim *= m_blkstot[i];
		}
		
		LOG.os<>("-- Blocks per ndim slice: ", nblk_per_ndim, '\n');
		
		double nblk_ndim_per_batch = m_maxnblk_tot / nblk_per_ndim;
		int nblk_ndim_per_batch_last = m_maxnblk_tot % nblk_per_ndim;
		
		if (nblk_ndim_per_batch < 1.0) throw std::runtime_error("Insufficient memory."); 
		
		m_nbatches = std::ceil((double)m_nblktot / (nblk_ndim_per_batch * (double)nblk_per_ndim));
		
		LOG.os<>("-- Dividing dimension(s)");
		for (auto n : ndim) {
			LOG.os<>(" ", n);
		}
		LOG.os<>(" into ", m_nbatches, " batches, with ", 
			nblk_ndim_per_batch, " blocks along that/those dimension(s).\n");
			
		int64_t idxblkoff = 0;
		
		// MAX blk super index
		int64_t max_idx_super = 1;
		
		for (int i = 0; i != ndim.size(); ++i) {
			max_idx_super *= m_blkstot[ndim[i]];
		}
		
		max_idx_super -= 1;
		
		LOG.os<>("MAX super idx: ", max_idx_super, '\n');
		
		m_boundsblk.resize(m_nbatches);
		
		for (int ibatch = 0; ibatch != m_nbatches; ++ibatch) {
			
			m_boundsblk[ibatch].resize(ndim.size());
			
			int64_t low_super = idxblkoff;
			idxblkoff += nblk_ndim_per_batch;
			int64_t high_super = (idxblkoff <= max_idx_super) ? 
				idxblkoff - 1 : max_idx_super;
				
			LOG.os<>("Batch ", ibatch, '\n');
			LOG.os<>("Low/high superidx ", low_super, " ", high_super, '\n');
			
			// unroll super index
			std::vector<int> sizes(ndim.size());
			for (int i = 0; i != ndim.size(); ++i) {
				sizes[i] = m_blkstot[ndim[i]];
			}
			
			auto low = unroll_index(low_super,sizes);
			auto high = unroll_index(high_super,sizes);
			
			LOG.os<>("Unrolled: \n");
			for (auto l : low) LOG.os<>(l, " ");
			LOG.os<>('\n');
			for (auto h : high) LOG.os<>(h, " ");
			LOG.os<>('\n');
			
			for (int i = 0; i != ndim.size(); ++i) {
				m_boundsblk[ibatch][i][0] = low[i];
				m_boundsblk[ibatch][i][1] = high[i];
			}	
			
		}
		
		LOG.os<>("Bounds: \n");
		for (auto b : m_boundsblk) {
			for (int i = 0; i != b.size(); ++i) {
				LOG.os<>(b[i][0], " -> ", b[i][1], " ");
			}
		} LOG.os<>('\n');
		
	}
	
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
	
	void create_file() {
		
		MPI_File fh;
		
		LOG.os<>("Creating files for ", m_filename, '\n');
		
		std::string data_fname = m_filename + ".dat";
		//std::string idx_fname = m_filename + ".info";
		
		int rc = MPI_File_open(m_comm,data_fname.c_str(),MPI_MODE_CREATE|MPI_MODE_WRONLY,
			MPI_INFO_NULL,&fh);
			
		if (rc) std::cout << "UNABLE TO OPEN FILE." << std::endl;
		
		MPI_File_close(&fh);
		
		//rc = MPI_File_open(m_comm,idx_fname.c_str(),MPI_MODE_CREATE|MPI_MODE_WRONLY,
		//	MPI_INFO_NULL,&fh);
			
		//if (rc) std::cout << "UNABLE TO OPEN FILE." << std::endl;
		
		//MPI_File_close(&fh);
		m_nblktot = 0;
		m_nzetot = 0;
		m_nblkloc = 0;
		m_nzeloc = 0;
		
	}
		
	void delete_file() {
		
		std::string data_fname = m_filename + ".dat";
		//std::string idx_fname = m_filename + ".info";
		
		int rc = MPI_File_delete(data_fname.c_str(),MPI_INFO_NULL);
		//int rc = MPI_File_delete(idx_fname.c_str(),MPI_INFO_NULL);
		
		// reset variables
		m_nblktot = 0;
		m_nzetot = 0;
		
	}
		
	
	void write(int ibatch) {
		
		m_stored_batchdim = m_current_batchdim;
				
		// writes the local blocks of batch ibatch to file
		// should only be called in order
		// blocks of a batch are stored as follows:
		
		// (batch1)                    (batch2)                .... 
		// (blks node1)(blks node2)....(blks node1)(blks node2)....
		
		// also stores indices and block offsets in .info file
		
		// allocate data
		
		LOG.os<>("Writing data of tensor ", m_filename, " to files.\n");
		LOG.os<>("Batch ", ibatch, '\n');
		
		int nze = m_stensor->num_nze();
		int nblocks = m_stensor->num_blocks();
		T* data = new T[nze]();
		
		m_nblksprocbatch.resize(m_nbatches);
		m_nzeprocbatch.resize(m_nbatches);
		
		m_nblksprocbatch[ibatch] = vec<int>(m_mpisize); 
		m_nzeprocbatch[ibatch] = vec<int>(m_mpisize);
				
		// communicate nzes + blks to other processes
		
		LOG.os<>("Gathering nze and nblocks...\n");
		
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
		
		LOG(-1).os<>("Local number of blocks: ", m_nblkloc, '\n');
		LOG.os<>("Total number of blocks: ", nblktotbatch, " out of ", m_maxnblk_tot, '\n');
		LOG.os<>("Total number of nze: ", nzetotbatch, '\n');
		
		// read blocks

		LOG.os<>("Writing blocks...\n");

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
			
			bool found = false;
			m_stensor->get_block(&data[offset], idx, size, found);
			
			blkoffbatch[iblk++] = offset;
			
			offset += ntot;
			
		}
		
		iter.stop();
		
		// filenames
		
		std::string data_fname = m_filename + ".dat";
		//std::string idx_fname = m_filename + ".info";
		
		LOG.os<>("Computing offsets...\n");
	
		// offsets
		//MPI_Offset idx_batch_offset = 0;
		MPI_Offset data_batch_offset = 0;
		
		//int64_t nblkprev = 0;
		int64_t ndataprev = 0;
		
		// global batch offset
		for (int i = 0; i != ibatch; ++i) {
			//nblkprev += std::accumulate(m_nblksprocbatch[i].begin(),
			//	m_nblksprocbatch[i].end(),int64_t(0));
				
			ndataprev += std::accumulate(m_nzeprocbatch[i].begin(),
				m_nzeprocbatch[i].end(),int64_t(0));
			
		}
		
		//LOG.os<>("Blocks in previous batch: ", nblkprev, '\n');
		
		// local processor dependent offset
		for (int i = 0; i < m_mpirank; ++i) {
			//nblkprev += m_nblksprocbatch[ibatch][i];
			ndataprev += m_nzeprocbatch[ibatch][i];
		}
		
		//idx_batch_offset = nblkprev * (N * sizeof(int) + sizeof(MPI_Offset));	
		data_batch_offset = ndataprev;
		
		// add it to blkoffsets
		for (auto& off : blkoffbatch) {
			off = off + data_batch_offset;
			
		}
		
		// concatenate indices and offsets
		LOG.os<>("Adding offsets to vector...\n");
		
		for (int i = 0; i != N; ++i) {
			m_locblkidx[i].insert(m_locblkidx[i].end(),
				std::make_move_iterator(blkidxbatch[i].begin()),
				std::make_move_iterator(blkidxbatch[i].end()));
			
		}
		
		m_locblkoff.insert(m_locblkoff.end(),
			std::make_move_iterator(blkoffbatch.begin()),
			std::make_move_iterator(blkoffbatch.end()));
		
		/*
		LOG.os<>("Writing to .info file...\n");
		LOG.os<>("Offset: ", idx_batch_offset, '\n', 
			"Size: ", nblocks, '\n');
		
		// write to info file
		MPI_File fh_idx;
		
		int rc = MPI_File_open(m_comm,idx_fname.c_str(),MPI_MODE_WRONLY,
			MPI_INFO_NULL,&fh_idx);
		
		 first write inidices
		
		MPI_File_write_at(fh_idx,idx_batch_offset,idxdata,
			N*nblocks,MPI_INT,MPI_STATUS_IGNORE);
			
		idx_batch_offset += N*nblocks*sizeof(int);		
		
		// then offsets
			
		MPI_File_write_at(fh_idx,idx_batch_offset,blk_offsets.data(),
		nblocks,MPI_OFFSET,MPI_STATUS_IGNORE);
			
		MPI_File_close(&fh_idx);
		
		*/
		
		// write data to file 
		
		LOG.os<>("Writing tensor data...\n");
		
		//std::cout << "OFFSET: " << data_batch_offset << std::endl;
		//std::cout << "NZE: " << nze << std::endl;
		
		MPI_File fh_data;
		
		MPI_File_open(m_comm,data_fname.c_str(),MPI_MODE_WRONLY,
			MPI_INFO_NULL,&fh_data);
		
		MPI_File_write_at(fh_data,data_batch_offset*sizeof(T),data,
			nze,MPI_DOUBLE,MPI_STATUS_IGNORE);
			
		MPI_File_close(&fh_data);
		
		for (auto i : m_locblkidx) {
			for (auto n : i) {
				std::cout << n << " ";
			} std::cout << std::endl;
		}
		
		// deallocate
		
		delete [] data;
		
		LOG.os<>("Done with batch ", ibatch, '\n');
		
	}
	
	void read(int ibatch) {
		
		LOG.os<>("Reading batch ", ibatch, '\n');
		
		// if reading is done in same way as writing
		if (m_stored_batchdim == m_current_batchdim) {
		
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
				blkoff += m_nblksprocbatch[ibatch][m_mpirank];
			}
				
			arrvec<int,N> locblkidx;
			
			//// setting
			for (int i = 0; i != N; ++i) {
				locblkidx[i].insert(locblkidx[i].end(),
					m_locblkidx[i].begin() + blkoff,
					m_locblkidx[i].begin() + blkoff + nblktotbatch);
			}
			
			//// reserving
			m_stensor->reserve(locblkidx);
			
			for (auto& v : locblkidx) v.resize(0);
			
			// === Reading from file ===
			//// Allocating
			
			T* data = new T[nze];
			
			//// Offset
			MPI_Offset data_batch_offset = m_locblkoff[blkoff];
			//std::cout << "AINT: " << m_locblkoff[blkoff] << std::endl;
			//std::cout << "BLKOFFSET: " << blkoff << std::endl;
			//std::cout << "OFFSET: " << data_batch_offset << std::endl;
			
			//// Opening File
			std::string fname = m_filename + ".dat";
			
			MPI_File fh_data;
			
			MPI_File_open(m_comm,fname.c_str(),MPI_MODE_RDONLY,
				MPI_INFO_NULL,&fh_data);
		
			//// Reading
			MPI_File_read_at(fh_data,data_batch_offset*sizeof(T),data,
				nze,MPI_DOUBLE,MPI_STATUS_IGNORE);
			
			MPI_File_close(&fh_data);
			
			// === Putting into tensor ===
			int offset = 0;
			
			dbcsr::iterator_t<N,T> iter(*m_stensor);
		
			iter.start();
			
			while (iter.blocks_left()) {
				
				iter.next();
				
				auto& size = iter.size();
				auto& idx = iter.idx();
				
				int ntot = std::accumulate(size.begin(),size.end(),1,std::multiplies<int>());
				
				bool found = false;
				m_stensor->put_block(idx,&data[offset],size);
				
				offset += ntot;
				
			}
			
			iter.stop();

			for (int i = 0; i != nze; ++i) {
				std::cout << data[i] << "\n";
			} std::cout << std::endl;

			delete [] data;
		
		// needs special offsets
		} else {
			
			int i = 0;
			
			// === First we need to figure out the block indices
			// get current block bounds
			
			auto& ndim = m_current_batchdim;
			int ndimsize = ndim.size();
			
			vec<vec<int>> bbounds(ndimsize);
			
			for (int i = 0; i != ndimsize; ++i) {
				bbounds[i] = this->bounds_blk(ibatch,ndim[i]);
			}
			
			arrvec<int,N> res;
			vec<MPI_Aint> blkoff;
			
			int nzeloc = 0;
			
			for (int i = 0; i != N; ++i) {
				res[i].reserve(m_maxblk_per_node);
			}
			blkoff.reserve(m_maxblk_per_node);
			
			std::cout << "BLOCKS: " << m_nblkloc << " " << m_nblktot << std::endl;
			
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
							blkoff.push_back(m_locblkoff[i]);
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
							blkoff.push_back(m_locblkoff[i]);
						}
					}
					break;
				}
			}
			
			int nblk = res[0].size();
			vec<int> blksizes(nblk);
			
			LOG.os<>("Computing block sizes.\n");
			
			for (int i = 0; i != nblk; ++i) {
				int size = 1;
				for (int n = 0; n != N; ++n) {
					size *= m_blksizes[n][res[n][i]];
				}
				blksizes[i] = size;
			}
			
			LOG.os<>("Reserving.\n");
			
			m_stensor->reserve(res);
			
			for (auto& a : res) a.clear();
			
			// now adjust the file view
			int nze = m_stensor->num_nze();
			
			LOG.os<>("Creating MPI TYPE.\n");
			
			MPI_Datatype MPI_HINDEXED;
			
			MPI_Type_create_hindexed(nblk,blksizes.data(),blkoff.data(),MPI_DOUBLE,&MPI_HINDEXED);
			MPI_Type_commit(&MPI_HINDEXED);
			
			std::string fname = m_filename + ".dat";
			MPI_File fh_data;
			
			MPI_File_open(m_comm,fname.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&fh_data);
			
			LOG.os<>("Setting file view.\n");
			
			MPI_File_set_view(fh_data, 0, MPI_DOUBLE, MPI_HINDEXED, "native", MPI_INFO_NULL);
			
			T* data = new T[nze];
			
			LOG.os<>("Reading from file...\n");
			
			MPI_File_read_at(fh_data,0,data,nze,MPI_DOUBLE,MPI_STATUS_IGNORE);
			
			MPI_File_close(&fh_data);
			
			LOG.os<>("Done!!\n");
			
			for (int i = 0; i != nze; ++i) {
				std::cout << data[i] << " ";
			} std::cout << std::endl;
			
			exit(0);
			
		} // end if
	}
		
		
	int nbatches() { return m_nbatches; }
	vec<int> bounds(int ibatch, int idim) { 
		
		vec<int> bblk = this->bounds_blk(ibatch,idim);
		vec<int> out(2);
		
		out[0] = m_blkoffsets[idim][bblk[0]];
		out[1] = m_blkoffsets[idim][bblk[1]] + m_blksizes[idim][bblk[1]] - 1;

		return out;
			
	}
	
	vec<int> bounds_blk(int ibatch, int idim) { 
		
		vec<int> out(2);
		
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
	
			
		
	
}; //end class disktensor
	
} //end namespace

#endif
