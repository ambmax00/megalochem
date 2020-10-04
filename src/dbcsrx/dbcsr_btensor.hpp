#ifndef DBCSR_BTENSOR_HPP
#define DBCSR_BTENSOR_HPP

#include <dbcsr_tensor_ops.hpp>
#include "utils/mpi_time.h"

#include <map>
#include <mpi.h>
#include <cstdlib>
#include <memory>
#include <filesystem>
#include <functional>

namespace dbcsr {
	
enum class btype {
	invalid,
	core,
	disk,
	direct
};

inline btype get_btype(std::string str) {
	if (str == "core") return btype::core;
	if (str == "disk") return btype::disk;
	if (str == "direct") return btype::direct;
	return btype::invalid;
};

inline vec<vec<int>> make_blk_bounds(std::vector<int> blksizes, int nbatches) {
		
	int nblks = blksizes.size();
	int nele = std::accumulate(blksizes.begin(),blksizes.end(),0);
	
	if (nblks < nbatches) nbatches = nblks;
			
	double nele_per_batch = (double) nele / (double) nbatches;
	
	vec<vec<int>> out;
	int current_sum = 0;
	int ibatch = 1;
	int first_blk = 0;
	int last_blk = 0;
	
	for (int i = 0; i != nblks; ++i) {
		current_sum += blksizes[i];
		
		if (current_sum >= ibatch * nele_per_batch) {
			last_blk = i;
			vec<int> b = {first_blk,last_blk};
			out.push_back(b);
			first_blk = i+1;
			++ibatch;
		}
	}
	
	return out;
		
}
	
template <int N, typename T, 
	typename = typename std::enable_if<(N >= 2 && N <= 4)>::type>
class btensor {
protected:

	MPI_Comm m_comm;
	util::mpi_log LOG;
	int m_mpirank;
	int m_mpisize;
	
	dbcsr::stensor<N,T> m_read_tensor;
	// tensor which keeps original dist for reading
	
	dbcsr::stensor<N,T> m_work_tensor;
	// underlying shared tensor for contraction/copying
	
	std::string m_name;
	dbcsr::shared_pgrid<N> m_spgrid_N;
	arrvec<int,N> m_blk_sizes;
	
	std::string m_filename;
	// root name for files
	
	btype m_type;
	
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
	
	/* ======== BATCHING INFO ========== */

	int m_write_current_batch;
	int m_read_current_batch;
	vec<int> m_read_current_dims;
	bool m_read_current_is_contiguous;

	arrvec<vec<int>,N> m_blk_bounds;
	arrvec<vec<int>,N> m_bounds;
	arrvec<int,N> m_full_blk_bounds;
	arrvec<int,N> m_full_bounds;
	
	std::array<int,N> m_nbatches_dim;
	
	struct view {
		vec<int> dims;
		bool is_contiguous;
		int nbatches;
		vec<int> map1, map2;
		arrvec<int,N> locblkidx;
		vec<MPI_Aint> locblkoff;
		vec<vec<int>> nblksprocbatch;
		vec<vec<int>> nzeprocbatch;
	};
	
	view m_wrview;
	std::map<vec<int>,view> m_rdviewmap;
	
	/* =========== FUNCTIONS ========== */
	
	using generator_type = 
	std::function<void(dbcsr::stensor<N,T>&,vec<vec<int>>&)>;
	
	generator_type m_generator;		
	
public:

	/* batchsize: given in MB */
	btensor(std::string name, shared_pgrid<N> spgrid, arrvec<int,N> blksizes, 
		std::array<int,N> nbatches, btype mytype, int print = -1) : 
		m_comm(spgrid->comm()),
		m_name(name),
		m_spgrid_N(spgrid),
		m_blk_sizes(blksizes),
		m_filename(),
		m_mpirank(-1), m_mpisize(-1),
		LOG(spgrid->comm(),print),
		m_type(mytype)
	{
		
		if (m_type == btype::invalid) {
			throw std::runtime_error("Invalid mode for batchtensor.\n");
		}
		
		MPI_Comm_rank(m_comm, &m_mpirank);
		MPI_Comm_size(m_comm, &m_mpisize);
		
		LOG.os<1>("Setting up batch tensor information.\n");
		
		std::string path = std::filesystem::current_path();
		path += "/batching/";
		
		m_filename = path + name + ".dat";
		
		// divide dimensions
		
		arrvec<int,N> blkoffsets;
		
		for (int i = 0; i != N; ++i) {
			
			// offsets
			int off = 0;
			for (auto blksize : m_blk_sizes[i]) {
				blkoffsets[i].push_back(off);
				off += blksize;
			}
			
			// batch bounds
			m_blk_bounds[i] = make_blk_bounds(blksizes[i],nbatches[i]);
	
			m_bounds[i] = m_blk_bounds[i];
			auto& ibds = m_bounds[i];
			auto& iblkoffs = blkoffsets[i];
			auto& iblksizes = m_blk_sizes[i];
			
			for (int x = 0; x != ibds.size(); ++x) {
				ibds[x][0] = iblkoffs[ibds[x][0]];
				ibds[x][1] = iblkoffs[ibds[x][1]] + iblksizes[ibds[x][1]] - 1;
			}
			
			// full bounds
			
			int nfull_blk = m_blk_sizes[i].size();
			m_full_blk_bounds[i] = vec<int>{0, nfull_blk-1};
			
			int nfull = std::accumulate(m_blk_sizes[i].begin(), m_blk_sizes[i].end(),0);
			m_full_bounds[i] =vec<int>{0, nfull-1};
			
			// nbatches
			
			m_nbatches_dim[i] = m_blk_bounds[i].size();
			
			// check if memory sufficient
			
			int nele = std::accumulate(blksizes[i].begin(),blksizes[i].end(),0);
			double nele_per_batch = (double) nele / (double) m_nbatches_dim[i];
			
			if (nele_per_batch * sizeof(T) >= 2 * 1e+9) {
				LOG.os<>(
					"WARNING: Batch sizes might be too large!\n",
					"Either increase number of batches, ",
					"or increase number of MPI processes.\n");
			}
			
		}
		
		LOG.os<1>("Batch bounds: \n");
		for (int i = 0; i != N; ++i) {
			LOG.os<1>("Dimension ", i, ":\n");
			for (auto b : m_blk_bounds[i]) {
				LOG.os<1>(b[0], " -> ", b[1], " ");
			} LOG.os<1>('\n');
		}
		
		if (m_type == btype::disk) {
			create_file();
		}
		
		LOG.os<1>("Finished setting up batchtensor.\n");
		
		reset_var();
		
	}
	
	btensor(const btensor& in) = default;
	
	void set_generator(generator_type& func) {
		m_generator = func;
	}
	
	/* Create all necessary files. 
	 * !!! Deletes previous files and resets all variables !!! */
	void create_file() {
		
		delete_file();
		
		MPI_File fh;
		
		LOG.os<1>("Creating files for ", m_filename, '\n');
		
		int rc = MPI_File_open(m_comm,m_filename.c_str(),MPI_MODE_CREATE|MPI_MODE_WRONLY,
			MPI_INFO_NULL,&fh);
		
		MPI_File_close(&fh);
		
	}
	
	/* Deletes all files asscoiated with tensor. 
	 * !!! Resets variables !!! */
	void delete_file() {
		
		LOG.os<1>("Deleting files for ", m_filename, '\n');
		int rc = MPI_File_delete(m_filename.c_str(),MPI_INFO_NULL);
		
	}
	
	void reset_var() {
		
		m_nblkloc_global = 0;
		m_nblktot = 0;
		m_nblkloc = 0; 
		m_nzeloc = 0; 
		m_nblktot = 0; 
		m_nzetot = 0; 
		
		for (auto& v : m_wrview.locblkidx) {
			v.clear();
		}
		
		m_wrview.locblkoff.clear();		
		m_wrview.nblksprocbatch.clear();
		m_wrview.nzeprocbatch.clear();
	
		m_rdviewmap.clear();
	
	}
	
	void reset() {
		
		if (m_work_tensor) m_work_tensor->clear();
		if (m_read_tensor) m_read_tensor->clear();
		reset_var();
		if (m_type == btype::disk) {
			delete_file();
			create_file();
		}
		
	}
	
	~btensor() { if (m_type == btype::disk) delete_file(); }
	
	int flatten(vec<int>& idx, vec<int>& dims) {
		
		vec<int> bsizes(dims.size());
		for (int i = 0; i != bsizes.size(); ++i) {
			bsizes[i] = m_nbatches_dim[dims[i]];
		}
		
		int flat_idx = 0;
		
		for (int i = 0; i != bsizes.size(); ++i) {
			int off = 1;
			for (int n = bsizes.size() - 1; n > i; --n) {
				off *= bsizes[i];
			}
			flat_idx += idx[i] * off;
		}
				
		return flat_idx;
		
	}
	
	int get_nbatches(vec<int> dims) {
		/* get total number of batches */
		int nbatches = 1;
		for (int i = 0; i != dims.size(); ++i) {
			int idx = dims[i];
			nbatches *= m_nbatches_dim[idx];
		}
		return nbatches;
	}
	
	void compress_init(std::initializer_list<int> dim_list, vec<int> map1, vec<int> map2) 
	{
		
		LOG.os<1>("Initializing compression...\n");
		
		vec<int> dims = dim_list;
		
		m_wrview.nbatches = 1;
		m_wrview.dims = dims;
		m_wrview.is_contiguous = true;
		m_wrview.map1 = map1;
		m_wrview.map2 = map2;
		
		m_wrview.nbatches = get_nbatches(dims);
		
		LOG.os<1>("Total batches for writing: ", m_wrview.nbatches, '\n');
		
		reset_var();
		
		if (m_type == btype::core) {
			LOG.os<1>("Allocating core work tensor.\n");
			m_work_tensor = tensor_create<N,T>()
				.name(m_name + "_work")
				.pgrid(m_spgrid_N)
				.map1(map1).map2(map2)
				.blk_sizes(m_blk_sizes)
				.get();
			
			m_work_tensor->batched_contract_init();	
				
		}
		
	}
	
	vec<vec<int>> get_bounds(vec<int> idx, vec<int> dims) {
		
		vec<vec<int>> b(N);
		
		for (int i = 0; i != idx.size(); ++i) {
			b[dims[i]] = this->bounds(dims[i], idx[i]);
		}
		
		LOG.os<1>("Copy bounds.\n");
		for (int i = 0; i != N; ++i) {
			auto iter = std::find(dims.begin(),dims.end(),i);
			b[i] = (iter == dims.end()) ? full_bounds(i) : b[i];
			LOG.os<1>(b[i][0], " -> ", b[i][1], '\n');
		}
		
		return b;
		
	}
	
	vec<vec<int>> get_blk_bounds(vec<int> idx, vec<int> dims) {
		
		vec<vec<int>> b(N);
		
		for (int i = 0; i != idx.size(); ++i) {
			b[dims[i]] = this->blk_bounds(dims[i], idx[i]);
		}
		
		LOG.os<1>("Copy bounds.\n");
		for (int i = 0; i != N; ++i) {
			auto iter = std::find(dims.begin(),dims.end(),i);
			b[i] = (iter == dims.end()) ? full_blk_bounds(i) : b[i];
			LOG.os<1>(b[i][0], " -> ", b[i][1], '\n');
		}
		
		return b;
		
	}
			
	
	/* ... */
	void compress(std::initializer_list<int> idx_list, stensor<N,T> tensor_in) {
	
		vec<int> idx = idx_list;
		
		switch(m_type) {
			case btype::disk: compress_disk(idx,tensor_in);
			break;
			case btype::core: compress_core(idx,tensor_in);
			break;
			case btype::direct: compress_direct(tensor_in);
			break;
		}
		
	}
	
	void compress_direct(stensor<N,T> tensor_in) {
		tensor_in->clear();
		return;
	}
	
	void compress_core(vec<int> idx, stensor<N,T> tensor_in) {
		
		LOG.os<1>("Compressing into core memory...\n");
		
		auto b = get_bounds(idx, m_wrview.dims);
    
		LOG.os<1>("Copying\n");
		
		m_nzetot += tensor_in->num_nze_total();
    
		copy(*tensor_in, *m_work_tensor)
			.bounds(b)
			.move_data(true)
			.sum(true)
			.perform();
		
		LOG.os<1>("DONE.\n");
		
	}
    
	void compress_disk(vec<int> idx, stensor<N,T> tensor_in) {
		
		auto dims = m_wrview.dims;
		int ibatch = flatten(idx,dims);
		int nbatches = m_wrview.nbatches;
		
		auto write_tensor = tensor_in;
						
		// writes the local blocks of batch ibatch to file
		// should only be called in order
		// blocks of a batch are stored as follows:
		
		// (batch1)                    (batch2)                .... 
		// (blks node1)(blks node2)....(blks node1)(blks node2)....
		
		// allocate data
		
		LOG.os<1>("Writing data of tensor ", m_filename, " to file.\n");
		LOG.os<1>("Batch ", ibatch, '\n');
		
		int nze = write_tensor->num_nze();
		int nblocks = write_tensor->num_blocks();
		
		LOG.os<1>("NZE/NBLOCKS: ", nze, "/", nblocks, '\n');
		
		auto& nblksprocbatch = m_wrview.nblksprocbatch;
		auto& nzeprocbatch = m_wrview.nzeprocbatch;
		
		nblksprocbatch.resize(nbatches);
		nzeprocbatch.resize(nbatches);
		
		nblksprocbatch[ibatch] = vec<int>(m_mpisize); 
		nzeprocbatch[ibatch] = vec<int>(m_mpisize);
				
		// communicate nzes + blks to other processes
		
		LOG.os<1>("Gathering nze and nblocks...\n");
		
		MPI_Allgather(&nze,1,MPI_INT,nzeprocbatch[ibatch].data(),1,MPI_INT,m_comm);
		MPI_Allgather(&nblocks,1,MPI_INT,nblksprocbatch[ibatch].data(),1,MPI_INT,m_comm);
		
		int64_t nblktotbatch = std::accumulate(nblksprocbatch[ibatch].begin(),
			nblksprocbatch[ibatch].end(),int64_t(0));
		
		int64_t nzetotbatch = std::accumulate(nzeprocbatch[ibatch].begin(),
			nzeprocbatch[ibatch].end(),int64_t(0));
		
		m_nblkloc += nblksprocbatch[ibatch][m_mpirank];
		m_nzeloc += nzeprocbatch[ibatch][m_mpirank];
		
		m_nblktot += nblktotbatch;
			
		m_nzetot += nzetotbatch;
		
		LOG(-1).os<1>("Local number of blocks: ", m_nblkloc, '\n');
		LOG.os<1>("Total number of blocks: ", nblktotbatch, '\n');
		LOG.os<1>("Total number of nze: ", nzetotbatch, '\n');
		
		// read blocks

		LOG.os<1>("Writing blocks...\n");

		dbcsr::iterator_t<N,T> iter(*write_tensor);
		
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
		
		std::string data_fname = m_filename;
		
		LOG.os<1>("Computing offsets...\n");
	
		// offsets
		MPI_Offset data_batch_offset = 0;
		
		//int64_t nblkprev = 0;
		int64_t ndataprev = 0;
		
		// global batch offset
		for (int i = 0; i != ibatch; ++i) {
				
			ndataprev += std::accumulate(nzeprocbatch[i].begin(),
				nzeprocbatch[i].end(),int64_t(0));
			
		}
		
		// local processor dependent offset
		for (int i = 0; i < m_mpirank; ++i) {
			ndataprev += nzeprocbatch[ibatch][i];
		}
		
		data_batch_offset = ndataprev;
		
		// add it to blkoffsets
		for (auto& off : blkoffbatch) {
			off += data_batch_offset;
		}
		
		// concatenate indices and offsets
		LOG.os<1>("Adding offsets to vector...\n");
		
		auto& locblkidx = m_wrview.locblkidx;
		auto& locblkoff = m_wrview.locblkoff;
		
		for (int i = 0; i != N; ++i) {
			locblkidx[i].insert(locblkidx[i].end(),
				blkidxbatch[i].begin(),blkidxbatch[i].end());
			blkidxbatch[i].clear();
		}
		
		locblkoff.insert(locblkoff.end(),
			blkoffbatch.begin(),blkoffbatch.end());
		blkoffbatch.clear();
	
		// write data to file 
		
		LOG.os<1>("Writing tensor data...\n");
		
		long long int datasize;
		T* data = write_tensor->data(datasize);
		
		MPI_File fh_data;
		
		MPI_File_open(m_comm,data_fname.c_str(),MPI_MODE_WRONLY,
			MPI_INFO_NULL,&fh_data);
		
		MPI_File_write_at_all(fh_data,data_batch_offset*sizeof(T),data,
			nze,MPI_DOUBLE,MPI_STATUS_IGNORE);
			
		MPI_File_close(&fh_data);
		
		if (LOG.global_plev() >= 10) {
			std::cout << "LOC BLK IDX AND OFFSET." << std::endl;
			for (int i = 0; i != locblkidx[0].size(); ++i) {
				for (int j = 0; j != N; ++j) {
					std::cout << locblkidx[j][i] << " ";
				}
				std::cout << locblkoff[i] << std::endl;
			}
		}
		
		write_tensor->clear();
		
		LOG.os<1>("Done with batch ", ibatch, '\n');
		
	}
	
	void compress_finalize() {
		
		LOG.os<1>("Finalizing compression...\n");
		if (m_type == dbcsr::btype::core) m_work_tensor->batched_contract_finalize();
				
	}
	
	/*
	void reorder(vec<int> map1, vec<int> map2) {
		
		auto this_map1 = m_stensor->map1_2d();
		auto this_map2 = m_stensor->map2_2d();
		
		/*
		std::cout << "MAP1 vs MAP1 new" << std::endl;
		auto prt = [](vec<int> m) {
			for (auto i : m) {
				std::cout << i << " ";
			} std::cout << std::endl;
		};
		
		prt(this_map1);
		prt(map1);
		
		std::cout << "MAP2 vs MAP2 new" << std::endl;
		prt(this_map2);
		prt(map2);
		
		
		if (map1 == this_map1 && map2 == this_map2) return;
		
		//std::cout << "REO" << std::endl;
		
		stensor<N,T> newtensor = tensor_create_template<N,T>(m_stensor)
			.name(m_stensor->name()).map1(map1).map2(map2).get();
			
		if (m_type == btype::core) {
			dbcsr::copy(*m_stensor, *newtensor).move_data(true).perform();
		}
		
		m_stensor = newtensor;
		
	}
	
	void reorder(shared_tensor<N,T> mytensor) {
		
		stensor<N,T> newtensor = tensor_create_template<N,T>(mytensor)
			.name(m_stensor->name()).get();
			
		if (m_type == btype::core) {
			dbcsr::copy(*m_stensor, *newtensor).move_data(true).perform();
		}
		
		m_stensor = newtensor;
		
	}*/
	
	view set_view(vec<int> dims) {
		
		view rview;
		
		rview.is_contiguous = false;
		rview.dims = dims;
		
		auto w_dims = m_wrview.dims;
		
		// get idx speed for writing
		auto map1 = m_wrview.map1;
		auto map2 = m_wrview.map2;
		
		map2.insert(map2.end(),map1.begin(),map1.end());
		
		auto order = map2;
		
		std::reverse(order.begin(),order.end());
		
		// reorder indices
		for (auto i : order) {
			LOG.os<1>(i, " ");
		} LOG.os<1>('\n');
		
		for (auto rd : rview.dims) {
			for (auto wd : w_dims) {
				
				LOG.os<1>("rd/wd ", rd, " ", wd, '\n');
				
				auto w_iter = std::find(order.begin(), order.end(), wd);
				int w_npos = order.end() - w_iter;
				
				auto r_iter = std::find(order.begin(), order.end(), rd);
				int r_npos = order.end() - r_iter;
				
				LOG.os<1>("w/rpos ", w_npos, " ", r_npos, '\n');
				
				if (r_npos > w_npos) 
					throw std::runtime_error("Reading dimension(s) are too slow. \n Use direct or core.\n");
				
			}
		}	
		
		// compute super indices
		
		auto& nblksprocbatch = rview.nblksprocbatch;
		auto& nzeprocbatch = rview.nzeprocbatch;
		auto& rlocblkidx = rview.locblkidx;
		auto& rlocblkoff = rview.locblkoff;
		auto& wlocblkidx = m_wrview.locblkidx;
		auto& wlocblkoff = m_wrview.locblkoff;
		
		vec<size_t> perm(wlocblkidx[0].size()), superidx(wlocblkidx[0].size());
		
		std::iota(perm.begin(),perm.end(),(size_t)0);
		
		// get nblk_per_batch for all dims
		vec<double> nele_per_batch(N);
		
		auto& blksizes = m_blk_sizes;
		
		for (int i = 0; i != N; ++i) {
			int nele = std::accumulate(blksizes[i].begin(),
				blksizes[i].end(),0);
			int nbatch = m_nbatches_dim[i];
			nele_per_batch[i] = (double) nele / (double) nbatch;
			//std::cout << "NELE: " << nele_per_batch[i] << std::endl;
		}	
		
		if (N == 3) {
			
			const size_t ix0 = order[0];
			const size_t ix1 = order[1];
			const size_t ix2 = order[2];
			
			const size_t bsize0 = (dims.size() > 0) ? m_nbatches_dim[dims[0]] : 1;
			const size_t bsize1 = (dims.size() > 1) ? m_nbatches_dim[dims[1]] : 1;
			const size_t bsize2 = (dims.size() > 2) ? m_nbatches_dim[dims[2]] : 1;
			
			const size_t batchsize = bsize0 * bsize1 * bsize2;
			
			//std::cout << "BATCHSIZE: " << batchsize << std::endl;
			
			rview.nbatches = batchsize;
			
			auto sizes = m_read_tensor->blk_sizes();
			auto full = m_read_tensor->nblks_total();
			auto off = m_read_tensor->blk_offsets();
			
			size_t nblks = std::accumulate(full.begin(),full.end(),
				1, std::multiplies<size_t>());
				
			const size_t blksize1 = full[ix1];
			const size_t blksize2 = full[ix2];
			
			nblksprocbatch = vec<vec<int>>(batchsize,vec<int>(m_mpisize));
			nzeprocbatch = vec<vec<int>>(batchsize,vec<int>(m_mpisize));
			
			#pragma omp parallel for
			for (size_t i = 0; i != superidx.size(); ++i) {	
									
					const size_t i0 = wlocblkidx[ix0][i];
					const size_t i1 = wlocblkidx[ix1][i];
					const size_t i2 = wlocblkidx[ix2][i];
					
					// compute batch indices
					const size_t b0 = std::floor(
						(double)off[dims[0]][wlocblkidx[dims[0]][i]]
						/ nele_per_batch[dims[0]]);
					const size_t b1 = (dims.size() > 1) ? 
						std::floor(
						(double)off[dims[1]][wlocblkidx[dims[1]][i]] 
						/ nele_per_batch[dims[1]]) : 0;
					const size_t b2 = (dims.size() > 2) ?
						std::floor(
						(double)off[dims[2]][wlocblkidx[dims[2]][i]] 
						/ nele_per_batch[dims[2]]) : 0;
						
					const size_t batch_idx = b0 * bsize1 * bsize2 + b1 * bsize2 + b2;
					
					#pragma omp critical 
					{
						nblksprocbatch[batch_idx][m_mpirank]++;
						nzeprocbatch[batch_idx][m_mpirank]++;
					}
					
					//std::cout << "BATCHIDX: " << batch_idx << std::endl;
									
					superidx[i] = batch_idx * nblks
					 + i0 * blksize2 * blksize1
					 + i1 * blksize2
					 + i2;
					 
					// std::cout << "INTEGER: " << superidx[i] << " "
					//	<< " BATCHES " << b0 << " " << b1 << " " << b2
					//	<< " IDX " << i0 << " " << i1 << " " << i2 << std::endl;
					 
			}
				 
			// communicate
			LOG.os<1>("Gathering nze and nblocks...\n");
		
			for (int i = 0; i != batchsize; ++i) {
				
				int nblk = nblksprocbatch[i][m_mpirank];
				int nze = nzeprocbatch[i][m_mpirank];
				
				MPI_Allgather(&nze,1,MPI_INT,nzeprocbatch[i].data(),1,MPI_INT,m_comm);
				MPI_Allgather(&nblk,1,MPI_INT,nblksprocbatch[i].data(),1,MPI_INT,m_comm);
				
			}
			
			if (LOG.global_plev() >= 10 && m_mpirank == 0) {
				std::cout << "BATCHSIZES:" << std::endl;
				for (int i = 0; i != nblksprocbatch.size(); ++i) {
					std::cout << "BATCH " << i << " " << nblksprocbatch[i][0] << std::endl;
				}
			}
				 
		}
		
		std::sort(perm.begin(),perm.end(),
			[&](size_t i0, size_t i1) {
				return superidx[i0] < superidx[i1];
			});
		
		rlocblkidx = wlocblkidx;
		rlocblkoff = wlocblkoff;
		
		for (size_t i = 0; i != perm.size(); ++i) {
			for (int n = 0; n != N; ++n) {
				rlocblkidx[n][i] = wlocblkidx[n][perm[i]];
			}
			rlocblkoff[i] = wlocblkoff[perm[i]];
		}
		
		if (LOG.global_plev() >= 10) {
			for (auto a : rlocblkidx) {
				for (auto i : a) {
					LOG.os<1>(i, " ");
				} LOG.os<1>('\n');
			}
			for (auto a : rlocblkoff) {
				LOG.os<1>(a, " ");
			} LOG.os<1>('\n');
		}
		
		return rview;
			
	}
		
	
	void decompress_init(std::initializer_list<int> dims_list, vec<int> map1, vec<int> map2) 
	{
		
		vec<int> dims = dims_list;
		
		LOG.os<1>("Initializing decompression...\n");
		
		m_read_current_is_contiguous = (dims == m_wrview.dims) ? true : false;
		m_read_current_dims = dims;
		
		if (m_type == btype::core) {
			
			if (m_wrview.map1 != map1 || m_wrview.map2 != map2) {
				
				LOG.os<1>("Reordering core tensor.\n");
				auto new_work_tensor = tensor_create_template<N,T>(m_work_tensor)
					.name(m_name + "_work_core").get();
				copy(*m_work_tensor, *new_work_tensor)
					.move_data(true)
					.perform();
					
				// update
				m_wrview.dims = dims;
				m_wrview.map1 = map1;
				m_wrview.map2 = map2;
				m_wrview.nbatches = get_nbatches(dims);
				
				m_work_tensor = new_work_tensor;
					
			} else {
				
				LOG.os<1>("Core tensor is compatible.\n");
				
			}
			
		}
		
		if (m_type == btype::direct) {
			
			LOG.os<1>("Creating direct work and read tensors.\n");
			
			m_work_tensor = tensor_create<N,T>()
				.name(m_name + "_work_direct")
				.pgrid(m_spgrid_N)
				.map1(map1).map2(map2)
				.blk_sizes(m_blk_sizes)
				.get();
				
			m_read_tensor = tensor_create_template(m_work_tensor)
				.name(m_name + "_read_direct")
				.get();
				
			// update 
			m_wrview.dims = dims;
			m_wrview.map1 = map1;
			m_wrview.map2 = map2;
			m_wrview.nbatches = get_nbatches(dims);
				
		}
		
		if (m_type == btype::disk) {
			
			LOG.os<1>("Creating disk work and read tensors.\n");
			
			m_work_tensor = tensor_create<N,T>()
				.name(m_name + "_work_disk")
				.pgrid(m_spgrid_N)
				.map1(map1).map2(map2)
				.blk_sizes(m_blk_sizes)
				.get();
				
			m_read_tensor = tensor_create_template(m_work_tensor)
				.name(m_name + "_read_disk")
				.map1(m_wrview.map1)
				.map2(m_wrview.map2)
				.get();
			
		}
		
		if (m_type == btype::disk && !m_read_current_is_contiguous) {
			
			LOG.os<1>("Reading will be non-contiguous. ",
				"Setting view of local block indices.\n");
				
			if (m_rdviewmap.find(dims) != m_rdviewmap.end()) {
				LOG.os<1>("View was already set previously. Using it.\n");
			} else {
				LOG.os<1>("View not yet computed. Doing it now...\n");
				m_rdviewmap[dims] = set_view(dims);	
			}
			
		}
		
		m_work_tensor->batched_contract_init();
		//std::cout << "NATCHED." << std::endl;
		
	}
	
	// if tensor_in nullptr, then gives back m_stensor
	void decompress(std::initializer_list<int> idx_list) {
		
		vec<int> idx = idx_list;
		
		switch(m_type) {
			case btype::disk : decompress_disk(idx);
			break;
			case btype::direct : decompress_direct(idx);
			break;
			case btype::core : decompress_core(idx);
			break;
		}
		
	}
	
	void decompress_direct(vec<int> idx) {
		
		LOG.os<1>("Generating tensor entries.\n");
		
		auto b = get_blk_bounds(idx, m_read_current_dims);
		
		m_generator(m_read_tensor, b);
		
		vec<vec<int>> copy_bounds = get_bounds(idx, m_wrview.dims);
		
		LOG.os<1>("Copying to work tensor.\n");
		
		copy(*m_read_tensor, *m_work_tensor)
			.bounds(copy_bounds)
			.move_data(true)
			.perform();
		
	}
	
	void decompress_core(vec<int> idx) {
		
		LOG.os<1>("Decompressing from core.\n");
		
		return;
				
	}

	void decompress_disk(vec<int> idx) {
			
		vec<int> dims = (m_read_current_is_contiguous) ?
			m_wrview.dims : m_rdviewmap[m_read_current_dims].dims;
		
		int ibatch = flatten(idx, dims);
		LOG.os<1>("Reading batch ", ibatch, '\n');
		
		// if reading is done in same way as writing
		if (m_read_current_is_contiguous) {
			
			LOG.os<1>("Same dimensions for writing and reading.\n");
			
			auto& nzeprocbatch = m_wrview.nzeprocbatch;
			auto& nblksprocbatch = m_wrview.nblksprocbatch;
			auto& locblkidx = m_wrview.locblkidx;
			auto& locblkoff = m_wrview.locblkoff;
		
			int nze = nzeprocbatch[ibatch][m_mpirank];
			int nblk = nblksprocbatch[ibatch][m_mpirank];
			
			int64_t nblktotbatch = std::accumulate(nblksprocbatch[ibatch].begin(),
				nblksprocbatch[ibatch].end(),int64_t(0));
		
			int64_t nzetotbatch = std::accumulate(nzeprocbatch[ibatch].begin(),
				nzeprocbatch[ibatch].end(),int64_t(0));
			
			// === Allocating blocks for tensor ===
			//// offsets
			
			int64_t blkoff = 0;
			
			for (int i = 0; i < ibatch; ++i) {
				blkoff += nblksprocbatch[i][m_mpirank];
			}
				
			arrvec<int,N> rlocblkidx;
			
			LOG(-1).os<1>("Offset: ", blkoff, '\n');
			LOG(-1).os<1>("NBLKTOTBATCH: ", nblktotbatch, '\n');
			//// setting
			for (int i = 0; i != N; ++i) {
				rlocblkidx[i].insert(rlocblkidx[i].end(),
					locblkidx[i].begin() + blkoff,
					locblkidx[i].begin() + blkoff + nblk);
			}
			
			if (LOG.global_plev() >= 10) {
			
				MPI_Barrier(m_comm);
				
				for (int i = 0; i != m_mpisize; ++i) {
				
					if (i == m_mpirank) {
					
						for (auto a : rlocblkidx) {
							for (auto l : a) {
								std::cout << l << " ";
							} std::cout << std::endl;
						}
						
					}
				
				MPI_Barrier(m_comm);
				
				}
				
			}
			
			//// reserving
			m_read_tensor->reserve(rlocblkidx);
			
			for (auto& v : rlocblkidx) v.resize(0);
			
			// === Reading from file ===
			//// Allocating
			
			//// Offset
			MPI_Offset data_batch_offset = locblkoff[blkoff];
			
			//// Opening File
			std::string fname = m_filename;
			
			MPI_File fh_data;
			
			MPI_File_open(m_comm,fname.c_str(),MPI_MODE_RDONLY,
				MPI_INFO_NULL,&fh_data);
		
			//// Reading
			long long int datasize;
			T* data = m_read_tensor->data(datasize);
			
			MPI_File_read_at_all(fh_data,data_batch_offset*sizeof(T),data,
				nze,MPI_DOUBLE,MPI_STATUS_IGNORE);
			
			MPI_File_close(&fh_data);
		
		// needs special offsets
		} else {
			
			LOG.os<1>("Different dimension.\n");
			
			auto& rdview = m_rdviewmap[m_read_current_dims];
			
			auto& nzeprocbatch = rdview.nzeprocbatch;
			auto& nblksprocbatch = rdview.nblksprocbatch;
			auto& locblkidx = rdview.locblkidx;
			auto& locblkoff = rdview.locblkoff;
		
			int nze = nzeprocbatch[ibatch][m_mpirank];
			int nblk = nblksprocbatch[ibatch][m_mpirank];
			
			int64_t nblktotbatch = std::accumulate(nblksprocbatch[ibatch].begin(),
				nblksprocbatch[ibatch].end(),int64_t(0));
		
			int64_t nzetotbatch = std::accumulate(nzeprocbatch[ibatch].begin(),
				nzeprocbatch[ibatch].end(),int64_t(0));
			
			// === Allocating blocks for tensor ===
			//// offsets
			
			int64_t blkoff = 0;
			
			for (int i = 0; i < ibatch; ++i) {
				blkoff += nblksprocbatch[i][m_mpirank];
			}
				
			arrvec<int,N> rlocblkidx;
			vec<MPI_Aint> blkoffsets(nblk);
			
			LOG(-1).os<1>("Offset: ", blkoff, '\n');
			LOG(-1).os<1>("NBLKTOTBATCH: ", nblktotbatch, '\n');
			//// setting
			for (int i = 0; i != N; ++i) {
				rlocblkidx[i].insert(rlocblkidx[i].end(),
					locblkidx[i].begin() + blkoff,
					locblkidx[i].begin() + blkoff + nblk);
			}
			
			for (size_t i = 0; i != nblk; ++i) {
				blkoffsets[i] = locblkoff[i + blkoff] * sizeof(T);
			}
			
			if (LOG.global_plev() >= 10) {
			
				MPI_Barrier(m_comm);
				
				LOG.os<>("LOC IDX AND OFFS\n");
				
				for (int i = 0; i != m_mpisize; ++i) {
				
					if (i == m_mpirank) {
						
						for (auto a : rlocblkidx) {
							for (auto l : a) {
								std::cout << l << " ";
							} std::cout << std::endl;
						}
						
						for (auto a : blkoffsets) {
							std::cout << a << " ";
						} std::cout << std::endl;
						
						
					}
				
				MPI_Barrier(m_comm);
				
				}
				
			}
			
			//// reserving
			m_read_tensor->reserve(rlocblkidx);
			
			vec<int> blksizes(nblk);
			auto& tsizes = m_blk_sizes;
			
			LOG.os<1>("Computing block sizes.\n");
			
			for (int i = 0; i != nblk; ++i) {
				int size = 1;
				for (int n = 0; n != N; ++n) {
					size *= tsizes[n][rlocblkidx[n][i]];
				}
				blksizes[i] = size;
			}
			
			LOG.os<1>("Reserving.\n");
						
			for (auto& a : rlocblkidx) a.clear();
			
			// now adjust the file view
			nze = m_read_tensor->num_nze();
			
			LOG.os<1>("Creating MPI TYPE.\n");
			
			if (LOG.global_plev() >= 10) {
			
				std::cout << "BLOCK SIZES + OFF: " << std::endl;
				for (int i = 0; i != nblk; ++i) {
					std::cout << blksizes[i] << " " << blkoffsets[i] << std::endl;
				}
				
			}
			
			MPI_Datatype MPI_HINDEXED;
			
			MPI_Type_create_hindexed(nblk,blksizes.data(),blkoffsets.data(),MPI_DOUBLE,&MPI_HINDEXED);
			MPI_Type_commit(&MPI_HINDEXED);
			
			std::string fname = m_filename;
			MPI_File fh_data;
			
			MPI_File_open(m_comm,fname.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&fh_data);
			
			LOG.os<1>("Setting file view.\n");
			
			MPI_File_set_view(fh_data, 0, MPI_DOUBLE, MPI_HINDEXED, "native", MPI_INFO_NULL);
			
			long long int datasize;
			T* data = m_read_tensor->data(datasize);
			
			LOG.os<1>("Reading from file...\n");
			
			MPI_File_read_at_all(fh_data,0,data,nze,MPI_DOUBLE,MPI_STATUS_IGNORE);
			
			MPI_File_close(&fh_data);
			
			LOG.os<1>("Done!!\n");
			
			/*std::cout << "READTENSOR" << std::endl;
			dbcsr::print(*read_tensor);
			
			for (int i = 0; i != 1000; ++i) {
				std::cout << data[i] << " ";
			} std::cout << std::endl;*/
			
		} // end if
		
		LOG.os<1>("Copying to work tensor\n");
		
		auto copy_bounds = get_bounds(idx, m_read_current_dims);
		dbcsr::copy(*m_read_tensor, *m_work_tensor)
			.move_data(true)
			.bounds(copy_bounds)
			.perform();

	}
	
	void decompress_finalize() {
		m_work_tensor->batched_contract_finalize();
		if (m_type != btype::core) {
			m_work_tensor->clear();
			m_read_tensor->clear();
		}
	}
		
	dbcsr::stensor<N,T> get_work_tensor() { return m_work_tensor; }
		
	int nbatches(int idim) {
		return m_nbatches_dim[idim];
	}
	
	vec<int> blk_bounds(int idim, int ibatch) {
		return m_blk_bounds[idim][ibatch];
	}
	
	vec<int> bounds(int idim, int ibatch) {
		return m_bounds[idim][ibatch];
	}
	
	vec<int> full_blk_bounds(int idim) {
		return m_full_blk_bounds[idim];
	}
	
	vec<int> full_bounds(int idim) {
		return m_full_bounds[idim];
	}
	
	double occupation() {
		
		std::array<int,N> full;
		for (int i = 0; i != N; ++i) {
			full[i] = std::accumulate(m_blk_sizes[i].begin(), m_blk_sizes[i].end(), 0);
		}
		int64_t tot = std::accumulate(full.begin(),full.end(),(int64_t)1,
			std::multiplies<int64_t>());
		
		return (double)m_nzetot / (double) tot;
		
	}
	
	arrvec<int,N> blk_sizes() {
		return m_blk_sizes;
	}
	
	MPI_Comm comm() {
		return m_spgrid_N->comm();
	}
	
	shared_pgrid<N> spgrid() {
		return m_spgrid_N;
	}
	
	btype get_type() { return m_type; }
	
	std::array<int,N> batch_dims() {
		return m_nbatches_dim;
	}
	
	dbcsr::shared_tensor<N,T> get_template(std::string name, vec<int> map1,
		vec<int> map2) 
	{
		auto out = dbcsr::tensor_create<N,T>()
			.name(name)
			.pgrid(m_spgrid_N)
			.blk_sizes(m_blk_sizes)
			.map1(map1).map2(map2)
			.get();
			
		return out;
	}
			
		
	
}; //end class btensor

template <int N, typename T>
using sbtensor = std::shared_ptr<btensor<N,T>>;

template <int N, typename T>
class btensor_create_base {
private:

	shared_pgrid<N> c_pgrid;
	arrvec<int,N> c_blk_sizes;
	std::array<int,N> c_bdims;
	std::string c_name;
	dbcsr::btype c_btype;
	int c_print;
	
public:

	btensor_create_base() {}
	
	btensor_create_base& name(std::string t_name) {
		c_name = t_name; return *this;
	}
	
	btensor_create_base& pgrid(shared_pgrid<N>& t_pgrid) {
		c_pgrid = t_pgrid; return *this;
	}
	
	btensor_create_base& blk_sizes(arrvec<int,N> t_blk_sizes) {
		c_blk_sizes = t_blk_sizes; return *this;
	}
	
	btensor_create_base& batch_dims(std::array<int,N> t_bdims) {
		c_bdims = t_bdims; return *this;
	}
	
	btensor_create_base& btensor_type(btype t_btype) {
		c_btype = t_btype; return *this;
	}
	
	btensor_create_base& print(int n) {
		c_print = n; return *this;
	}
	
	sbtensor<N,T> get() {
		auto out = std::make_shared<btensor<N,T>>(c_name,c_pgrid,c_blk_sizes,c_bdims,c_btype,c_print);
		return out;
	}

};

template <int N, typename T = double>
btensor_create_base<N,T> btensor_create() {
	return btensor_create_base<N,T>();
}

} //end namespace

#endif
