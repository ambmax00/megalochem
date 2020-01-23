#ifndef DBCSR_HPP
#define DBCSR_HPP

#define MAXDIM 4

#define really_unparen(...) __VA_ARGS__
#define invoke_(expr) expr
#define unparen(args) invoke_(really_unparen args)
#define concat(text,var) text##var

#define PLUS_ONE(x) INCREMENT##x
#define INCREMENT0 1
#define INCREMENT1 2
#define INCREMENT2 3
#define INCREMENT3 4
#define INCREMENT4 5
#define INCREMENT5 6
#define INCREMENT6 7

#define STEP0(func, x, n, del) 
#define STEP1(func, x, n, del) func(x,n) STEP0(func, x, PLUS_ONE(n), del)
#define STEP2(func, x, n, del) func(x,n)unparen(del) STEP1(func, x, PLUS_ONE(n), del)
#define STEP3(func, x, n, del) func(x,n)unparen(del) STEP2(func, x, PLUS_ONE(n), del)
#define STEP4(func, x, n, del) func(x,n)unparen(del) STEP3(func, x, PLUS_ONE(n), del)
#define STEP5(func, x, n, del) func(x,n)unparen(del) STEP4(func, x, PLUS_ONE(n), del)
#define STEP6(func, x, n, del) func(x,n)unparen(del) STEP5(func, x, PLUS_ONE(n), del)
#define REPEAT(func, x, n, del, start) concat(STEP,n) (func, x, start, del)

#define VARANDSIZE(x,n) x[n], x##_size[n]

#include <dbcsr.h>
#include <dbcsr_tensor.h>
#include <utility>
#include <string>
#include <numeric>
#include <vector>
#include <memory>
#include <stdexcept>
#include <random>
#include <algorithm>

#include "utils/params.hpp"

// debug
#include <iostream>
#include <typeinfo>
#include <thread>
#include <unistd.h>


template <typename T>
using vec = std::vector<T>;

namespace dbcsr {

template <int N>
using index = std::array<int,N>;

typedef index<2> idx2;
typedef index<3> idx3;
typedef index<4> idx4;

template <int N>
class pgrid;

template <int N>
class dist;

template <int N, typename T>
class tensor;

template <int N, typename T>
class iterator;
	
static void init(MPI_Comm comm = MPI_COMM_WORLD, int* io_unit = nullptr) {

	c_dbcsr_init_lib(comm, io_unit);
	
};

static void finalize() {
	
	c_dbcsr_finalize_lib();
	
};

static auto default_dist = 
	[](int nel, int nbin, vec<int> weights) {
		
		std::vector<int> distvec(weights.size(),0);
		
		//std::cout << "nel, nbin: " << nel << " " << nbin << std::endl;
		
		vec<int> occup(nbin, 0);
		int ibin = 0;
		
		for (int iel = 0; iel != nel; ++iel) {
			int niter = 0;
			ibin = abs((ibin + 1)%nbin);
			//std::cout << "ibin: " << ibin << std::endl;
			while(occup[ibin] + weights[iel] >= *std::max_element(occup.begin(), occup.end())) {
				//std::cout << occup[ibin] << " " << weights[iel] << " " << *std::max_element(occup.begin(), occup.end()) << std::endl;
				int minloc = std::min_element(occup.begin(),occup.end()) - occup.begin();
				//std::cout << minloc << std::endl;
				if (minloc == ibin) break;
				ibin = abs((ibin+1)%nbin);
				++niter;
			}
			
			distvec[iel] = ibin;
			occup[ibin] = occup[ibin] + weights[iel];
			
		}
		
		int myrank = 0;
		
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	    
	    /*
	    if (myrank == 0) {
	    
			std::cout << "Dist: " << std::endl;
			for (auto x : distvec) {
				std::cout << x << " ";
			} std::cout << std::endl;
			
		}*/
		
		return distvec;
		
	};
				
				

template <int N>
class pgrid {
protected:

	void* m_pgrid_ptr;
	vec<int> m_dims;
	MPI_Comm m_comm;
	
	template <int M>
	friend class dist;
	
public:

	struct pgrid_params{
		required<MPI_Comm, val> 	comm;
		optional<vec<int>, val> 	map1, map2;
		optional<vec<int>, val>	    tensor_dims;
		optional<int, val>			nsplit, dimsplit;
	};

	pgrid(pgrid_params&& p) : 
		m_pgrid_ptr(nullptr),
		m_dims(N),
		m_comm(*p.comm)	
	{
	
		// handle "optional" arguments
		int* fmap1 = (p.map1) ? p.map1->data() : nullptr;
		int* fmap2 = (p.map2) ? p.map2->data() : nullptr;
		int map1size = (p.map1) ? p.map1->size() : 0;
		int map2size = (p.map2) ? p.map2->size() : 0;
		int* ftendims = (p.tensor_dims) ? p.tensor_dims->data() : nullptr;

		int* fnsplit = (p.nsplit) ? &*p.nsplit : nullptr;
		int* fdimsplit = (p.dimsplit) ? &*p.dimsplit : nullptr;
		
		MPI_Fint fmpi = MPI_Comm_c2f(m_comm);
		
		if (p.map1 && p.map2) {
			c_dbcsr_t_pgrid_create_expert(&fmpi, m_dims.data(), N, &m_pgrid_ptr, fmap1, map1size, fmap2, map2size, ftendims, fnsplit, fdimsplit);
		} else {
			c_dbcsr_t_pgrid_create(&fmpi, m_dims.data(), N, &m_pgrid_ptr, ftendims);
		}
		
	}
	
	pgrid(pgrid<N>& rhs) = delete;
	
	pgrid<N>& operator=(pgrid<N>& rhs) = delete;
		
	
	vec<int> dims() {
		
		return m_dims;
		
	}
	
	void destroy(bool keep_comm = false) {
		
		if (m_pgrid_ptr != nullptr)
			c_dbcsr_t_pgrid_destroy(&m_pgrid_ptr, &keep_comm);
			
		m_pgrid_ptr = nullptr;
		
	}
	
	MPI_Comm comm() {
		return m_comm;
	}
	
	~pgrid() {
		
		destroy(true);
		
	}
		
	
	
};

static vec<int> random_dist(int dist_size, int nbins)
{
    vec<int> d(dist_size);

    for(int i=0; i < dist_size; i++)
        d[i] = abs((nbins-i+1) % nbins);

    return std::move(d);
};


template <int N>
class dist {
private:

	void* m_dist_ptr;
	
	vec<vec<int>> m_nd_dists;
	MPI_Comm m_comm;
	
	template <int M, typename T>
	friend class tensor;
	
public:

	struct dist_params {
		required<pgrid<N>,ref>			pgridN;
		required<vec<vec<int>>,val>		nd_dists;
	};

	dist(dist_params p) :
		m_dist_ptr(nullptr),
		m_nd_dists(*p.nd_dists)
	{
		
		void* pgrid_ptr = p.pgridN->m_pgrid_ptr;
		m_comm = p.pgridN->m_comm;
		
		vec<int*> f_dists(MAXDIM, nullptr);
		vec<int> f_dists_size(MAXDIM, 0);
		
		for (int i = 0; i != N; ++i) {
			f_dists[i] = p.nd_dists->at(i).data();
			f_dists_size[i] = p.nd_dists->at(i).size();
		}
		
		c_dbcsr_t_distribution_new(&m_dist_ptr, pgrid_ptr, 
                                   REPEAT(VARANDSIZE, f_dists, MAXDIM, (,), 0)
                                   );
                                   
	}
	
	dist(dist<N>& rhs) = delete;
		
		
	dist<N>& operator=(dist<N>& rhs) = delete;
	
	~dist() {
		
		//std::cout << "DEST1START" << std::endl;
		
		if (m_dist_ptr != nullptr)
			c_dbcsr_t_distribution_destroy(&m_dist_ptr);
			
		m_dist_ptr = nullptr;
		
		//std::cout << "DEST1END" << std::endl;
		
	}
	
	void destroy() {
		
		//std::cout << "DEST2START" << std::endl;
		
		if (m_dist_ptr != nullptr) {
			c_dbcsr_t_distribution_destroy(&m_dist_ptr);
		}
			
		m_dist_ptr = nullptr;
		
		//std::cout << "DEST2END" << std::endl;
		
	}
	
	
};


template <int N, typename T = double>
class block {
private:

	T* m_data;
	vec<int> m_sizes;
	int m_nfull;
	bool m_own = true;
	
public:

	block() : m_data(nullptr), m_sizes(N,0), m_nfull(0) {}
	
	block(const vec<int> sizes, T* ptr = nullptr) : m_sizes(sizes) {
		
		if (sizes.size() != N) {
			throw std::runtime_error("Wrong block dimensions passed to constr.");
		}
		
		m_nfull = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());
		
		if (ptr) {
			m_data = ptr; 
			m_own = false;
		} else {
			m_data = new T[m_nfull]();
		}
		
	}
	
	block(const block<N,T>& blk_in) : m_data(nullptr) {
		
		if (m_data != nullptr) 
			delete [] m_data;
			
		m_nfull = blk_in.m_nfull;
		m_data = new T[m_nfull];
		m_sizes = blk_in.m_sizes;
		
		std::copy(blk_in.m_data, blk_in.m_data + m_nfull, m_data);
		
	}
	
	block& operator=(const block & rhs)
	{
		if(this == &rhs)
		   return *this;
		   
		if (m_data != nullptr)  
			delete [] m_data;
			
		m_nfull = rhs.m_nfull;
		m_data = new T[m_nfull];
		m_sizes = rhs.m_sizes;
		
		std::copy(rhs.m_data, rhs.m_data + m_nfull, m_data);
		   
		return *this;
	}
	
	int dim() { return N; }
	int ntot() { return m_nfull; }
	vec<int> sizes() { return m_sizes; }
	
	T* data() {
		return m_data;
	}
	
	T& operator() (int id) { 
		//std::cout << id << std::endl;
		return m_data[id]; 
	}
	
	template <typename D = T, int M = N>
	typename std::enable_if<M == 2, D&>::type
	operator()(int i, int j) {
		return m_data[i + j * m_sizes[0]];
	}
	
	template <typename D = T, int M = N>
	typename std::enable_if<M == 3, D&>::type
	operator()(int i, int j, int k) {
		return m_data[i + j*m_sizes[0] + k*m_sizes[0]*m_sizes[1]];
	}
	
	template <typename D = T, int M = N>
	typename std::enable_if<M == 4, D&>::type
	operator()(int i, int j, int k, int l) {
		return m_data[i + j*m_sizes[0] + k*m_sizes[0]*m_sizes[1] 
			+ l*m_sizes[0]*m_sizes[1]*m_sizes[2]];
	}
	
	void fill_rand() {
		
		std::random_device rd; 
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(-1.0, 1.0);
		
		for (int i = 0; i != m_nfull; ++i) {
			m_data[i] = dis(gen);
			//std::cout << m_data[i] << std::endl;
		}
		
	} 
	
	//new constructor with raw pointer
	
	~block() {
		
		if (m_data != nullptr && m_own) {
			delete [] m_data;
			m_data = nullptr;
		}
		
	}
	
};	

template <int N, typename T>
struct tensor_copy;

template <int N, typename T = double>
class tensor {
protected:

	void* m_tensor_ptr;
	MPI_Comm m_comm;
	
	tensor(void* ptr) : m_tensor_ptr(ptr) {}
	
public:

	template <int M, typename D>
	friend void copy(tensor_copy<M,D>&& p);

	typedef T value_type;
	
	struct tensor_get_info_params {
		optional<vec<int>, ref> 	nblks_total, nfull_total, 
									nblks_local, nfull_local, 
									pdims, my_ploc;
		optional<vec<vec<int>>,ref> blks_local, proc_dist,
									blk_size, blk_offset;
		optional<std::string,ref>	name;
	};
	void get_info(tensor_get_info_params&& p) const {
	
		auto reserve = [] (optional<vec<int>,ref>& opt) -> int* {
			int* ptr = nullptr;
			if (opt) {
				std::cout << "Here" << opt-> size() << std::endl;
				opt->reserve(N);
				ptr = opt->data();
			} 
			return ptr;
		};
		
		int *nblks_total_p, *nfull_total_p, *nblks_local_p, *nfull_local_p, *pdims_p, *my_ploc_p;	
		
		nblks_total_p = 	reserve(p.nblks_total);
		nfull_total_p = 	reserve(p.nfull_total);
		nblks_local_p = 	reserve(p.nblks_local);
		nfull_local_p = 	reserve(p.nfull_local);
		pdims_p = 			reserve(p.pdims);
		my_ploc_p = 		reserve(p.my_ploc);
		
		vec<int**> blks_local_v(MAXDIM,nullptr), proc_dist_v(MAXDIM,nullptr),
			blk_size_v(MAXDIM,nullptr), blk_offset_v(MAXDIM,nullptr);
		
		vec<int*> blks_local_v_size(MAXDIM,nullptr), proc_dist_v_size(MAXDIM,nullptr),
			blk_size_v_size(MAXDIM,nullptr), blk_offset_v_size(MAXDIM,nullptr);
	
		auto reserve_vec = [] (vec<int**>& v, vec<int*>& vsize) {
			for (int i = 0; i != N; ++i) {
				v[i] = new int*;
				vsize[i] = new int;
			}
		};
		
		if (p.blks_local) { reserve_vec(blks_local_v, blks_local_v_size); std::cout << "reserved" << std::endl;}
		if (p.proc_dist) { reserve_vec(proc_dist_v, proc_dist_v_size); }
		if (p.blk_size) { reserve_vec(blk_size_v, blk_size_v_size); }
		if (p.blk_offset) { reserve_vec(blk_offset_v, blk_offset_v_size); }
	
		char* name = nullptr;
		int name_size = 0;
		
		std::cout << "IN HERE" << std::endl;
		
		if (nfull_total_p == nullptr) { std::cout << "its null" << std::endl; } 
		else { std::cout << "Not null!" << std::endl; }
	
	    c_dbcsr_t_get_info(m_tensor_ptr, N, 
							   nblks_total_p,
                               nfull_total_p,
                               nblks_local_p,
                               nfull_local_p,
                               pdims_p, my_ploc_p, 
                               REPEAT(VARANDSIZE,blks_local_v,MAXDIM,(,),0),
                               REPEAT(VARANDSIZE,proc_dist_v,MAXDIM,(,),0),
                               REPEAT(VARANDSIZE,blk_size_v,MAXDIM,(,),0),
                               REPEAT(VARANDSIZE,blk_offset_v,MAXDIM,(,),0),
                               nullptr, &name, &name_size,
                               nullptr);
		
		auto printvec = [](std::string msg, vec<int>& v) {
			std::cout << msg;
			for (auto x : v) {
				std::cout << x << " ";
			}
			std::cout << std::endl;
		};
		
		//std::cout << "OUT" << std::endl;
		
		//if(p.nblks_total) printvec("Total number of blocks:\n",*p.nblks_total);
		//if(p.nfull_total) printvec("Total number of elements:\n",*p.nfull_total);
		//if(p.nblks_local) printvec("Total number of local blocks:\n",*p.nblks_local);
		//if(p.nfull_local) printvec("Total number of local elements:\n",*p.nfull_local);
		
		auto makevec = [](vec<int**>& v, vec<int*>& vsize) {
			vec<vec<int>> out(N);
			for (int i = 0; i != N; ++i) {
				vec<int> veci(*vsize[i]);
				
				for (int j = 0; j != *vsize[i]; ++j) {
					veci[j] = (*(v[i]))[j];
				}
				
				free(*(v[i]));
				delete v[i];
				delete vsize[i];
				
				out[i] = veci;
			}
			
			return out;
			
		};
				
        if (p.blks_local) {
			//std::cout << "Local block indices:" << std::endl;
			*p.blks_local = makevec(blks_local_v, blks_local_v_size);
			//for (int i = 0; i != N; ++i) printvec("", out[i]);
		}
		
		if (p.proc_dist) {
			//std::cout << "Distribution vectors:" << std::endl;
			*p.proc_dist = makevec(proc_dist_v, proc_dist_v_size);
			//for (int i = 0; i != N; ++i) printvec("", out[i]);
		}
		
		if (p.blk_size) {
			//std::cout << "Block sizes:" << std::endl;
			*p.blk_size = makevec(blk_size_v, blk_size_v_size);
			//for (int i = 0; i != N; ++i) printvec("", out[i]);
		}
		
		if (p.blk_offset) {
			//std::cout << "Block offsets:" << std::endl;
			*p.blk_offset = makevec(blk_offset_v, blk_offset_v_size);
			//for (int i = 0; i != N; ++i) printvec("", out[i]);
		} 
		
		if (p.name) {
			*p.name = name;
        }
        
        free(name); 
         
	}
	
	tensor() : m_tensor_ptr(nullptr) {}
	
	//copy constructor
	tensor(const tensor<N,T>& rhs): m_tensor_ptr(nullptr) {
		
		std::cout << "Copying..." << std::endl;
		
		if (this != &rhs) {
		
			const char* name = rhs.name().c_str();
		
			c_dbcsr_t_create_template(rhs.m_tensor_ptr, &m_tensor_ptr, name);
			c_dbcsr_t_copy(rhs.m_tensor_ptr, N, m_tensor_ptr, nullptr, nullptr, nullptr, nullptr);
	
			m_comm = rhs.m_comm;
			
		}
		
	}
	
	// construct template only
	tensor(const tensor<N,T>& rhs, std::string name) : m_tensor_ptr(nullptr), m_comm(rhs.m_comm) {
		
		const char* cname = rhs.name().c_str();
		
		c_dbcsr_t_create_template(rhs.m_tensor_ptr, &m_tensor_ptr, cname);
		
	}
		
	//move constructor
	tensor(tensor<N,T>&& rhs) : m_comm(rhs.m_comm) {
		std::cout << "Moving" << std::endl;
		//bool move = true;
		m_tensor_ptr = rhs.m_tensor_ptr;
		m_comm = rhs.m_comm;
		//c_dbcsr_t_create_template(t.m_tensor_ptr, &this->m_tensor_ptr, nullptr);
		rhs.m_tensor_ptr = nullptr;
		std::cout << "Done." << std::endl;
		
	}
	
	struct tensor_params {
		required<std::string, val>	name;
		required<dist<N>, ref>  	distN;
		required<vec<int>,val>		map1,map2;
		required<vec<vec<int>>,val>	blk_sizes;
	};
	tensor(tensor_params&& p) :
		m_tensor_ptr(nullptr)
	{
		
		void* dist_ptr = p.distN->m_dist_ptr;
		const char* f_name = p.name->c_str();
		int* c_data_type = nullptr;
		
		vec<int*> f_blks(MAXDIM, nullptr);
		vec<int> f_blks_size(MAXDIM, 0);
		
		for (int i = 0; i != N; ++i) {

			f_blks[i] = p.blk_sizes->at(i).data();
			f_blks_size[i] = p.blk_sizes->at(i).size();
			
		}
		
		c_dbcsr_t_create_new(&m_tensor_ptr, f_name, dist_ptr, 
						p.map1->data(), p.map1->size(),
						p.map2->data(), p.map2->size(), c_data_type, 
						f_blks[0], f_blks_size[0],
						f_blks[1], f_blks_size[1],
						f_blks[2], f_blks_size[2],
						f_blks[3], f_blks_size[3]);
					
	}
	
	struct tensor_params2 {
		required<std::string, val>	name;
		required<pgrid<N>, ref>  	pgridN;
		required<vec<int>,val>		map1,map2;
		required<vec<vec<int>>,val>	blk_sizes;
	};
	tensor(tensor_params2&& p) :
		m_tensor_ptr(nullptr),
		m_comm(p.pgridN->comm())
	{
		
		vec<vec<int>> distvecs(N);
		
		for (int i = 0; i != N; ++i) {
			distvecs[i] = default_dist(p.blk_sizes->at(i).size(),
				p.pgridN->dims()[i], p.blk_sizes->at(i));
		}
		
		dist<N> d({.pgridN = *p.pgridN, .nd_dists = distvecs});
		
		void* dist_ptr = d.m_dist_ptr;
		const char* f_name = p.name->c_str();
		int* c_data_type = nullptr;
		
		vec<int*> f_blks(MAXDIM, nullptr);
		vec<int> f_blks_size(MAXDIM, 0);
		
		for (int i = 0; i != N; ++i) {
			f_blks[i] = p.blk_sizes->at(i).data();
			f_blks_size[i] = p.blk_sizes->at(i).size();
		}
		
		c_dbcsr_t_create_new(&m_tensor_ptr, f_name, dist_ptr, 
						p.map1->data(), p.map1->size(),
						p.map2->data(), p.map2->size(), c_data_type, 
						f_blks[0], f_blks_size[0],
						f_blks[1], f_blks_size[1],
						f_blks[2], f_blks_size[2],
						f_blks[3], f_blks_size[3]);
						
		d.destroy();
					
	}
		
	~tensor() {
		
		destroy();
		
	}
	
	static constexpr int dim() { return N; }
	
	void* data() {
		
		return m_tensor_ptr;
		
	}
	
	void destroy() {
		
		if (m_tensor_ptr != nullptr) {
			c_dbcsr_t_destroy(&m_tensor_ptr);
		}
		
		m_tensor_ptr = nullptr;
		
	}
	
	void reserve(vec<vec<int>> nzblks) {
		
		if (nzblks.size() != N) 
			throw std::runtime_error("Reserve: Indices don't match tensor dimension.");
			
		if (nzblks[0].size() == 0) return;
	
		vec<int*> f_blks(MAXDIM, nullptr);
		
		for (int i = 0; i != N; ++i) {
			f_blks[i] = nzblks[i].data();
		}
		
		c_dbcsr_t_reserve_blocks_index(m_tensor_ptr, nzblks[0].size(), 
			f_blks[0], f_blks[1], f_blks[2], f_blks[3]);
			
	}
	
	struct tensor_put_block_params {
		required<index<N>,val>		idx;
		required<block<N,T>,ref>	blk;
		optional<bool,val>			sum;
		optional<double,val>		scale;
	};
	
	void put_block(tensor_put_block_params&& p) {
		
		bool * c_summation = (p.sum) ? &*p.sum : nullptr;
		T * c_scale = (p.scale) ? &*p.scale : nullptr;
		
		auto blksizes = p.blk->sizes();
		
		c_dbcsr_t_put_block(m_tensor_ptr, p.idx->data(), blksizes.data(), 
			p.blk->data(), c_summation, c_scale);
			
	}
	
	struct tensor_get_block_params {
		required<const index<N>,val> idx;
		required<const vec<int>,val> blk_size;
		required<bool,ref> found;
	};	
	block<N,T> get_block(tensor_get_block_params&& p) {
		
		//std::cout << std::endl;
		
		block<N,T> blk_out(*p.blk_size);
		
		c_dbcsr_t_get_block(m_tensor_ptr, p.idx->data(), p.blk_size->data(), 
			blk_out.data(), &*p.found);
			
		return blk_out;
			
	}
	
	struct tensor_get_stored_params {
		required<index<N>,ref> idx;
		required<int,ref> proc;
	};	
	void get_stored_coordinates(tensor_get_stored_params&& p) {
		c_dbcsr_t_get_stored_coordinates(m_tensor_ptr, N, p.idx->data(), &*p.proc);
	}
	
	struct tensor_mapping_info {
		optional<int,ref> ndim_nd;
		optional<int,ref> ndim1_2d;
		optional<int,ref> ndim2_2d;
		optional<vec<long long int>,ref> dims_2d_i8;
		optional<vec<int>,ref> dims_2d;
		optional<vec<int>,ref> dims_nd;
		optional<vec<int>,ref> dims1_2d;
		optional<vec<int>,ref> dims2_2d;
		optional<vec<int>,ref> map1_2d;
		optional<vec<int>,ref> map2_2d;
		optional<vec<int>,ref> map_nd;
		optional<int,ref> base;
		optional<bool,ref> col_major;
	};
	
private:
	void get_mapping_info_base(void* nd, tensor_mapping_info&& p) {
		
		int* c_ndim_nd = (p.ndim_nd) ? &*p.ndim_nd : nullptr;
		int* c_ndim1_2d = (p.ndim1_2d) ? &*p.ndim1_2d : nullptr;
		int* c_ndim2_2d = (p.ndim2_2d) ? &*p.ndim1_2d : nullptr;
		
		long long int* c_dims_2d_i8 = (p.dims_2d_i8) ? p.dims_2d_i8->data() : nullptr;
		int* c_dims_2d = (p.dims_2d) ? p.dims_2d->data() : nullptr;

#define set_array(name) \
	int** c_##name = (p.name) ? new int*() : nullptr; \
	int* name##_size = (p.name) ? new int() : nullptr;

		set_array(dims_nd)
		set_array(dims1_2d)
		set_array(dims2_2d)
		set_array(map1_2d)
		set_array(map2_2d)
		set_array(map_nd)
		
		int* c_base = (p.base) ? &*p.base : nullptr;
		bool* c_col_major = (p.col_major) ? &*p.col_major : nullptr;
		
		c_dbcsr_t_get_mapping_info(nd, c_ndim_nd, c_ndim1_2d, c_ndim2_2d, 
                        c_dims_2d_i8, c_dims_2d, c_dims_nd, dims_nd_size, 
                        c_dims1_2d, dims1_2d_size, c_dims2_2d, dims2_2d_size, 
                        c_map1_2d, map1_2d_size, c_map2_2d, map2_2d_size, 
                        c_map_nd, map_nd_size, c_base, c_col_major);
		
		
		auto make_vector = [](int** arr, int* size) 
		{
			vec<int> out(arr[0], arr[0] + *size); 
			return out;
		};
		
#define set_arg(name) if (p.name) {  \
		*p.name = make_vector(c_##name, name##_size); \
		free(c_##name[0]); \
		delete c_##name; \
		delete name##_size; \
		}
	  
		set_arg(dims_nd)
		set_arg(dims1_2d)
		set_arg(dims2_2d)
		set_arg(map1_2d)
		set_arg(map2_2d)
		set_arg(map_nd)
		
	}
		

public:	

	void get_mapping_info(tensor_mapping_info&& p) {	
		void* nd = nullptr;
		c_dbcsr_t_get_nd_index(m_tensor_ptr, &nd);
		get_mapping_info_base(nd, std::forward<tensor_mapping_info>(p));	
	}
	
	void get_blk_mapping_info(tensor_mapping_info&& p) {
		void* nd = nullptr;
		c_dbcsr_t_get_nd_index_blk(m_tensor_ptr, &nd);
		get_mapping_info_base(nd, std::forward<tensor_mapping_info>(p));
	}
	
	void scale(T alpha) {
		c_dbcsr_t_scale(m_tensor_ptr, alpha);
	}
	
	void set(T alpha) {
		c_dbcsr_t_set(m_tensor_ptr, alpha);
	}
	
	tensor<N,T>& operator+=(const tensor<N,T>& t) {
		
		bool sum = true;
		//c_dbcsr_t_create_template(t.m_tensor_ptr, &m_tensor_ptr, nullptr);
		c_dbcsr_t_copy(t.m_tensor_ptr, N, m_tensor_ptr, nullptr, &sum, nullptr, nullptr);
		return *this;
		
	}
	
	tensor<N,T>& operator=(const tensor<N,T>& rhs) {	
		
		if (&rhs != this) {
			
			this->destroy();
			
			std::cout << "&" << std::endl;
			m_comm = rhs.m_comm;
			
			if (m_tensor_ptr == nullptr) {
				c_dbcsr_t_create_template(rhs.m_tensor_ptr, &m_tensor_ptr, nullptr); 
			}
			
			c_dbcsr_t_copy(rhs.m_tensor_ptr, N, m_tensor_ptr, nullptr, nullptr, nullptr, nullptr);
			
		}
		
		return *this;
	}
		
	tensor<N,T>& operator=(tensor<N,T>&& rhs) {
		std::cout << "&&" << std::endl;
		
		if (&rhs != this) {
		
			std::cout << "Now..." << std::endl;
			this->destroy();
			m_comm = rhs.m_comm;
			m_tensor_ptr = rhs.m_tensor_ptr;
			rhs.m_tensor_ptr = nullptr;
			
		}
		
		return *this;
	}
	
	int proc(index<N> idx) {
		int p = -1;
		c_dbcsr_t_get_stored_coordinates(m_tensor_ptr, idx.data(), &p);
		return p;
	}
	
	void clear() {
		c_dbcsr_t_clear(m_tensor_ptr);
	}
	
	MPI_Comm comm() {
		return m_comm;
	}
	
	std::string name() const {
		std::string out;
		get_info({.name=out});
		std::cout << "NAME: " << out << std::endl;
		return out;
	}
	
	vec<int> nblks_tot() const {
		vec<int> out(N);
		get_info({.nblks_total=out});
		return out;
	}
	
	vec<int> nblks_loc() const {
		vec<int> out(N);
		get_info({.nblks_local=out});
		return out;
	}
	
	vec<int> nfull_tot() const {
		vec<int> out(N);
		get_info({.nfull_total=out});
		return out;
	}
	
	vec<int> nfull_loc() const {
		vec<int> out(N);
		get_info({.nfull_local=out});
		return out;
	}
	
	vec<int> pdims() const {
		vec<int> out(N); 
		get_info({.pdims=out});
		return out;
	}
	
	vec<int> my_ploc() const {
		vec<int> out(N);
		get_info({.my_ploc=out});
		return out;
	}
	
	vec<vec<int>> blks_local() const {
		vec<vec<int>> out;
		get_info({.blks_local=out});
		return out;
	}
	
	vec<vec<int>> proc_dist() const {
		vec<vec<int>> out;
		get_info({.proc_dist=out});
		return out;
	}
	
	vec<vec<int>> blk_size() const {
		vec<vec<int>> out;
		get_info({.blk_size=out});
		return out;
	}
	
	vec<vec<int>> blk_offset() const {
		vec<vec<int>> out;
		get_info({.blk_offset=out});
		return out;
	}
	
	int num_blocks() const {
		return c_dbcsr_t_get_num_blocks(m_tensor_ptr);
	}
	
	long long int num_blocks_total() const {
		return c_dbcsr_t_get_num_blocks_total(m_tensor_ptr);
	}
	
};

template <int N, typename T = double>
using ptensor = std::shared_ptr<tensor<N,T>>;

template <int N, typename T = double>
tensor<N,T> operator+(const tensor<N,T>& t1, const tensor<N,T>& t2) {
		
		std::cout << "+1" << std::endl;
		tensor<N,T> out(t1);
		std::cout << "+2" << std::endl;
		out += t2;
		
		return out;
		
}

template <int N, typename T = double>
tensor<N,T> operator*(const T alpha, const tensor<N,T>& t) {
	
	tensor<N,T> out = t;
	out.scale(alpha);
	return out;
	
}


template <int N, typename T = double>
class iterator {
private:

	void* m_iter_ptr;
	void* m_tensor_ptr;
	
	index<N> m_idx;
	int m_blk_n;
	int m_blk_p;
	vec<int> m_sizes;
	vec<int> m_offset;
	
public:

	iterator(tensor<N,T>& t_tensor) :
		m_iter_ptr(nullptr), m_tensor_ptr(t_tensor.data()),
		m_sizes(N), m_offset(N), m_blk_n(0), m_blk_p(0)
	{
		c_dbcsr_t_iterator_start(&m_iter_ptr, m_tensor_ptr);
	}
	
	~iterator() {
		
		c_dbcsr_t_iterator_stop(&m_iter_ptr);
		//also empty vectors
		
	}
	
	void next() {
		
		c_dbcsr_t_iterator_next_block(m_iter_ptr, m_idx.data(), &m_blk_n, &m_blk_p,
			m_sizes.data(), m_offset.data());
		
	}
	
	bool blocks_left() {	
		return c_dbcsr_t_iterator_blocks_left(m_iter_ptr);	
	}
	
	const index<N>& idx() {
		return m_idx;	
	}
	
	const vec<int>& sizes() {	
		return m_sizes;	
	}
	
	const vec<int>& offset() {		
		return m_offset;		
	}
	
	int blk_n() {		
		return m_blk_n;		
	}
	
	int blk_p() {	
		return m_blk_p;	
	}
		
};

template <int N, typename T = double>
struct tensor_copy {
		required<tensor<N,T>,ref>  	t_in;
		required<tensor<N,T>,ref>	t_out;
		optional<index<N>,val>		order;
		optional<bool,val>			sum, move_data;
		optional<int,val>			unit_nr;
};
template <int N, typename T = double>
void copy(tensor_copy<N,T>&& p) {
		
		int* forder = (p.order) ? p.order->data() : nullptr;
		bool* fsum = (p.sum) ? &*p.sum : nullptr;
		bool* fmove = (p.move_data) ? &*p.move_data : nullptr;
		int* funit = (p.unit_nr) ? &*p.unit_nr : nullptr;
		
		c_dbcsr_t_copy(p.t_in->m_tensor_ptr,N,p.t_out->m_tensor_ptr,forder,fsum,fmove,funit);
	
}

template <typename T, int N1, int N2, int N3>
struct contract_param {
	required<T,val>				alpha;
	required<tensor<N1,T>,ref>	t1;
	required<tensor<N2,T>,ref>  t2;
	required<T,val>				beta;
	required<tensor<N3,T>,ref>	t3;
	required<vec<int>,val>		con1, ncon1;
	required<vec<int>,val>		con2, ncon2;
	required<vec<int>,val>		map1, map2;
	optional<vec<vec<int>>,val>		b1, b2, b3;
	optional<double, val>		filter;
	optional<long long int, ref> flop;
	optional<bool, val>			move;
	optional<int, val>			unit_nr;
	optional<bool, val>			log;
};

template <typename T, int N1, int N2, int N3, typename ... Arg>
contract_param(T alpha, tensor<N1,T>& t1, tensor<N2,T>& t2, 
	T beta, tensor<N3,T>& t3,  Arg && ... args) -> contract_param<T,N1,N2,N3>;


template <int N1, int N2, int N3, typename T  = double>
void contract(contract_param<T,N1,N2,N3>&& p) {
	
	auto unfold_bounds = [](vec<vec<int>>& v) {
			int b_size = v.size();
			int* f_bounds = new int[2 * b_size];
			for (int j = 0; j != b_size; ++j) {
				for (int i = 0; i != 2; ++i) {
					f_bounds[i + j * 2] = v[j][i];
				}
			}
			return f_bounds;
	};
	
	int* f_b1 = (p.b1) ? unfold_bounds(*p.b1) : nullptr;
	int* f_b2 = (p.b2) ? unfold_bounds(*p.b2) : nullptr;
	int* f_b3 = (p.b3) ? unfold_bounds(*p.b3) : nullptr;
	
	std::cout << "In here..." << std::endl;
	
	double* f_filter = (p.filter) ? &*p.filter : nullptr;
	long long int* f_flop = (p.flop) ? &*p.flop : nullptr;
	bool* f_move = (p.move) ? &*p.move : nullptr;
	int* f_unit = (p.unit_nr) ? &*p.unit_nr : nullptr;
	bool* f_log = (p.log) ? &*p.log : nullptr;
	
	if (p.log) {
		if (*f_log) {
			std::cout << "LOGGING!!!" << std::endl;
			
		}}

	c_dbcsr_t_contract_r_dp(*p.alpha, p.t1->data(), p.t2->data(), 
						*p.beta, p.t3->data(), p.con1->data(), p.con1->size(),
						p.ncon1->data(), p.ncon1->size(),
						p.con2->data(), p.con2->size(), p.ncon2->data(), p.ncon2->size(),
						p.map1->data(), p.map1->size(), p.map2->data(), p.map2->size(), 
						f_b1, f_b2, f_b3, nullptr, nullptr, nullptr, nullptr, f_filter, 
						f_flop, f_move, f_unit, f_log);
       
     delete[] f_b1, f_b2, f_b3;               
	
}

static void eval(std::string str, std::vector<int>& con1, std::vector<int>& con2, 
	std::vector<int>& ncon1, std::vector<int>& ncon2,
	std::vector<int>& map1, std::vector<int>& map2) {
	
	std::cout << str << std::endl;
	
	std::vector<std::string> idxs(4);
	
	int i = 0;
	
	// Parsing input
	for (int ic = 0; ic != str.size(); ++ic) {
		
		char c = str[ic];
		
		if (c == ' ') continue;
			
		if (c == ',') {
			if (++i > 2) throw std::runtime_error("Invalid synatax: " + str); 
		} else if ((c == '-')) {
			
			
			if (str[++ic] == '>') {
				++i;
			} else {
				throw std::runtime_error("Invalid syntax: " + str);
			}
			
			if (i != 2) throw std::runtime_error("Invalid syntax."+str);
		
		} else {
			idxs[i].push_back(c);
		}
		
	}
	
	if (idxs[2].size() == 0) throw std::runtime_error("Implicit mode not implemented.");
	
	for (auto v : idxs) {
		std::cout << v << std::endl;
	}
	
	// evaluating input
	auto t1 = idxs[0];
	auto t2 = idxs[1];
	auto t3 = idxs[2];
	
	std::cout << "t3 map" << std::endl;
	
	if ((std::unique(t1.begin(), t1.end()) != t1.end()) || 
		(std::unique(t2.begin(), t2.end()) != t2.end()) ||
		(std::unique(t3.begin(), t3.end()) != t3.end()))
			throw std::runtime_error("Duplicate tensor indices: "+str);
			
	
	std::string scon, sncon1, sncon2;
	
	for (int i1 = 0; i1 != t1.size(); ++i1)  {
		auto c1 = t1[i1];
		for (int i2 = 0; i2 != t2.size(); ++i2) {
			auto c2 = t2[i2];
			if (c1 == c2) { 
				scon.push_back(c1);
				con1.push_back(i1);
				con2.push_back(i2);
			}
		}
	}
			
	std::cout << "To be contrcated: " << scon << std::endl;	
	std::cout << "Maps:" << std::endl;
	for (auto v : con1) std::cout << v << " ";
	std::cout << std::endl;
	for (auto v : con2) std::cout << v << " ";
	std::cout << std::endl;
			
	for (int i = 0; i != t1.size(); ++i) {
		auto found = std::find(scon.begin(), scon.end(), t1[i]);
		if (found == scon.end()) {
			sncon1.push_back(t1[i]);
			ncon1.push_back(i);
		}
	}
	
	for (int i = 0; i != t2.size(); ++i) {
		auto found = std::find(scon.begin(), scon.end(), t2[i]);
		if (found == scon.end()) {
			sncon2.push_back(t2[i]);
			ncon2.push_back(i);
		}
	}
	
	std::cout << "not con1: " << sncon1 << std::endl;
	std::cout << "not con2: " << sncon2 << std::endl;
	std::cout << "Maps:" << std::endl;
	for (auto v : ncon1) std::cout << v << " ";
	std::cout << std::endl;
	for (auto v : ncon2) std::cout << v << " ";
	std::cout << std::endl;
	
	if (ncon1.size() + ncon2.size() != t3.size()) throw std::runtime_error("Wrong tensor dimensions: "+str);
	
	for (int i = 0; i != t3.size(); ++i) {
		auto found1 = std::find(sncon1.begin(),sncon1.end(),t3[i]);
		if (found1 != sncon1.end()) {
			map1.push_back(i);
		}
		auto found2 = std::find(sncon2.begin(),sncon2.end(),t3[i]);
		if (found2 != sncon2.end()) {
			map2.push_back(i);
		}
	}
	
	std::cout << "Maps tensor 3" << std::endl;
	for (auto v : map1) std::cout << v << " ";
	std::cout << std::endl;
	for (auto v : map2) std::cout << v << " ";
	std::cout << std::endl;
	
	if (map1.size() + map2.size() != t3.size()) 
		throw std::runtime_error("Incompatible tensor dimensions: "+str);
	
}

template <typename T, int N1, int N2, int N3>
struct einsum_param {
	required<std::string,val>	x;
	required<tensor<N1,T>,ref>	t1;
	required<tensor<N2,T>,ref>  t2;
	required<tensor<N3,T>,ref> 	t3;
	required<T,val>				alpha = 1.0;
	required<T,val>				beta = T();
	optional<vec<vec<int>>,val>		b1, b2, b3;
	optional<double, val>		filter;
	optional<long long int, ref> flop;
	optional<bool, val>			move;
	optional<int, val>			unit_nr;
	optional<bool, val>			log;
};

template <int N, typename T = double>
dbcsr::tensor<N+1,T> add_dummy(tensor<N,T>& t) {
	
	std::cout << "REARRANGING!" << std::endl;
	
	// get maps of tensor
	vec<int> map1, map2; 
	
	t.get_mapping_info({.map1_2d = map1, .map2_2d = map2});
	
	vec<int> new_map1 = map1;
	new_map1.insert(new_map1.end(), map2.begin(), map2.end());
	
	vec<int> new_map2 = {N};
	
	pgrid<N+1> grid({.comm = t.comm()});
	
	auto blksizes = t.blk_size();
	blksizes.push_back({1});
	
	tensor<N+1,T> new_t({.name = t.name(), .pgridN = grid, .map1 = new_map1, .map2 = new_map2,
		.blk_sizes = blksizes});
	
	iterator<N> it(t);
	
	// parallelize this:
	
	std::cout << "NUMBER OF BLOCKS: " << std::endl;
	std::cout << t.num_blocks() << std::endl;
	
	while (it.blocks_left()) {
		
		it.next();
		
		index<N> idx = it.idx();
		index<N+1> new_idx;
		
		for (int i = 0; i != N; ++i) {
			new_idx[i] = idx[i];
		}
		
		new_idx[N] = 0;
		
		std::cout << "INDEX: " << std::endl;
		for (auto x  : new_idx) {
			std::cout << x << " ";
		} std::cout << std::endl;
		
		vec<int> blksz(N);
		
		for (int i = 0; i != N; ++i) {
			blksz[i] = blksizes[i][idx[i]];
		}
	
		bool found = false;
		auto blk = t.get_block({.idx = idx, .blk_size = blksz, .found = found});
		blksz.push_back(1);
		
		block<N+1,T> new_blk(blksz, blk.data());
		
		std::cout << "IDX: " << new_idx.size() << std::endl;
		std::cout << "BLK: " << new_blk.sizes().size() << std::endl;
		
		vec<vec<int>> res(N+1, vec<int>(1));
		for (int i = 0; i != N; ++i) {
			res[i][0] = idx[i];
		}
		res[N][0] = 0;
		
		new_t.reserve(res); 
				
		new_t.put_block({.idx = new_idx, .blk = new_blk});
		
	}
		
	print(new_t);
	
	grid.destroy();
	
	return new_t;
	
}

template <int N, typename T = double>
dbcsr::tensor<N-1,T> remove_dummy(tensor<N,T>& t, vec<int> map1, vec<int> map2) {
	
	std::cout << "Removing dummy dim." << std::endl;
	
	pgrid<N-1> grid({.comm = t.comm()});
	
	auto blksizes = t.blk_size();
	blksizes.pop_back();
	
	tensor<N-1,T> new_t({.name = t.name(), .pgridN = grid, .map1 = map1, .map2 = map2,
		.blk_sizes = blksizes});
	
	iterator<N> it(t);
	
	// parallelize this:
	while (it.blocks_left()) {
		
		it.next();
		
		index<N> idx = it.idx();
		index<N-1> new_idx;
		
		for (int i = 0; i != N-1; ++i) {
			new_idx[i] = idx[i];
		}
		
		vec<int> blksz(N);
		
		for (int i = 0; i != N-1; ++i) {
			blksz[i] = blksizes[i][new_idx[i]];
		}
		blksz[N-1] = 1;
		
		std::cout << "INDEX: " << std::endl;
		for (auto x  : new_idx) {
			std::cout << x << " ";
		} std::cout << std::endl;
	
		bool found = false;
		auto blk = t.get_block({.idx = idx, .blk_size = blksz, .found = found});
		auto sizes = blk.sizes();
		sizes.pop_back();
		
		const int M = N - 1;
		
		block<M,T> new_blk(sizes, blk.data());
		
		std::cout << "IDX: " << new_idx.size() << std::endl;
		std::cout << "BLK: " << new_blk.sizes().size() << std::endl;
		
		vec<vec<int>> res(N-1, vec<int>(1));
		for (int i = 0; i != N-1; ++i) {
			res[i][0] = idx[i];
		}
		
		new_t.reserve(res); 
				
		new_t.put_block({.idx = new_idx, .blk = new_blk});
		
	}
		
	print(new_t);
	
	grid.destroy();
	
	return new_t;
	
}

template <int N1, int N2, int N3, typename T  = double>
void einsum(einsum_param<T,N1,N2,N3>&& p) {
	
	vec<int> c1(0), c2(0), nc1(0), nc2(0), m1(0), m2(0);
	eval(*p.x, c1, c2, nc1, nc2, m1, m2);
	
	contract<N1,N2,N3,T>({p.alpha, p.t1, p.t2, p.beta, p.t3, c1, nc1,
		c2, nc2, m1, m2, p.b1, p.b2, p.b3, p.filter, p.flop, p.move, p.unit_nr, p.log});
		
	std::cout << "OUT" << std::endl;
		
}

template <int N, typename T = double>
T dot(tensor<N,T>& t1, tensor<N,T>& t2) {
	
	// dot product only for N = 2 at the moment
	assert(N == 2);
	
	iterator<N,T> it(t1);
	
	T sum = T();
	
	while (it.blocks_left()) {
		
		it.next();
		
		bool found = false;
		auto b1 = t1.get_block({.idx = it.idx(), .blk_size = it.sizes(), .found = found});
		auto b2 = t2.get_block({.idx = it.idx(), .blk_size = it.sizes(), .found = found});
		
		if (!found) continue;
		
		std::cout  << std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), T()) << std::endl;
		
		std::cout << "ELE: " << std::endl;
		for (int i = 0; i != b1.ntot(); ++i) { std::cout << b1(i) << " " << b2(i) << std::endl; }
		
		
		sum += std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), T());
		
		//std::cout << "SUM: " << std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), T()) << std::endl;
		
	}
	
	return sum;
		
}

template <int N, typename T>
void print(tensor<N,T>& t_in) {
	
	int myrank, mpi_size;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	auto blkszs = t_in.blk_size();
	
	iterator<N,T> iter(t_in);
    
    for (int p = 0; p != mpi_size; ++p) {
		if (myrank == p) {
			while (iter.blocks_left()) {
				  
				iter.next();
				bool found = false;
				auto idx = iter.idx();
				
				vec<int> sizes(N);
				
				for (int i = 0; i != N; ++i) {
					sizes[i] = blkszs[i][idx[i]];
				}
				
				auto blk = t_in.get_block({.idx = idx, .blk_size = sizes, .found = found});
				
				std::cout << myrank << ": [";
				for (int s = 0; s != sizes.size(); ++s) {
					std::cout << sizes[s];
					if (s != sizes.size() - 1) { std::cout << ","; }
				}
				std::cout << "] (";
				for (int s = 0; s != idx.size(); ++s) {
					std::cout << idx[s];
					if (s != sizes.size() - 1) { std::cout << ","; }
				}
				std::cout << ") {";
				
				
				for (int i = 0; i != blk.ntot(); ++i) {
					std::cout << blk(i) << " ";
				}
				std::cout << "}" << std::endl;
					
				
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}
                      

} // end namespace dbcsr

using v_ = vec<int>;

#endif
