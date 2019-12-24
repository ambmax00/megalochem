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

#define STEP0(func, x, n, del) 
#define STEP1(func, x, n, del) func(x,n) STEP0(func, x, PLUS_ONE(n), del)
#define STEP2(func, x, n, del) func(x,n)unparen(del) STEP1(func, x, PLUS_ONE(n), del)
#define STEP3(func, x, n, del) func(x,n)unparen(del) STEP2(func, x, PLUS_ONE(n), del)
#define STEP4(func, x, n, del) func(x,n)unparen(del) STEP3(func, x, PLUS_ONE(n), del)
#define STEP5(func, x, n, del) func(x,n)unparen(del) STEP4(func, x, PLUS_ONE(n), del)
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

auto default_dist = 
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
	    
	    if (myrank == 0) {
	    
			std::cout << "Dist: " << std::endl;
			for (auto x : distvec) {
				std::cout << x << " ";
			} std::cout << std::endl;
			
		}
		
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

		int* fnsplit = (p.nsplit) ? &*p.nsplit : nullptr;
		int* fdimsplit = (p.dimsplit) ? &*p.dimsplit : nullptr;
		
		MPI_Fint fmpi = MPI_Comm_c2f(m_comm);
		c_dbcsr_t_pgrid_create(&fmpi, m_dims.data(), N, &m_pgrid_ptr, nullptr, fmap1, map1size, fmap2, map2size, fnsplit, fdimsplit);
		
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
	
	~pgrid() {
		
		destroy(true);
		
	}
		
	
	
};

vec<int> random_dist(int dist_size, int nbins)
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
	
	vec<int> m_map1;
	vec<int> m_map2;
	vec<vec<int>> m_nd_dists;
	MPI_Comm m_comm;
	
	template <int M, typename T>
	friend class tensor;
	
public:

	struct dist_params {
		required<pgrid<N>,ref>			pgridN;
		required<vec<int>,val>			map1, map2;
		required<vec<vec<int>>,val>		nd_dists;
		optional<bool,val>				own_comm;
	};

	dist(dist_params p) :
		m_dist_ptr(nullptr),
		m_map1(*p.map1),
		m_map2(*p.map2),
		m_nd_dists(*p.nd_dists)
	{
		
		void* pgrid_ptr = p.pgridN->m_pgrid_ptr;
		m_comm = p.pgridN->m_comm;
		
		bool* f_own_comm = (p.own_comm) ? &*p.own_comm : nullptr;
		
		vec<int*> f_dists(MAXDIM, nullptr);
		vec<int> f_dists_size(MAXDIM, 0);
		
		for (int i = 0; i != N; ++i) {
			f_dists[i] = p.nd_dists->at(i).data();
			f_dists_size[i] = p.nd_dists->at(i).size();
		}
		
		c_dbcsr_t_distribution_new(&m_dist_ptr, pgrid_ptr, 
								   m_map1.data(), m_map1.size(),
                                   m_map2.data(), m_map2.size(), 
                                   REPEAT(VARANDSIZE, f_dists, MAXDIM, (,), 0),
                                   f_own_comm);
                                   
	}
	
	dist(dist<N>& rhs) = delete;
		
		
	dist<N>& operator=(dist<N>& rhs) = delete;
	
	~dist() {
		
		std::cout << "DEST1START" << std::endl;
		
		if (m_dist_ptr != nullptr)
			c_dbcsr_t_distribution_destroy(&m_dist_ptr);
			
		m_dist_ptr = nullptr;
		
		std::cout << "DEST1END" << std::endl;
		
	}
	
	void destroy() {
		
		std::cout << "DEST2START" << std::endl;
		
		if (m_dist_ptr != nullptr) {
			c_dbcsr_t_distribution_destroy(&m_dist_ptr);
		}
			
		m_dist_ptr = nullptr;
		
		std::cout << "DEST2END" << std::endl;
		
	}
	
	
};


template <int N, typename T>
class block {
private:

	T* m_data;
	vec<int> m_sizes;
	int m_nfull;
	
public:

	block() : m_data(nullptr), m_sizes(N,0), m_nfull(0) {}
	
	block(const vec<int> sizes) : m_sizes(sizes) {
		
		if (sizes.size() != N) {
			throw std::runtime_error("Wrong block dimensions passed to constr.");
		}
		
		m_nfull = 1;
		for (int i = 0; i != N; ++i) m_nfull *= sizes[i];
		
		m_data = new T[m_nfull];
		
	}
	
	block(const block<N,T>& blk_in) : m_data(nullptr) {
		
		if (m_data != nullptr) 
			delete [] m_data;
			
		m_nfull = blk_in.m_nfull;
		m_data = new T[m_nfull];
		m_sizes = blk_in.m_sizes;
		
		memcpy ( m_data, blk_in.m_data, sizeof(T) * m_nfull );
		
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
		
		memcpy ( m_data, rhs.m_data, sizeof(T) * m_nfull );
		   
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
		
	~block() {
		
		if (m_data != nullptr)
			delete [] m_data;
		
	}
	
};		

template <int N, typename T>
class tensor {
protected:

	void* m_tensor_ptr;
	std::string m_name;
	vec<int> m_map1;
	vec<int> m_map2;
	vec<vec<int>> m_blk_sizes;
	vec<vec<int>> m_nzblks;
	MPI_Comm m_comm;
	
	int m_n_blocks;
	
	tensor(void* ptr) : m_tensor_ptr(ptr) {}
	
public:

	typedef T value_type;
	
	struct tensor_get_info_params {
		optional<vec<int>, ref> 	nblks_total, nfull_total, 
									nblks_local, nfull_local, 
									pdims, my_ploc;
		optional<vec<vec<int>>,ref> blks_local, proc_dist,
									blk_size, blk_offset;
		optional<std::string,ref>	name;
	};
	void get_info(tensor_get_info_params&& p) {
	
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
	
		char* name;
		int name_size;
		
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
		
		std::cout << "OUT" << std::endl;
		
		if(p.nblks_total) printvec("Total number of blocks:\n",*p.nblks_total);
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
         
        free(name); 
         
	}
	
	tensor() : m_tensor_ptr(nullptr) {}
	
	//copy constructor
	tensor(const tensor<N,T>& t): m_tensor_ptr(nullptr) {
		c_dbcsr_t_create_template(t.m_tensor_ptr, &this->m_tensor_ptr, nullptr);
		c_dbcsr_t_copy(t.m_tensor_ptr, N, this->m_tensor_ptr, nullptr, nullptr, nullptr, nullptr);
	}
		
	//move constructor
	tensor(tensor<N,T>&& t) {
		std::cout << "Moving" << std::endl;
		bool move = true;
		this->m_tensor_ptr = t.m_tensor_ptr;
		//c_dbcsr_t_create_template(t.m_tensor_ptr, &this->m_tensor_ptr, nullptr);
		t.m_tensor_ptr = nullptr;
	}
	
	struct tensor_params {
		required<std::string, val>	name;
		required<dist<N>, ref>  	distN;
		required<vec<int>,val>		map1,map2;
		required<vec<vec<int>>,ref>	blk_sizes;
	};
	tensor(tensor_params&& p) :
		m_tensor_ptr(nullptr),
		m_name(*p.name),
		m_map1(*p.map1),
		m_map2(*p.map2),
		m_blk_sizes(*p.blk_sizes)
	{
		
		void* dist_ptr = p.distN->m_dist_ptr;
		const char* f_name = m_name.c_str();
		int* c_data_type = nullptr;
		
		vec<int*> f_blks(MAXDIM, nullptr);
		vec<int> f_blks_size(MAXDIM, 0);
		
		m_n_blocks = 1;
		
		for (int i = 0; i != N; ++i) {
			//std::cout << i << " " << m_blk_sizes[i].size() << std::endl;
			f_blks[i] = m_blk_sizes[i].data();
			f_blks_size[i] = m_blk_sizes[i].size();
			m_n_blocks *= m_blk_sizes[i].size();
		}
		
		c_dbcsr_t_create_new(&m_tensor_ptr, f_name, dist_ptr, 
						m_map1.data(), m_map1.size(),
						m_map2.data(), m_map2.size(), c_data_type, 
						f_blks[0], f_blks_size[0],
						f_blks[1], f_blks_size[1],
						f_blks[2], f_blks_size[2],
						f_blks[3], f_blks_size[3]);
					
	}
	
	struct tensor_params2 {
		required<std::string, val>	name;
		required<pgrid<N>, ref>  	pgridN;
		required<vec<int>,val>		map1,map2;
		required<vec<vec<int>>,ref>	blk_sizes;
	};
	tensor(tensor_params2&& p) :
		m_tensor_ptr(nullptr),
		m_name(*p.name),
		m_map1(*p.map1),
		m_map2(*p.map2),
		m_blk_sizes(*p.blk_sizes)
	{
		
		vec<vec<int>> distvecs(N);
		
		for (int i = 0; i != N; ++i) {
			distvecs[i] = default_dist(p.blk_sizes->at(i).size(),
				p.pgridN->dims()[i], p.blk_sizes->at(i));
		}
		
		dist<N> d({.pgridN = *p.pgridN, .map1 = *p.map1, .map2 = *p.map2, 
				.nd_dists = distvecs});
		
		void* dist_ptr = d.m_dist_ptr;
		const char* f_name = m_name.c_str();
		int* c_data_type = nullptr;
		
		vec<int*> f_blks(MAXDIM, nullptr);
		vec<int> f_blks_size(MAXDIM, 0);
		
		m_n_blocks = 1;
		
		for (int i = 0; i != N; ++i) {
			//std::cout << i << " " << m_blk_sizes[i].size() << std::endl;
			f_blks[i] = m_blk_sizes[i].data();
			f_blks_size[i] = m_blk_sizes[i].size();
			m_n_blocks *= m_blk_sizes[i].size();
		}
		
		c_dbcsr_t_create_new(&m_tensor_ptr, f_name, dist_ptr, 
						m_map1.data(), m_map1.size(),
						m_map2.data(), m_map2.size(), c_data_type, 
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
		required<index<N>,val> idx;
		required<bool,ref> found;
	};	
	block<N,T> get_block(tensor_get_block_params&& p) {
	
		vec<int> loc_sizes;
		for (int i = 0; i != N; ++i) {
			int idx_i = p.idx->at(i);
			loc_sizes.push_back(m_blk_sizes[i][idx_i]);
			//std::cout << loc_sizes[i];
		}		
		
		//std::cout << std::endl;
		
		block<N,T> blk_out(loc_sizes);
		
		c_dbcsr_t_get_block(m_tensor_ptr, p.idx->data(), loc_sizes.data(), 
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
	
	void scale(T alpha) {
		c_dbcsr_t_scale(m_tensor_ptr, alpha);
	}
	
	void set(T alpha) {
		c_dbcsr_t_set(m_tensor_ptr, alpha);
	}
	
	tensor<N,T>& operator+=(tensor<N,T>& t) {
		
		bool sum = true;
		//c_dbcsr_t_create_template(t.m_tensor_ptr, &m_tensor_ptr, nullptr);
		c_dbcsr_t_copy(t.m_tensor_ptr, N, m_tensor_ptr, nullptr, &sum, nullptr, nullptr);
		return *this;
		
	}

	tensor<N,T> operator+(tensor<N,T>& t) {
		
		tensor<N,T> out(*this);
		return out += t;
		
	}
	
	tensor<N,T> operator=(const tensor<N,T>& t) {	
		std::cout << "&" << std::endl;
		tensor<N,T> out(std::forward<tensor<N,T>>(t));
		return out;
	}
		
	tensor<N,T> operator=(tensor<N,T>&& t) {
		std::cout << "&&" << std::endl;
		tensor<N,T> out(std::forward<tensor<N,T>>(t));
		return out;
	}
	
	int proc(index<N> idx) {
		int p = -1;
		c_dbcsr_t_get_stored_coordinates(m_tensor_ptr, idx.data(), &p);
		return p;
	}
	
	void clear() {
		c_dbcsr_t_clear(m_tensor_ptr);
	}
	
	vec<int> nblks_tot(){
		vec<int> out(N);
		get_info({.nblks_total=out});
		return out;
	}
	
	vec<int> nblks_loc(){
		vec<int> out(N);
		get_info({.nblks_local=out});
		return out;
	}
	
	vec<int> nfull_tot(){
		vec<int> out(N);
		get_info({.nfull_total=out});
		return out;
	}
	
	vec<int> nfull_loc(){
		vec<int> out(N);
		get_info({.nfull_local=out});
		return out;
	}
	
	vec<int> pdims() {
		vec<int> out(N); 
		get_info({.pdims=out});
		return out;
	}
	
	vec<int> my_ploc() {
		vec<int> out(N);
		get_info({.my_ploc=out});
		return out;
	}
	
	vec<vec<int>> blks_local() {
		vec<vec<int>> out;
		get_info({.blks_local=out});
		return out;
	}
	
	vec<vec<int>> proc_dist() {
		vec<vec<int>> out;
		get_info({.proc_dist=out});
		return out;
	}
	
	vec<vec<int>> blk_size() {
		vec<vec<int>> out;
		get_info({.blk_size=out});
		return out;
	}
	
	vec<vec<int>> blk_offset() {
		vec<vec<int>> out;
		get_info({.blk_offset=out});
		return out;
	}
	
	vec<int> map1() {
		
		std::cout << "Here" << std::endl;
		
		int *map1;
		int map1size;
		
		c_get_nd_index(m_tensor_ptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
		nullptr, nullptr, nullptr, &map1, &map1size, nullptr, nullptr, nullptr, nullptr, nullptr, 
		nullptr);
		
		vec<int> out(map1size);
		for (int n = 0; n != map1size; ++n) {
			out[n] = map1[n];
		}
		
		std::cout << "There" << std::endl;
		
		free(map1);
		
		std::cout << "End" << std::endl;
		
		return out;
		
	}
	
	vec<int> map2() {
		
		int *map2;
		int map2size;
		
		c_get_nd_index(m_tensor_ptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
		nullptr, nullptr, nullptr, nullptr, nullptr, &map2, &map2size, nullptr, nullptr, nullptr, 
		nullptr);
		
		vec<int> out(map2, map2 + map2size);
		
		free(map2);
		
		return out;
		
	}
	
};


template <int N, typename T>
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
	
	index<N> idx() {
		return m_idx;	
	}
	
	vec<int> sizes() {	
		return m_sizes;	
	}
	
	vec<int> offset() {		
		return m_offset;		
	}
	
	int blk_n() {		
		return m_blk_n;		
	}
	
	int blk_p() {	
		return m_blk_p;	
	}
		
};

template <int N, typename T>
struct tensor_copy {
		required<tensor<N,T>,ref>  	tensor_in;
		required<tensor<N,T>,ref>	tensor_out;
		optional<index<N>,val>		order;
		optional<bool,val>			summation, move_data;
		optional<int,val>			unit_nr;
	};
template <int N, typename T = double>
void copy(tensor_copy<N,T>&& p) {
		
		int* forder = (p.order) ? p.order->data() : nullptr;
		bool* fsum = (p.summation) ? &*p.summation : nullptr;
		bool* fmove = (p.move_data) ? &*p.move_data : nullptr;
		int* funit = (p.unit_nr) ? &*p.unit_nr : nullptr;
		
		c_dbcsr_t_copy(p.tensor_in->m_tensor_ptr,N,p.tensor_out->m_tensor_ptr,forder,fsum,fmove,funit);
	
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
	optional<vec<int>,val>		b1, b2, b3;
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
	
	int* f_b1 = (p.b1) ? p.b1->data() : nullptr;
	int* f_b2 = (p.b2) ? p.b2->data() : nullptr;
	int* f_b3 = (p.b3) ? p.b3->data() : nullptr;
	
	std::cout << "In here..." << std::endl;
	
	double* f_filter = (p.filter) ? &*p.filter : nullptr;
	long long int* f_flop = (p.flop) ? &*p.flop : nullptr;
	bool* f_move = (p.move) ? &*p.move : nullptr;
	int* f_unit = (p.unit_nr) ? &*p.unit_nr : nullptr;
	bool* f_log = (p.log) ? &*p.log : nullptr;

	c_dbcsr_t_contract_r_dp(*p.alpha, p.t1->data(), p.t2->data(), 
						*p.beta, p.t3->data(), p.con1->data(), p.con1->size(),
						p.ncon1->data(), p.ncon1->size(),
						p.con2->data(), p.con2->size(), p.ncon2->data(), p.ncon2->size(),
						p.map1->data(), p.map1->size(), p.map2->data(), p.map2->size(), 
						f_b1, f_b2, f_b3, nullptr, nullptr, nullptr, nullptr, f_filter, 
						f_flop, f_move, f_unit, f_log);
                       
	
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
	optional<vec<int>,val>		b1, b2, b3;
	optional<double, val>		filter;
	optional<long long int, ref> flop;
	optional<bool, val>			move;
	optional<int, val>			unit_nr;
	optional<bool, val>			log;
};


template <int N1, int N2, int N3, typename T  = double>
void einsum(einsum_param<T,N1,N2,N3>&& p) {
	
	vec<int> c1(0), c2(0), nc1(0), nc2(0), m1(0), m2(0);
	eval(*p.x, c1, c2, nc1, nc2, m1, m2);
	
	if (p.b1) std::cout << "THERE!" << std::endl;
	
	contract<N1,N2,N3,T>({p.alpha, p.t1, p.t2, p.beta, p.t3, c1, nc1,
		c2, nc2, m1, m2, p.b1, p.b2, p.b3, p.filter, p.flop, p.move, p.unit_nr, p.log});
		
	std::cout << "OUT" << std::endl;
		
}

template <int N, typename T>
void print(tensor<N,T>& t_in) {
	
	int myrank, mpi_size;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	iterator<N,T> iter(t_in);
    
    for (int p = 0; p != mpi_size; ++p) {
		if (myrank == p) {
			while (iter.blocks_left()) {
				  
				iter.next();
				bool found = false;
				auto idx = iter.idx();
				
				auto blk = t_in.get_block({.idx = idx, .found = found});
				auto sizes = blk.sizes();
				
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
