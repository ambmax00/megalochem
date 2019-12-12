#ifndef DBCSR_HPP
#define DBCSR_HPP

#define MAXDIM 4
#define MAXIDX 3

#define BOOST_PARAMETER_MAX_ARITY 20

#define really_unparen(...) __VA_ARGS__
#define invoke(expr) expr
#define unparen(args) invoke(really_unparen args)
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
#include <boost/parameter.hpp>
#include <optional>
#include <utility>
#include <string>
#include <numeric>
#include <vector>
#include <memory>
#include <stdexcept>
#include <random>

#include "params/params.hpp"

// debug
#include <iostream>
#include <typeinfo>
#include <boost/mpl/placeholders.hpp>
#include <thread>
#include <unistd.h>


template <typename T>
using vec = std::vector<T>;

template <typename T>
using opt = std::optional<T>;

auto nullopt = std::nullopt;

namespace parameter = boost::parameter;
using boost::mpl::placeholders::_;

BOOST_PARAMETER_NAME(pgrid)
BOOST_PARAMETER_NAME(dist)
BOOST_PARAMETER_NAME(map12)
BOOST_PARAMETER_NAME(map22)
BOOST_PARAMETER_NAME(comm)
BOOST_PARAMETER_NAME(nsplit)
BOOST_PARAMETER_NAME(dimsplit)
BOOST_PARAMETER_NAME(nd_dists)
BOOST_PARAMETER_NAME(own_comm)
BOOST_PARAMETER_NAME(blk_sizes)
BOOST_PARAMETER_NAME(name)
BOOST_PARAMETER_NAME(blk)
BOOST_PARAMETER_NAME(idx)
BOOST_PARAMETER_NAME(sum)
BOOST_PARAMETER_NAME(scale)
BOOST_PARAMETER_NAME(found)
BOOST_PARAMETER_NAME(nblks_total)
BOOST_PARAMETER_NAME(nfull_total)
BOOST_PARAMETER_NAME(nblks_local)
BOOST_PARAMETER_NAME(nfull_local)
BOOST_PARAMETER_NAME(pdims)
BOOST_PARAMETER_NAME(my_ploc)
BOOST_PARAMETER_NAME(blks_local)
BOOST_PARAMETER_NAME(proc_dist)
BOOST_PARAMETER_NAME(blk_offset)
BOOST_PARAMETER_NAME(NIL)

namespace dbcsr {

template <int N>
using index = std::array<int,N>;

typedef index<2> idx2;
typedef index<3> idx3;
typedef index<4> idx4;
	
static void init(MPI_Comm comm = MPI_COMM_WORLD, int* io_unit = nullptr) {

	c_dbcsr_init_lib(comm, io_unit);
	
};

static void finalize() {
	
	c_dbcsr_finalize_lib();
	
};

template <int N>
class dist_impl;

template <int N, typename T>
class tensor_impl;

template <int N, typename T>
class iterator;

template <int N>
class pgrid_impl {
protected:

	void* 			m_pgrid_ptr;
	vec<int> 		m_dims;
	MPI_Comm		m_comm;
	
	std::optional<vec<int>> 	m_map12;
	std::optional<vec<int>> 	m_map22;
	std::optional<int> 			m_dimsplit;
	std::optional<int> 			m_nsplit;
	
	template <int M>
	friend class dist_impl;
	
public:

	template <typename ArgPack>
	pgrid_impl(ArgPack const& args) : 
		m_pgrid_ptr(nullptr),
		m_dims(N),
		m_comm(args[_comm]), 
		m_map12(args[_map12 | nullopt]),
		m_map22(args[_map22 | nullopt]),
		m_nsplit(args[_nsplit | nullopt]),
		m_dimsplit(args[_dimsplit | nullopt])	
	{
	
		// handle "optional" arguments
		int* fmap1 = (m_map12) ? m_map12.value().data() : nullptr;
		int* fmap2 = (m_map22) ? m_map22.value().data() : nullptr;
		int map1size = (m_map12) ? m_map12.value().size() : 0;
		int map2size = (m_map22) ? m_map22.value().size() : 0;

		int* fnsplit = (m_nsplit) ? &*m_nsplit : nullptr;
		int* fdimsplit = (m_dimsplit) ? &*m_dimsplit : nullptr;
		
		MPI_Fint fmpi = MPI_Comm_c2f(m_comm);
		c_dbcsr_t_pgrid_create(&fmpi, m_dims.data(), N, &m_pgrid_ptr, nullptr, fmap1, map1size, fmap2, map2size, fnsplit, fdimsplit);
		
		
	}
	
	vec<int> dims() {
		
		return m_dims;
		
	}
	
	void destroy() {
		
		if (m_pgrid_ptr != nullptr)
			c_dbcsr_t_pgrid_destroy(&m_pgrid_ptr, nullptr);
			
		m_pgrid_ptr = nullptr;
		
	}
	
	~pgrid_impl() {
		
		if (m_pgrid_ptr != nullptr) {
			c_dbcsr_t_pgrid_destroy(&m_pgrid_ptr, nullptr);
		}
			
		m_pgrid_ptr = nullptr;
		
	}
		
	
	
};

template <int N>
class pgrid : public pgrid_impl<N> {
public:

		BOOST_PARAMETER_CONSTRUCTOR(
			pgrid, (pgrid_impl<N>), tag,
			(required 
				(comm,		(MPI_Comm))
			) 
			(optional 
				(map12,		(std::optional<vec<int>>))
				(map22,		(std::optional<vec<int>>))
				(nsplit,	(std::optional<int>))
				(dimsplit,	(std::optional<int>))
			)
		)
		
		
				
};			
	

vec<int> random_dist(int dist_size, int nbins)
{
    vec<int> d(dist_size);

    for(int i=0; i < dist_size; i++)
        d[i] = abs((nbins-i+1) % nbins);

    return std::move(d);
};


template <int N>
class dist_impl {
private:

	void* m_dist_ptr;
	//pgrid<4> m_pgrid;
	vec<int> m_map12;
	vec<int> m_map22;
	vec<vec<int>> m_nd_dists;
	std::optional<bool> m_own_comm;
	MPI_Comm m_comm;
	
	template <int M, typename U>
	friend class tensor_impl;
	
public:

	template <typename ArgPack>
	dist_impl(ArgPack const& args) :
		m_dist_ptr(nullptr),
		m_map12(args[_map12]),
		m_map22(args[_map22]),
		m_nd_dists(args[_nd_dists]),
		m_own_comm(args[_own_comm | nullopt])
	{
		
		void* pgrid_ptr = args[_pgrid].m_pgrid_ptr;
		m_comm = args[_pgrid].m_comm;
		
		bool* f_own_comm = (m_own_comm) ? &*m_own_comm : nullptr;
		
		vec<int*> f_dists(MAXDIM, nullptr);
		vec<int> f_dists_size(MAXDIM, 0);
		
		for (int i = 0; i != N; ++i) {
			f_dists[i] = m_nd_dists[i].data();
			f_dists_size[i] = m_nd_dists[i].size();
		}
		
		c_dbcsr_t_distribution_new(&m_dist_ptr, pgrid_ptr, 
								   m_map12.data(), m_map12.size(),
                                   m_map22.data(), m_map22.size(), 
                                   REPEAT(VARANDSIZE, f_dists, MAXDIM, (,), 0),
                                   f_own_comm);
                                   
	}
	
	~dist_impl() {
		
		std::cout << "DEST1START" << std::endl;
		
		if (m_dist_ptr != nullptr)
			c_dbcsr_t_distribution_destroy(&m_dist_ptr);
			
		m_dist_ptr = nullptr;
		
		std::cout << "DEST1END" << std::endl;
		
	}
	
	void destroy() {
		
		std::cout << "DEST2START" << std::endl;
		
		if (m_dist_ptr != nullptr)
			c_dbcsr_t_distribution_destroy(&m_dist_ptr);
			
		m_dist_ptr = nullptr;
		
		std::cout << "DEST2END" << std::endl;
		
	}
	
	/* TO DO:
	 * 
	 * reserve() {}
	 * begin? end?*/
	
};

template <int N>
class dist : public dist_impl<N> {
public:

		BOOST_PARAMETER_CONSTRUCTOR(
			dist, (dist_impl<N>), tag,
			(required 
				(pgrid,			(pgrid<N>))
				(map12,			(vec<int>))
				(map22, 		(vec<int>))
				(nd_dists,		(vec<vec<int>>))
			) 
			(optional 
				(own_comm,		(std::optional<bool>))
			)
		)
				
};	

/*
template <int N, typename T = double>
class iterator {
 
 // gets info from iter itself, doesnt need tensor
 
};
*/


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
	
	T& operator()(int i, int j) {
		static_assert(N == 2);
		return m_data[i + j * m_sizes[0]];
	}
	
	T& operator()(int i, int j, int k) {
		static_assert(N == 3);
		return m_data[i + j*m_sizes[0] + k*m_sizes[0]*m_sizes[1]];
	}
	
	T& operator()(int i, int j, int k, int l) {
		static_assert(N == 4);
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
class tensor_impl {
protected:

	void* m_tensor_ptr;
	std::string m_name;
	vec<int> m_map12;
	vec<int> m_map22;
	vec<vec<int>> m_blk_sizes;
	vec<vec<int>> m_nzblks;
	MPI_Comm m_comm;
	
	int m_n_blocks;
	
	template <int M, typename U>
	friend class iterator;
	
	
	BOOST_PARAMETER_MEMBER_FUNCTION(
		(void), get_info, tag,
		(required (NIL,*))
		(optional (out(nblks_total),*,nullopt)
				  (out(nfull_total),*,nullopt)
				  (out(nblks_local),*,nullopt)
				  (out(nfull_local),*,nullopt)
				  (out(pdims),*,nullopt)
				  (my_ploc,*,nullopt)
				  (blks_local,*,nullopt)
				  (proc_dist,*,nullopt)
				  (blk_sizes,*,nullopt)
				  (out(blk_offset),*,nullopt)
				  (out(name),*,nullopt)
		)
	){
		
	
		
		/*
		auto reserve = [] (std::optional<vec<int>>& o) -> int* {
			
			int* ptr = nullptr;
			
			if (o) {
				o->reserve(N);
				ptr = o->data();
			} 
			
			return ptr;
		
		};
		
		int *nblks_total_p, *nfull_total_p, *nblks_local_p, *nfull_local_p, *pdims_p, *my_ploc_p;
		
		nblks_total_p = reserve(nblks_total);
		nfull_total_p = reserve(nfull_total);
		nblocks_local_p = reserve(nblocks_local);
		nfull_local_p = reserve(nfull_local);
		pdims_p = reserve(pdims);
		my_ploc_p = reserve(my_ploc);
		
		vec<int*> blks_local_v(MAX_DIM,nullptr), proc_dist_v(MAX_DIM,nullptr);
			blk_size_v(MAX_DIM,nullptr), blk_offset_v(MAX_DIM,nullptr);
		
		vec<int> blks_local_v_size(MAX_DIM,0), proc_dist_v_size(MAX_DIM,0);
			blk_size_v_size(MAX_DIM,0), blk_offset_v_size(MAX_DIM,0);
	
	/*
	   void c_dbcsr_t_get_info(m_tensor_ptr, N, 
							   nblks_total_p,
                               nfull_total_p,
                               nblks_local_p,
                               nfull_local_p,
                               pdims_p, my_ploc_p, 
                               blks_local_p
                               REPEAT(VARANDSIZE,blks_local_v,MAXDIM,(,),0),
                               ${extern_alloc_varlist_and_size("c_proc_dist")}$, 
                               ${extern_alloc_varlist_and_size("c_blk_size")}$, 
                               ${extern_alloc_varlist_and_size("c_blk_offset")}$, 
                               void** c_distribution, 
                               char** name, int* name_size,
                               int* data_type);
                               */
                               
	   }
	
public:

	template <typename ArgPack>
	tensor_impl(ArgPack const& args) :
		m_tensor_ptr(nullptr),
		m_name(args[_name]),
		m_map12(args[_map12]),
		m_map22(args[_map22]),
		m_blk_sizes(args[_blk_sizes])
	{
		
		m_comm = args[_dist].m_comm;
		void* dist_ptr = args[_dist].m_dist_ptr;
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
						m_map12.data(), m_map12.size(),
						m_map22.data(), m_map22.size(), c_data_type, 
						f_blks[0], f_blks_size[0],
						f_blks[1], f_blks_size[1],
						f_blks[2], f_blks_size[2],
						f_blks[3], f_blks_size[3]);
					
	}
		
	~tensor_impl() {
		
		destroy();
		
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
	
		vec<int*> f_blks(MAXDIM, nullptr);
		
		for (int i = 0; i != N; ++i) {
			f_blks[i] = nzblks[i].data();
		}
		
		c_dbcsr_t_reserve_blocks_index(m_tensor_ptr, nzblks[0].size(), 
			f_blks[0], f_blks[1], f_blks[2], f_blks[3]);
			
	}
	
	
	BOOST_PARAMETER_MEMBER_FUNCTION(
		(void), put_block, tag, 
		(required (idx,(index<N>))
				  (in_out(blk),*)
		)
		(optional (sum,(std::optional<bool>),nullopt)
				  (scale,(std::optional<T>),nullopt)
		)
	){
		
		bool * c_summation = (sum) ? &*sum : nullptr;
		T * c_scale = (scale) ? &*scale : nullptr;
		
		auto blks = blk.sizes();
		
		c_dbcsr_t_put_block(m_tensor_ptr, idx.data(), blks.data(), blk.data(),
			c_summation, c_scale);
			
	}
	
	BOOST_PARAMETER_MEMBER_FUNCTION(
		(block<N,T>), get_block, tag, 
		(required (idx,(index<N>))
				  (found,*)
		)
	){
		
		vec<int> loc_sizes;
		for (int i = 0; i != N; ++i) {
			int idx_i = idx[i];
			loc_sizes.push_back(m_blk_sizes[i][idx_i]);
			std::cout << loc_sizes[i];
		}		
		
		std::cout << std::endl;
		
		block<N,T> blk_out(loc_sizes);
		
		c_dbcsr_t_get_block(m_tensor_ptr, idx.data(), loc_sizes.data(), 
			blk_out.data(), &found);
			
		return blk_out;
			
	}
	
	int proc(index<N> idx) {
		int p = -1;
		c_dbcsr_t_get_stored_coordinates(m_tensor_ptr, idx.data(), &p);
		return p;
	}
	
	void clear() {
		c_dbcsr_t_clear(m_tensor_ptr);
	}
	
	
	
	
	
	
	
	
	
	
	
	
};

template <int N, typename T>
class tensor : public tensor_impl<N,T> {
public:

		BOOST_PARAMETER_CONSTRUCTOR(
			tensor, (tensor_impl<N,T>), tag,
			(required 
				(name,			(std::string))
				(dist,			(dist<N>))
				(map12, 		(vec<int>))
				(map22, 		(vec<int>))
				(blk_sizes,		(vec<vec<int>>))
			) 
		)
				
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
		m_iter_ptr(nullptr), m_tensor_ptr(t_tensor.m_tensor_ptr),
		m_sizes(N), m_offset(N)
	{
		c_dbcsr_t_iterator_start(&m_iter_ptr, m_tensor_ptr);
	}
	
	~iterator() {
		
		c_dbcsr_t_iterator_stop(&m_iter_ptr);
		
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

} // end namespace dbcsr


#endif
