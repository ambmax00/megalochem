#ifndef DBCSR_HPP
#define DBCSR_HPP

#define MAXDIM 4

#include "argo.hpp"
#include <dbcsr.h>
#include <dbcsr_tensor.h>
#include <optional>
#include <utility>
#include <vector>
#include <memory>

// debug
#include <iostream>

using namespace argo::literals;
using namespace std::placeholders;

template <typename T>
using vec = std::vector<T>;

template <typename T>
using opt = std::optional<T>;

auto nullopt = std::nullopt;


namespace dbcsr {

template <int N>
class dist;

template <int N>
class tensor;
	
void init(MPI_Comm comm = MPI_COMM_WORLD, int* io_unit = nullptr) {

	c_dbcsr_init_lib(comm, io_unit);
	
}

void finalize() {
	
	c_dbcsr_finalize_lib();
	
}
	
/* named parameters for pgrid */

const auto p_pgrid = argo::argspec(
            "comm"_arg,
            "map1_2d"_arg 	= nullopt,
            "map2_2d"_arg 	= nullopt,
            "nsplit"_arg 	= nullopt,
            "dimsplit"_arg 	= nullopt
);


template <int N>
class pgrid {
private:

	void* 			m_pgrid_ptr;
	vec<int> 		m_dims;
	MPI_Comm		m_comm;
	
	opt<vec<int>> 	m_map1;
	opt<vec<int>> 	m_map2;
	opt<int> 		m_dimsplit;
	opt<int> 		m_nsplit;
	
	pgrid init_pgrid(MPI_Comm comm, opt<vec<int>> map1, opt<vec<int>> map2, opt<int> nsplit, opt<int> dimsplit) {
		std::cout << "Assigning..." << std::endl;
		pgrid out(comm,map1,map2,nsplit,dimsplit);
		std::cout << "Done" << std::endl;
		return out;
	}
	
	pgrid(MPI_Comm comm, opt<vec<int>> map1, opt<vec<int>> map2, opt<int> nsplit, opt<int> dimsplit) 
		: m_comm(comm), m_dims(N), m_map1(map1), m_map2(map2), m_nsplit(nsplit), m_dimsplit(dimsplit),
		  m_pgrid_ptr(nullptr) {
		
		// handle "optional" arguments
		int* fmap1 = (map1) ? map1.value().data() : nullptr;
		int* fmap2 = (map2) ? map2.value().data() : nullptr;
		int map1size = (map1) ? map1.value().size() : 0;
		int map2size = (map2) ? map2.value().size() : 0;

		int* fnsplit = (nsplit) ? &*nsplit : nullptr;
		int* fdimsplit = (dimsplit) ? &*dimsplit : nullptr;
		
		MPI_Fint fmpi = MPI_Comm_c2f(comm);
		
		c_dbcsr_t_pgrid_create(&fmpi, m_dims.data(), N, &m_pgrid_ptr, fmap1, map1size, fmap2, map2size, fnsplit, fdimsplit);
		
	}
	
	friend dist<N>;
	
public:

	template <typename... Arg>
	pgrid(Arg ... args) : m_pgrid_ptr(nullptr) {
		
		const auto bind_pgrid = std::bind(std::mem_fn(&pgrid::init_pgrid), this, _1, _2, _3, _4, _5);
		const auto create = argo::adapt(p_pgrid, bind_pgrid);
		
		this->~pgrid();
		new (this) pgrid(create(args...));
		
	}
	
	vec<int> dims() {
		
		return m_dims;
		
	}
	
	void destroy() {
		
		if (m_pgrid_ptr != nullptr)
			c_dbcsr_t_pgrid_destroy(&m_pgrid_ptr, nullptr);;
		
	}
	
	~pgrid() {
		
		if (m_pgrid_ptr != nullptr) 
			c_dbcsr_t_pgrid_destroy(&m_pgrid_ptr, nullptr);
		
	}
		
	
	
};

vec<int> random_dist(int dist_size, int nbins)
{
    vec<int> dist(dist_size);

    for(int i=0; i < dist_size; i++)
        dist[i] = abs((nbins-i+1) % nbins);

    return std::move(dist);
}

const auto p_dist = argo::argspec(
            "pgrid"_arg,
            "map1_2d"_arg,
            "map2_2d"_arg,
            "nd_dist"_arg,
            "own_comm"_arg 	= nullopt
);


template <int N>
class dist {
private:

	void* 			m_dist_ptr;
	pgrid<N>		m_pgrid;
	vec<int>		m_map1;
	vec<int>		m_map2;
	vec<vec<int>>	m_nd_dist;
	opt<bool>		m_own_comm;
	
	
	dist init_dist(pgrid<N> p, vec<int> map1, vec<int> map2, vec<vec<int>> nd_dist, opt<bool> own_comm) {
		dist out(p,map1,map2,nd_dist,own_comm);
		return out;
	}
	
	dist(pgrid<N> p, vec<int> map1, vec<int> map2, vec<vec<int>> nd_dist, opt<bool> own_comm) 
		: m_pgrid(p), m_map1(map1), m_map2(map2), m_nd_dist(nd_dist), m_own_comm(own_comm),
		  m_dist_ptr(nullptr) {
		
		// handle "optional" arguments
		bool* f_own_comm = (own_comm) ? &*own_comm : nullptr;
		
		vec<int*> f_nd_dist(MAXDIM, nullptr);
		vec<int> f_nd_dist_size(MAXDIM, 0);
		
		for (int i = 0; i != N; ++i) {
			f_nd_dist[i] = nd_dist[i].data();
			f_nd_dist_size[i] = nd_dist[i].size();
		}
				
		c_dbcsr_t_distribution_new(m_dist_ptr, p.m_pgrid_ptr, 
				map1.data(), map1.size(),
                map2.data(), map2.size(), 
                f_nd_dist[0], f_nd_dist_size[0],
                f_nd_dist[1], f_nd_dist_size[1],
                f_nd_dist[2], f_nd_dist_size[2],
                f_nd_dist[3], f_nd_dist_size[3],
                f_own_comm);
		
	}
		
	
public:

	template <typename... Arg>
	dist(Arg ... args) : m_dist_ptr(nullptr) {
		
		const auto bind_dist = std::bind(std::mem_fn(&dist::init_dist), this, _1, _2, _3, _4, _5);
		const auto create = argo::adapt(p_dist, bind_dist);
		
		this->~dist();
		new (this) dist(create(args...));
		
	}
	
	void destroy() {
		
		if (m_dist_ptr != nullptr)
			c_dbcsr_t_distribution_destroy(&m_dist_ptr);
		
	}
	
	~dist() {
		
		if (m_dist_ptr != nullptr)
			c_dbcsr_t_distribution_destroy(&m_dist_ptr);
		
	}
		
	
	
};
	
	
} // end namespace dbcsr

#endif
