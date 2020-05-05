#ifndef DBCSR_FWD_HPP
#define DBCSR_FWD_HPP

#:include "dbcsr.fypp"

#include <vector>
#include <array>
#include <utility>
#include <string>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iostream>

#include "utils/params.hpp"

// ----------------------------------------
//               typedefs                 
// ----------------------------------------

template <typename T>
using svector = std::shared_ptr<std::vector<T>>;

template <typename T>
using vec = std::vector<T>;

template <typename T, int N>
using arr = std::array<T,N>;

template <typename T, int N>
using arrvec = std::array<std::vector<T>,N>;

namespace dbcsr {

template <int N>
using index = std::array<int,N>;

typedef index<2> idx2;
typedef index<3> idx3;
typedef index<4> idx4;

// ----------------------------------------
//       structs for SFINAE purposes
// ----------------------------------------

template <int N>
struct is_valid_dim {
    const static bool value = (N >= 2 && N <= ${MAXDIM}$);
};

template <typename T> 
struct is_valid_type {
    const static bool value = (
	std::is_same<double,T>::value || 
	std::is_same<float,T>::value ||
	std::is_same<double _Complex,T>::value ||
	std::is_same<float _Complex,T>::value);
};

template <typename T>
struct dbcsr_type {
	static const int value =  
			dbcsr_type_real_4 * std::is_same<T,float>::value
			+ dbcsr_type_real_8 * std::is_same<T,double>::value 
			+ dbcsr_type_complex_4 * std::is_same<T,float _Complex>::value 
			+ dbcsr_type_complex_8 * std::is_same<T,double _Complex>::value;
};

// ---------------------------------------------------------------------
//                     forward declarations (matrix) 
// ---------------------------------------------------------------------

class group;
class dist;
template <typename T = double>
class iterator;
template <typename T, typename = typename std::enable_if<is_valid_type<T>::value>::type>
class matrix;

template <typename T = double>
class multiply;

// ---------------------------------------------------------------------
//                     forward declarations (tensor) 
// ---------------------------------------------------------------------

template <int N, typename = typename std::enable_if<is_valid_dim<N>::value>::type>
class pgrid;
template <int N, typename = typename std::enable_if<is_valid_dim<N>::value>::type>
class dist_t;
template <int N, typename T = double>
class iterator_t;
template <int N, typename T, typename = typename std::enable_if<
is_valid_dim<N>::value && is_valid_type<T>::value>::type>
class tensor;
template <int N, typename T = double>
using stensor = std::shared_ptr<tensor<N,T>>;

template <int N1, int N2, int N3, typename T = double>
class contract;
template <int N1, typename T = double>
class copy;

inline double block_threshold = 1e-16;

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
		
		return distvec;
		
};

// ---------------------------------------------------------------------
//                library functions              
// ---------------------------------------------------------------------

inline void init(MPI_Comm comm = MPI_COMM_WORLD, int* io_unit = nullptr) {
	c_dbcsr_init_lib(comm, io_unit);
};

inline void finalize() {
	c_dbcsr_finalize_lib();
};

inline void print_statistics(const bool print_timers = false) {
	c_dbcsr_print_statistics(&print_timers, nullptr);
}

/* ---------------------------------------------------------------------
/					BLOCK: multi-dimensional array (wrapper) class
/ ----------------------------------------------------------------------
*/

template <int N, typename T = double>
class block {
private:

	T* m_data;
	std::array<int,N> m_size;
	int m_nfull;
	bool m_own = true;
	
public:

	block() : m_data(nullptr), m_size(N,0), m_nfull(0) {}
    
	block(const std::array<int,N> sizes, T* ptr = nullptr) : m_size(sizes) {
		
		m_nfull = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());
		
		if (ptr) {
			m_data = ptr; 
			m_own = false;
		} else {
			m_data = new T[m_nfull]();
		}
		
	}
	
	block(const block<N,T>& blk_in) : m_data(nullptr) {
		
		m_nfull = blk_in.m_nfull;
		m_data = new T[m_nfull];
		m_size = blk_in.m_size;
		
		std::copy(blk_in.m_data, blk_in.m_data + m_nfull, m_data);
		
	}
	
	block(block<N,T>&& blk_in) : m_data(nullptr) {
			
		m_nfull = blk_in.m_nfull;
		m_data = blk_in.m_data;
		m_size = blk_in.m_size;
		
		blk_in.m_nfull = 0;
		blk_in.m_data = nullptr;
		
	}
	
	block& operator=(const block & rhs)
	{
		if(this == &rhs)
		   return *this;
		   
		if (m_data != nullptr && m_own)  
			delete [] m_data;
			
		m_nfull = rhs.m_nfull;
		m_data = new T[m_nfull];
		m_size = rhs.m_size;
		
		std::copy(rhs.m_data, rhs.m_data + m_nfull, m_data);
		   
		return *this;
	}
	
	block& operator=(block && rhs)
	{
		if(this == &rhs)
		   return *this;
		   
		if (m_data != nullptr)  
			delete [] m_data;
			
		m_nfull = rhs.m_nfull;
		m_data = rhs.m_data;
		m_size = rhs.m_size;
		
		rhs.m_data = nullptr;
		   
		return *this;
	}
	
	void print() {
		
		std::cout << "(";
		for (int i = 0; i != N; ++i) {
			std::cout << m_size[i];
			if (i != N - 1) std::cout << ",";
		} std::cout << ") [";
		
		for (int i = 0; i != m_nfull; ++i) {
			std::cout << m_data[i];
			if (i != m_nfull - 1) std::cout << " ";
		} std::cout << "]" << std::endl;
		
	}
	 
	int ntot() { return m_nfull; }
	const index<N> size() { return m_size; }
	
	T* data() {
		return m_data;
	}
	
	T& operator[] (int id) { 
		//std::cout << id << std::endl;
		return m_data[id]; 
	}

#:def make_idx(n)
    #:if n == 0
        i_0
    #:else
        + i_${n}$
        #:for i in range(0,n)
            * m_size[${i}$]
        #:endfor
    #:endif
#:enddef

#:for idim in range(2,MAXDIM+1)
    template <typename D = T, int M = N>
    typename std::enable_if<M == ${idim}$, D&>::type
    operator()(
    #:for n in range(0,idim)
        int i_${n}$
        #:if n < idim-1
        ,
        #:endif
    #:endfor
    ) { return m_data[
    #:for n in range(0,idim)
        ${make_idx(n)}$
    #:endfor
        ];
    }
#:endfor
	
	void fill_rand(T a, T b) {
		
		std::random_device rd; 
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(a,b);
		
		for (int i = 0; i != m_nfull; ++i) {
			m_data[i] = dis(gen);
			//std::cout << m_data[i] << std::endl;
		}
		
	} 
	
	T norm() {
		
		T out = 0;
		for (int i = 0; i != m_nfull; ++i) {
			out += pow(m_data[i],2);
		}
		
		return sqrt(out);
		
	}
	
	~block() {
		
		if (m_data != nullptr && m_own) {
			delete [] m_data;
			m_data = nullptr;
		}
		
	}
	
};

} // end namespace

#endif
