#ifndef DBCSR_HPP
#define DBCSR_HPP

#:set MAXDIM = 4 

#:def make_param(structname, params)
    private:
    #:for name,type,ro,rv in params
        ${ro}$<${type}$,${rv}$> c_${name}$; 
    #:endfor
    public:
    #:for name,type,ro,rv in params
        inline ${structname}$& ${name}$(${ro}$<${type}$,${rv}$> i_${name}$)
        {
            c_${name}$ = std::move(i_${name}$);
            return *this;
        }
    #:endfor
#:enddef

#:def make_struct(structname, friend, params)
struct ${structname}$ {
    
    ${make_param(structname,params)}$
    
    ${structname}$() = default;
    friend class ${friend}$;
};
#:enddef 

#:def datasize(name, nmin, nmax)
#!    expand into list of elements "name[0].data(), name[0].size(), ...
$:    ", ".join([name + "[" + str(i) + "].data(), " + name + "[" + str(i) + "].size()" for i in range(nmin, nmax)])
#:enddef

#:def datasizeptr(name, nmin, nmax)
#!    expand into list of elements "name->at(0).data(), name->at(0).size(), ...
$:    ", ".join([name + "->at(" + str(i) + ").data(), " + name + "->at(" + str(i) + ").size()" for i in range(nmin, nmax)])
#:enddef

#:def datasize0(nmax)
#!    repeat nullptr, 0, ...
$:    ", ".join(["nullptr, 0" for i in range(0, nmax)])
#:enddef

#:def repeat(name,nmax,end='')
#! repeat name, name, ...
#:if nmax > 0
$:     ", ".join([name for i in range(0,nmax)]) + end
#:endif
#:enddef

#:def repeatvar(name,nmin,nmax,suffix,end='')
#! repeat name0, name1, ...
#:if nmax != nmin
$:    ", ".join([name + str(i) + suffix for i in range(nmin, nmax+1)]) + end
#:endif
#:enddef  

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

class group;

template <int N, typename = typename std::enable_if<is_valid_dim<N>::value>::type>
class pgrid;

class dist;

template <int N, typename = typename std::enable_if<is_valid_dim<N>::value>::type>
class dist_t;

template <typename T, typename = typename std::enable_if<is_valid_type<T>::value>::type>
class matrix;

template <typename T = double>
class iterator;

template <int N, typename T, typename = typename std::enable_if<
is_valid_dim<N>::value && is_valid_type<T>::value>::type>
class tensor;

template <int N, typename T = double>
using stensor = std::shared_ptr<tensor<N,T>>;

template <int N, typename T = double>
class iterator_t;

template <typename T = double>
class multiply;

template <int N1, int N2, int N3, typename T = double>
class contract;

template <int N1, typename T = double>
class copy;

static const int dbcsr_type_real_4 = 1;
static const int dbcsr_type_real_8 = 3;
static const int dbcsr_type_complex_4 = 5;
static const int dbcsr_type_complex_8 = 7;

template <typename T>
struct dbcsr_type {
	static const int value =  
			dbcsr_type_real_4 * std::is_same<T,float>::value
			+ dbcsr_type_real_8 * std::is_same<T,double>::value 
			+ dbcsr_type_complex_4 * std::is_same<T,float _Complex>::value 
			+ dbcsr_type_complex_8 * std::is_same<T,double _Complex>::value;
};
	
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
		
		return distvec;
		
};

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
			
	//new constructor with raw pointer
	
	~block() {
		
		if (m_data != nullptr && m_own) {
			delete [] m_data;
			m_data = nullptr;
		}
		
	}
	
};

class world {
private:

    MPI_Comm m_comm;
    MPI_Comm m_group;
    int m_rank;
    int m_size;
    std::array<int,2> m_dims;
    std::array<int,2> m_coord;
    
    friend class dist;
    
public:

    world(MPI_Comm comm) : m_comm(comm), m_group(MPI_COMM_NULL) {
        
        MPI_Comm_rank(m_comm, &m_rank);
        MPI_Comm_size(m_comm, &m_size);
        int dims[2] = {0};
        MPI_Dims_create(m_size, 2, dims);
        int periods[2] = {1};
        int reorder = 0;
        MPI_Comm m_group;
        MPI_Cart_create(m_comm, 2, dims, periods, reorder, &m_group);
        
        int coord[2];
        MPI_Cart_coords(m_group, m_rank, 2, coord);
        
        m_dims[0] = dims[0];
        m_dims[1] = dims[1];
        m_coord[0] = coord[0];
        m_coord[1] = coord[1];
        
    }
    
    void destroy(bool keep_comm = false) {
        if (m_group != MPI_COMM_NULL && !keep_comm) 
            MPI_Comm_free(&m_group);
        m_comm = MPI_COMM_NULL;
    }
    
    ~world() { destroy(true); }
    
    MPI_Comm comm() { return m_comm; }
    MPI_Comm group() { return m_group; }
    
    int rank() { return m_rank; }
    int size() { return m_size; }
    
    std::array<int,2> dims() { return m_dims; }
    std::array<int,2> coord() { return m_coord; }
    
};

class dist {
private:

    void* m_dist_ptr;
    MPI_Comm m_comm;
    
public:

    #:set list = [ &
        ['set_world', 'world', 'required', 'ref'],&
        ['row_dist', 'vec<int>', 'required', 'ref'],&
        ['col_dist', 'vec<int>', 'required', 'ref']]
    ${make_struct(structname='create',friend='dist',params=list)}$

    dist(const create& p) {
        m_comm = p.c_set_world->m_comm;
        c_dbcsr_distribution_new(&m_dist_ptr, p.c_set_world->m_group, 
            p.c_row_dist->data(), p.c_row_dist->size(),
            p.c_col_dist->data(), p.c_col_dist->size());
    }
    
    void release() {
        c_dbcsr_distribution_release(&m_dist_ptr);
    }
    
    ~dist() { release(); }
    
    template <typename T, typename>
	friend class matrix;
    
};

template <typename T = double, typename>
class matrix {
private:
    
    void* m_matrix_ptr;
    MPI_Comm m_comm;
    
    const int m_data_type = dbcsr_type<T>::value;
    
    matrix(void* ptr, MPI_Comm comm) : m_matrix_ptr(ptr), m_comm(comm) {}
    
public:

    matrix() : m_matrix_ptr(nullptr), m_comm(MPI_COMM_NULL) {}

    typedef T value_type;
    
    template <typename D>
    friend class multiply;

    #:set list = [ &
        ['name', 'std::string', 'required', 'val'], &
        ['set_world', 'world', 'optional', 'ref'],&
        ['set_dist', 'dist', 'optional', 'ref'],&
        ['type', 'char', 'required', 'val'],&
        ['row_blk_sizes', 'vec<int>', 'required', 'ref'],&
        ['col_blk_sizes', 'vec<int>', 'required', 'ref'],&
        ['nze', 'int', 'optional', 'val'],&
        ['reuse', 'bool', 'optional', 'val'],&
        ['reuse_arrays', 'bool', 'optional', 'val'],&
        ['mutable_work', 'bool', 'optional', 'val'],&
        ['replication_type', 'char', 'optional', 'val']]
    ${make_struct(structname='create',friend='matrix',params=list)}$
    
    matrix(create& p) : m_matrix_ptr(nullptr) {
        
        void* dist_ptr = nullptr;
        
        if (!p.c_set_dist) {
            vec<int> rowdist, coldist; 
            auto dims = p.c_set_world->dims();
            
            auto rdist = default_dist(p.c_row_blk_sizes->size(),
                            dims[0], *p.c_row_blk_sizes);
            auto cdist = default_dist(p.c_col_blk_sizes->size(),
                            dims[1], *p.c_col_blk_sizes);
                            
            dist_ptr = new dist(dist::create().set_world(*p.c_set_world).row_dist(rdist).col_dist(cdist));
        } else {
            dist_ptr = p.c_set_dist->m_dist_ptr;
        }
        
        c_dbcsr_create_new(&m_matrix_ptr, p.c_name->c_str(), dist_ptr, *p.c_type, 
                               p.c_row_blk_sizes->data(), p.c_row_blk_sizes->size(), 
                               p.c_col_blk_sizes->data(), p.c_col_blk_sizes->size(), 
                               (p.c_nze) ? &*p.c_nze : nullptr, &m_data_type, 
                               (p.c_reuse) ? &*p.c_reuse : nullptr,
                               (p.c_reuse_arrays) ? &*p.c_reuse_arrays : nullptr, 
                               (p.c_mutable_work) ? &*p.c_mutable_work : nullptr, 
                               (p.c_replication_type) ? &*p.c_replication_type : nullptr);
    
        if (!p.c_set_dist) delete dist_ptr;
        
    }
    
    #:set list = [ &
        ['name', 'std::string', 'required', 'val'], &
        ['set_dist', 'dist', 'optional', 'ref'],&
        ['type', 'char', 'optional', 'val'],&
        ['row_blk_sizes', 'vec<int>', 'optional', 'ref'],&
        ['col_blk_sizes', 'vec<int>', 'optional', 'ref'],&
        ['nze', 'int', 'optional', 'val'],&
        ['reuse', 'bool', 'optional', 'val'],&
        ['reuse_arrays', 'bool', 'optional', 'val'],&
        ['mutable_work', 'bool', 'optional', 'val'],&
        ['replication_type', 'char', 'optional', 'val']]
    struct create_template {
        ${make_param(structname='create_template',params=list)}$
        private:
            
        required<matrix,ref> c_template;
        
        public:
        
        create_template(matrix& temp) : c_template(temp) {}
        friend class matrix;
    };
    
    matrix(create_template& p) : m_matrix_ptr(nullptr) {
        
        m_comm = p.c_template->m_comm;
        c_dbcsr_create_template(&m_matrix_ptr, p.c_name->c_str(), p.c_template->m_matrix_ptr, 
                                   (p.c_dist) ? p.c_dist->m_dist_ptr : nullptr, 
                                   (p.c_type) ? *p.c_type : nullptr, 
                                   (p.c_row_blk_sizes) ? p.c_row_blk_sizes->data() : nullptr,
                                   (p.c_row_blk_sizes) ? p.c_row_blk_sizes->size() : 0,
                                   (p.c_col_blk_sizes) ? p.c_col_blk_sizes->data() : nullptr,
                                   (p.c_col_blk_sizes) ? p.c_col_blk_sizes->size() : 0,
                                   (p.c_nze) ? &*p.c_nze : nullptr, &m_data_type, 
                                   (p.c_reuse) ? &*p.c_reuse : nullptr,
                                   (p.c_reuse_arrays) ? &*p.c_reuse_arrays : nullptr, 
                                   (p.c_mutable_work) ? &*p.c_mutable_work : nullptr, 
                                   (p.c_replication_type) ? &*p.c_replication_type : nullptr);
                                   
    }
    
    void set(const T alpha) {
        c_dbcsr_set(m_matrix_ptr, alpha);
    }
   
    void add(const matrix& matrix_a, const T alpha = (T)1.0, const T beta = T()) {
        c_dbcsr_add(matrix_a.m_matrix_ptr, this->matrix_ptr, alpha, beta);
    }

    void scale(T alpha, std::optional<int> last_column = std::nullopt) {
        c_dbcsr_scale(m_matrix_ptr, alpha, (last_column) ? &*last_column : nullptr);
    }
    
    void scale(const std::vector<T>& alpha, const std::string side) {
        c_dbcsr_scale_by_vector (m_matrix_ptr, alpha.data(), alpha.size(), side.c_str());
    }
    
    void add_on_diag(const T alpha) {
        c_dbcsr_add_on_diag(m_matrix_ptr, alpha);
    }
   
    void set_diag(const std::vector<T>& diag) {
        c_dbcsr_set_diag(m_matrix_ptr, diag.data(), diag.size());
    }
   
    //std::vector get_diag(); // TO DO
     
    T trace() const {
        T out;
        c_dbcsr_trace(m_matrix_ptr, out);
        return out;
    }
    
    T dot(const matrix& b) const {
        T out;
        c_dbcsr_dot(m_matrix_ptr, b.m_matrix_ptr, out);
        return out;
    }
    
    void hadamard_product(const matrix& a, const matrix& b, std::optional<T> assume_value) {
        c_dbcsr_hadamard_product(a.m_matrix_ptr, b.m_matrix_ptr, m_matrix_ptr, (assume_value) ? &*assume_value : nullptr); 
    }
    
    block<2,T> get_block_p(const int row, const int col, bool& found) {
        T* data = nullptr;
        std::array<int,2> size = {0,0};
    
        c_dbcsr_get_block_notrans_p(m_matrix_ptr, row, col, &data, &found, &size[0], &size[1]);
        
        return block<2,T>(size, data);
    }
    
    void complete_redistribute(matrix<T>& in, std::optional<bool> keep_sparsity = std::nullopt, 
                                std::optional<bool> summation = std::nullopt) {
    
        c_dbcsr_complete_redistribute(in.m_matrix_ptr, &m_matrix_ptr, 
            (keep_sparsity) ? &*keep_sparsity : nullptr, 
            (summation) ? &*summation : nullptr);
            
    }
    
    void filter(double eps, std::optional<int> method = std::nullopt, 
                std::optional<bool> use_absolute = std::nullopt, 
                std::optional<bool> filter_diag = std::nullopt) {
        c_dbcsr_filter(m_matrix_ptr, &eps, (method) ? &*method : nullptr, 
                        (use_absolute) ? &*use_absolute : nullptr, 
                        (filter_diag) ? &*filter_diag : nullptr);
    }

    matrix<T> get_block_diag(matrix<T>& in) { 
        void* diag;
        c_dbcsr_get_block_diag(in.m_matrix_ptr, &diag);
        return matrix<T>(diag, in.m_comm);
    }
   
#:set list = [ &
    ['shallow_copy', 'bool', 'optional', 'val'],&
    ['transpose_data', 'bool', 'optional', 'val'],& 
    ['transpose_dist', 'bool', 'optional', 'val'],& 
    ['use_dist', 'bool', 'optional', 'val']]
    struct transpose {
        ${make_param('transpose', list)}$
        private:
            matrix<T>& c_in;
        public:
            transpose(matrix<T>& in) : c_in(in) {}
            friend class matrix;
    };
    
    matrix(transpose& p) {
        
        m_comm = p.c_in->m_comm();
        c_dbcsr_transposed(&m_matrix_ptr, p.c_in->m_matrix_ptr, 
                            (p.c_shallow_copy) ? &*p.c_shallow_copy : nullptr,
                            (p.c_transpose_data) ? &*p.c_transpose_data : nullptr, 
                            (p.c_transpose_distribution) ? &*p.c_transpose_distribution : nullptr, 
                            (p.c_use_distribution) ? &*p.c_use_distribution : nullptr);
                            
    }

#:set list = [ &
    ['name', 'std::string', 'optional', 'val'],&
    ['keep_sparsity', 'bool', 'optional', 'val'],&
    ['shallow_data', 'bool', 'optional', 'val'],&
    ['keep_imaginary', 'bool', 'optional', 'val']]
    
    template <typename D>
    struct copy {
        ${make_param('copy', list)}$
        private:
            matrix<D>& c_in;
        public:
            copy(matrix<D>& in) : c_in(in) {}
            friend class matrix;
    };
    
    template <typename D>
    matrix(copy<D>& p) {
        
        m_comm = p.c_in->m_comm;
        c_dbcsr_copy(&m_matrix_ptr, p.c_in->m_matrix_ptr, (p.c_name) ? p.c_name->c_str() : nullptr, 
                      (p.c_keep_sparsity) ? &*p.c_keep_sparsity : nullptr, 
                      (p.c_shallow_data) ? &*p.c_shallow_data : nullptr, 
                      (p.c_keep_imaginary) ? &*p.c_keep_imaginary : nullptr, 
                      &m_data_type);
                      
    }
    
    void copy_in(matrix& in) {
        c_dbcsr_copy_into_existing(m_matrix_ptr, in.m_matrix_ptr);
    }
    
    struct desymmetrize {
        private:
        matrix& c_in;
        public:
        desymmetrize(matrix& in) : c_in(in) {}
        friend class matrix;
    };
    
    matrix(desymmetrize& p) {
        m_comm = p.c_in->m_comm;
        c_dbcsr_desymmetrize(p.c_in->m_matrix_ptr, &m_matrix_ptr);
    }
    
    void clear() {
        c_dbcsr_clear(m_matrix_ptr);
    }
    
    void reserve_diag_blocks() {
        c_dbcsr_reserve_diag_blocks(m_matrix_ptr);
    }
   
    void reserve_blocks(const std::vector<int>& rows, const std::vector<int>& cols) {
        c_dbcsr_reserve_blocks(m_matrix_ptr, rows.data(), cols.data(), rows.size());
    }

    void reserve_all() {
        c_dbcsr_reserve_all_blocks(m_matrix_ptr);
    }
    
    // reserve block 2d
    
    void put_block(const int row, const int col, block<2,T>& blk, std::optional<bool> sum = std::nullopt,
                    std::optional<T> scale = std::nullopt) {
        c_dbcsr_put_block2d(m_matrix_ptr, row, col, blk.data(), blk.size()[0], blk.size()[1], 
                            (sum) ? &*sum : nullptr, (scale) ? &*scale : nullptr);
    }
    
    T* data(long long int& data_size, std::optional<int> lb = std::nullopt, std::optional<int> ub = std::nullopt) {
        T data_type = T();
        T* ptr = nullptr;
        c_dbcsr_get_data(m_matrix_ptr, &ptr, &data_size, &data_type, 
            (lb) ? &*lb : nullptr, (ub) ? &*ub : nullptr);
        return ptr;
    }
    
    void replicate_all() {
        c_dbcsr_replicate_all(m_matrix_ptr);
    }

    void distribute(std::optional<bool> fast = std::nullopt) {
        c_dbcsr_distribute(m_matrix_ptr, (fast) ? &*fast : nullptr);
    }
    
    void sum_replicated() {
        c_dbcsr_sum_replicated(m_matrix_ptr);
    }
    
    void finalize() {
        c_dbcsr_finalize(m_matrix_ptr);
    }

    void release() {
        m_comm = MPI_COMM_NULL;
        if (m_matrix_ptr != nullptr) c_dbcsr_release(&m_matrix_ptr);
    }
    
    ~matrix() { release(); }
    
    int nblkrows_total() const {
       return c_dbcsr_nblkrows_total(m_matrix_ptr);
    }
    
    int nblkcols_total() const {
       return c_dbcsr_nblkcols_total(m_matrix_ptr);
    }
    
    int nblkrows_local() const {
       return c_dbcsr_nblkrows_local(m_matrix_ptr);
    }
    
    int nblkcols_local() const {
       return c_dbcsr_nblkcols_local(m_matrix_ptr);
    }

#:set vars = ['local_rows', 'local_cols', 'proc_row_dist', 'c_proc_col_dist', &
                'row_blk_sizes', 'col_blk_sizes', 'row_blk_offsets', 'col_blk_offsets']
#:for i in range(0,len(vars))
#:set var = vars[i]

    std::vector<int> ${var}$() const {
#:set rowcol = 'row'
#:set loctot = 'total'
#:if 'local' in var
    #:set loctot = 'local'
#:endif
#:if i % 2 != 0
    #:set rowcol = 'col'
#:endif
        std::vector<int> out(this->nblk${rowcol}$_${loctot}$());
        
        c_dbcsr_get_info(m_matrix_ptr, ${repeat('nullptr',10)}$, 
                             ${repeat('nullptr',i,',')}$
                             out.data(),
                             ${repeat('nullptr',len(vars) - i -1,',')}$
                             ${repeat('nullptr',5)}$);
                             
        return out;
        
    }
#:endfor
     
    std::string name() const {
        char* cname;
        c_dbcsr_get_info(m_matrix_ptr, ${repeat('nullptr',19)}$, &cname, nullptr, nullptr, nullptr);
        
        std::string out(cname);
        c_free_string(&cname);
        return out;
    }
    
    char matrix_type() const {
        char out;
        c_dbcsr_get_info(m_matrix_ptr, ${repeat('nullptr',20)}$, &out, nullptr, nullptr);
        return out;
    }
    
    int proc(const int row, const int col) const {
        int p = -1;
        c_dbcsr_get_stored_coordinates(m_matrix_ptr, row, col, &p);
        return p;
    }
    
    void setname(std::string name) {
        c_dbcsr_setname(m_matrix_ptr, name.c_str());
    }
   
    double occupation() const {
        return c_dbcsr_get_occupation(m_matrix_ptr);
    }
   
    int num_blocks() const {
        return c_dbcsr_get_num_blocks(m_matrix_ptr);
    } 
    
    int data_size() const {
        return c_dbcsr_get_data_size(m_matrix_ptr);
    }
   
    bool has_symmetry() const {
        return c_dbcsr_has_symmetry(m_matrix_ptr);
    }
   
    int nfullrows_total() const {
        return c_dbcsr_nfullrows_total(m_matrix_ptr);
    }
    
    int nfullcols_total() const {
        return c_dbcsr_nfullcols_total(m_matrix_ptr);
    }
   
    bool valid_index() const {
        return c_dbcsr_valid_index(m_matrix_ptr);
    }
    
    template <typename D>
    friend class iterator;
    
    void write(std::string file_path) const {
        c_dbcsr_binary_write(m_matrix_ptr, file_path.c_str());
    }
    
#:set list = [&
    ['filepath', 'std::string', 'required', 'val'],&
    ['distribution', 'dist', 'required', 'ref'],&
    ['set_world', 'world', 'required', 'ref']]
    ${make_struct(structname='read',friend='matrix',params=list)}$
    
    matrix(read& p) {
        m_comm = p.c_set_world->comm();
        c_dbcsr_binary_read(p.c_filepath->c_str(), p.c_distribution->m_dist_ptr, 
        p.c_set_world->comm(), m_matrix_ptr);
    }
    
};  

template <typename T>
class iterator {
#:set list = [&
    ['shared', 'bool', 'optional', 'val'],&
    ['dynamic', 'bool', 'optional', 'val'],&
    ['dynamic_byrows', 'bool', 'optional', 'val'],&
    ['contiguous_ptrs', 'bool', 'optional', 'val'],&
    ['read_only', 'bool', 'optional', 'val']]
    ${make_param('iterator',list)}$

private:
    void* m_iter_ptr;
    void* m_matrix_ptr;
    
    int m_row;
    int m_col;
    int m_iblk;
    int m_blk_p;
    bool m_transposed;
    int m_row_size;
    int m_col_size;
    int m_row_offset;
    int m_col_offset;
    T* m_blk_ptr;
    
public:

    iterator(matrix<T>& in) : m_iter_ptr(nullptr), m_matrix_ptr(in.m_matrix_ptr) {} 
    ~iterator() {}
    
    void start() {
        
        c_dbcsr_iterator_start(&m_iter_ptr, m_matrix_ptr, 
                                (c_shared) ? &*c_shared : nullptr, 
                                (c_dynamic) ? &*c_dynamic : nullptr, 
                                (c_dynamic_byrows) ? &*c_dynamic_byrows : nullptr, 
                                (c_contiguous_ptrs) ? &*c_shared : nullptr, 
                                (c_read_only) ? &*c_shared : nullptr);
                                
    }

    void stop() {
       c_dbcsr_iterator_stop(&m_iter_ptr);
    }

    bool blocks_left() {
        return c_dbcsr_iterator_blocks_left(m_iter_ptr);
    }
    
    void next_block_index() {
        c_dbcsr_iterator_next_block_index(m_iter_ptr, &m_row, &m_col, &m_iblk, &m_blk_p);
    }

    void next_block() {
        c_dbcsr_iterator_next_2d_block(m_iter_ptr, &m_row, &m_col, &m_blk_ptr, &m_transposed, &m_iblk, 
             &m_row_size, &m_col_size, &m_row_offset, &m_col_offset);
    }
    
    inline T& operator()(const int i, const int j) {
        return m_blk_ptr[i + m_row_size * j];
    }
    
#:set list = ['row', 'col', 'iblk', 'blk_p', 'row_size', 'col_size', 'row_offset', 'col_offset']
#:for var in list
    int ${var}$() { return m_${var}$; }
#:endfor

    T* data() { return m_blk_ptr; }
    
};

template <typename T>
class multiply {

#:set list = [ &
    ['alpha', 'T', 'optional', 'val'],&
    ['beta', 'T', 'optional', 'val'],&
    ['first_row', 'int', 'optional', 'val'],&
    ['last_row', 'int', 'optional', 'val'],&
    ['first_col', 'int', 'optional', 'val'],&
    ['last_col', 'int', 'optional', 'val'],&
    ['first_k', 'int', 'optional', 'val'],&
    ['last_k', 'int', 'optional', 'val'],&
    ['retain_sparsity', 'bool', 'optional', 'val'],&
    ['filter_eps', 'double', 'optional', 'val'],&
    ['flop', 'long long int', 'optional', 'ref']] 
    
    ${make_param('multiply',list)}$
    
private:

    char m_transa, m_transb;
    matrix<T>& m_A; 
    matrix<T>& m_B; 
    matrix<T>& m_C;
    
public:

    multiply(char transa, char transb, matrix<T>& A, matrix<T>& B, matrix<T>& C) :
        m_transa(transa), m_transb(transb), m_A(A), m_B(B), m_C(C) {}
    
    ~multiply() {}

    void perform() {
        
        c_dbcsr_multiply(m_transa, m_transb, 
                        (c_alpha) ? *c_alpha : (T)1.0, 
                        m_A.m_matrix_ptr, m_B.m_matrix_ptr,
                        (c_beta) ? *c_beta : T(),
                        m_C.matrix_ptr,
                        (c_first_row) ? &*c_first_row : nullptr,
                        (c_last_row) ? &*c_last_row : nullptr,
                        (c_first_col) ? &*c_first_col : nullptr,
                        (c_last_col) ? &*c_last_col : nullptr,
                        (c_first_k) ? &*c_first_k : nullptr,
                        (c_last_k) ? &*c_last_k : nullptr,
                        (c_retain_sparsity) ? &*c_retain_sparsity : nullptr,
                        (c_filter_eps) ? &*c_filter_eps : nullptr, 
                        (c_flop) ? &*c_flop : nullptr);
    }

};

template <typename T>
multiply(char transa, char transb, matrix<T>& A, matrix<T>& B, matrix<T>& C) -> multiply<typename matrix<T>::value_type>;

template <int N, typename>
class pgrid {
protected:

	void* m_pgrid_ptr;
	vec<int> m_dims;
	MPI_Comm m_comm;
	
	template <int M, typename>
	friend class dist_t;
	
public:
	
    #:set list = [ &
        ['comm', 'MPI_Comm', 'required', 'val'],&
        ['map1', 'vec<int>', 'optional', 'val'],&
        ['map2', 'vec<int>', 'optional', 'val'],&
        ['tensor_dims', 'arr<int,N>', 'optional', 'val'],&
        ['nsplit', 'int', 'optional', 'val'],&
        ['dimsplit', 'int', 'optional', 'val']]
	${make_struct(structname='create',friend='pgrid',params=list)}$

	pgrid() {}
	
	pgrid(MPI_Comm comm) : pgrid(create().comm(comm)) {}	
	
	pgrid(const create& p) : m_dims(N), m_comm(*p.c_comm) {
		
		MPI_Fint fmpi = MPI_Comm_c2f(m_comm);
		
		if (p.c_map1 && p.c_map2) {
			c_dbcsr_t_pgrid_create_expert(&fmpi, m_dims.data(), N, &m_pgrid_ptr, 
                (p.c_map1) ? p.c_map1->data() : nullptr, 
                (p.c_map1) ? p.c_map1->size() : 0,
                (p.c_map2) ? p.c_map2->data() : nullptr,
                (p.c_map2) ? p.c_map2->size() : 0,
                (p.c_tensor_dims) ? p.c_tensor_dims->data() : nullptr,
                (p.c_nsplit) ? &*p.c_nsplit : nullptr, 
                (p.c_dimsplit) ? &*p.c_dimsplit : nullptr);
		} else {
			c_dbcsr_t_pgrid_create(&fmpi, m_dims.data(), N, &m_pgrid_ptr, 
                (p.c_tensor_dims) ? p.c_tensor_dims->data() : nullptr);
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

template <int N, typename>
class dist_t {
private:

	void* m_dist_ptr;
	
	arrvec<int,N> m_nd_dists;
	MPI_Comm m_comm;
	
	template <int M, typename T, typename>
	friend class tensor;
	
public:

    #:set list = [ &
        ['ngrid', 'pgrid<N>', 'required', 'ref'],&
        ['nd_dists', 'arrvec<int,N>', 'required', 'val']]
    ${make_struct(structname='create',friend='dist_t',params=list)}$
	
	dist_t() : m_dist_ptr(nullptr) {}

#:for idim in range(2,MAXDIM+1)
    template<int M = N, typename std::enable_if<M == ${idim}$,int>::type = 0>
	dist_t(create& p) :
		m_dist_ptr(nullptr),
		m_nd_dists(std::move(*p.c_nd_dists)),
        m_comm(p.c_ngrid->m_comm)
	{
		
		c_dbcsr_t_distribution_new(&m_dist_ptr, p.c_ngrid->m_pgrid_ptr,
                ${datasize('m_nd_dists',0,idim)}$
                #:if idim != MAXDIM
                ,
                #:endif
                ${datasize0(MAXDIM-idim)}$);
                                   
	}
#:endfor
	
	dist_t(dist_t<N>& rhs) = delete;
		
	dist_t<N>& operator=(dist_t<N>& rhs) = delete;
	
	~dist_t() {

		if (m_dist_ptr != nullptr)
			c_dbcsr_t_distribution_destroy(&m_dist_ptr);
			
		m_dist_ptr = nullptr;
		
	}
	
	void destroy() {
		
		if (m_dist_ptr != nullptr) {
			c_dbcsr_t_distribution_destroy(&m_dist_ptr);
		}
			
		m_dist_ptr = nullptr;
		
	}
	
	
};

inline double block_threshold = 1e-16;

template <int N, typename T = double, typename>
class tensor {
protected:

	void* m_tensor_ptr;
	MPI_Comm m_comm;
	
	const int m_data_type = dbcsr_type<T>::value;
	
	tensor(void* ptr) : m_tensor_ptr(ptr) {}
	
public:

    friend class iterator_t<N,T>;
    
	template <int N1, int N2, int N3, typename D>
    friend class contract;
    
    template <int M, typename D>
	friend class copy;

	typedef T value_type;
    const static int dim = N;
    
// ======== get_info =======
#:set vars = ['nblks_total', 'nfull_total', 'nblks_local', 'nfull_local', 'pdims', 'my_ploc']
#:for i in range(0,len(vars))
    #:set var = vars[i]
    
    arr<int,N> ${var}$() {
        arr<int,N> out;
        c_dbcsr_t_get_info(m_tensor_ptr, N, 
        #:for n in range(0,i)
        nullptr,
        #:endfor
        out.data(),
        #:for n in range(i+1,len(vars))
        nullptr,
        #:endfor
        ${repeat('0',2*MAXDIM)}$, // nblksloc, nblkstot
        ${repeat('nullptr',4*MAXDIM)}$, // blks, proc ...
        nullptr, nullptr, nullptr);
        
       return out;
   
   }
#:endfor

#:set vars = ['blks_local', 'proc_dist', 'blk_sizes', 'blk_offsets']
#:for i in range(0,len(vars))
    #:set var = vars[i]
    
    #:for idim in range(2,MAXDIM+1)
    template <int M = N>
    typename std::enable_if<M == ${idim}$,arrvec<int,N>>::type
    ${var}$() {
        
        arrvec<int,N> out;
        vec<int> sizes(N);
    
    #:for n in range(0,idim)
        #:if var == 'blks_local'
        sizes[${n}$] = c_dbcsr_t_nblks_local(m_tensor_ptr,${n}$);
        #:else
        sizes[${n}$] = c_dbcsr_t_nblks_total(m_tensor_ptr,${n}$);
        #:endif
        
        out[${n}$] = vec<int>(sizes[${n}$]);
        
    #:endfor
    
        c_dbcsr_t_get_info(m_tensor_ptr, N, ${repeat('nullptr',6)}$,
            #:if var != 'blks_local'
                ${repeat('0',MAXDIM)}$,
            #:endif
                ${repeatvar('sizes[',0,idim-1,']')}$, ${repeat('0',MAXDIM-idim,',')}$
            #:if var == 'blks_local'
                ${repeat('0',MAXDIM)}$,
            #:endif
                ${repeat('nullptr',i*MAXDIM,',')}$
                ${repeatvar('out[',0,idim-1,'].data()')}$, ${repeat('nullptr',MAXDIM-idim,',')}$
                ${repeat('nullptr',(len(vars)-i-1)*MAXDIM,',')}$
                nullptr, nullptr, nullptr);
            
        return out;
        
    }
    
    #:endfor
#:endfor

// =========0 end get_info =========

// ========= map info ==========

#:set vars = ['ndim_nd', 'ndim1_2d', 'ndim2_2d']
#:for i in range(0,len(vars))
    #:set var = vars[i]
    
    int ${var}$() {
        
        int c_${var}$;
        
        c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, 0, 0, 
                        ${repeat('nullptr',i,',')}$
                        &c_${var}$,
                        ${repeat('nullptr',2-i,',')}$
                        ${repeat('nullptr',10)}$);
                        
        return c_${var}$;
        
    }
    
#:endfor

#:set vars = ['dims_2d_i8', 'dims_2d', 'dims_nd', 'dims1_2d', 'dims2_2d', 'map1_2d', 'map2_2d', 'map_nd']
#:for i in range(0,len(vars))
    #:set var = vars[i]
    
    #:if var == 'dims_2d_i8'
    vec<long long int> ${var}$() {
    #:else
    vec<int> ${var}$() {
    #:endif
        
        int nd_size = N;
        int nd_row_size = c_dbcsr_t_ndims_matrix_row(m_tensor_ptr);
        int nd_col_size = c_dbcsr_t_ndims_matrix_column(m_tensor_ptr);
        
        #:if var in ['dims_2d_i8', 'dims_2d']
        int vsize = 2;
        #:elif var in ['dims_nd', 'map_nd']
        int vsize = nd_size;
        #:elif var in ['dims1_2d', 'map1_2d']
        int vsize = nd_row_size;
        #:else
        int vsize = nd_col_size;
        #:endif
        
        #:if var == 'dims_2d_i8'
        vec<long long int> out(vsize);
        #:else 
        vec<int> out(vsize);
        #:endif
        
        c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, nd_row_size, nd_col_size, nullptr, nullptr, nullptr,
                        ${repeat('nullptr',i,',')}$
                        out.data(),
                        ${repeat('nullptr',len(vars) - i - 1,',')}$
                        nullptr, nullptr);
                        
        return out;
        
    }
    
#:endfor
        
// ======= end map info =========
    
	tensor() : m_tensor_ptr(nullptr) {}
	
    #:set list = [ &
        ['name', 'std::string', 'required', 'val'],&
        ['ndist','dist_t<N>','optional','ref'],&
        ['ngrid', 'pgrid<N>', 'optional', 'ref'],&
        ['map1', 'vec<int>', 'required', 'val'],&
        ['map2', 'vec<int>', 'required', 'val'],&
        ['blk_sizes', 'arrvec<int,N>', 'required', 'val']]
    ${make_struct(structname='create',friend='tensor',params=list)}$
    
#:for idim in range(2,MAXDIM+1)
    template<int M = N, typename D = T, typename std::enable_if<M == ${idim}$,int>::type = 0>
	tensor(create& p) : m_tensor_ptr(nullptr) {
		
		void* dist_ptr;
		dist_t<N>* distn = nullptr;
		
		if (p.c_ndist) {
			
			dist_ptr = p.c_ndist->m_dist_ptr;
            m_comm = p.c_ndist->m_comm;
			
		} else {
			
			arrvec<int,N> distvecs;
			
			for (int i = 0; i != N; ++i) {
				distvecs[i] = default_dist(p.c_blk_sizes->at(i).size(),
				p.c_ngrid->dims()[i], p.c_blk_sizes->at(i));
			}
			
			distn = new dist_t<N>(typename dist_t<N>::create().ngrid(*p.c_ngrid).nd_dists(distvecs));
			dist_ptr = distn->m_dist_ptr;
            m_comm = distn->m_comm;
		}

		c_dbcsr_t_create_new(&m_tensor_ptr, p.c_name->c_str(), dist_ptr, 
						p.c_map1->data(), p.c_map1->size(),
						p.c_map2->data(), p.c_map2->size(), &m_data_type, 
						${datasizeptr('p.c_blk_sizes', 0, idim)}$
                        #:if idim != MAXDIM
                        ,
                        #:endif
                        ${datasize0(MAXDIM-idim)}$);
						
		if (distn) delete distn;
					
	}
#:endfor

    #:set list = [ &
        ['tensor_in', 'tensor', 'required', 'ref'],&
        ['name', 'std::string', 'required', 'val'],&
        ['ndist', 'dist_t<N>', 'optional', 'ref'],&
        ['map1', 'vec<int>', 'optional', 'val'],&
        ['map2', 'vec<int>', 'optional', 'val']]
    ${make_struct(structname='create_template',friend='tensor',params=list)}$
	
	tensor(create_template& p) : m_tensor_ptr(nullptr) {
        
        m_comm = p.c_tensor_in->m_comm;
		
		c_dbcsr_t_create_template(p.c_tensor_in->m_tensor_ptr, &m_tensor_ptr, 
            p.c_name->c_str(), (p.c_ndist) ? p.c_ndist->m_dist_ptr : nullptr,
            (p.c_map1) ? p.c_map1->data() : nullptr,
            (p.c_map1) ? p.c_map1->size() : 0,
            (p.c_map2) ? p.c_map2->data() : nullptr,
            (p.c_map2) ? p.c_map2->size() : 0,
            &m_data_type);
		
	}    
		
	~tensor() {
		
		//if (m_tensor_ptr != nullptr) std::cout << "Destroying: " << this->name() << std::endl;
		destroy();
		
	}
	
	void destroy() {
		
		if (m_tensor_ptr != nullptr) {
			c_dbcsr_t_destroy(&m_tensor_ptr);
		}
		
		m_tensor_ptr = nullptr;
		
	}
	
#:for idim in range(2,MAXDIM+1)
    template <int M = N>
    typename std::enable_if<M == ${idim}$>::type
	reserve(arrvec<int,N> nzblks) {
			
		if (nzblks[0].size() == 0) return;
		
		c_dbcsr_t_reserve_blocks_index(m_tensor_ptr, nzblks[0].size(), 
			${repeatvar('nzblks[',0,idim-1,'].data()')}$
            #:if idim != MAXDIM
            ,
            #:endif
            ${repeat('nullptr',MAXDIM-idim)}$);
			
	}
#:endfor
	
	void reserve_all() {
		
		auto blks = this->blks_local();
		arrvec<int,N> res;
		
		std::function<void(int,int*)> loop;
		
		int* arr = new int[N];
		
		loop = [&res,&blks,&loop](int depth, int* vals) {
			for (auto eleN : blks[depth]) {
				vals[depth] = eleN;
				if (depth == N-1) {
					for (int i = 0; i != N; ++i) {
						res[i].push_back(vals[i]);
					}
				} else {
					loop(depth+1,vals);
				}
			}
		};
		
		loop(0,arr);
		this->reserve(res);
        delete[] arr;
		
	}
	
	void put_block(const index<N>& idx, block<N,T>& blk, std::optional<bool> sum = std::nullopt, std::optional<double> scale = std::nullopt) {
		
		c_dbcsr_t_put_block(m_tensor_ptr, idx.data(), blk.size().data(), 
			blk.data(), (sum) ? &*sum : nullptr, (scale) ? &*scale : nullptr);
			
	}
	
	block<N,T> get_block(const index<N>& idx, const index<N>& blk_size, bool& found) {
        
		block<N,T> blk_out(blk_size);
        
		c_dbcsr_t_get_block(m_tensor_ptr, idx.data(), blk_size.data(), 
			blk_out.data(), &found);
			
		return blk_out;
			
	}
	
	int proc(index<N>& idx) {
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
        char* cstring;
        c_dbcsr_t_get_info(m_tensor_ptr, N, ${repeat('nullptr',6)}$, 
							   ${repeat('0',2*MAXDIM)}$,
                               ${repeat('nullptr',4*MAXDIM)}$,
                               nullptr, &cstring, nullptr);
		std::string out(cstring);
        return out;
	}
	
	int num_blocks() const {
		return c_dbcsr_t_get_num_blocks(m_tensor_ptr);
	}
	
	long long int num_blocks_total() const {
		return c_dbcsr_t_get_num_blocks_total(m_tensor_ptr);
	}
	
	void filter(std::optional<T> ieps = std::nullopt, 
        std::optional<int> method = std::nullopt, 
        std::optional<bool> use_absolute = std::nullopt) {
        
		c_dbcsr_t_filter(m_tensor_ptr, (ieps) ? *ieps : block_threshold,
            (method) ? &*method : nullptr, (use_absolute) ? &*use_absolute : nullptr);
		
	}
    
    void scale(T factor) {
        c_dbcsr_t_scale(m_tensor_ptr, factor);
    }
    
    void set(T factor) {
        c_dbcsr_t_set(m_tensor_ptr, factor);
    }
    
    void finalize() {
        c_dbcsr_t_finalize(m_tensor_ptr);
    }
	
	stensor<N,T> get_stensor() {
		
		tensor<N,T>* t = new tensor<N,T>();
		
		t->m_tensor_ptr = m_tensor_ptr;
		t->m_comm = m_comm;
		
		m_tensor_ptr = nullptr;
		
		return stensor<N,T>(t);
		
	} 
    
};

template <int N, typename T = double>
stensor<N,T> make_stensor(typename tensor<N,T>::create& p) {
    tensor<N,T> t(p);
    return t.get_stensor();
}

template <int N, typename T = double>
stensor<N,T> make_stensor(typename tensor<N,T>::create_template& p) {
    tensor<N,T> t(p);
    return t.get_stensor();
}

template <int N, typename T>
class iterator_t {
private:

	void* m_iter_ptr;
	void* m_tensor_ptr;
	
	index<N> m_idx;
	int m_blk_n;
	int m_blk_p;
	std::array<int,N> m_size;
	std::array<int,N> m_offset;
	
public:

	iterator_t(tensor<N,T>& t_tensor) :
		m_iter_ptr(nullptr), m_tensor_ptr(t_tensor.m_tensor_ptr),
		m_blk_n(0), m_blk_p(0)
	{}
	
	~iterator_t() {}
    
    void start() {
        c_dbcsr_t_iterator_start(&m_iter_ptr, m_tensor_ptr);
    }
    
    void stop() {
        c_dbcsr_t_iterator_stop(&m_iter_ptr);
        m_iter_ptr = nullptr;
    }
	
	void next() {
		
		c_dbcsr_t_iterator_next_block(m_iter_ptr, m_idx.data(), &m_blk_n, &m_blk_p,
			m_size.data(), m_offset.data());
		
	}
	
	bool blocks_left() {	
		return c_dbcsr_t_iterator_blocks_left(m_iter_ptr);	
	}
	
	const index<N>& idx() {
		return m_idx;	
	}
	
	const index<N>& size() {	
		return m_size;	
	}
	
	const index<N>& offset() {		
		return m_offset;		
	}
	
	int blk_n() {		
		return m_blk_n;		
	}
	
	int blk_p() {	
		return m_blk_p;	
	}
		
};

template <typename D>
D* unfold_bounds(vec<vec<D>>& v) {
	int b_size = v.size();
	D* f_bounds = new D[2 * b_size];
	for (int j = 0; j != b_size; ++j) {
		for (int i = 0; i != 2; ++i) {
			f_bounds[i + j * 2] = v[j][i];
		}
	}
	return f_bounds;
}

template <int N, typename T>
class copy {

    #:set list = [ &
        ['order','vec<int>','optional','val'],&
        ['sum','bool','optional','val'],&
        ['bounds','vec<vec<int>>', 'optional', 'ref'],&
        ['move_data','bool','optional','val']]
    ${make_param('copy',list)}$
        
private:

    tensor<N,T>& c_t_in;
    tensor<N,T>& c_t_out;
    
public:

    copy(tensor<N,T>& t1, tensor<N,T>& t2) : c_t_in(t1), c_t_out(t2) {}
    
    void perform() {
        
        int* fbounds = (c_bounds) ? unfold_bounds<int>(*c_bounds) : nullptr;
        
        c_dbcsr_t_copy(c_t_in.m_tensor_ptr, N, c_t_out.m_tensor_ptr,
            (c_order) ? c_order->data() : nullptr, 
            (c_sum) ? &*c_sum : nullptr, fbounds,
            (c_move_data) ? &*c_move_data : nullptr, nullptr);
            
    }
            
};

template <int N, typename T>
copy(tensor<N,T>& t1, tensor<N,T>& t2) -> copy<tensor<N,T>::dim, typename tensor<N,T>::value_type>;

template <int N1, int N2, int N3, typename T>
class contract {

    #:set list = [ &
        ['alpha','T','optional','val'],&
        ['beta', 'T', 'optional', 'val'],&
        ['con1', 'vec<int>', 'required', 'val'],&
        ['ncon1', 'vec<int>', 'required', 'val'],&
        ['con2', 'vec<int>', 'required', 'val'],&
        ['ncon2', 'vec<int>', 'required', 'val'],&
        ['map1', 'vec<int>', 'required', 'val'],&
        ['map2', 'vec<int>', 'required', 'val'],&
        ['bounds1','vec<vec<int>>','optional','ref'],&
        ['bounds2','vec<vec<int>>','optional','ref'],&
        ['bounds3','vec<vec<int>>','optional','ref'],&
        ['filter','double','optional','val'],&
        ['flop','long long int','optional','ref'],&
        ['move','bool','optional','val'],&
        ['retain_sparsity','bool','optional','val'],&
        ['print','bool','optional','val'],&
        ['log','bool','optional','val']]
    ${make_param('contract',list)}$

private:

    dbcsr::tensor<N1,T>& c_t1;
    dbcsr::tensor<N2,T>& c_t2;
    dbcsr::tensor<N3,T>& c_t3;
    
public:
    
    contract(dbcsr::tensor<N1,T>& t1, dbcsr::tensor<N2,T>& t2, dbcsr::tensor<N3,T>& t3) 
        : c_t1(t1), c_t2(t2), c_t3(t3) {}
    
    void perform() {
        
        int* f_b1 = (c_bounds1) ? unfold_bounds<int>(*c_bounds1) : nullptr;
        int* f_b2 = (c_bounds2) ? unfold_bounds<int>(*c_bounds2) : nullptr;
        int* f_b3 = (c_bounds3) ? unfold_bounds<int>(*c_bounds3) : nullptr;
        
        int out = 6;
        int* unit_nr = (c_print) ? ((*c_print) ? &out : nullptr) : nullptr;  
        
        c_dbcsr_t_contract_r_dp(
            (c_alpha) ? *c_alpha : 1,
            c_t1.m_tensor_ptr, c_t2.m_tensor_ptr,
            (c_beta) ? *c_beta : 0,
            c_t3.m_tensor_ptr, 
            c_con1->data(), c_con1->size(), 
            c_ncon1->data(), c_ncon1->size(),
            c_con2->data(), c_con2->size(),
            c_ncon2->data(), c_ncon2->size(),
            c_map1->data(), c_map1->size(),
            c_map2->data(), c_map2->size(), 
            f_b1, f_b2, f_b3,
            nullptr, nullptr, nullptr, nullptr, 
            (c_filter) ? &*c_filter : nullptr,
            (c_flop) ? &*c_flop : nullptr,
            (c_move) ? &*c_move : nullptr,
            (c_retain_sparsity) ? &*c_retain_sparsity : nullptr,
            unit_nr, (c_log) ? &*c_log : nullptr);
            
        delete [] f_b1, f_b2, f_b3;
             
    }
    
    void perform (std::string formula) {
        
        eval(formula);
        
        perform();
        
    }
    
    void eval(std::string str) {
        
        std::vector<std::string> idxs(4);
        std::vector<int> con1, con2, ncon1, ncon2, map1, map2;
	
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
        
        //for (auto v : idxs) {
        //	std::cout << v << std::endl;
        //}
        
        // evaluating input
        auto t1 = idxs[0];
        auto t2 = idxs[1];
        auto t3 = idxs[2];
        
        //std::cout << "t3 map" << std::endl;
        
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
        
        /*
        std::cout << "To be contrcated: " << scon << std::endl;	
        std::cout << "Maps:" << std::endl;
        for (auto v : con1) std::cout << v << " ";
        std::cout << std::endl;
        for (auto v : con2) std::cout << v << " ";
        std::cout << std::endl;
        */
        
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
        
        /*
        std::cout << "not con1: " << sncon1 << std::endl;
        std::cout << "not con2: " << sncon2 << std::endl;
        std::cout << "Maps:" << std::endl;
        for (auto v : ncon1) std::cout << v << " ";
        std::cout << std::endl;
        for (auto v : ncon2) std::cout << v << " ";
        std::cout << std::endl;
        */
        
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

        /*
        std::cout << "Maps tensor 3" << std::endl;
        for (auto v : map1) std::cout << v << " ";
        std::cout << std::endl;
        for (auto v : map2) std::cout << v << " ";
        std::cout << std::endl;
        */
        
        if (map1.size() + map2.size() != t3.size()) 
            throw std::runtime_error("Incompatible tensor dimensions: "+str);
            
        this->map1(map1);
        this->map2(map2);
        this->con1(con1);
        this->con2(con2);
        this->ncon1(ncon1);
        this->ncon2(ncon2);
        
    }
        
    
};

template <int N1, int N2, int N3, typename T>
contract(tensor<N1,T>& t1, tensor<N2,T>& t2, tensor<N3,T>& t3) 
-> contract<tensor<N1,T>::dim,tensor<N2,T>::dim,tensor<N3,T>::dim, T>;

template <int N, typename T = double>
dbcsr::tensor<N+1,T> add_dummy(tensor<N,T>& t) {
	
	vec<int> map1 = t.map1_2d();
    vec<int> map2 = t.map2_2d();   
	
	vec<int> new_map1 = map1;
	new_map1.insert(new_map1.end(), map2.begin(), map2.end());
	
	vec<int> new_map2 = {N};
	
	pgrid<N+1> grid(t.comm());
	
	arrvec<int,N> blksizes = t.blk_sizes();
    arrvec<int,N+1> blksizesnew;
    
    std::copy(blksizes.begin(),blksizes.end(),blksizesnew.begin());
    blksizesnew[N] = vec<int>{1};
	
	tensor<N+1,T> new_t = typename tensor<N+1,T>::create().name(t.name() + " dummy").ngrid(grid)
        .map1(new_map1).map2(new_map2).blk_sizes(blksizesnew);
	
	iterator_t<N> it(t);
    index<N+1> newidx;
    index<N+1> newsize;
    
    newidx[N] = 0;
    newsize[N] = 1;
    
    auto locblks = t.blks_local();
    arrvec<int,N+1> newlocblks;
    
    std::copy(locblks.begin(),locblks.end(),newlocblks.begin());
    newlocblks[N] = vec<int>(locblks[0].size(),0);
    
    new_t.reserve(newlocblks);
    
    it.start();
	
	while (it.blocks_left()) {
		
		it.next();
		
		auto& idx = it.idx();
        auto& size = it.size();
				
		std::copy(size.begin(),size.end(),newsize.begin());
		std::copy(idx.begin(),idx.end(),newidx.begin());
        
		bool found = false;
		auto blk = t.get_block(idx, size, found);
		
		block<N+1,T> newblk(newsize,blk.data());
		
		//std::cout << "IDX: " << new_idx.size() << std::endl;
		//std::cout << "BLK: " << new_blk.sizes().size() << std::endl;
				
		new_t.put_block(newidx, newblk);
		
	}
    
    it.stop();
		
	//print(new_t);
    t.finalize();
    new_t.finalize();
	
	grid.destroy();
	
	return new_t;
	
}

template <int N, typename T = double>
dbcsr::tensor<N-1,T> remove_dummy(tensor<N,T>& t, vec<int> map1, vec<int> map2, std::optional<std::string> name = std::nullopt) {
	
	//std::cout << "Removing dummy dim." << std::endl;
	
	pgrid<N-1> grid(t.comm());
	
	auto blksizes = t.blk_sizes();
	arrvec<int,N-1> newblksizes;
    
    std::copy(blksizes.begin(),blksizes.end()-1,newblksizes.begin());
	std::string newname = (name) ? *name : t.name();
    
	tensor<N-1,T> new_t = typename dbcsr::tensor<N-1,T>::create().name(newname).ngrid(grid).map1(map1).map2(map2)
		.blk_sizes(newblksizes);
	
	iterator_t<N> it(t);
    
    // reserve 
    auto locblks = t.blks_local();
    arrvec<int,N-1> newlocblks;
    
    std::copy(locblks.begin(),locblks.end()-1,newlocblks.begin());
    
    new_t.reserve(newlocblks);
	
    it.start();
    
	// parallelize this:
	while (it.blocks_left()) {
		
		it.next();
		
		auto& idx = it.idx();
        auto& size = it.size();
        
		index<N-1> newidx;
        index<N-1> newsize;
        
        std::copy(idx.begin(),idx.end()-1,newidx.begin());
        std::copy(size.begin(),size.end()-1,newsize.begin());
    
		bool found = false;
		auto blk = t.get_block(idx, size, found);
		
		block<N-1,T> newblk(newsize, blk.data());
				
		new_t.put_block(newidx, newblk);
		
	}
		
	t.finalize();
    new_t.finalize();
    
    it.stop();
	
	grid.destroy();
	
	return new_t;
	
}

template <int N>
double dot(tensor<N,double>& t1, tensor<N,double>& t2) {

    // have to have same pgrid!
	double sum = 0.0;
	
    //#pragma omp parallel 
    //{
        
        iterator_t<2> iter(t1);
        iter.start();
        
		while (iter.blocks_left()) {
			
			iter.next();
            
            auto& idx = iter.idx();
            auto& size = iter.size();
			
			bool found = false;
            
			auto b1 = t1.get_block(idx, size, found);
			auto b2 = t2.get_block(idx, size, found);
			
			if (!found) continue;
			
			sum += std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), 0.0);
			
		}
        
        t1.finalize();
        t2.finalize();
        
        iter.stop();
        
    //}
		
    
	
	double MPIsum = 0.0;
	
	MPI_Allreduce(&sum,&MPIsum,1,MPI_DOUBLE,MPI_SUM,t1.comm());
	
	return MPIsum;
		
}


template <int N, typename T = double>
void ewmult(tensor<N,T>& t1, tensor<N,T>& t2, tensor<N,T>& tout) {
	
	// elementwise multiplication
	// make sure tensors have same grid and dimensions, caus I sure don't do it here
	
	//#pragma omp parallel 
   // {
        dbcsr::iterator_t<N> it(t1);
        
        it.start();
        
        while (it.blocks_left()) {
                
            it.next();
            auto& idx = it.idx();
            auto& blksize = it.size();
                
            bool found = false;
            bool found3 = false;
            
            auto b1 = t1.get_block(idx, blksize, found);
            auto b2 = t2.get_block(idx, blksize, found);
            auto b3 = tout.get_block(idx, blksize, found3);
                
            if (!found) continue;
            if (!found3) {
                arrvec<int,N> res;
                for (int i = 0; i != N; ++i) {
                    res[i].push_back(idx[i]);
                }
               
                tout.reserve(res);
                
            }
                
            std::transform(b1.data(), b1.data() + b1.ntot(), b2.data(), b3.data(), std::multiplies<T>());
            
            tout.put_block(idx, b3);
            
        }
        
        it.stop();
        t1.finalize();
        t2.finalize();
        tout.finalize();
        
  //  }
		
	tout.filter();
	
}

/*
template <int N>
void ewmult(tensor<N,double>& t1, tensor<N,double>& t2, tensor<N,double>& t3) {
	
	// dot product only for N = 2 at the moment
	//assert(N == 2);

	double sum = 0.0;
	
	dbcsr::iterator_t<N> it(t1);
	
	while (it.blocks_left()) {
			
			it.next();
			
			bool found = false;
			auto b1 = t1.get_block({.idx = it.idx(), .blk_size = it.sizes(), .found = found});
			auto b2 = t2.get_block({.idx = it.idx(), .blk_size = it.sizes(), .found = found});
			
			if (!found) continue;
			
			//std::cout  << std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), T()) << std::endl;
			
			//std::cout << "ELE: " << std::endl;
			//for (int i = 0; i != b1.ntot(); ++i) { std::cout << b1(i) << " " << b2(i) << std::endl; }
			
			
			sum += std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), 0.0);
			
			//std::cout << "SUM: " << std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), T()) << std::endl;
			
	}
		
	
	
	double MPIsum = 0.0;
	
	MPI_Allreduce(&sum,&MPIsum,1,MPI_DOUBLE,MPI_SUM,t1.comm());
	
	return MPIsum;
		
}*/ /*

*/
template <int N, typename T>
T RMS(tensor<N,T>& t_in) {
	
	T prod = dot(t_in, t_in);
	
	// get total number of elements
	auto tot = t_in.nfull_total();
	
	size_t ntot = 1.0;
	for (auto i : tot) {
		ntot *= i;
	}
	
	return sqrt(prod/ntot);
	
}

template <int N, typename T>
void print(tensor<N,T>& t_in) {
	
	int myrank, mpi_size;
	
	MPI_Comm_rank(t_in.comm(), &myrank); 
	MPI_Comm_size(t_in.comm(), &mpi_size);
	
	iterator_t<N,T> iter(t_in);
    iter.start();
	
	if (myrank == 0) std::cout << "Tensor: " << t_in.name() << std::endl;
    
    for (int p = 0; p != mpi_size; ++p) {
		if (myrank == p) {
            int nblk = 0;
			while (iter.blocks_left()) {
                
                nblk++;
                
				iter.next();
				bool found = false;
				auto idx = iter.idx();
				auto size = iter.size();
				
				auto blk = t_in.get_block(idx, size, found);
                
                std::cout << myrank << ": [";
                
				for (int s = 0; s != N; ++s) {
					std::cout << idx[s];
					if (s != N - 1) { std::cout << ","; }
				}
				
				std::cout << "] (";
				
                for (int s = 0; s != N; ++s) {
					std::cout << size[s];
					if (s != N - 1) { std::cout << ","; }
				}
				std::cout << ") {";
				
				
				for (int i = 0; i != blk.ntot(); ++i) {
					std::cout << blk[i] << " ";
				}
				std::cout << "}" << std::endl;
					
				
			}
            
            if (nblk == 0) {
                std::cout << myrank << ": {empty}" << std::endl;
            }
            
		}
		MPI_Barrier(t_in.comm());
	}
    
    iter.stop();
}
/*
template <typename T>
vec<T> diag(tensor<2,T>& t) {
	
	int myrank = -1;
	int commsize = 0;
	
	MPI_Comm_rank(t.comm(), &myrank); 
	MPI_Comm_size(t.comm(), &commsize);
	
	auto nfull = t.nfull_tot();
	
	int n = nfull[0];
	int m = nfull[1];
	
	if (n != m) throw std::runtime_error("Cannot take diagonal of non-square matrix.");
	
	vec<T> dvec(n, T());
	
	auto blksize = t.blk_size();
	auto blkoff = t.blk_offset();
	
	// loop over diagonal blocks
	for (int D = 0; D != blksize[0].size(); ++D) {
		
		int proc = -1;
		idx2 idx = {D,D};
		
		//std::cout << "BLOCK: " << D << " " << D << std::endl;
		
		t.get_stored_coordinates({.idx = idx, .proc = proc});
		
		//std::cout << "RANK: " << proc << std::endl;
		
		block<2,T> blk;
		int size = blksize[0][D];
		
		int off = blkoff[0][D];
		
		if (proc == myrank) {
			
			bool found = false;
			blk = t.get_block({.idx = idx, .blk_size = {size,size}, .found = found});	
			
			if (found) {
				//std::cout << "FOUND" << std::endl;
			
				for (int d = 0; d != size; ++d) {
					dvec[off + d] = blk(d,d);
				}
				
			}
			
		}
			
		MPI_Bcast(&dvec[off],size,MPI_DOUBLE,proc,t.comm());
		
	}
	
	//print(t);
	
	//if (myrank == 0) {
	//	for (auto x : dvec) {
	//		std::cout << x << " ";
	//	} std::cout << std::endl;
	//}
	
	return dvec;
	
}
                      
*/

} // end namespace dbcsr

#endif
