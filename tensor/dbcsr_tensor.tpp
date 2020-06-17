#ifndef DBCSR_TENSOR_HPP
#define DBCSR_TENSOR_HPP

#:include "dbcsr.fypp"
#include <dbcsr_common.hpp>
#include <dbcsr_tensor.h>

namespace dbcsr {

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
    friend class contract_base;
    
    template <int M, typename D>
	friend class copy_base;
	
	template <typename D>
	friend void copy_tensor_to_matrix(tensor<2,D>& t_in, matrix<D>& m_out, std::optional<bool> summation);

	template <typename D>
	friend void copy_matrix_to_tensor(matrix<D>& m_in, tensor<2,D>& t_out, std::optional<bool> summation);

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
    struct create_template {
        ${make_param(structname='create_template',params=list)}$
        private:
            
        required<tensor,ref> c_template;
        
        public:
        
        create_template(tensor& temp) : c_template(temp) {}
        friend class tensor;
    };
    
	tensor(create_template& p) : m_tensor_ptr(nullptr) {
        
        m_comm = p.c_template->m_comm;
		
		c_dbcsr_t_create_template(p.c_template->m_tensor_ptr, &m_tensor_ptr, 
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
		if (!(nzblks[0].size() == nzblks[1].size()
	#:for n in range(2,idim)
		&& nzblks[0].size() == nzblks[${n}$].size()
	#:endfor
		)) throw std::runtime_error("tensor.reserve : wrong dimensions.");
		
		
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
	
	void reserve_template(tensor& t_template) {
		c_dbcsr_t_reserve_blocks_template(t_template.m_tensor_ptr, this->m_tensor_ptr);
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
	
	long long int num_nze_total() const {
		return c_dbcsr_t_get_nze_total(m_tensor_ptr);
	}
	
	void filter(std::optional<T> ieps = std::nullopt, 
        std::optional<int> method = std::nullopt, 
        std::optional<bool> use_absolute = std::nullopt) {
        
		c_dbcsr_t_filter(m_tensor_ptr, (ieps) ? *ieps : filter_eps,
            (method) ? &*method : nullptr, (use_absolute) ? &*use_absolute : &filter_use_absolute);
		
	}
	
	double occupation() {
		
		auto nfull = this->nfull_total();
		long long int tote = std::accumulate(nfull.begin(), nfull.end(), 1, std::multiplies<long long int>());
		long long int nze = this->num_nze_total();
		
		return (double)nze/(double)tote;
		
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
				
				auto blk = t_in.get_block(idx, size, found);
				
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

// typedefs
#:for idim in range(2,MAXDIM+1)
#:for i in range(0,4)
#:set type = typelist[i]
#:set suffix = typesuffix[i]
typedef tensor<${idim}$,${type}$> tensor${idim}$_${suffix}$;
typedef iterator_t<${idim}$,${type}$> iterator_t${idim}$_${suffix}$;
typedef block<${idim}$,${type}$> block${idim}$_${suffix}$;
typedef stensor<${idim}$,${type}$> stensor${idim}$_${suffix}$;
#:endfor
#:endfor

} // end namespace

#endif
