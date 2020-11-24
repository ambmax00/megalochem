#ifndef DBCSR_TENSOR_HPP
#define DBCSR_TENSOR_HPP

#:include "dbcsr.fypp"
#include <dbcsr_common.hpp>
#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.h>

namespace dbcsr {

template <int N, typename>
class pgrid : std::enable_shared_from_this<pgrid<N>> {
protected:

	void* m_pgrid_ptr;
	vec<int> m_dims;
	MPI_Comm m_comm;
	
	template <int M, typename>
	friend class dist_t;
	
	template <int M>
	friend class create_pgrid_base;
	
public:

	pgrid() {}
	
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
		destroy(false);
	}
	
	std::shared_ptr<pgrid<N>> get_ptr() {
		return this->shared_from_this();
	}
	
	template <int M>
	friend class create_pgrid_base;

};

template <int N>
using shared_pgrid = std::shared_ptr<pgrid<N>>;

template <int N>
class create_pgrid_base {
	
	#:set list = [ &
        ['map1', 'vec<int>', 'optional', 'val'],&
        ['map2', 'vec<int>', 'optional', 'val'],&
        ['tensor_dims', 'arr<int,N>', 'optional', 'val'],&
        ['nsplit', 'int', 'optional', 'val'],&
        ['dimsplit', 'int', 'optional', 'val']]
	${make_param(structname='create_pgrid_base',params=list)}$

private:

	MPI_Comm c_comm;
	
public:

	create_pgrid_base(MPI_Comm comm) : c_comm(comm) {}	
	
	shared_pgrid<N> get() {
		
		shared_pgrid<N> spgrid_out = std::make_shared<pgrid<N>>();
		spgrid_out->m_comm = c_comm;
		spgrid_out->m_dims.resize(N);
		
		MPI_Fint fmpi = MPI_Comm_c2f(c_comm);
		
		if (c_map1 && c_map2) {
			c_dbcsr_t_pgrid_create_expert(&fmpi, spgrid_out->m_dims.data(), 
				N, &spgrid_out->m_pgrid_ptr, 
                (c_map1) ? c_map1->data() : nullptr, 
                (c_map1) ? c_map1->size() : 0,
                (c_map2) ? c_map2->data() : nullptr,
                (c_map2) ? c_map2->size() : 0,
                (c_tensor_dims) ? c_tensor_dims->data() : nullptr,
                (c_nsplit) ? &*c_nsplit : nullptr, 
                (c_dimsplit) ? &*c_dimsplit : nullptr);
		} else {
			c_dbcsr_t_pgrid_create(&fmpi, spgrid_out->m_dims.data(), N, 
				&spgrid_out->m_pgrid_ptr, 
                (c_tensor_dims) ? c_tensor_dims->data() : nullptr);
		}
		
		return spgrid_out;
	}
	
};

template <int N>
inline create_pgrid_base<N> create_pgrid(MPI_Comm comm) { 
	return create_pgrid_base<N>(comm);
}

template <int N, typename>
class dist_t {
private:

	void* m_dist_ptr;
	
	arrvec<int,N> m_nd_dists;
	MPI_Comm m_comm;
	
	template <int M, typename T, typename>
	friend class tensor;
	
	template <int M, typename T>
	friend class tensor_create_base;
	
	template <int M, typename T>
	friend class tensor_create_template_base;
	
public:
	
	dist_t() : m_dist_ptr(nullptr) {}

#:for idim in range(2,MAXDIM+1)
    template<int M = N, typename std::enable_if<M == ${idim}$,int>::type = 0>
	dist_t(shared_pgrid<N> spgrid, arrvec<int,N> nd_dists) :
		m_dist_ptr(nullptr),
		m_nd_dists(nd_dists),
        m_comm(spgrid->m_comm)
	{
		
		c_dbcsr_t_distribution_new(&m_dist_ptr, spgrid->m_pgrid_ptr,
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
class tensor : std::enable_shared_from_this<tensor<N,T>> {
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
	friend class tensor_copy_base;
	
	template <int M, typename D>
	friend class tensor_create_base;
	
	template <typename D>
	friend class tensor_create_matrix_base;
	
	template <int M, typename D>
	friend class tensor_create_template_base;
	
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
	reserve(arrvec<int,N>& nzblks) {
			
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
	
	void put_block(const index<N>& idx, T* data, const index<N>& size) {
		c_dbcsr_t_put_block(m_tensor_ptr, idx.data(), size.data(), data, nullptr, nullptr);
	}
	
	block<N,T> get_block(const index<N>& idx, const index<N>& blk_size, bool& found) {
        
		block<N,T> blk_out(blk_size);
        
		c_dbcsr_t_get_block(m_tensor_ptr, idx.data(), blk_size.data(), 
			blk_out.data(), &found);
			
		return blk_out;
			
	}
	
	void get_block(T* data_ptr, const index<N>& idx, const index<N>& blk_size, bool& found) {
		
		c_dbcsr_t_get_block(m_tensor_ptr, idx.data(), blk_size.data(), data_ptr, &found);
		
	}
	

	T* get_block_p(const index<N>& idx, bool& found) {
		
		T* out = nullptr;
		
		c_dbcsr_t_get_block_p (m_tensor_ptr, idx.data(), &out, &found);
		
		return out;
		
	}

	int proc(const index<N>& idx) {
		int p = -1;
		c_dbcsr_t_get_stored_coordinates(m_tensor_ptr, idx.data(), &p);
		return p;
	}
	
	T* data(long long int& data_size) {
		
		T* data_ptr;
		T data_type = T();
		
		c_dbcsr_t_get_data_p(m_tensor_ptr, &data_ptr, &data_size, data_type, nullptr, nullptr);
		
		return data_ptr;
		
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
		c_free_string(&cstring);
        return out;
	}
	
	int num_blocks() const {
		return c_dbcsr_t_get_num_blocks(m_tensor_ptr);
	}
	
	long long int num_blocks_total() const {
		return c_dbcsr_t_get_num_blocks_total(m_tensor_ptr);
	}
	
	int num_nze() {
		return c_dbcsr_t_get_nze(m_tensor_ptr);
	}
	
	long long int num_nze_total() const {
		return c_dbcsr_t_get_nze_total(m_tensor_ptr);
	}
	
	void filter(T eps, 
        std::optional<filter> method = std::nullopt, 
        std::optional<bool> use_absolute = std::nullopt) {
        
        int fmethod = (method) ? static_cast<int>(*method) : 0;
        
		c_dbcsr_t_filter(m_tensor_ptr, eps,
            (method) ? &fmethod : nullptr, 
            (use_absolute) ? &*use_absolute : nullptr);
		
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
    
    void batched_contract_init() {
		//c_dbcsr_t_batched_contract_init(m_tensor_ptr);
	}
	
	void batched_contract_finalize() {
		//c_dbcsr_t_batched_contract_finalize(m_tensor_ptr,nullptr);
	}
	
	vec<int> idx_speed() {
		// returns order of speed of indices
		auto map1 = this->map1_2d();
		auto map2 = this->map2_2d();
		map2.insert(map2.end(),map1.begin(),map1.end());
		
		return map2;
	}
	
	std::shared_ptr<tensor<N,T>> get_ptr() {
		return this->shared_from_this();
	}
    
};


template <int N, typename T = double>
using shared_tensor = std::shared_ptr<tensor<N,T>>;

template <int N, typename T>
class tensor_create_base {

    #:set list = [ &
        ['name', 'std::string', 'required', 'val'],&
        ['ndist','dist_t<N>','optional','ref'],&
        ['pgrid', 'shared_pgrid<N>', 'optional', 'ref'],&
        ['map1', 'vec<int>', 'required', 'val'],&
        ['map2', 'vec<int>', 'required', 'val'],&
        ['blk_sizes', 'arrvec<int,N>', 'required', 'val']]
    ${make_param(structname='tensor_create_base',params=list)}$

public:

    tensor_create_base() = default;

#:for idim in range(2,MAXDIM+1)
    template<int M = N, typename D = T, typename std::enable_if<M == ${idim}$,int>::type = 0>
    shared_tensor<M,D> get() {
		
		shared_tensor<M,D> stensor_out = std::make_shared<tensor<M,D>>();
 
		void* dist_ptr;
		dist_t<N>* distn = nullptr;
		
		if (c_ndist) {
			
			dist_ptr = c_ndist->m_dist_ptr;
            stensor_out->m_comm = c_ndist->m_comm;
			
		} else {
			
			arrvec<int,N> distvecs;
			
			for (int i = 0; i != N; ++i) {
				distvecs[i] = default_dist(c_blk_sizes->at(i).size(),
				(*c_pgrid)->dims()[i], c_blk_sizes->at(i));
			}
			
			distn = new dist_t<N>(*c_pgrid, distvecs);
			dist_ptr = distn->m_dist_ptr;
            stensor_out->m_comm = distn->m_comm;
		}

		c_dbcsr_t_create_new(&stensor_out->m_tensor_ptr, c_name->c_str(), dist_ptr, 
						c_map1->data(), c_map1->size(),
						c_map2->data(), c_map2->size(), &stensor_out->m_data_type, 
						${datasizeptr('c_blk_sizes', 0, idim)}$
                        #:if idim != MAXDIM
                        ,
                        #:endif
                        ${datasize0(MAXDIM-idim)}$);
		
		if (distn) delete distn;
		
		return stensor_out;
					
	}
#:endfor

};

template <int N, typename T = double>
inline tensor_create_base<N,T> tensor_create() {
	return tensor_create_base<N,T>();
}

template <int N, typename T>
class tensor_create_template_base {
	
	 #:set list = [ &
        ['name', 'std::string', 'required', 'val'],&
        ['ndist', 'dist_t<N>', 'optional', 'ref'],&
        ['map1', 'vec<int>', 'optional', 'val'],&
        ['map2', 'vec<int>', 'optional', 'val']]
	
	${make_param(structname='tensor_create_template_base',params=list)}$

private:

	shared_tensor<N,T> c_template;
	
public:
    
    tensor_create_template_base(shared_tensor<N,T> templt) :
		c_template(templt) {}
    
    shared_tensor<N,T> get() {
   
		shared_tensor<N,T> stensor_out 
			= std::make_shared<tensor<N,T>>();
        
        stensor_out->m_comm = c_template->m_comm;
		
		c_dbcsr_t_create_template(c_template->m_tensor_ptr, 
			&stensor_out->m_tensor_ptr, 
            c_name->c_str(), (c_ndist) ? c_ndist->m_dist_ptr : nullptr,
            (c_map1) ? c_map1->data() : nullptr,
            (c_map1) ? c_map1->size() : 0,
            (c_map2) ? c_map2->data() : nullptr,
            (c_map2) ? c_map2->size() : 0,
            &stensor_out->m_data_type);
            
       return stensor_out;
		
	}
	
};    

template <int N, typename T = double>
inline tensor_create_template_base<N,T> 
	tensor_create_template(shared_tensor<N,T> tensor_in) 
{
	return tensor_create_template_base<N,T>(tensor_in);
}

template <typename T>
class tensor_create_matrix_base {
	
	 #:set list = [ &
        ['name', 'std::string', 'optional', 'val'],&
        ['order', 'vec<int>', 'optional', 'val']]
	
	${make_param(structname='tensor_create_matrix_base',params=list)}$

private:

	shared_matrix<T> c_matrix;
	
public:
    
    tensor_create_matrix_base(shared_matrix<T> mat) :
		c_matrix(mat) {}
    
    shared_tensor<2,T> get() {
   
		shared_tensor<2,T> stensor_out 
			= std::make_shared<tensor<2,T>>();
        
        stensor_out->m_comm = c_matrix->get_world().comm();
		
		c_dbcsr_t_create_matrix(
			c_matrix->m_matrix_ptr, &stensor_out->m_tensor_ptr, 
			(c_order) ? c_order->data() : nullptr, 
			(c_name) ? c_name->c_str() : nullptr);
            
       return stensor_out;
		
	}
	
};    

template <typename T = double>
inline tensor_create_matrix_base<T> 
	tensor_create_matrix(shared_matrix<T> mat_in) 
{
	return tensor_create_matrix_base<T>(mat_in);
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
	
	void next_block() {
		
		c_dbcsr_t_iterator_next_block(m_iter_ptr, m_idx.data(), &m_blk_n, nullptr,
			nullptr, nullptr);
		
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
typedef shared_tensor<${idim}$,${type}$> shared_tensor${idim}$_${suffix}$;
#:endfor
#:endfor

} // end namespace

#endif
