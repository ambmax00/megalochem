#ifndef DBCSR_MATRIX_HPP
#define DBCSR_MATRIX_HPP

#:include "dbcsr.fypp"
#include <dbcsr_common.hpp>

namespace dbcsr {

class dist {
protected:

    void* m_dist_ptr;
    world m_world;
    
public:

	template <typename D>
    friend class create_base;
    
    template <typename D>
    friend class create_template_base;
    
    template <typename D>
    friend class transpose_base;
    
    template <typename D>
    friend class matrix_copy_base;

	dist() {}
	
	dist(const dist& d) = delete;
	
	dist(dist&& d) : m_dist_ptr(d.m_dist_ptr), m_world(d.m_world) 
	{
		d.m_dist_ptr = nullptr;
	}

    #:set list = [ &
        ['set_world', 'world', 'required', 'ref'],&
        ['row_dist', 'vec<int>', 'required', 'ref'],&
        ['col_dist', 'vec<int>', 'required', 'ref']]
    ${make_struct(structname='create',friend='dist',params=list)}$

    dist(const create& p) : m_world(*p.c_set_world) {
        c_dbcsr_distribution_new(&m_dist_ptr, p.c_set_world->group(), 
            p.c_row_dist->data(), p.c_row_dist->size(),
            p.c_col_dist->data(), p.c_col_dist->size());
    }
    
    void release() {
		if (m_dist_ptr != nullptr) 
			c_dbcsr_distribution_release(&m_dist_ptr);
    }
    
    ~dist() { release(); }
    
    template <typename T, typename>
	friend class matrix;
    
};

template <typename T = double, typename>
class matrix : std::enable_shared_from_this<matrix<T>> {
protected:
    
    void* m_matrix_ptr;
    world m_world;
    
    const int m_data_type = dbcsr_type<T>::value;
    
    matrix(void* ptr, world& w) : m_matrix_ptr(ptr), m_world(w) {}
    
public:

    matrix() : m_matrix_ptr(nullptr) {}
    
    matrix(const matrix& m) = delete;
    
    matrix(matrix&& m) : m_matrix_ptr(nullptr)
	{
		this->release();
		this->m_matrix_ptr = m.m_matrix_ptr;
		this->m_world = m.m_world;
		m.m_matrix_ptr = nullptr;
	}

	matrix& operator=(matrix&& m) {
		
		if (&m == this) return *this;
		
		this->release();
		this->m_matrix_ptr = m.m_matrix_ptr;
		this->m_world = m.m_world;
		
		m.m_matrix_ptr = nullptr;
		return *this;
		
	}

    typedef T value_type;
    
    template <typename D>
    friend class create_base;
    
    template <typename D>
    friend class create_template_base;
    
    template <typename D>
    friend class transpose_base;
    
    template <typename D>
    friend class matrix_copy_base;
    
    template <typename D>
    friend class multiply_base;
    
	template <typename D>
	friend void copy_tensor_to_matrix(tensor<2,D>& t_in, matrix<D>& m_out, std::optional<bool> summation);

	template <typename D>
	friend void copy_matrix_to_tensor(matrix<D>& m_in, tensor<2,D>& t_out, std::optional<bool> summation);
    
    void set(const T alpha) {
        c_dbcsr_set(m_matrix_ptr, alpha);
    }
    
    // this = alpha * this + beta * mat 
    void add(const T alpha, const T beta, const matrix& matrix_in) {
        c_dbcsr_add(this->m_matrix_ptr, matrix_in.m_matrix_ptr, alpha, beta);
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
   
    std::vector<T> get_diag() {
		int diagsize = this->nfullrows_total();
		std::vector<T> diag(diagsize);
		c_dbcsr_get_diag (m_matrix_ptr, diag.data(), diagsize);
		return diag;
	}
     
    T trace() const {
        T out;
        c_dbcsr_trace(m_matrix_ptr, &out);
        return out;
    }
    
    T dot(const matrix& b) const {
        T out;
        c_dbcsr_dot(m_matrix_ptr, b.m_matrix_ptr, &out);
        return out;
    }
    
    void hadamard_product(const matrix& a, const matrix& b, 
		std::optional<T> assume_value = std::nullopt) {
        c_dbcsr_hadamard_product(a.m_matrix_ptr, b.m_matrix_ptr, m_matrix_ptr, (assume_value) ? &*assume_value : nullptr); 
    }
    
    block<2,T> get_block_p(const int row, const int col, bool& found) {
        T* data = nullptr;
        std::array<int,2> size = {0,0};
    
        c_dbcsr_get_block_p(m_matrix_ptr, row, col, &data, &found, &size[0], &size[1]);
        
        std::cout << "SIZES: " << size[0] << " " << size[1] << std::endl;
        return block<2,T>(size, data);
    }
    
    T* get_block_data(const int row, const int col, bool& found) {
        T* data = nullptr;
        std::array<int,2> size = {0,0};
    
        c_dbcsr_get_block_p(m_matrix_ptr, row, col, &data, &found, &size[0], &size[1]);
        
        return data;
    }
    
    void complete_redistribute(matrix<T>& in, std::optional<bool> keep_sparsity = std::nullopt, 
                                std::optional<bool> summation = std::nullopt) {
    
        c_dbcsr_complete_redistribute(in.m_matrix_ptr, m_matrix_ptr, 
            (keep_sparsity) ? &*keep_sparsity : nullptr, 
            (summation) ? &*summation : nullptr);
            
    }
    
    void filter(double eps, 
				std::optional<filter> method = std::nullopt, 
                std::optional<bool> use_absolute = std::nullopt, 
                std::optional<bool> filter_diag = std::nullopt) {
					
		int method_int = (method) ? static_cast<int>(*method) : 0;
					
        c_dbcsr_filter(m_matrix_ptr, &eps, (method) ? &method_int : nullptr, 
                        (use_absolute) ? &*use_absolute : nullptr, 
                        (filter_diag) ? &*filter_diag : nullptr);
    }

    std::shared_ptr<matrix<T>> get_block_diag(matrix<T>& in) { 
		auto out = std::make_shared<matrix<T>>();
        c_dbcsr_get_block_diag(in.m_matrix_ptr, &out->m_matrix_ptr);
        out->m_world = in.m_world;
        return out;
    }
    
    void copy_in(const matrix& in, std::optional<bool> keep_sp = std::nullopt,
		std::optional<bool> shallow = std::nullopt,
		std::optional<bool> keep_im = std::nullopt,
		std::optional<char> type = std::nullopt) 
	{
		std::string name = this->name();
        c_dbcsr_copy(&m_matrix_ptr, in.m_matrix_ptr, name.c_str(), 
                      (keep_sp) ? &*keep_sp : nullptr, 
                      (shallow) ? &*shallow : nullptr, 
                      (keep_im) ? &*keep_im : nullptr, 
                      (type) ? &*type : nullptr);
    }
    
    std::shared_ptr<matrix<T>> desymmetrize() {
		auto out = std::make_shared<matrix<T>>();
        c_dbcsr_desymmetrize(m_matrix_ptr, &out->m_matrix_ptr);
        out->m_world = this->m_world;
        return out;
    }
    
    void clear() {
        c_dbcsr_clear(&m_matrix_ptr);
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
    
    void reserve_sym() {
		
		int nrowblk = this->nblkrows_total();
		int ncolblk = this->nblkcols_total();
		
		if (nrowblk != ncolblk) throw std::runtime_error("Cannot allocate matrix blocks in a symmatric fashion.");
		
		vec<int> resrows, rescols;
		
		for (int j = 0; j != ncolblk; ++j) {
			for (int i = 0; i <= j; ++i) {
				if (this->proc(i,j) == this->m_world.rank()) {
					resrows.push_back(i);
					rescols.push_back(j);
				}
			}
		}
		
		this->reserve_blocks(resrows,rescols);
		
	}
    
    // reserve block 2d
    
    void put_block(const int row, const int col, block<2,T>& blk, std::optional<bool> sum = std::nullopt,
                    std::optional<T> scale = std::nullopt) {
						
        c_dbcsr_put_block2d(m_matrix_ptr, row, col, blk.data(), blk.size()[0], blk.size()[1], 
                            (sum) ? &*sum : nullptr, (scale) ? &*scale : nullptr);
    }
    
    void put_block_p(const int row, const int col, T* data, const int rowsize, const int colsize,
					std::optional<bool> sum = std::nullopt, std::optional<T> scale = std::nullopt) {
						
        c_dbcsr_put_block2d(m_matrix_ptr, row, col, data, rowsize, colsize, 
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
    
    void fill_random(std::optional<bool> keep_sparsity = std::nullopt) {
		c_dbcsr_init_random(m_matrix_ptr, (keep_sparsity) ? &*keep_sparsity : nullptr);
	}

    void distribute(std::optional<bool> fast = std::nullopt) {
        c_dbcsr_distribute(m_matrix_ptr, (fast) ? &*fast : nullptr);
    }
    
    void redistribute(matrix& mat_in, std::optional<bool> keep_sparsity = std::nullopt,
						std::optional<bool> summation = std::nullopt) {
		c_dbcsr_complete_redistribute(mat_in.m_matrix_ptr, m_matrix_ptr,
			(keep_sparsity) ? &*keep_sparsity : nullptr, (summation) ? &*summation : nullptr);
	}
    
    void sum_replicated() {
        c_dbcsr_sum_replicated(m_matrix_ptr);
    }
    
    void finalize() {
        c_dbcsr_finalize(m_matrix_ptr);
    }

    void release() {
        if (m_matrix_ptr != nullptr) c_dbcsr_release(&m_matrix_ptr);
        m_matrix_ptr = nullptr;
    }
    
    ~matrix() { release(); }
    
    world get_world() const {
		return m_world;
	}
    
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

#:set vars = ['local_rows', 'local_cols', 'proc_row_dist', 'proc_col_dist', &
                'row_blk_size', 'col_blk_size', 'row_blk_offset', 'col_blk_offset']
#:for i in range(0,len(vars))
#:set var = vars[i]
#:set name = var
#:if var in ['row_blk_size', 'col_blk_size', 'row_blk_offset', 'col_blk_offset']
#:set name = var + 's'
#:endif
    std::vector<int> ${name}$() const {
#:set rowcol = 'rows'
#:set loctot = 'total'
#:if 'local' in var
    #:set loctot = 'local'
#:endif
#:if i % 2 != 0
    #:set rowcol = 'cols'
#:endif
        std::vector<int> out(this->nblk${rowcol}$_${loctot}$(),0);
        //std::cout << out.size() << std::endl;
        
        c_dbcsr_get_${var}$(m_matrix_ptr, out.data());
                             
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
    
    type matrix_type() const {
        char out;
        c_dbcsr_get_info(m_matrix_ptr, ${repeat('nullptr',20)}$, &out, nullptr, nullptr);
        return static_cast<type>(out);
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
    
    double norm(const int method) const {
		double out;
		c_dbcsr_norm_scalar(m_matrix_ptr, method, &out);
		return out;
	}
	
	void apply(func my_func, std::optional<double> a0 = std::nullopt, 
		std::optional<double> a1 = std::nullopt,
		std::optional<double> a2 = std::nullopt) 
	{
	
		int f = static_cast<int>(my_func);
			
		c_dbcsr_function_of_elements(m_matrix_ptr, f,
			(a0) ? &*a0 : nullptr, (a1) ? &*a1 : nullptr, (a2) ? &*a2 : nullptr);		
	}
	
	std::shared_ptr<matrix<T>> get_ptr() {
		return this->shared_from_this();
	}
    
};  

template <typename T>
using shared_matrix = std::shared_ptr<matrix<T>>;

template <typename T>
class create_base {
	
#:set list = [ &
        ['name', 'std::string', 'required', 'val'], &
        ['set_world', 'world', 'optional', 'ref'],&
        ['set_dist', 'dist', 'optional', 'ref'],&
        ['matrix_type', 'type', 'required', 'val'],&
        ['row_blk_sizes', 'vec<int>', 'required', 'ref'],&
        ['col_blk_sizes', 'vec<int>', 'required', 'ref'],&
        ['nze', 'int', 'optional', 'val'],&
        ['reuse', 'bool', 'optional', 'val'],&
        ['reuse_arrays', 'bool', 'optional', 'val'],&
        ['mutable_work', 'bool', 'optional', 'val'],&
        ['replication_type', 'repl', 'optional', 'val']]
    ${make_param(structname='create_base',params=list)}$
    
public:

	create_base() = default;

	shared_matrix<T> get() {
		
		shared_matrix<T> m_out = std::make_shared<matrix<T>>();
		
		m_out->m_matrix_ptr = nullptr;
		m_out->m_world = (c_set_world) ? (*c_set_world) : c_set_dist->m_world;
		
        dist* dist_ptr = nullptr;
        
        if (!c_set_dist) {
			
            vec<int> rowdist, coldist; 
            auto dims = c_set_world->dims();
            
            auto rdist = default_dist(c_row_blk_sizes->size(),
                            dims[0], *c_row_blk_sizes);
            auto cdist = default_dist(c_col_blk_sizes->size(),
                            dims[1], *c_col_blk_sizes);
                            
            dist_ptr = new dist(dist::create().set_world(*c_set_world).row_dist(rdist).col_dist(cdist));
        } else {
            dist_ptr = &*c_set_dist;
        }
        
        char mat_type = static_cast<char>(*c_matrix_type);
        char repl_type = (c_replication_type) ? static_cast<char>(*c_replication_type) : '0';
        
        c_dbcsr_create_new(&m_out->m_matrix_ptr, c_name->c_str(), dist_ptr->m_dist_ptr, mat_type, 
                               c_row_blk_sizes->data(), c_row_blk_sizes->size(), 
                               c_col_blk_sizes->data(), c_col_blk_sizes->size(), 
                               (c_nze) ? &*c_nze : nullptr, &m_out->m_data_type, 
                               (c_reuse) ? &*c_reuse : nullptr,
                               (c_reuse_arrays) ? &*c_reuse_arrays : nullptr, 
                               (c_mutable_work) ? &*c_mutable_work : nullptr, 
                               (c_replication_type) ? &repl_type : nullptr);
    
        if (!c_set_dist) delete dist_ptr;
        
        return m_out;
        
    }
    
};

template <typename T>
inline create_base<T> create() {
	return create_base<T>();
}

template <typename T>
class create_template_base {

#:set list = [ &
	['name', 'std::string', 'required', 'val'], &
	['set_dist', 'dist', 'optional', 'ref'],&
	['matrix_type', 'type', 'optional', 'val'],&
	['row_blk_sizes', 'vec<int>', 'optional', 'ref'],&
	['col_blk_sizes', 'vec<int>', 'optional', 'ref'],&
	['nze', 'int', 'optional', 'val'],&
	['reuse', 'const bool', 'optional', 'val'],&
	['reuse_arrays', 'const bool', 'optional', 'val'],&
	['mutable_work', 'const bool', 'optional', 'val'],&
	['replication_type', 'repl', 'optional', 'val']]
	
	${make_param(structname='create_template_base',params=list)}$

private:
		
	shared_matrix<T> c_template;
	
public:
	
	create_template_base(shared_matrix<T>& temp) : c_template(temp) {}

	shared_matrix<T> get() {
		
		shared_matrix<T> m_out = std::make_shared<matrix<T>>();
		
		m_out->m_matrix_ptr = nullptr;
		m_out->m_world = c_template->m_world;
		
		char mat_type = (c_matrix_type) ? static_cast<char>(*c_matrix_type) : '0';
		char repl_type = (c_replication_type) ? static_cast<char>(*c_replication_type) : '0';
 
        c_dbcsr_create_template(&m_out->m_matrix_ptr, 
			c_name->c_str(), c_template->m_matrix_ptr, 
			(c_set_dist) ? c_set_dist->m_dist_ptr : nullptr, 
			(c_matrix_type) ? &mat_type : nullptr, 
			(c_row_blk_sizes) ? c_row_blk_sizes->data() : nullptr,
			(c_row_blk_sizes) ? c_row_blk_sizes->size() : 0,
			(c_col_blk_sizes) ? c_col_blk_sizes->data() : nullptr,
			(c_col_blk_sizes) ? c_col_blk_sizes->size() : 0,
			(c_nze) ? &*c_nze : nullptr, &m_out->m_data_type, 
			(c_reuse_arrays) ? &*c_reuse_arrays : nullptr, 
			(c_mutable_work) ? &*c_mutable_work : nullptr, 
			(c_replication_type) ? &repl_type : nullptr);
                                   
		return m_out;

    }
    
};

template <typename T>
inline create_template_base<T> 
	create_template(shared_matrix<T>& m_in) {
		return create_template_base<T>(m_in);
}

template <typename T>
class transpose_base {
	
#:set list = [ &
    ['shallow_copy', 'bool', 'optional', 'val'],&
    ['transpose_data', 'bool', 'optional', 'val'],& 
    ['transpose_dist', 'bool', 'optional', 'val'],& 
    ['use_dist', 'bool', 'optional', 'val']]
    
    ${make_param('transpose_base', list)}$

private:
	
	shared_matrix<T> c_in;

public:
    
	transpose_base(shared_matrix<T>& m_in) : c_in(m_in) {}
    
    shared_matrix<T> get() {
		
		shared_matrix<T> m_out = std::make_shared<matrix<T>>();
    
        m_out->m_world = c_in->m_world;
        c_dbcsr_transposed(&m_out->m_matrix_ptr, c_in->m_matrix_ptr, 
                            (c_shallow_copy) ? &*c_shallow_copy : nullptr,
                            (c_transpose_data) ? &*c_transpose_data : nullptr, 
                            (c_transpose_dist) ? &*c_transpose_dist : nullptr, 
                            (c_use_dist) ? &*c_use_dist : nullptr);
                            
        return m_out;                            
    
    }
    
};

template <typename T>
inline transpose_base<T> transpose(shared_matrix<T>& m_in) {
	return transpose_base<T>(m_in);
}


template <typename T>
class matrix_copy_base {

#:set list = [ &
    ['name', 'std::string', 'optional', 'val'],&
    ['keep_sparsity', 'bool', 'optional', 'val'],&
    ['shallow_data', 'bool', 'optional', 'val'],&
    ['keep_imaginary', 'bool', 'optional', 'val'],&
    ['matrix_type', 'type', 'optional', 'val']]
    
    ${make_param('matrix_copy_base', list)}$

private:
    
	shared_matrix<T> c_in;
	
public:
	
	matrix_copy_base(shared_matrix<T>& in) : c_in(in) {}
	
	shared_matrix<T> get() {
		
		shared_matrix<T> m_out = std::make_shared<matrix<T>>();
		
		m_out->m_matrix_ptr = nullptr;
		m_out->m_world = c_in->m_world;
		
		char mat_type = (c_matrix_type) ? static_cast<char>(*c_matrix_type) : '0';
		
		c_dbcsr_copy(&m_out->m_matrix_ptr, c_in->m_matrix_ptr, 
			(c_name) ? c_name->c_str() : nullptr, 
			(c_keep_sparsity) ? &*c_keep_sparsity : nullptr, 
			(c_shallow_data) ? &*c_shallow_data : nullptr, 
			(c_keep_imaginary) ? &*c_keep_imaginary : nullptr, 
			(c_matrix_type) ? &mat_type : nullptr);
			
		return m_out;
		
	}

};

template <typename T>
inline matrix_copy_base<T> copy(shared_matrix<T>& m_in) {
	return matrix_copy_base<T>(m_in);
}

template <typename T>
class read_base {

#:set list = [&
    ['filepath', 'std::string', 'required', 'val'],&
    ['distribution', 'dist', 'required', 'ref'],&
    ['set_world', 'world', 'required', 'ref']]
    ${make_param(structname='read_base',params=list)}$

public:

	read_base() = default;
	
	shared_matrix<T> get() {
		
		shared_matrix<T> m_out = std::make_shared<matrix<T>>();
		
		m_out->m_matrix_ptr = nullptr;
		m_out->m_world = c_set_world;
		
        c_dbcsr_binary_read(c_filepath->c_str(), c_distribution->m_dist_ptr, 
			c_set_world->comm(), &m_out->m_matrix_ptr);
			
		return m_out;
    }
    
};

template <typename T>
inline read_base<T> read() {
	return read_base<T>();
}

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
    
    double norm() {
		double out = 0.0;
		for (int n = 0; n != m_row_size*m_col_size; ++n) {
			out += pow(m_blk_ptr[n],2);
		}
		return sqrt(out);
	}
    
#:set list = ['row', 'col', 'iblk', 'blk_p', 'row_size', 'col_size', 'row_offset', 'col_offset']
#:for var in list
    int ${var}$() { return m_${var}$; }
#:endfor

    T* data() { return m_blk_ptr; }	
    
};

template <typename T>
void print(matrix<T>& mat) {
		
	world w = mat.get_world();
	
	iterator<T> iter(mat);
	
	iter.start();
	
	if (w.rank() == 0) {
		std::cout << "Matrix " << mat.name() << std::endl;
	}
	
	for (int irank = 0; irank != w.size(); ++irank) {
		
		int nblk = 0;
		
		if (irank == w.rank()) {
	
			while (iter.blocks_left()) {
				
				++nblk;
				
				iter.next_block();
				
				std::cout << "Rank: " << irank << " ";
				std::cout << "(" << iter.row() << "," << iter.col() << ") ";
				std::cout << "[" << iter.row_size() << "," << iter.col_size() << "] {";
				
				for (int i = 0; i != iter.row_size() * iter.col_size(); ++i) {
					std::cout << iter.data()[i] << " ";
				} std::cout << "}" << std::endl;
				
			}
			
			if (nblk == 0) {
				std::cout << "Rank: " << irank << " " << "{empty}" << std::endl;
			}
			
		}
		
		MPI_Barrier(w.comm());
		
	}
	
	iter.stop();
		
}


// typedefs
#:for i in range(0,4)
#:set type = typelist[i]
#:set suffix = typesuffix[i]
typedef block<2,${type}$> block_${suffix}$;
typedef matrix<${type}$> mat_${suffix}$;
typedef smatrix<${type}$> smat_${suffix}$;
typedef iterator<${type}$> iter_${suffix}$;
#:endfor


} // end namespace

#endif
