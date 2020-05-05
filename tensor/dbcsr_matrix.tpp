#ifndef DBCSR_MATRIX_HPP
#define DBCSR_MATRIX_HPP

#:include "dbcsr.fypp"
#include <dbcsr_fwd.hpp>

namespace dbcsr {

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
    
    double norm(const int method) const {
		double out;
		c_dbcsr_norm_scalar(m_matrix_ptr, method, &out);
		return out;
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

// typedefs
#:for i in range(0,4)
#:set type = typelist[i]
#:set suffix = typesuffix[i]
typedef block<2,${type}$> block_${suffix}$;
typedef matrix<${type}$> matrix_${suffix}$;
typedef iterator<${type}$> iterator_${suffix}$;
#:endfor


} // end namespace

#endif