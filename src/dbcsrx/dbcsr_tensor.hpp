#ifndef DBCSR_TENSOR_HPP
#define DBCSR_TENSOR_HPP

#ifndef TEST_MACRO
  #include <dbcsr_tensor.h>
  #include <dbcsr_common.hpp>
  #include <dbcsr_matrix.hpp>
#endif

#include "utils/ppdirs.hpp"

#define MAXDIM 4

#define DATASIZE(x, n) \
  (n < N) ? x[n].data() : nullptr, (n < N) ? x[n].size() : 0

namespace dbcsr {

template <int N, typename>
class pgrid : std::enable_shared_from_this<pgrid<N>> {
 protected:
  void* m_pgrid_ptr;
  vec<int> m_dims;
  MPI_Comm m_comm;

  template <int M, typename>
  friend class dist_t;

 public:
#define PGRID_PARAMS \
  (((util::optional<vec<int>>), map1), ((util::optional<vec<int>>), map2), \
   ((util::optional<arr<int, N>>), tensor_dims), \
   ((util::optional<int>), nsplit), ((util::optional<int>), dimsplit))

#define PGRID_INIT (((MPI_Comm), comm))

  MAKE_PARAM_STRUCT(create, PGRID_PARAMS, PGRID_INIT)
  MAKE_BUILDER_CLASS(pgrid, create, PGRID_PARAMS, PGRID_INIT)

  pgrid(create_pack&& p)
  {
    m_comm = p.p_comm;
    m_dims.resize(N);

    MPI_Fint fmpi = MPI_Comm_c2f(p.p_comm);

    if (p.p_map1 && p.p_map2) {
      c_dbcsr_t_pgrid_create_expert(
          &fmpi, m_dims.data(), N, &m_pgrid_ptr,
          (p.p_map1) ? p.p_map1->data() : nullptr,
          (p.p_map1) ? p.p_map1->size() : 0,
          (p.p_map2) ? p.p_map2->data() : nullptr,
          (p.p_map2) ? p.p_map2->size() : 0,
          (p.p_tensor_dims) ? p.p_tensor_dims->data() : nullptr,
          (p.p_nsplit) ? &*p.p_nsplit : nullptr,
          (p.p_dimsplit) ? &*p.p_dimsplit : nullptr);
    }
    else {
      c_dbcsr_t_pgrid_create(
          &fmpi, m_dims.data(), N, &m_pgrid_ptr,
          (p.p_tensor_dims) ? p.p_tensor_dims->data() : nullptr);
    }
  }

  pgrid()
  {
  }

  pgrid(pgrid<N>& rhs) = delete;

  pgrid<N>& operator=(pgrid<N>& rhs) = delete;

  vec<int> dims() const
  {
    return m_dims;
  }

  void destroy(bool keep_comm = false)
  {
    if (m_pgrid_ptr != nullptr)
      c_dbcsr_t_pgrid_destroy(&m_pgrid_ptr, &keep_comm);

    m_pgrid_ptr = nullptr;
  }

  MPI_Comm comm() const
  {
    return m_comm;
  }

  ~pgrid()
  {
    destroy(false);
  }

  std::shared_ptr<pgrid<N>> get_ptr() const
  {
    return this->shared_from_this();
  }
};

template <int N>
using shared_pgrid = std::shared_ptr<pgrid<N>>;

template <int N, typename>
class dist_t {
 private:
  void* m_dist_ptr;

  arrvec<int, N> m_nd_dists;
  MPI_Comm m_comm;

  template <int M, typename T, typename>
  friend class tensor;

  template <int M, typename T>
  friend class tensor_create_base;

  template <int M, typename T>
  friend class tensor_create_template_base;

 public:
  dist_t() : m_dist_ptr(nullptr)
  {
  }

#define DIST_T_PLIST (((pgrid<N>&), set_pgrid), ((arrvec<int, N>&), nd_dists))

  MAKE_PARAM_STRUCT(create, DIST_T_PLIST, ())
  MAKE_BUILDER_CLASS(dist_t, create, DIST_T_PLIST, ())

  dist_t(create_pack&& p)
  {
    m_dist_ptr = nullptr;
    m_nd_dists = p.p_nd_dists;
    m_comm = p.p_set_pgrid.comm();

    c_dbcsr_t_distribution_new(
        &m_dist_ptr, p.p_set_pgrid.m_pgrid_ptr,
        REPEAT_FIRST(DATASIZE, m_nd_dists, 0, MAXDIM, (PPDIRS_COMMA), ()));
  }

  dist_t(dist_t<N>& rhs) = delete;

  dist_t<N>& operator=(dist_t<N>& rhs) = delete;

  ~dist_t()
  {
    if (m_dist_ptr != nullptr)
      c_dbcsr_t_distribution_destroy(&m_dist_ptr);

    m_dist_ptr = nullptr;
  }

  void destroy()
  {
    if (m_dist_ptr != nullptr) {
      c_dbcsr_t_distribution_destroy(&m_dist_ptr);
    }

    m_dist_ptr = nullptr;
  }
};

template <int N, typename T = double, typename>
class tensor : std::enable_shared_from_this<tensor<N, T>> {
 protected:
  void* m_tensor_ptr;
  MPI_Comm m_comm;

  const int m_data_type = dbcsr_type<T>::value;

  tensor(void* ptr) : m_tensor_ptr(ptr)
  {
  }

 public:
  friend class iterator_t<N, T>;

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
  friend void copy_tensor_to_matrix(
      tensor<2, D>& t_in, matrix<D>& m_out, std::optional<bool> summation);

  template <typename D>
  friend void copy_matrix_to_tensor(
      matrix<D>& m_in, tensor<2, D>& t_out, std::optional<bool> summation);

  typedef T value_type;
  const static int dim = N;

  // =====================================================================
  //                             CONSTRUCTORS
  // =====================================================================

#define TENSOR_CREATE_PLIST \
  (((std::string), name), ((util::optional<dist_t<N>&>), set_dist), \
   ((util::optional<pgrid<N>&>), set_pgrid), ((vec<int>), map1), \
   ((vec<int>), map2), ((arrvec<int, N>), blk_sizes))

  MAKE_PARAM_STRUCT(create, TENSOR_CREATE_PLIST, ())
  MAKE_BUILDER_CLASS(tensor, create, TENSOR_CREATE_PLIST, ())

  tensor(create_pack&& p)
  {
    void* dist_ptr = nullptr;
    std::shared_ptr<dist_t<N>> distn;

    if (p.p_set_dist) {
      dist_ptr = p.p_set_dist->m_dist_ptr;
      m_comm = p.p_set_dist->m_comm;
    }
    else {
      arrvec<int, N> distvecs;
      auto pgrid_dims = p.p_set_pgrid->dims();

      for (int i = 0; i != N; ++i) {
        distvecs[i] = default_dist(
            p.p_blk_sizes[i].size(), pgrid_dims[i], p.p_blk_sizes[i]);
      }

      distn = dist_t<N>::create()
                  .set_pgrid(*p.p_set_pgrid)
                  .nd_dists(distvecs)
                  .build();

      dist_ptr = distn->m_dist_ptr;
      m_comm = distn->m_comm;
    }

    c_dbcsr_t_create_new(
        &m_tensor_ptr, p.p_name.c_str(), dist_ptr, p.p_map1.data(),
        p.p_map1.size(), p.p_map2.data(), p.p_map2.size(), &m_data_type,
        REPEAT_FIRST(DATASIZE, p.p_blk_sizes, 0, MAXDIM, (PPDIRS_COMMA), ()));
  }

#define TENSOR_TEMPLATE_ILIST (((tensor<N, T>&), templet))

#define TENSOR_TEMPLATE_PLIST \
  (((std::string), name), ((util::optional<dist_t<N>&>), ndist), \
   ((util::optional<vec<int>>), map1), ((util::optional<vec<int>>), map2))

  MAKE_PARAM_STRUCT(
      create_template, TENSOR_TEMPLATE_PLIST, TENSOR_TEMPLATE_ILIST)
  MAKE_BUILDER_CLASS(
      tensor, create_template, TENSOR_TEMPLATE_PLIST, TENSOR_TEMPLATE_ILIST)

  tensor(create_template_pack&& p)
  {
    m_comm = p.p_templet.m_comm;

    c_dbcsr_t_create_template(
        p.p_templet.m_tensor_ptr, &m_tensor_ptr, p.p_name.c_str(),
        (p.p_ndist) ? p.p_ndist->m_dist_ptr : nullptr,
        (p.p_map1) ? p.p_map1->data() : nullptr,
        (p.p_map1) ? p.p_map1->size() : 0,
        (p.p_map2) ? p.p_map2->data() : nullptr,
        (p.p_map2) ? p.p_map2->size() : 0, &m_data_type);
  }

#define TENSOR_CREATE_MATRIX_ILIST (((matrix<T>&), matrix_in))

#define TENSOR_CREATE_MATRIX_PLIST \
  (((util::optional<std::string>), name), ((util::optional<vec<int>>), order))

  MAKE_PARAM_STRUCT(
      create_matrix, TENSOR_CREATE_MATRIX_PLIST, TENSOR_CREATE_MATRIX_ILIST)
  MAKE_BUILDER_CLASS(
      tensor,
      create_matrix,
      TENSOR_CREATE_MATRIX_PLIST,
      TENSOR_CREATE_MATRIX_ILIST)

  tensor(create_matrix_pack&& p)
  {
    m_comm = p.p_matrix_in.get_cart().comm();

    c_dbcsr_t_create_matrix(
        p.p_matrix.m_matrix_ptr, &m_tensor_ptr,
        (p.p_order) ? p.p_order->data() : nullptr,
        (p.p_name) ? p.p_name->c_str() : nullptr);
  }

  tensor() : m_tensor_ptr(nullptr)
  {
  }

  tensor(const tensor& in) = delete;

  tensor& operator=(const tensor& in) = delete;

  tensor(tensor&& in)
  {
    this->m_tensor_ptr = in.m_tensor_ptr;
    this->m_comm = in.m_comm;
    in.m_tensor_ptr = nullptr;
    in.m_comm = MPI_COMM_NULL;
  }

  tensor& operator=(tensor&& in)
  {
    if (this == &in)
      return *this;
    this->m_tensor_ptr = in.m_tensor_ptr;
    this->m_comm = in.m_comm;
    in.m_tensor_ptr = nullptr;
    in.m_comm = MPI_COMM_NULL;
  }

  ~tensor()
  {
    // if (m_tensor_ptr != nullptr) std::cout << "Destroying: " << this->name()
    // << std::endl;
    destroy();
  }

  void destroy()
  {
    if (m_tensor_ptr != nullptr) {
      c_dbcsr_t_destroy(&m_tensor_ptr);
    }

    m_tensor_ptr = nullptr;
  }

  // =====================================================================
  //                             GET INFO
  // =====================================================================

#define TENSOR_INFO_1 \
  (nblks_total, nfull_total, nblks_local, nfull_local, pdims, my_ploc)

#define GEN_TENSOR_INFO_1(x, n) \
  arr<int, N> ECHO(CAT(GET_, n) TENSOR_INFO_1)() const \
  { \
    arr<int, N> out; \
    c_dbcsr_t_get_info( \
        m_tensor_ptr, N, \
        REPEAT_SECOND( \
            ECHO_P, nullptr, 0, SUB(n, 1), (PPDIRS_COMMA), (PPDIRS_COMMA)) \
            out.data(), \
        REPEAT_SECOND( \
            ECHO_P, nullptr, 0, SUB(LISTSIZE(TENSOR_INFO_1), n), (, ), \
            (, )) REPEAT_SECOND(ECHO_P, 0, 0, MAXDIM, (, ), (, )) \
            REPEAT_SECOND(ECHO_P, 0, 0, MAXDIM, (, ), (, )) REPEAT_SECOND( \
                ECHO_P, nullptr, 0, MAXDIM, (, ), (, )) \
                REPEAT_SECOND(ECHO_P, nullptr, 0, MAXDIM, (, ), (, )) \
                    REPEAT_SECOND(ECHO_P, nullptr, 0, MAXDIM, (, ), (, )) \
                        REPEAT_SECOND( \
                            ECHO_P, nullptr, 0, MAXDIM, (, ), (, )) nullptr, \
        nullptr, nullptr); \
    return out; \
  }

  REPEAT_FIRST(GEN_TENSOR_INFO_1, UNUSED, 1, LISTSIZE(TENSOR_INFO_1), (), ())

#define TENSOR_INFO_2 (blks_local, proc_dist, blk_sizes, blk_offsets)

#define ECHO_LOC_NBLKS(x, n) sizes[n] = c_dbcsr_t_nblks_local(m_tensor_ptr, n);

#define ECHO_TOT_NBLKS(x, n) sizes[n] = c_dbcsr_t_nblks_total(m_tensor_ptr, n);

#define ECHO_OUTVEC(x, n) out[n] = vec<int>(sizes[n]);

#define ECHO_SIZE(x, i) sizes[i]

#define REPEAT_X(x, i) REPEAT_THIRD(ECHO_P, nullptr, 0, MAXDIM, (, ), ())

#define OUT_DATA(x, i) out[i].data()

#define GEN_TENSOR_INFO_2_BASE(idim, ivar) \
  template <int M = N> \
  typename std::enable_if<M == idim, arrvec<int, N>>::type ECHO( \
      CAT(GET_, ivar) TENSOR_INFO_2)() const \
  { \
    arrvec<int, N> out; \
    vec<int> sizes(N); \
    if constexpr (STRING_EQUAL(GET(TENSOR_INFO_2, ivar), blks_local)) { \
      REPEAT_THIRD(ECHO_LOC_NBLKS, UNUSED, 0, idim, (), ()) \
    } \
    else { \
      REPEAT_THIRD(ECHO_TOT_NBLKS, UNUSED, 0, idim, (), ()) \
    } \
    REPEAT_THIRD(ECHO_OUTVEC, UNUSED, 0, idim, (), ()) \
    if constexpr (STRING_EQUAL(GET(TENSOR_INFO_2, ivar), blks_local)) { \
      c_dbcsr_t_get_info( \
          m_tensor_ptr, N, \
          REPEAT_SECOND(ECHO_P, nullptr, 0, 6, (, ), (, )) REPEAT_SECOND( \
              ECHO_SIZE, UNUSED, 0, idim, (, ), \
              (, )) REPEAT_SECOND(ECHO_P, 0, 0, SUB(MAXDIM, idim), (, ), (, )) \
              REPEAT_SECOND(ECHO_P, 0, 0, MAXDIM, (, ), (, )) REPEAT_SECOND( \
                  REPEAT_X, UNUSED, 0, DEC(ivar), (, ), (, )) \
                  REPEAT_SECOND(OUT_DATA, UNUSED, 0, idim, (, ), (, )) \
                      REPEAT_SECOND( \
                          ECHO_P, nullptr, 0, SUB(MAXDIM, idim), (, ), (, )) \
                          REPEAT_SECOND( \
                              REPEAT_X, UNUSED, 0, \
                              SUB(LISTSIZE(TENSOR_INFO_2), ivar), (, ), \
                              (, )) nullptr, \
          nullptr, nullptr); \
    } \
    else { \
      c_dbcsr_t_get_info( \
          m_tensor_ptr, N, \
          REPEAT_SECOND(ECHO_P, nullptr, 0, 6, (, ), (, )) REPEAT_SECOND( \
              ECHO_P, 0, 0, MAXDIM, (, ), \
              (, )) REPEAT_SECOND(ECHO_SIZE, UNUSED, 0, idim, (, ), (, )) \
              REPEAT_SECOND(ECHO_P, 0, 0, SUB(MAXDIM, idim), (, ), (, )) \
                  REPEAT_SECOND(REPEAT_X, UNUSED, 0, DEC(ivar), (, ), (, )) \
                      REPEAT_SECOND(OUT_DATA, UNUSED, 0, idim, (, ), (, )) \
                          REPEAT_SECOND( \
                              ECHO_P, nullptr, 0, SUB(MAXDIM, idim), (, ), \
                              (, )) \
                              REPEAT_SECOND( \
                                  REPEAT_X, UNUSED, 0, \
                                  SUB(LISTSIZE(TENSOR_INFO_2), ivar), (, ), \
                                  (, )) nullptr, \
          nullptr, nullptr); \
    } \
\
    return out; \
  }

  REPEAT_FIRST(GEN_TENSOR_INFO_2_BASE, 2, 1, LISTSIZE(TENSOR_INFO_2), (), ())
  REPEAT_FIRST(GEN_TENSOR_INFO_2_BASE, 3, 1, LISTSIZE(TENSOR_INFO_2), (), ())
  REPEAT_FIRST(GEN_TENSOR_INFO_2_BASE, 4, 1, LISTSIZE(TENSOR_INFO_2), (), ())

  // =====================================================================
  //                             GET MAP INFO
  // =====================================================================

#define TENSOR_MAPINFO_1 (ndim_nd, ndim1_2d, ndim2_2d)

#define GEN_TENSOR_MAPINFO_1(x, n) \
  int GET(TENSOR_MAPINFO_1, n)() const \
  { \
    int c_out; \
    c_dbcsr_t_get_mapping_info( \
        m_tensor_ptr, N, 0, 0, \
        REPEAT_SECOND(ECHO_P, nullptr, 0, SUB(n, 1), (, ), (, )) & c_out, \
        REPEAT_SECOND( \
            ECHO_P, nullptr, 0, SUB(LISTSIZE(TENSOR_MAPINFO_1), n), (, ), \
            (, )) REPEAT_SECOND(ECHO_P, nullptr, 0, 10, (, ), ())); \
    return c_out; \
  }

  REPEAT_FIRST(
      GEN_TENSOR_MAPINFO_1, UNUSED, 1, LISTSIZE(TENSOR_MAPINFO_1), (), ())

#define TENSOR_MAPINFO_2 \
  ((long long int, dims_2d_i8), (int, dims_2d), (int, dims_nd), \
   (int, dims1_2d), (int, dims2_2d), (int, map1_2d), (int, map2_2d), \
   (int, map_nd))

#define GEN_TENSOR_MAPINFO_2_BASE(ctype, var, i) \
  vec<ctype> var() const \
  { \
    int nd_size = N; \
    int nd_row_size = c_dbcsr_t_ndims_matrix_row(m_tensor_ptr); \
    int nd_col_size = c_dbcsr_t_ndims_matrix_column(m_tensor_ptr); \
\
    int vsize; \
    if constexpr ( \
        STRING_EQUAL(var, dims_2d_i8) || STRING_EQUAL(var, dims_2d)) { \
      vsize = 2; \
    } \
    else if (STRING_EQUAL(var, dims_nd) || STRING_EQUAL(var, map_nd)) { \
      vsize = nd_size; \
    } \
    else if (STRING_EQUAL(var, dims1_2d) || STRING_EQUAL(var, map1_2d)) { \
      vsize = nd_row_size; \
    } \
    else { \
      vsize = nd_col_size; \
    } \
    vec<ctype> out(vsize); \
    c_dbcsr_t_get_mapping_info( \
        m_tensor_ptr, N, nd_row_size, nd_col_size, nullptr, nullptr, nullptr, \
        REPEAT_SECOND(ECHO_P, nullptr, 0, SUB(i, 1), (, ), (, )) out.data(), \
        REPEAT_SECOND( \
            ECHO_P, nullptr, 0, SUB(LISTSIZE(TENSOR_MAPINFO_2), i), (, ), \
            (, )) nullptr, \
        nullptr); \
    return out; \
  }

#define GEN_TENSOR_MAPINFO_2(x, i) \
  GEN_TENSOR_MAPINFO_2_BASE( \
      GET_1 GET(TENSOR_MAPINFO_2, i), GET_2 GET(TENSOR_MAPINFO_2, i), i)

  REPEAT_FIRST(
      GEN_TENSOR_MAPINFO_2, UNUSED, 1, LISTSIZE(TENSOR_MAPINFO_2), (), ())

#define TENSOR_RESERVE_SUB0(x, ii) nzblks[0].size() == nzblks[ii].size()

#define TENSOR_RESERVE_SUB1(x, ii) nzblks[ii].data()

#define TENSOR_RESERVE(x, idim) \
  template <int M = N> \
  typename std::enable_if<M == idim>::type reserve(arrvec<int, N>& nzblks) \
  { \
    if (nzblks[0].size() == 0) \
      return; \
    if (!(REPEAT_SECOND(TENSOR_RESERVE_SUB0, UNUSED, 0, idim, (&&), ()))) { \
      throw std::runtime_error("tensor.reserve : wrong dimensions."); \
    } \
\
    c_dbcsr_t_reserve_blocks_index( \
        m_tensor_ptr, nzblks[0].size(), \
        REPEAT_SECOND(TENSOR_RESERVE_SUB1, UNUSED, 0, idim, (, ), ()) \
            REPEAT_SECOND(ECHO_NONE, UNUSED, 0, SUB(MAXDIM, idim), (), (, )) \
                REPEAT_SECOND( \
                    ECHO_P, nullptr, 0, SUB(MAXDIM, idim), (, ), ())); \
  }

  REPEAT_FIRST(TENSOR_RESERVE, UNUSED, 2, SUB(MAXDIM, 1), (), ())

  void reserve_all()
  {
    auto blks = this->blks_local();
    arrvec<int, N> res;

    std::function<void(int, int*)> loop;

    int* arr = new int[N];

    loop = [&res, &blks, &loop](int depth, int* vals) {
      for (auto eleN : blks[depth]) {
        vals[depth] = eleN;
        if (depth == N - 1) {
          for (int i = 0; i != N; ++i) { res[i].push_back(vals[i]); }
        }
        else {
          loop(depth + 1, vals);
        }
      }
    };

    loop(0, arr);
    this->reserve(res);

    delete[] arr;
  }

  void reserve_template(tensor& t_template)
  {
    c_dbcsr_t_reserve_blocks_template(
        t_template.m_tensor_ptr, this->m_tensor_ptr);
  }

  void put_block(
      const index<N>& idx,
      block<N, T>& blk,
      std::optional<bool> sum = std::nullopt,
      std::optional<double> scale = std::nullopt)
  {
    c_dbcsr_t_put_block(
        m_tensor_ptr, idx.data(), blk.size().data(), blk.data(),
        (sum) ? &*sum : nullptr, (scale) ? &*scale : nullptr);
  }

  void put_block(const index<N>& idx, T* data, const index<N>& size)
  {
    c_dbcsr_t_put_block(
        m_tensor_ptr, idx.data(), size.data(), data, nullptr, nullptr);
  }

  block<N, T> get_block(
      const index<N>& idx, const index<N>& blk_size, bool& found) const
  {
    block<N, T> blk_out(blk_size);

    c_dbcsr_t_get_block(
        m_tensor_ptr, idx.data(), blk_size.data(), blk_out.data(), &found);

    return blk_out;
  }

  void get_block(
      T* data_ptr,
      const index<N>& idx,
      const index<N>& blk_size,
      bool& found) const
  {
    c_dbcsr_t_get_block(
        m_tensor_ptr, idx.data(), blk_size.data(), data_ptr, &found);
  }

  T* get_block_p(const index<N>& idx, bool& found) const
  {
    T* out = nullptr;

    c_dbcsr_t_get_block_p(m_tensor_ptr, idx.data(), &out, &found);

    return out;
  }

  int proc(const index<N>& idx) const
  {
    int p = -1;
    c_dbcsr_t_get_stored_coordinates(m_tensor_ptr, idx.data(), &p);
    return p;
  }

  T* data(long long int& data_size) const
  {
    T* data_ptr;
    T data_type = T();

    c_dbcsr_t_get_data_p(
        m_tensor_ptr, &data_ptr, &data_size, data_type, nullptr, nullptr);

    return data_ptr;
  }

  void clear()
  {
    c_dbcsr_t_clear(m_tensor_ptr);
  }

  MPI_Comm comm() const
  {
    return m_comm;
  }

  std::string name() const
  {
    char* cstring;
    c_dbcsr_t_get_info(
        m_tensor_ptr, N,
        REPEAT_FIRST(ECHO_P, nullptr, 0, 6, (, ), (, ))
            REPEAT_FIRST(ECHO_P, 0, 0, MAXDIM, (, ), (, ))
                REPEAT_FIRST(ECHO_P, 0, 0, MAXDIM, (, ), (, )) REPEAT_FIRST(
                    ECHO_P, 0, 0, MAXDIM, (, ), (, ))
                    REPEAT_FIRST(ECHO_P, 0, 0, MAXDIM, (, ), (, )) REPEAT_FIRST(
                        ECHO_P, 0, 0, MAXDIM, (, ), (, ))
                        REPEAT_FIRST(ECHO_P, 0, 0, MAXDIM, (, ), (, )) nullptr,
        &cstring, nullptr);
    std::string out(cstring);
    c_free_string(&cstring);
    return out;
  }

  int num_blocks() const
  {
    return c_dbcsr_t_get_num_blocks(m_tensor_ptr);
  }

  long long int num_blocks_total() const
  {
    return c_dbcsr_t_get_num_blocks_total(m_tensor_ptr);
  }

  int num_nze() const
  {
    return c_dbcsr_t_get_nze(m_tensor_ptr);
  }

  long long int num_nze_total() const
  {
    return c_dbcsr_t_get_nze_total(m_tensor_ptr);
  }

  void filter(
      T eps,
      std::optional<filter> method = std::nullopt,
      std::optional<bool> use_absolute = std::nullopt)
  {
    int fmethod = (method) ? static_cast<int>(*method) : 0;

    c_dbcsr_t_filter(
        m_tensor_ptr, eps, (method) ? &fmethod : nullptr,
        (use_absolute) ? &*use_absolute : nullptr);
  }

  double occupation() const
  {
    auto nfull = this->nfull_total();
    long long int tote = std::accumulate(
        nfull.begin(), nfull.end(), 1, std::multiplies<long long int>());
    long long int nze = this->num_nze_total();

    return (double)nze / (double)tote;
  }

  double long_sum() const
  {
    long long int num_nze;
    T* data = this->data(num_nze);

    double local_sum = std::accumulate(data, data + num_nze, 0.0);
    double total_sum = 0.0;

    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, m_comm);

    return total_sum;
  }

  void scale(T factor)
  {
    c_dbcsr_t_scale(m_tensor_ptr, factor);
  }

  void set(T factor)
  {
    c_dbcsr_t_set(m_tensor_ptr, factor);
  }

  void finalize()
  {
    c_dbcsr_t_finalize(m_tensor_ptr);
  }

  void batched_contract_init()
  {
    c_dbcsr_t_batched_contract_init(m_tensor_ptr);
  }

  void batched_contract_finalize()
  {
    c_dbcsr_t_batched_contract_finalize(m_tensor_ptr, nullptr);
  }

  vec<int> idx_speed() const
  {
    // returns order of speed of indices
    auto map1 = this->map1_2d();
    auto map2 = this->map2_2d();
    map2.insert(map2.end(), map1.begin(), map1.end());

    return map2;
  }

  std::shared_ptr<tensor<N, T>> get_ptr() const
  {
    return this->shared_from_this();
  }
};

template <int N, typename T = double>
using shared_tensor = std::shared_ptr<tensor<N, T>>;

template <int N, typename T>
class iterator_t {
 private:
  void* m_iter_ptr;
  void* m_tensor_ptr;

  index<N> m_idx;
  int m_blk_n;
  int m_blk_p;
  std::array<int, N> m_size;
  std::array<int, N> m_offset;

 public:
  iterator_t(tensor<N, T>& t_tensor) :
      m_iter_ptr(nullptr), m_tensor_ptr(t_tensor.m_tensor_ptr), m_blk_n(0),
      m_blk_p(0)
  {
  }

  ~iterator_t()
  {
  }

  void start()
  {
    c_dbcsr_t_iterator_start(&m_iter_ptr, m_tensor_ptr);
  }

  void stop()
  {
    c_dbcsr_t_iterator_stop(&m_iter_ptr);
    m_iter_ptr = nullptr;
  }

  void next()
  {
    c_dbcsr_t_iterator_next_block(
        m_iter_ptr, m_idx.data(), &m_blk_n, &m_blk_p, m_size.data(),
        m_offset.data());
  }

  void next_block()
  {
    c_dbcsr_t_iterator_next_block(
        m_iter_ptr, m_idx.data(), &m_blk_n, nullptr, nullptr, nullptr);
  }

  bool blocks_left() const
  {
    return c_dbcsr_t_iterator_blocks_left(m_iter_ptr);
  }

  const index<N>& idx()
  {
    return m_idx;
  }

  const index<N>& size()
  {
    return m_size;
  }

  const index<N>& offset()
  {
    return m_offset;
  }

  int blk_n() const
  {
    return m_blk_n;
  }

  int blk_p() const
  {
    return m_blk_p;
  }
};

template <int N, typename T>
void print(tensor<N, T>& t_in)
{
  int myrank, mpi_size;

  MPI_Comm_rank(t_in.comm(), &myrank);
  MPI_Comm_size(t_in.comm(), &mpi_size);

  iterator_t<N, T> iter(t_in);
  iter.start();

  if (myrank == 0)
    std::cout << "Tensor: " << t_in.name() << std::endl;

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
          if (s != N - 1) {
            std::cout << ",";
          }
        }

        std::cout << "] (";

        for (int s = 0; s != N; ++s) {
          std::cout << size[s];
          if (s != N - 1) {
            std::cout << ",";
          }
        }
        std::cout << ") {";

        auto blk = t_in.get_block(idx, size, found);

        for (int i = 0; i != blk.ntot(); ++i) { std::cout << blk[i] << " "; }
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

}  // namespace dbcsr

#endif
