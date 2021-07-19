#ifndef DBCSR_COMMON_HPP
#define DBCSR_COMMON_HPP

#define MAXDIM 4

#ifndef TEST_MACRO

  #include <dbcsr.h>
  #include <algorithm>
  #include <array>
  #include <iostream>
  #include <memory>
  #include <numeric>
  #include <random>
  #include <stdexcept>
  #include <string>
  #include <utility>
  #include <vector>

#endif

#include "utils/ppdirs.hpp"

// ----------------------------------------
//               typedefs
// ----------------------------------------

template <typename T>
using svector = std::shared_ptr<std::vector<T>>;

template <typename T>
using vec = std::vector<T>;

template <typename T, int N>
using arr = std::array<T, N>;

template <typename T, int N>
using arrvec = std::array<std::vector<T>, N>;

namespace dbcsr {

template <int N>
using index = std::array<int, N>;

typedef index<2> idx2;
typedef index<3> idx3;
typedef index<4> idx4;

// ----------------------------------------
//       structs for SFINAE purposes
// ----------------------------------------

template <int N>
struct is_valid_dim {
  const static bool value = (N >= 2 && N <= MAXDIM);
};

template <typename T>
struct is_valid_type {
  const static bool value =
      (std::is_same<double, T>::value || std::is_same<float, T>::value ||
       std::is_same<double _Complex, T>::value ||
       std::is_same<float _Complex, T>::value);
};

template <typename T>
struct dbcsr_type {
  static const int value = dbcsr_type_real_4 * std::is_same<T, float>::value +
      dbcsr_type_real_8 * std::is_same<T, double>::value +
      dbcsr_type_complex_4 * std::is_same<T, float _Complex>::value +
      dbcsr_type_complex_8 * std::is_same<T, double _Complex>::value;
};

// ---------------------------------------------------------------------
//                                enums
// ---------------------------------------------------------------------

enum class filter { frobenius = 1 };

enum class norm { frobenius = 1, maxabsnorm = 2, gershgorin = 3, column = 4 };

enum class func {
  inverse = 0,
  tanh = 1,
  dtanh = 2,
  ddtanh = 3,
  artanh = 4,
  inverse_special = 5,
  spread_from_zero = 6,
  sin = 7,
  dsin = 8,
  ddsin = 9,
  asin = 10,
  cos = 11,
  truncate = 12
};

enum class type {
  invalid = '0',
  no_symmetry = 'N',
  symmetric = 'S',
  antisymmetric = 'A',
  hermitian = 'H',
  antihermitian = 'K'
};

enum class repl { none = 'N', row = 'R', col = 'C', full = 'A' };

// ---------------------------------------------------------------------
//                     forward declarations (matrix)
// ---------------------------------------------------------------------

class dist;
template <typename T = double>
class iterator;
template <
    typename T,
    typename = typename std::enable_if<is_valid_type<T>::value>::type>
class matrix;
template <typename T>
using smatrix = std::shared_ptr<matrix<T>>;

template <typename T = double>
class multiply_base;

// ---------------------------------------------------------------------
//                     forward declarations (tensor)
// ---------------------------------------------------------------------

template <
    int N,
    typename = typename std::enable_if<is_valid_dim<N>::value>::type>
class pgrid;
template <
    int N,
    typename = typename std::enable_if<is_valid_dim<N>::value>::type>
class dist_t;
template <int N, typename T = double>
class iterator_t;
template <
    int N,
    typename T,
    typename = typename std::enable_if<
        is_valid_dim<N>::value && is_valid_type<T>::value>::type>
class tensor;
template <int N, typename T = double>
using stensor = std::shared_ptr<tensor<N, T>>;

template <int N1, int N2, int N3, typename T = double>
class contract_base;
template <int N1, typename T = double>
class copy_base;

// other

template <int N>
class create_pgrid_base;

template <int N, typename T>
class tensor_create_base;

template <int N, typename T>
class tensor_create_template_base;

template <typename T>
class tensor_create_matrix_base;

template <typename T>
void copy_tensor_to_matrix(
    tensor<2, T>& t_in,
    matrix<T>& m_out,
    std::optional<bool> summation = std::nullopt);

template <typename T>
void copy_matrix_to_tensor(
    matrix<T>& m_in,
    tensor<2, T>& t_out,
    std::optional<bool> summation = std::nullopt);

struct global {
  static inline double filter_eps = 1e-9;
  static inline bool filter_use_absolute = true;
};

inline auto default_dist = [](int nel, int nbin, vec<int> weights) {
  std::vector<int> distvec(weights.size(), 0);

  // std::cout << "nel, nbin: " << nel << " " << nbin << std::endl;

  vec<int> occup(nbin, 0);
  int ibin = 0;

  for (int iel = 0; iel != nel; ++iel) {
    int niter = 0;
    ibin = abs((ibin + 1) % nbin);
    // std::cout << "ibin: " << ibin << std::endl;
    while (occup[ibin] + weights[iel] >=
           *std::max_element(occup.begin(), occup.end())) {
      // std::cout << occup[ibin] << " " << weights[iel] << " " <<
      // *std::max_element(occup.begin(), occup.end()) << std::endl;
      int minloc = std::min_element(occup.begin(), occup.end()) - occup.begin();
      // std::cout << minloc << std::endl;
      if (minloc == ibin)
        break;
      ibin = abs((ibin + 1) % nbin);
      ++niter;
    }

    distvec[iel] = ibin;
    occup[ibin] = occup[ibin] + weights[iel];
  }

  return distvec;
};

inline auto split_range = [](int n, int split) {
  // number of intervals
  int nblock = n % split == 0 ? n / split : n / split + 1;
  bool even = n % split == 0 ? true : false;

  if (even) {
    std::vector<int> out(nblock, split);
    return out;
  }
  else {
    std::vector<int> out(nblock, split);
    out[nblock - 1] = n % split;
    return out;
  }
};

inline auto cyclic_dist = [](int dist_size, int nbins) {
  std::vector<int> distv(dist_size);
  for (int i = 0; i < dist_size; i++) distv[i] = i % nbins;

  return distv;
};

// ---------------------------------------------------------------------
//                library functions
// ---------------------------------------------------------------------

inline void init(MPI_Comm comm = MPI_COMM_WORLD, int* io_unit = nullptr)
{
  c_dbcsr_init_lib(comm, io_unit);
}

inline void finalize()
{
  c_dbcsr_finalize_lib();
}

inline void print_statistics(const bool print_timers = false)
{
  c_dbcsr_print_statistics(&print_timers, nullptr);
}

class cart {
 private:
  MPI_Comm m_comm;
  MPI_Comm m_group;
  int m_rank;
  int m_size;
  std::array<int, 2> m_dims;
  std::array<int, 2> m_coord;

  friend class dist;

 public:
  cart(MPI_Comm comm) : m_comm(comm)
  {
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);

    int dims[2] = {0};
    MPI_Dims_create(m_size, 2, dims);
    int periods[2] = {1};
    int reorder = 0;
    MPI_Cart_create(m_comm, 2, dims, periods, reorder, &m_group);

    int coord[2];
    MPI_Cart_coords(m_group, m_rank, 2, coord);

    m_dims[0] = dims[0];
    m_dims[1] = dims[1];
    m_coord[0] = coord[0];
    m_coord[1] = coord[1];
  }

  cart(MPI_Comm comm, MPI_Comm group) : m_comm(comm), m_group(group)
  {
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);

    int dims[2] = {0};
    MPI_Dims_create(m_size, 2, dims);

    int coord[2];
    MPI_Cart_coords(m_group, m_rank, 2, coord);

    m_dims[0] = dims[0];
    m_dims[1] = dims[1];
    m_coord[0] = coord[0];
    m_coord[1] = coord[1];
  }

  cart(const cart& w) = default;

  void free()
  {
    if (m_group != MPI_COMM_NULL)
      MPI_Comm_free(&m_group);
    // if (*m_comm_ptr != MPI_COMM_NULL) MPI_Comm_free(&*m_comm_ptr);
  }

  ~cart()
  {
  }

  MPI_Comm comm() const
  {
    return m_comm;
  }
  MPI_Comm group() const
  {
    return m_group;
  }

  int rank() const
  {
    return m_rank;
  }
  int size() const
  {
    return m_size;
  }

  int myprow() const
  {
    return m_coord[0];
  }
  int mypcol() const
  {
    return m_coord[1];
  }
  int nprow() const
  {
    return m_dims[0];
  }
  int npcol() const
  {
    return m_dims[1];
  }

  std::array<int, 2> dims() const
  {
    return m_dims;
  }
  std::array<int, 2> coord() const
  {
    return m_coord;
  }
};

/* ---------------------------------------------------------------------
/					BLOCK: multi-dimensional array (wrapper)
class / ----------------------------------------------------------------------
*/

template <int N, typename T = double>
class block {
 private:
  T* m_data;
  std::array<int, N> m_size;
  int m_nfull;
  bool m_own = true;

 public:
  block() : m_data(nullptr), m_size{N, 0}, m_nfull(0)
  {
  }

  block(const std::array<int, N> sizes, T* ptr = nullptr) : m_size(sizes)
  {
    m_nfull =
        std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());

    if (ptr) {
      m_data = ptr;
      m_own = false;
    }
    else {
      m_data = new T[m_nfull]();
    }
  }

  template <
      int M = N,
      typename D = T,
      typename = typename std::enable_if<M == 2, int>::type>
  block(const int nrows, const int ncols, T* ptr = nullptr)
  {
    m_nfull = nrows * ncols;
    m_size = {nrows, ncols};

    if (ptr) {
      m_data = ptr;
      m_own = false;
    }
    else {
      m_data = new T[m_nfull]();
    }
  }

  block(const block<N, T>& blk_in) : m_data(nullptr)
  {
    m_nfull = blk_in.m_nfull;
    m_data = new T[m_nfull];
    m_size = blk_in.m_size;

    std::copy(blk_in.m_data, blk_in.m_data + m_nfull, m_data);
  }

  block(block<N, T>&& blk_in) : m_data(nullptr)
  {
    m_nfull = blk_in.m_nfull;
    m_data = blk_in.m_data;
    m_size = blk_in.m_size;

    blk_in.m_nfull = 0;
    blk_in.m_data = nullptr;
  }

  block& operator=(const block& rhs)
  {
    if (this == &rhs)
      return *this;

    if (m_data != nullptr && m_own)
      delete[] m_data;

    m_nfull = rhs.m_nfull;
    m_data = new T[m_nfull];
    m_size = rhs.m_size;

    std::copy(rhs.m_data, rhs.m_data + m_nfull, m_data);

    return *this;
  }

  block& operator=(block&& rhs)
  {
    if (this == &rhs)
      return *this;

    if (m_data != nullptr)
      delete[] m_data;

    m_nfull = rhs.m_nfull;
    m_data = rhs.m_data;
    m_size = rhs.m_size;

    rhs.m_data = nullptr;

    return *this;
  }

  void print() const
  {
    std::cout << "(";
    for (int i = 0; i != N; ++i) {
      std::cout << m_size[i];
      if (i != N - 1)
        std::cout << ",";
    }
    std::cout << ") [";

    for (int i = 0; i != m_nfull; ++i) {
      std::cout << m_data[i];
      if (i != m_nfull - 1)
        std::cout << " ";
    }
    std::cout << "]" << std::endl;
  }

  int ntot() const
  {
    return m_nfull;
  }

  const index<N> size() const
  {
    return m_size;
  }

  T* data() const
  {
    return m_data;
  }

  T& operator[](int id)
  {
    // std::cout << id << std::endl;
    return m_data[id];
  }

#define MSIZE(x, n) m_size[DEC(n)]

#define ARRAYINDEX(var, n) \
  CAT(i_, n) \
  REPEAT_THIRD(ECHO_NONE, UNUSED, 0, n, (), (*)) \
  REPEAT_THIRD(MSIZE, UNUSED, 1, n, (*), ())

#define OPERATOR(var, n) \
  template <typename D = T, int M = N> \
  typename std::enable_if<M == n, D&>::type operator()( \
      REPEAT_SECOND(CAT, int i_, 0, n, (PPDIRS_COMMA), ())) \
  { \
    return m_data[REPEAT_SECOND(ARRAYINDEX, UNUSED, 0, n, (+), ())]; \
  }

  REPEAT_FIRST(OPERATOR, UNUSED, 2, 3, (), ())

#define RESHAPE(var, n) \
  template <typename D = T, int M = N> \
  typename std::enable_if<M == n, void>::type reshape( \
      D* data, std::array<int, 2> sizes, vec<int> map1, vec<int> map2) \
  { \
    CAT(fortran_reshape_, n) \
    (data, sizes.data(), m_data, m_size.data(), map1.data(), map1.size(), \
     map2.data(), map2.size()); \
  }

  // REPEAT_FIRST(RESHAPE, UNUSED, 2, 3, (), ())

#define ASSIGN(var, n) \
  int& CAT(i_, n) = idx[n]; \
  int& CAT(r_, n) = idx[rev_order[n]]; \
  int& CAT(rsize_, n) = m_size[rev_order[n]];

#define FOR_CLAUSE(var, n) \
  for (CAT(r_, n) = 0; CAT(r_, n) != CAT(rsize_, n); ++CAT(r_, n)) PPDIRS_OB

#define ECHO_PARA(var, n) CAT(i_, n)

#define RESHAPE_2D(var, n) \
  template <typename D = T, int M = N> \
  typename std::enable_if<M == n, void>::type reshape_2d( \
      T* data, vec<int> map1, vec<int> map2) \
  { \
    vec<int> order = map1; \
    order.insert(order.end(), map2.begin(), map2.end()); \
\
    vec<int> order_ref(N); \
    std::iota(order_ref.begin(), order_ref.end(), 0); \
\
    if (order == order_ref) { \
      std::copy(data, data + m_nfull, m_data); \
    } \
    else { \
      int off = 0; \
      std::vector<int> idx(N, 0); \
      vec<int> rev_order(order.rbegin(), order.rend()); \
      REPEAT_SECOND(ASSIGN, UNUSED, 0, n, (), ()) \
      REPEAT_SECOND(FOR_CLAUSE, i_, 0, n, (), ()) \
      this->operator()(REPEAT_SECOND(ECHO_PARA, UNUSED, 0, n, (, ), ())) = \
          data[off++]; \
      REPEAT_SECOND(ECHO_P, PPDIRS_CB, 0, n, (), ()) \
    } \
  }

  REPEAT_FIRST(RESHAPE_2D, UNUSED, 2, SUB(MAXDIM, 1), (), ())

  void fill_rand(T a, T b)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(a, b);

    for (int i = 0; i != m_nfull; ++i) {
      m_data[i] = dis(gen);
      // std::cout << m_data[i] << std::endl;
    }
  }

  T norm(norm norm_type = norm::frobenius) const
  {
    T out = 0;
    
    switch (norm_type) {
      case norm::frobenius:
      {
        for (int i = 0; i != m_nfull; ++i) { out += pow(m_data[i], 2); }
        out = std::sqrt(out);
        break;
      }
      case norm::maxabsnorm:
      {
        for (int i = 0; i != m_nfull; ++i) { out = std::max(out, std::fabs(m_data[i])); }
        break;
      }
      default:
        throw std::runtime_error("Norm type not yet implemented");
    }
    
    return out;
  }

  T max_abs() const
  {
    T* max_it =
        std::max_element(m_data, m_data + m_nfull, [](const T a, const T b) {
          return fabs(a) < fabs(b);
        });
    return *max_it;
  }

  T* begin() const
  {
    return m_data;
  }

  T* end() const
  {
    return m_data + m_nfull;
  }

  ~block()
  {
    if (m_data != nullptr && m_own) {
      delete[] m_data;
      m_data = nullptr;
    }
  }
};

}  // namespace dbcsr

#endif
