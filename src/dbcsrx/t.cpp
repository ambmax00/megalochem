# 1 "dbcsr_tensor.hpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "dbcsr_tensor.hpp"
# 10 "dbcsr_tensor.hpp"
# 1 "../utils/ppdirs.hpp" 1
# 439 "../utils/ppdirs.hpp"
namespace util {

template <typename T> struct is_optional : public std::false_type {};

template <typename T>
struct is_optional<util::optional<T>> : public std::true_type {};

template <typename T, typename T2 = void> struct builder_type;

template <typename T>
struct builder_type<
    T, typename std::enable_if<!util::is_optional<T>::value>::type> {
  typedef typename util::optional<T> type;
};

template <typename T>
struct builder_type<
    T, typename std::enable_if<util::is_optional<T>::value>::type> {
  typedef T type;
};

template <typename T, typename T2 = void> struct base_type;

template <typename T>
struct base_type<T,
                 typename std::enable_if<util::is_optional<T>::value>::type> {
  typedef typename T::value_type type;
};

template <typename T>
struct base_type<T,
                 typename std::enable_if<!util::is_optional<T>::value>::type> {
  typedef T type;
};

template <typename T,
          typename = typename std::enable_if<!is_optional<T>::value>::type>
inline T evaluate(util::optional<T> &val) {
  return *val;
}

template <typename T,
          typename = typename std::enable_if<is_optional<T>::value>::type>
inline T evaluate(T &val) {
  return (val) ? val : util::nullopt;
}

} // namespace util
# 11 "dbcsr_tensor.hpp" 2

namespace dbcsr {

template <int N, typename>
class pgrid : std::enable_shared_from_this<pgrid<N>> {
protected:
  void *m_pgrid_ptr;
  vec<int> m_dims;
  MPI_Comm m_comm;

  template <int M, typename> friend class dist_t;

public:
# 36 "dbcsr_tensor.hpp"
  struct create_pack {
    MPI_Comm p_comm;
    util::optional<vec<int>> p_map1;
    util::optional<vec<int>> p_map2;
    util::optional<arr<int, N>> p_tensor_dims;
    util::optional<int> p_dimsplit;
  };
  class create_base {
    typedef create_base _create_base;
  private:
    util::builder_type<MPI_Comm>::type c_comm;
    util::builder_type<util::optional<vec<int>>>::type c_map1;
    util::builder_type<util::optional<vec<int>>>::type c_map2;
    util::builder_type<util::optional<arr<int, N>>>::type c_tensor_dims;
    util::builder_type<util::optional<int>>::type c_dimsplit;
  public:
    _create_base &
    map1(util::optional<util::base_type<util::optional<vec<int>>>::type>
             i_map1) {
      c_map1 = std::move(i_map1);
      return *this;
    }
    _create_base &
    map2(util::optional<util::base_type<util::optional<vec<int>>>::type>
             i_map2) {
      c_map2 = std::move(i_map2);
      return *this;
    }
    _create_base &tensor_dims(
        util::optional<util::base_type<util::optional<arr<int, N>>>::type>
            i_tensor_dims) {
      c_tensor_dims = std::move(i_tensor_dims);
      return *this;
    }
    _create_base &dimsplit(
        util::optional<util::base_type<util::optional<int>>::type> i_dimsplit) {
      c_dimsplit = std::move(i_dimsplit);
      return *this;
    }
    create_base(MPI_Comm i_comm) : c_comm(i_comm) {}
    std::shared_ptr<pgrid> build() {
      if constexpr (!util::is_optional<util::optional<vec<int>>>::value) {
        if (!c_map1) {
          throw std::runtime_error("Parameter "
                                   "map1"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<util::optional<vec<int>>>::value) {
        if (!c_map2) {
          throw std::runtime_error("Parameter "
                                   "map2"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<util::optional<arr<int, N>>>::value) {
        if (!c_tensor_dims) {
          throw std::runtime_error("Parameter "
                                   "tensor_dims"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<util::optional<int>>::value) {
        if (!c_dimsplit) {
          throw std::runtime_error("Parameter "
                                   "dimsplit"
                                   " not present!");
        }
      }
      create_pack p = {
          util::evaluate<MPI_Comm>(c_comm),
          util::evaluate<util::optional<vec<int>>>(c_map1),
          util::evaluate<util::optional<vec<int>>>(c_map2),
          util::evaluate<util::optional<arr<int, N>>>(c_tensor_dims),
          util::evaluate<util::optional<int>>(c_dimsplit)};
      return std::make_shared<pgrid>(std::move(p));
    }
  };
  template <typename... Types> static create_base create(Types &&... p) {
    return create_base(std::forward<Types>(p)...);
  };

  pgrid(create_pack &&p) {

    m_comm = p.p_comm;
    m_dims.resize(N);

    MPI_Fint fmpi = MPI_Comm_c2f(p.p_comm);

    if (p.p_map1 && p.p_map2) {
      c_dbcsr_t_pgrid_create_expert(&fmpi, m_dims.data(), N, &m_pgrid_ptr,
                                    (p.p_map1) ? p.p_map1->data() : nullptr,
                                    (p.p_map1) ? p.p_map1->size() : 0,
                                    (p.p_map2) ? p.p_map2->data() : nullptr,
                                    (p.p_map2) ? p.p_map2->size() : 0,
                                    (p.p_tensor_dims) ? p.p_tensor_dims->data()
                                                      : nullptr,
                                    (p.p_nsplit) ? &*p.p_nsplit : nullptr,
                                    (p.p_dimsplit) ? &*p.p_dimsplit : nullptr);
    } else {
      c_dbcsr_t_pgrid_create(&fmpi, m_dims.data(), N, &m_pgrid_ptr,
                             (p.p_tensor_dims) ? p.p_tensor_dims->data()
                                               : nullptr);
    }
  }

  pgrid() {}

  pgrid(pgrid<N> &rhs) = delete;

  pgrid<N> &operator=(pgrid<N> &rhs) = delete;

  vec<int> dims() { return m_dims; }

  void destroy(bool keep_comm = false) {

    if (m_pgrid_ptr != nullptr)
      c_dbcsr_t_pgrid_destroy(&m_pgrid_ptr, &keep_comm);

    m_pgrid_ptr = nullptr;
  }

  MPI_Comm comm() { return m_comm; }

  ~pgrid() { destroy(false); }

  std::shared_ptr<pgrid<N>> get_ptr() { return this->shared_from_this(); }
};

template <int N> using shared_pgrid = std::shared_ptr<pgrid<N>>;

template <int N, typename> class dist_t {
private:
  void *m_dist_ptr;

  arrvec<int, N> m_nd_dists;
  MPI_Comm m_comm;

  template <int M, typename T, typename> friend class tensor;

  template <int M, typename T> friend class tensor_create_base;

  template <int M, typename T> friend class tensor_create_template_base;

public:
  dist_t() : m_dist_ptr(nullptr) {}

  struct create_pack {
    pgrid<N> &p_set_pgrid;
    arrvec<int, N> &p_nd_dists;
  };
  class create_base {
    typedef create_base _create_base;
  private:
    util::builder_type<pgrid<N> &>::type c_set_pgrid;
    util::builder_type<arrvec<int, N> &>::type c_nd_dists;
  public:
    _create_base &
    set_pgrid(util::optional<util::base_type<pgrid<N> &>::type> i_set_pgrid) {
      c_set_pgrid = std::move(i_set_pgrid);
      return *this;
    }
    _create_base &nd_dists(
        util::optional<util::base_type<arrvec<int, N> &>::type> i_nd_dists) {
      c_nd_dists = std::move(i_nd_dists);
      return *this;
    }
    create_base() {}
    std::shared_ptr<dist_t> build() {
      if constexpr (!util::is_optional<pgrid<N> &>::value) {
        if (!c_set_pgrid) {
          throw std::runtime_error("Parameter "
                                   "set_pgrid"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<arrvec<int, N> &>::value) {
        if (!c_nd_dists) {
          throw std::runtime_error("Parameter "
                                   "nd_dists"
                                   " not present!");
        }
      }
      create_pack p = {util::evaluate<pgrid<N> &>(c_set_pgrid),
                       util::evaluate<arrvec<int, N> &>(c_nd_dists)};
      return std::make_shared<dist_t>(std::move(p));
    }
  };
  template <typename... Types> static create_base create(Types &&... p) {
    return create_base(std::forward<Types>(p)...);
  };
# 155 "dbcsr_tensor.hpp"
  template <int M = N, typename std::enable_if<M == 2, int>::type = 0>
  dist_t(create_pack &&p) {
    m_dist_ptr = nullptr;
    m_nd_dists = p.p_nd_dists;
    m_comm = p.p_set_pgrid.comm();
    c_dbcsr_t_distribution_new(&m_dist_ptr, p.p_set_pgrid.m_pgrid_ptr,
                               m_nd_dists[0].data(), m_nd_dists[0].size(),
                               m_nd_dists[1].data(), m_nd_dists[1].size(),
                               nullptr, 0, nullptr, 0);
  }
  template <int M = N, typename std::enable_if<M == 3, int>::type = 0>
  dist_t(create_pack &&p) {
    m_dist_ptr = nullptr;
    m_nd_dists = p.p_nd_dists;
    m_comm = p.p_set_pgrid.comm();
    c_dbcsr_t_distribution_new(
        &m_dist_ptr, p.p_set_pgrid.m_pgrid_ptr, m_nd_dists[0].data(),
        m_nd_dists[0].size(), m_nd_dists[1].data(), m_nd_dists[1].size(),
        m_nd_dists[2].data(), m_nd_dists[2].size(), nullptr, 0);
  }
  template <int M = N, typename std::enable_if<M == 4, int>::type = 0>
  dist_t(create_pack &&p) {
    m_dist_ptr = nullptr;
    m_nd_dists = p.p_nd_dists;
    m_comm = p.p_set_pgrid.comm();
    c_dbcsr_t_distribution_new(&m_dist_ptr, p.p_set_pgrid.m_pgrid_ptr,
                               m_nd_dists[0].data(), m_nd_dists[0].size(),
                               m_nd_dists[1].data(), m_nd_dists[1].size(),
                               m_nd_dists[2].data(), m_nd_dists[2].size(),
                               m_nd_dists[3].data(), m_nd_dists[3].size());
  }
