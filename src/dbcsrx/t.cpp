# 1 "dbcsr_tensor.hpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "dbcsr_tensor.hpp"
# 10 "dbcsr_tensor.hpp"
# 1 "../utils/ppdirs.hpp" 1
# 483 "../utils/ppdirs.hpp"
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

template <class T, typename T2 = void> struct base_type;

template <class T>
struct base_type<T,
                 typename std::enable_if<util::is_optional<T>::value>::type> {
  typedef typename T::value_type type;
};

template <class T>
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
# 43 "dbcsr_tensor.hpp"
  struct create_pack {
    MPI_Comm p_comm;
    util::optional<vec<int>> p_map1;
    util::optional<vec<int>> p_map2;
    util::optional<arr<int, N>> p_tensor_dims;
    util::optional<int> p_nsplit;
    util::optional<int> p_dimsplit;
  };
  class create_base {
    typedef create_base _create_base;
  private:
    typename util::builder_type<MPI_Comm>::type c_comm;
    typename util::builder_type<util::optional<vec<int>>>::type c_map1;
    typename util::builder_type<util::optional<vec<int>>>::type c_map2;
    typename util::builder_type<util::optional<arr<int, N>>>::type
        c_tensor_dims;
    typename util::builder_type<util::optional<int>>::type c_nsplit;
    typename util::builder_type<util::optional<int>>::type c_dimsplit;
  public:
    _create_base &map1(
        util::optional<typename util::base_type<util::optional<vec<int>>>::type>
            i_map1) {
      c_map1 = std::move(i_map1);
      return *this;
    }
    _create_base &map2(
        util::optional<typename util::base_type<util::optional<vec<int>>>::type>
            i_map2) {
      c_map2 = std::move(i_map2);
      return *this;
    }
    _create_base &
    tensor_dims(util::optional<
                typename util::base_type<util::optional<arr<int, N>>>::type>
                    i_tensor_dims) {
      c_tensor_dims = std::move(i_tensor_dims);
      return *this;
    }
    _create_base &
    nsplit(util::optional<typename util::base_type<util::optional<int>>::type>
               i_nsplit) {
      c_nsplit = std::move(i_nsplit);
      return *this;
    }
    _create_base &
    dimsplit(util::optional<typename util::base_type<util::optional<int>>::type>
                 i_dimsplit) {
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
        if (!c_nsplit) {
          throw std::runtime_error("Parameter "
                                   "nsplit"
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
          util::evaluate<util::optional<int>>(c_nsplit),
          util::evaluate<util::optional<int>>(c_dimsplit)};
      return std::make_shared<pgrid>(std::move(p));
    }
  };
  template <typename... Types> static create_base create(Types &&... p) {
    return create_base(std::forward<Types>(p)...);
  }

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
    typename util::builder_type<pgrid<N> &>::type c_set_pgrid;
    typename util::builder_type<arrvec<int, N> &>::type c_nd_dists;
  public:
    _create_base &
    set_pgrid(util::optional<typename util::base_type<pgrid<N> &>::type>
                  i_set_pgrid) {
      c_set_pgrid = std::move(i_set_pgrid);
      return *this;
    }
    _create_base &
    nd_dists(util::optional<typename util::base_type<arrvec<int, N> &>::type>
                 i_nd_dists) {
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
  }

  dist_t(create_pack &&p) {

    m_dist_ptr = nullptr;
    m_nd_dists = p.p_nd_dists;
    m_comm = p.p_set_pgrid.comm();

    c_dbcsr_t_distribution_new(&m_dist_ptr, p.p_set_pgrid.m_pgrid_ptr,
                               (0 < N) ? m_nd_dists[0].data() : nullptr,
                               (0 < N) ? m_nd_dists[0].size() : 0,
                               (1 < N) ? m_nd_dists[1].data() : nullptr,
                               (1 < N) ? m_nd_dists[1].size() : 0,
                               (2 < N) ? m_nd_dists[2].data() : nullptr,
                               (2 < N) ? m_nd_dists[2].size() : 0,
                               (3 < N) ? m_nd_dists[3].data() : nullptr,
                               (3 < N) ? m_nd_dists[3].size() : 0);
  }

  dist_t(dist_t<N> &rhs) = delete;

  dist_t<N> &operator=(dist_t<N> &rhs) = delete;

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
class tensor : std::enable_shared_from_this<tensor<N, T>> {
protected:
  void *m_tensor_ptr;
  MPI_Comm m_comm;

  const int m_data_type = dbcsr_type<T>::value;

  tensor(void *ptr) : m_tensor_ptr(ptr) {}

public:
  friend class iterator_t<N, T>;

  template <int N1, int N2, int N3, typename D> friend class contract_base;

  template <int M, typename D> friend class tensor_copy_base;

  template <int M, typename D> friend class tensor_create_base;

  template <typename D> friend class tensor_create_matrix_base;

  template <int M, typename D> friend class tensor_create_template_base;

  template <typename D>
  friend void copy_tensor_to_matrix(tensor<2, D> &t_in, matrix<D> &m_out,
                                    std::optional<bool> summation);

  template <typename D>
  friend void copy_matrix_to_tensor(matrix<D> &m_in, tensor<2, D> &t_out,
                                    std::optional<bool> summation);

  typedef T value_type;
  const static int dim = N;
# 227 "dbcsr_tensor.hpp"
  struct create_pack {
    std::string p_name;
    util::optional<dist_t<N> &> p_set_dist;
    util::optional<pgrid<N> &> p_set_pgrid;
    vec<int> p_map1;
    vec<int> p_map2;
    arrvec<int, N> p_blk_sizes;
  };
  class create_base {
    typedef create_base _create_base;
  private:
    typename util::builder_type<std::string>::type c_name;
    typename util::builder_type<util::optional<dist_t<N> &>>::type c_set_dist;
    typename util::builder_type<util::optional<pgrid<N> &>>::type c_set_pgrid;
    typename util::builder_type<vec<int>>::type c_map1;
    typename util::builder_type<vec<int>>::type c_map2;
    typename util::builder_type<arrvec<int, N>>::type c_blk_sizes;
  public:
    _create_base &
    name(util::optional<typename util::base_type<std::string>::type> i_name) {
      c_name = std::move(i_name);
      return *this;
    }
    _create_base &
    set_dist(util::optional<
             typename util::base_type<util::optional<dist_t<N> &>>::type>
                 i_set_dist) {
      c_set_dist = std::move(i_set_dist);
      return *this;
    }
    _create_base &
    set_pgrid(util::optional<
              typename util::base_type<util::optional<pgrid<N> &>>::type>
                  i_set_pgrid) {
      c_set_pgrid = std::move(i_set_pgrid);
      return *this;
    }
    _create_base &
    map1(util::optional<typename util::base_type<vec<int>>::type> i_map1) {
      c_map1 = std::move(i_map1);
      return *this;
    }
    _create_base &
    map2(util::optional<typename util::base_type<vec<int>>::type> i_map2) {
      c_map2 = std::move(i_map2);
      return *this;
    }
    _create_base &
    blk_sizes(util::optional<typename util::base_type<arrvec<int, N>>::type>
                  i_blk_sizes) {
      c_blk_sizes = std::move(i_blk_sizes);
      return *this;
    }
    create_base() {}
    std::shared_ptr<tensor> build() {
      if constexpr (!util::is_optional<std::string>::value) {
        if (!c_name) {
          throw std::runtime_error("Parameter "
                                   "name"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<util::optional<dist_t<N> &>>::value) {
        if (!c_set_dist) {
          throw std::runtime_error("Parameter "
                                   "set_dist"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<util::optional<pgrid<N> &>>::value) {
        if (!c_set_pgrid) {
          throw std::runtime_error("Parameter "
                                   "set_pgrid"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<vec<int>>::value) {
        if (!c_map1) {
          throw std::runtime_error("Parameter "
                                   "map1"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<vec<int>>::value) {
        if (!c_map2) {
          throw std::runtime_error("Parameter "
                                   "map2"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<arrvec<int, N>>::value) {
        if (!c_blk_sizes) {
          throw std::runtime_error("Parameter "
                                   "blk_sizes"
                                   " not present!");
        }
      }
      create_pack p = {util::evaluate<std::string>(c_name),
                       util::evaluate<util::optional<dist_t<N> &>>(c_set_dist),
                       util::evaluate<util::optional<pgrid<N> &>>(c_set_pgrid),
                       util::evaluate<vec<int>>(c_map1),
                       util::evaluate<vec<int>>(c_map2),
                       util::evaluate<arrvec<int, N>>(c_blk_sizes)};
      return std::make_shared<tensor>(std::move(p));
    }
  };
  template <typename... Types> static create_base create(Types &&... p) {
    return create_base(std::forward<Types>(p)...);
  }

  tensor(create_pack &&p) {

    void *dist_ptr = nullptr;
    std::shared_ptr<dist_t<N>> distn;

    if (p.p_set_dist) {

      dist_ptr = p.p_set_dist->m_dist_ptr;
      m_comm = p.p_set_dist->m_comm;

    } else {

      arrvec<int, N> distvecs;
      auto pgrid_dims = p.p_set_pgrid->dims();

      for (int i = 0; i != N; ++i) {
        distvecs[i] = default_dist(p.p_blk_sizes[i].size(), pgrid_dims[i],
                                   p.p_blk_sizes[i]);
      }

      distn = typename dist_t<N>::create(5)
                  .set_pgrid(*p.p_set_pgrid)
                  .nd_dists(distvecs)
                  .build();

      dist_ptr = distn->m_dist_ptr;
      m_comm = distn->m_comm;
    }

    c_dbcsr_t_create_new(&m_tensor_ptr, p.p_name.c_str(), dist_ptr,
                         p.p_map1.data(), p.p_map1.size(), p.p_map2.data(),
                         p.p_map2.size(), &m_data_type,
                         (0 < N) ? p.p_blk_sizes[0].data() : nullptr,
                         (0 < N) ? p.p_blk_sizes[0].size() : 0,
                         (1 < N) ? p.p_blk_sizes[1].data() : nullptr,
                         (1 < N) ? p.p_blk_sizes[1].size() : 0,
                         (2 < N) ? p.p_blk_sizes[2].data() : nullptr,
                         (2 < N) ? p.p_blk_sizes[2].size() : 0,
                         (3 < N) ? p.p_blk_sizes[3].data() : nullptr,
                         (3 < N) ? p.p_blk_sizes[3].size() : 0);
  }
# 277 "dbcsr_tensor.hpp"
  struct create_template_pack {
    tensor<N, T> &p_templet;
    std::string p_name;
    util::optional<dist_t<N> &> p_ndist;
    util::optional<vec<int>> p_map1;
    util::optional<vec<int>> p_map2;
  };

  class create_template_base {
    typedef create_template_base _create_base;
  private:
    typename util::builder_type<tensor<N, T> &>::type c_templet;
    typename util::builder_type<std::string>::type c_name;
    typename util::builder_type<util::optional<dist_t<N> &>>::type c_ndist;
    typename util::builder_type<util::optional<vec<int>>>::type c_map1;
    typename util::builder_type<util::optional<vec<int>>>::type c_map2;
  public:
    _create_base &
    name(util::optional<typename util::base_type<std::string>::type> i_name) {
      c_name = std::move(i_name);
      return *this;
    }
    _create_base &
    ndist(util::optional<
          typename util::base_type<util::optional<dist_t<N> &>>::type>
              i_ndist) {
      c_ndist = std::move(i_ndist);
      return *this;
    }
    _create_base &map1(
        util::optional<typename util::base_type<util::optional<vec<int>>>::type>
            i_map1) {
      c_map1 = std::move(i_map1);
      return *this;
    }
    _create_base &map2(
        util::optional<typename util::base_type<util::optional<vec<int>>>::type>
            i_map2) {
      c_map2 = std::move(i_map2);
      return *this;
    }
    create_template_base(tensor<N, T> &i_templet) : c_templet(i_templet) {}
    std::shared_ptr<tensor> build() {
      if constexpr (!util::is_optional<std::string>::value) {
        if (!c_name) {
          throw std::runtime_error("Parameter "
                                   "name"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<util::optional<dist_t<N> &>>::value) {
        if (!c_ndist) {
          throw std::runtime_error("Parameter "
                                   "ndist"
                                   " not present!");
        }
      }
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
      create_template_pack p = {
          util::evaluate<tensor<N, T> &>(c_templet),
          util::evaluate<std::string>(c_name),
          util::evaluate<util::optional<dist_t<N> &>>(c_ndist),
          util::evaluate<util::optional<vec<int>>>(c_map1),
          util::evaluate<util::optional<vec<int>>>(c_map2)};
      return std::make_shared<tensor>(std::move(p));
    }
  };
  template <typename... Types>
  static create_template_base create_template(Types &&... p) {
    return create_template_base(std::forward<Types>(p)...);
  }

  tensor(create_template_pack &&p) {

    m_comm = p.p_templet.m_comm;

    c_dbcsr_t_create_template(p.p_templet.m_tensor_ptr, &m_tensor_ptr,
                              p.p_name.c_str(),
                              (p.p_ndist) ? p.p_ndist->m_dist_ptr : nullptr,
                              (p.p_map1) ? p.p_map1->data() : nullptr,
                              (p.p_map1) ? p.p_map1->size() : 0,
                              (p.p_map2) ? p.p_map2->data() : nullptr,
                              (p.p_map2) ? p.p_map2->size() : 0, &m_data_type);
  }
# 305 "dbcsr_tensor.hpp"
  struct create_matrix_pack {
    matrix<T> &p_matrix_in;
    util::optional<std::string> p_name;
    util::optional<vec<int>> p_order;
  };

  class create_matrix_base {
    typedef create_matrix_base _create_base;
  private:
    typename util::builder_type<matrix<T> &>::type c_matrix_in;
    typename util::builder_type<util::optional<std::string>>::type c_name;
    typename util::builder_type<util::optional<vec<int>>>::type c_order;
  public:
    _create_base &
    name(util::optional<
         typename util::base_type<util::optional<std::string>>::type>
             i_name) {
      c_name = std::move(i_name);
      return *this;
    }
    _create_base &order(
        util::optional<typename util::base_type<util::optional<vec<int>>>::type>
            i_order) {
      c_order = std::move(i_order);
      return *this;
    }
    create_matrix_base(matrix<T> &i_matrix_in) : c_matrix_in(i_matrix_in) {}
    std::shared_ptr<tensor> build() {
      if constexpr (!util::is_optional<util::optional<std::string>>::value) {
        if (!c_name) {
          throw std::runtime_error("Parameter "
                                   "name"
                                   " not present!");
        }
      }
      if constexpr (!util::is_optional<util::optional<vec<int>>>::value) {
        if (!c_order) {
          throw std::runtime_error("Parameter "
                                   "order"
                                   " not present!");
        }
      }
      create_matrix_pack p = {
          util::evaluate<matrix<T> &>(c_matrix_in),
          util::evaluate<util::optional<std::string>>(c_name),
          util::evaluate<util::optional<vec<int>>>(c_order)};
      return std::make_shared<tensor>(std::move(p));
    }
  };
  template <typename... Types>
  static create_matrix_base create_matrix(Types &&... p) {
    return create_matrix_base(std::forward<Types>(p)...);
  }

  tensor(create_matrix_pack &&p) {

    m_comm = p.p_matrix_in.get_world().comm();

    c_dbcsr_t_create_matrix(p.p_matrix.m_matrix_ptr, &m_tensor_ptr,
                            (p.p_order) ? p.p_order->data() : nullptr,
                            (p.p_name) ? p.p_name->c_str() : nullptr);
  }
# 350 "dbcsr_tensor.hpp"
  arr<int, N> nblks_total() {
    arr<int, N> out;
    c_dbcsr_t_get_info(m_tensor_ptr, N, out.data(), nullptr, nullptr, nullptr,
                       nullptr, nullptr, 0, 0, 0, 0, 0, 0, 0, 0, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    return out;
  }
  arr<int, N> nfull_total() {
    arr<int, N> out;
    c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, out.data(), nullptr, nullptr,
                       nullptr, nullptr, 0, 0, 0, 0, 0, 0, 0, 0, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    return out;
  }
  arr<int, N> nblks_local() {
    arr<int, N> out;
    c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, out.data(), nullptr,
                       nullptr, nullptr, 0, 0, 0, 0, 0, 0, 0, 0, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    return out;
  }
  arr<int, N> nfull_local() {
    arr<int, N> out;
    c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, out.data(),
                       nullptr, nullptr, 0, 0, 0, 0, 0, 0, 0, 0, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    return out;
  }
  arr<int, N> pdims() {
    arr<int, N> out;
    c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                       out.data(), nullptr, 0, 0, 0, 0, 0, 0, 0, 0, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    return out;
  }
  arr<int, N> my_ploc() {
    arr<int, N> out;
    c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                       nullptr, out.data(), 0, 0, 0, 0, 0, 0, 0, 0, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    return out;
  }
# 456 "dbcsr_tensor.hpp"
  template <int M = N>
  typename std::enable_if<M == 2, arrvec<int, N>>::type blks_local() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "blks_local";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, sizes[0], sizes[1], 0, 0, 0, 0, 0, 0,
                         out[0].data(), out[1].data(), nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, 0, 0, 0, 0, sizes[0], sizes[1], 0, 0,
                         out[0].data(), out[1].data(), nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 2, arrvec<int, N>>::type proc_dist() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "proc_dist";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, sizes[0], sizes[1], 0, 0, 0, 0, 0, 0,
                         nullptr, nullptr, nullptr, nullptr, out[0].data(),
                         out[1].data(), nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, 0, 0, 0, 0, sizes[0], sizes[1], 0, 0,
                         nullptr, nullptr, nullptr, nullptr, out[0].data(),
                         out[1].data(), nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 2, arrvec<int, N>>::type blk_sizes() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "blk_sizes";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, sizes[0], sizes[1], 0, 0, 0, 0, 0, 0,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, out[0].data(), out[1].data(),
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, 0, 0, 0, 0, sizes[0], sizes[1], 0, 0,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, out[0].data(), out[1].data(),
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 2, arrvec<int, N>>::type blk_offsets() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "blk_offsets";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, sizes[0], sizes[1], 0, 0, 0, 0, 0, 0,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         out[0].data(), out[1].data(), nullptr, nullptr,
                         nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, 0, 0, 0, 0, sizes[0], sizes[1], 0, 0,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         out[0].data(), out[1].data(), nullptr, nullptr,
                         nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 3, arrvec<int, N>>::type blks_local() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "blks_local";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_local(m_tensor_ptr, 2);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_total(m_tensor_ptr, 2);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, sizes[0], sizes[1], sizes[2], 0, 0,
                         0, 0, 0, out[0].data(), out[1].data(), out[2].data(),
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, 0, 0, 0, 0, sizes[0], sizes[1],
                         sizes[2], 0, out[0].data(), out[1].data(),
                         out[2].data(), nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 3, arrvec<int, N>>::type proc_dist() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "proc_dist";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_local(m_tensor_ptr, 2);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_total(m_tensor_ptr, 2);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, sizes[0], sizes[1], sizes[2], 0, 0,
                         0, 0, 0, nullptr, nullptr, nullptr, nullptr,
                         out[0].data(), out[1].data(), out[2].data(), nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, 0, 0, 0, 0, sizes[0], sizes[1],
                         sizes[2], 0, nullptr, nullptr, nullptr, nullptr,
                         out[0].data(), out[1].data(), out[2].data(), nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 3, arrvec<int, N>>::type blk_sizes() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "blk_sizes";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_local(m_tensor_ptr, 2);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_total(m_tensor_ptr, 2);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, sizes[0], sizes[1], sizes[2], 0, 0,
                         0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, out[0].data(),
                         out[1].data(), out[2].data(), nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, 0, 0, 0, 0, sizes[0], sizes[1],
                         sizes[2], 0, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, out[0].data(),
                         out[1].data(), out[2].data(), nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 3, arrvec<int, N>>::type blk_offsets() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "blk_offsets";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_local(m_tensor_ptr, 2);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_total(m_tensor_ptr, 2);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, sizes[0], sizes[1], sizes[2], 0, 0,
                         0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, out[0].data(), out[1].data(), out[2].data(),
                         nullptr, nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, 0, 0, 0, 0, sizes[0], sizes[1],
                         sizes[2], 0, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, out[0].data(), out[1].data(),
                         out[2].data(), nullptr, nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 4, arrvec<int, N>>::type blks_local() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "blks_local";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_local(m_tensor_ptr, 2);
      sizes[3] = c_dbcsr_t_nblks_local(m_tensor_ptr, 3);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_total(m_tensor_ptr, 2);
      sizes[3] = c_dbcsr_t_nblks_total(m_tensor_ptr, 3);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(
          m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          sizes[0], sizes[1], sizes[2], sizes[3], 0, 0, 0, 0, out[0].data(),
          out[1].data(), out[2].data(), out[3].data(), nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(
          m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          0, 0, 0, 0, sizes[0], sizes[1], sizes[2], sizes[3], out[0].data(),
          out[1].data(), out[2].data(), out[3].data(), nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 4, arrvec<int, N>>::type proc_dist() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "proc_dist";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_local(m_tensor_ptr, 2);
      sizes[3] = c_dbcsr_t_nblks_local(m_tensor_ptr, 3);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_total(m_tensor_ptr, 2);
      sizes[3] = c_dbcsr_t_nblks_total(m_tensor_ptr, 3);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(
          m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          sizes[0], sizes[1], sizes[2], sizes[3], 0, 0, 0, 0, nullptr, nullptr,
          nullptr, nullptr, out[0].data(), out[1].data(), out[2].data(),
          out[3].data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(
          m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          0, 0, 0, 0, sizes[0], sizes[1], sizes[2], sizes[3], nullptr, nullptr,
          nullptr, nullptr, out[0].data(), out[1].data(), out[2].data(),
          out[3].data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 4, arrvec<int, N>>::type blk_sizes() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "blk_sizes";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_local(m_tensor_ptr, 2);
      sizes[3] = c_dbcsr_t_nblks_local(m_tensor_ptr, 3);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_total(m_tensor_ptr, 2);
      sizes[3] = c_dbcsr_t_nblks_total(m_tensor_ptr, 3);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(
          m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          sizes[0], sizes[1], sizes[2], sizes[3], 0, 0, 0, 0, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, out[0].data(),
          out[1].data(), out[2].data(), out[3].data(), nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, 0, 0, 0, 0, sizes[0], sizes[1],
                         sizes[2], sizes[3], nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, out[0].data(),
                         out[1].data(), out[2].data(), out[3].data(), nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    return out;
  }
  template <int M = N>
  typename std::enable_if<M == 4, arrvec<int, N>>::type blk_offsets() {
    arrvec<int, N> out;
    vec<int> sizes(N);
    constexpr char nameleft[] = "blk_offsets";
    constexpr char nameright[] = "blks_local";
    if constexpr (nameleft == nameright) {
      sizes[0] = c_dbcsr_t_nblks_local(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_local(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_local(m_tensor_ptr, 2);
      sizes[3] = c_dbcsr_t_nblks_local(m_tensor_ptr, 3);
    } else {
      sizes[0] = c_dbcsr_t_nblks_total(m_tensor_ptr, 0);
      sizes[1] = c_dbcsr_t_nblks_total(m_tensor_ptr, 1);
      sizes[2] = c_dbcsr_t_nblks_total(m_tensor_ptr, 2);
      sizes[3] = c_dbcsr_t_nblks_total(m_tensor_ptr, 3);
    }
    if constexpr (nameleft == nameright) {
      c_dbcsr_t_get_info(
          m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          sizes[0], sizes[1], sizes[2], sizes[3], 0, 0, 0, 0, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, out[0].data(), out[1].data(),
          out[2].data(), out[3].data(), nullptr, nullptr, nullptr);
    } else {
      c_dbcsr_t_get_info(
          m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          0, 0, 0, 0, sizes[0], sizes[1], sizes[2], sizes[3], nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, out[0].data(), out[1].data(),
          out[2].data(), out[3].data(), nullptr, nullptr, nullptr);
    }
    return out;
  }
# 525 "dbcsr_tensor.hpp"
  int ndim_nd() {
    int c_out;
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, 0, 0, &c_out, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr);
    return c_out;
  }
  int ndim1_2d() {
    int c_out;
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, 0, 0, nullptr, &c_out, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr);
    return c_out;
  }
  int ndim2_2d() {
    int c_out;
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, 0, 0, nullptr, nullptr, &c_out,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr);
    return c_out;
  }
# 593 "dbcsr_tensor.hpp"
  vec<long long int> dims_2d_i8() {
    int nd_size = N;
    int nd_row_size = c_dbcsr_t_ndims_matrix_row(m_tensor_ptr);
    int nd_col_size = c_dbcsr_t_ndims_matrix_column(m_tensor_ptr);
    int vsize;
    constexpr char char_dims_2d_i8[] = "dims_2d_i8";
    constexpr char char_dims_2d[] = "dims_2d";
    constexpr char char_dims_nd[] = "dims_nd";
    constexpr char char_dims1_2d[] = "dims1_2d";
    constexpr char char_dims2_2d[] = "dims2_2d";
    constexpr char char_map1_2d[] = "map1_2d";
    constexpr char char_map2_2d[] = "map2_2d";
    constexpr char char_map_nd[] = "map_nd";
    if constexpr (char_dims_2d_i8 == char_dims_2d_i8 ||
                  char_dims_2d_i8 == char_dims_2d) {
      vsize = 2;
    } else if (char_dims_2d_i8 == char_dims_nd ||
               char_dims_2d_i8 == char_map_nd) {
      vsize = nd_size;
    } else if (char_dims_2d_i8 == char_dims1_2d ||
               char_dims_2d_i8 == char_map1_2d) {
      vsize = nd_row_size;
    } else {
      vsize = nd_col_size;
    }
    vec<long long int> out(vsize);
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, nd_row_size, nd_col_size,
                               nullptr, nullptr, nullptr, out.data(), nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr);
    return out;
  }
  vec<int> dims_2d() {
    int nd_size = N;
    int nd_row_size = c_dbcsr_t_ndims_matrix_row(m_tensor_ptr);
    int nd_col_size = c_dbcsr_t_ndims_matrix_column(m_tensor_ptr);
    int vsize;
    constexpr char char_dims_2d_i8[] = "dims_2d_i8";
    constexpr char char_dims_2d[] = "dims_2d";
    constexpr char char_dims_nd[] = "dims_nd";
    constexpr char char_dims1_2d[] = "dims1_2d";
    constexpr char char_dims2_2d[] = "dims2_2d";
    constexpr char char_map1_2d[] = "map1_2d";
    constexpr char char_map2_2d[] = "map2_2d";
    constexpr char char_map_nd[] = "map_nd";
    if constexpr (char_dims_2d == char_dims_2d_i8 ||
                  char_dims_2d == char_dims_2d) {
      vsize = 2;
    } else if (char_dims_2d == char_dims_nd || char_dims_2d == char_map_nd) {
      vsize = nd_size;
    } else if (char_dims_2d == char_dims1_2d || char_dims_2d == char_map1_2d) {
      vsize = nd_row_size;
    } else {
      vsize = nd_col_size;
    }
    vec<int> out(vsize);
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, nd_row_size, nd_col_size,
                               nullptr, nullptr, nullptr, nullptr, out.data(),
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr);
    return out;
  }
  vec<int> dims_nd() {
    int nd_size = N;
    int nd_row_size = c_dbcsr_t_ndims_matrix_row(m_tensor_ptr);
    int nd_col_size = c_dbcsr_t_ndims_matrix_column(m_tensor_ptr);
    int vsize;
    constexpr char char_dims_2d_i8[] = "dims_2d_i8";
    constexpr char char_dims_2d[] = "dims_2d";
    constexpr char char_dims_nd[] = "dims_nd";
    constexpr char char_dims1_2d[] = "dims1_2d";
    constexpr char char_dims2_2d[] = "dims2_2d";
    constexpr char char_map1_2d[] = "map1_2d";
    constexpr char char_map2_2d[] = "map2_2d";
    constexpr char char_map_nd[] = "map_nd";
    if constexpr (char_dims_nd == char_dims_2d_i8 ||
                  char_dims_nd == char_dims_2d) {
      vsize = 2;
    } else if (char_dims_nd == char_dims_nd || char_dims_nd == char_map_nd) {
      vsize = nd_size;
    } else if (char_dims_nd == char_dims1_2d || char_dims_nd == char_map1_2d) {
      vsize = nd_row_size;
    } else {
      vsize = nd_col_size;
    }
    vec<int> out(vsize);
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, nd_row_size, nd_col_size,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               out.data(), nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr);
    return out;
  }
  vec<int> dims1_2d() {
    int nd_size = N;
    int nd_row_size = c_dbcsr_t_ndims_matrix_row(m_tensor_ptr);
    int nd_col_size = c_dbcsr_t_ndims_matrix_column(m_tensor_ptr);
    int vsize;
    constexpr char char_dims_2d_i8[] = "dims_2d_i8";
    constexpr char char_dims_2d[] = "dims_2d";
    constexpr char char_dims_nd[] = "dims_nd";
    constexpr char char_dims1_2d[] = "dims1_2d";
    constexpr char char_dims2_2d[] = "dims2_2d";
    constexpr char char_map1_2d[] = "map1_2d";
    constexpr char char_map2_2d[] = "map2_2d";
    constexpr char char_map_nd[] = "map_nd";
    if constexpr (char_dims1_2d == char_dims_2d_i8 ||
                  char_dims1_2d == char_dims_2d) {
      vsize = 2;
    } else if (char_dims1_2d == char_dims_nd || char_dims1_2d == char_map_nd) {
      vsize = nd_size;
    } else if (char_dims1_2d == char_dims1_2d ||
               char_dims1_2d == char_map1_2d) {
      vsize = nd_row_size;
    } else {
      vsize = nd_col_size;
    }
    vec<int> out(vsize);
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, nd_row_size, nd_col_size,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, out.data(), nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr);
    return out;
  }
  vec<int> dims2_2d() {
    int nd_size = N;
    int nd_row_size = c_dbcsr_t_ndims_matrix_row(m_tensor_ptr);
    int nd_col_size = c_dbcsr_t_ndims_matrix_column(m_tensor_ptr);
    int vsize;
    constexpr char char_dims_2d_i8[] = "dims_2d_i8";
    constexpr char char_dims_2d[] = "dims_2d";
    constexpr char char_dims_nd[] = "dims_nd";
    constexpr char char_dims1_2d[] = "dims1_2d";
    constexpr char char_dims2_2d[] = "dims2_2d";
    constexpr char char_map1_2d[] = "map1_2d";
    constexpr char char_map2_2d[] = "map2_2d";
    constexpr char char_map_nd[] = "map_nd";
    if constexpr (char_dims2_2d == char_dims_2d_i8 ||
                  char_dims2_2d == char_dims_2d) {
      vsize = 2;
    } else if (char_dims2_2d == char_dims_nd || char_dims2_2d == char_map_nd) {
      vsize = nd_size;
    } else if (char_dims2_2d == char_dims1_2d ||
               char_dims2_2d == char_map1_2d) {
      vsize = nd_row_size;
    } else {
      vsize = nd_col_size;
    }
    vec<int> out(vsize);
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, nd_row_size, nd_col_size,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, out.data(), nullptr, nullptr,
                               nullptr, nullptr, nullptr);
    return out;
  }
  vec<int> map1_2d() {
    int nd_size = N;
    int nd_row_size = c_dbcsr_t_ndims_matrix_row(m_tensor_ptr);
    int nd_col_size = c_dbcsr_t_ndims_matrix_column(m_tensor_ptr);
    int vsize;
    constexpr char char_dims_2d_i8[] = "dims_2d_i8";
    constexpr char char_dims_2d[] = "dims_2d";
    constexpr char char_dims_nd[] = "dims_nd";
    constexpr char char_dims1_2d[] = "dims1_2d";
    constexpr char char_dims2_2d[] = "dims2_2d";
    constexpr char char_map1_2d[] = "map1_2d";
    constexpr char char_map2_2d[] = "map2_2d";
    constexpr char char_map_nd[] = "map_nd";
    if constexpr (char_map1_2d == char_dims_2d_i8 ||
                  char_map1_2d == char_dims_2d) {
      vsize = 2;
    } else if (char_map1_2d == char_dims_nd || char_map1_2d == char_map_nd) {
      vsize = nd_size;
    } else if (char_map1_2d == char_dims1_2d || char_map1_2d == char_map1_2d) {
      vsize = nd_row_size;
    } else {
      vsize = nd_col_size;
    }
    vec<int> out(vsize);
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, nd_row_size, nd_col_size,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, out.data(), nullptr,
                               nullptr, nullptr, nullptr);
    return out;
  }
  vec<int> map2_2d() {
    int nd_size = N;
    int nd_row_size = c_dbcsr_t_ndims_matrix_row(m_tensor_ptr);
    int nd_col_size = c_dbcsr_t_ndims_matrix_column(m_tensor_ptr);
    int vsize;
    constexpr char char_dims_2d_i8[] = "dims_2d_i8";
    constexpr char char_dims_2d[] = "dims_2d";
    constexpr char char_dims_nd[] = "dims_nd";
    constexpr char char_dims1_2d[] = "dims1_2d";
    constexpr char char_dims2_2d[] = "dims2_2d";
    constexpr char char_map1_2d[] = "map1_2d";
    constexpr char char_map2_2d[] = "map2_2d";
    constexpr char char_map_nd[] = "map_nd";
    if constexpr (char_map2_2d == char_dims_2d_i8 ||
                  char_map2_2d == char_dims_2d) {
      vsize = 2;
    } else if (char_map2_2d == char_dims_nd || char_map2_2d == char_map_nd) {
      vsize = nd_size;
    } else if (char_map2_2d == char_dims1_2d || char_map2_2d == char_map1_2d) {
      vsize = nd_row_size;
    } else {
      vsize = nd_col_size;
    }
    vec<int> out(vsize);
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, nd_row_size, nd_col_size,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, out.data(),
                               nullptr, nullptr, nullptr);
    return out;
  }
  vec<int> map_nd() {
    int nd_size = N;
    int nd_row_size = c_dbcsr_t_ndims_matrix_row(m_tensor_ptr);
    int nd_col_size = c_dbcsr_t_ndims_matrix_column(m_tensor_ptr);
    int vsize;
    constexpr char char_dims_2d_i8[] = "dims_2d_i8";
    constexpr char char_dims_2d[] = "dims_2d";
    constexpr char char_dims_nd[] = "dims_nd";
    constexpr char char_dims1_2d[] = "dims1_2d";
    constexpr char char_dims2_2d[] = "dims2_2d";
    constexpr char char_map1_2d[] = "map1_2d";
    constexpr char char_map2_2d[] = "map2_2d";
    constexpr char char_map_nd[] = "map_nd";
    if constexpr (char_map_nd == char_dims_2d_i8 ||
                  char_map_nd == char_dims_2d) {
      vsize = 2;
    } else if (char_map_nd == char_dims_nd || char_map_nd == char_map_nd) {
      vsize = nd_size;
    } else if (char_map_nd == char_dims1_2d || char_map_nd == char_map1_2d) {
      vsize = nd_row_size;
    } else {
      vsize = nd_col_size;
    }
    vec<int> out(vsize);
    c_dbcsr_t_get_mapping_info(m_tensor_ptr, N, nd_row_size, nd_col_size,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               out.data(), nullptr, nullptr);
    return out;
  }
# 641 "dbcsr_tensor.hpp"
  tensor() : m_tensor_ptr(nullptr) {}

  ~tensor() { destroy(); }

  void destroy() {

    if (m_tensor_ptr != nullptr) {
      c_dbcsr_t_destroy(&m_tensor_ptr);
    }

    m_tensor_ptr = nullptr;
  }
# 683 "dbcsr_tensor.hpp"
  template <int M = N>
  typename std::enable_if<M == 2>::type reserve(arrvec<int, N> &nzblks) {
    if (nzblks[0].size() == 0)
      return;
    if (!(nzblks[0].size() == nzblks[0].size() &&
          nzblks[0].size() == nzblks[1].size())) {
      throw std::runtime_error("tensor.reserve : wrong dimensions.");
    }
    c_dbcsr_t_reserve_blocks_index(m_tensor_ptr, nzblks[0].size(),
                                   nzblks[0].data(), nzblks[1].data(), nullptr,
                                   nullptr);
  }
  template <int M = N>
  typename std::enable_if<M == 3>::type reserve(arrvec<int, N> &nzblks) {
    if (nzblks[0].size() == 0)
      return;
    if (!(nzblks[0].size() == nzblks[0].size() &&
          nzblks[0].size() == nzblks[1].size() &&
          nzblks[0].size() == nzblks[2].size())) {
      throw std::runtime_error("tensor.reserve : wrong dimensions.");
    }
    c_dbcsr_t_reserve_blocks_index(m_tensor_ptr, nzblks[0].size(),
                                   nzblks[0].data(), nzblks[1].data(),
                                   nzblks[2].data(), nullptr);
  }
  template <int M = N>
  typename std::enable_if<M == 4>::type reserve(arrvec<int, N> &nzblks) {
    if (nzblks[0].size() == 0)
      return;
    if (!(nzblks[0].size() == nzblks[0].size() &&
          nzblks[0].size() == nzblks[1].size() &&
          nzblks[0].size() == nzblks[2].size() &&
          nzblks[0].size() == nzblks[3].size())) {
      throw std::runtime_error("tensor.reserve : wrong dimensions.");
    }
    c_dbcsr_t_reserve_blocks_index(m_tensor_ptr, nzblks[0].size(),
                                   nzblks[0].data(), nzblks[1].data(),
                                   nzblks[2].data(), nzblks[3].data());
  }

  void reserve_all() {

    auto blks = this->blks_local();
    arrvec<int, N> res;

    std::function<void(int, int *)> loop;

    int *arr = new int[N];

    loop = [&res, &blks, &loop](int depth, int *vals) {
      for (auto eleN : blks[depth]) {
        vals[depth] = eleN;
        if (depth == N - 1) {
          for (int i = 0; i != N; ++i) {
            res[i].push_back(vals[i]);
          }
        } else {
          loop(depth + 1, vals);
        }
      }
    };

    loop(0, arr);
    this->reserve(res);

    delete[] arr;
  }

  void reserve_template(tensor &t_template) {
    c_dbcsr_t_reserve_blocks_template(t_template.m_tensor_ptr,
                                      this->m_tensor_ptr);
  }

  void put_block(const index<N> &idx, block<N, T> &blk,
                 std::optional<bool> sum = std::nullopt,
                 std::optional<double> scale = std::nullopt) {

    c_dbcsr_t_put_block(m_tensor_ptr, idx.data(), blk.size().data(), blk.data(),
                        (sum) ? &*sum : nullptr, (scale) ? &*scale : nullptr);
  }

  void put_block(const index<N> &idx, T *data, const index<N> &size) {
    c_dbcsr_t_put_block(m_tensor_ptr, idx.data(), size.data(), data, nullptr,
                        nullptr);
  }

  block<N, T> get_block(const index<N> &idx, const index<N> &blk_size,
                        bool &found) {

    block<N, T> blk_out(blk_size);

    c_dbcsr_t_get_block(m_tensor_ptr, idx.data(), blk_size.data(),
                        blk_out.data(), &found);

    return blk_out;
  }

  void get_block(T *data_ptr, const index<N> &idx, const index<N> &blk_size,
                 bool &found) {

    c_dbcsr_t_get_block(m_tensor_ptr, idx.data(), blk_size.data(), data_ptr,
                        &found);
  }

  T *get_block_p(const index<N> &idx, bool &found) {

    T *out = nullptr;

    c_dbcsr_t_get_block_p(m_tensor_ptr, idx.data(), &out, &found);

    return out;
  }

  int proc(const index<N> &idx) {
    int p = -1;
    c_dbcsr_t_get_stored_coordinates(m_tensor_ptr, idx.data(), &p);
    return p;
  }

  T *data(long long int &data_size) {

    T *data_ptr;
    T data_type = T();

    c_dbcsr_t_get_data_p(m_tensor_ptr, &data_ptr, &data_size, data_type,
                         nullptr, nullptr);

    return data_ptr;
  }

  void clear() { c_dbcsr_t_clear(m_tensor_ptr); }

  MPI_Comm comm() { return m_comm; }

  std::string name() const {
    char *cstring;
    c_dbcsr_t_get_info(m_tensor_ptr, N, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nullptr, &cstring,
                       nullptr);
    std::string out(cstring);
    c_free_string(&cstring);
    return out;
  }

  int num_blocks() const { return c_dbcsr_t_get_num_blocks(m_tensor_ptr); }

  long long int num_blocks_total() const {
    return c_dbcsr_t_get_num_blocks_total(m_tensor_ptr);
  }

  int num_nze() { return c_dbcsr_t_get_nze(m_tensor_ptr); }

  long long int num_nze_total() const {
    return c_dbcsr_t_get_nze_total(m_tensor_ptr);
  }

  void filter(T eps, std::optional<filter> method = std::nullopt,
              std::optional<bool> use_absolute = std::nullopt) {

    int fmethod = (method) ? static_cast<int>(*method) : 0;

    c_dbcsr_t_filter(m_tensor_ptr, eps, (method) ? &fmethod : nullptr,
                     (use_absolute) ? &*use_absolute : nullptr);
  }

  double occupation() {

    auto nfull = this->nfull_total();
    long long int tote = std::accumulate(nfull.begin(), nfull.end(), 1,
                                         std::multiplies<long long int>());
    long long int nze = this->num_nze_total();

    return (double)nze / (double)tote;
  }

  double long_sum() {

    long long int num_nze;
    T *data = this->data(num_nze);

    double local_sum = std::accumulate(data, data + num_nze, 0.0);
    double total_sum = 0.0;

    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, m_comm);

    return total_sum;
  }

  void scale(T factor) { c_dbcsr_t_scale(m_tensor_ptr, factor); }

  void set(T factor) { c_dbcsr_t_set(m_tensor_ptr, factor); }

  void finalize() { c_dbcsr_t_finalize(m_tensor_ptr); }

  void batched_contract_init() {
    c_dbcsr_t_batched_contract_init(m_tensor_ptr);
  }

  void batched_contract_finalize() {
    c_dbcsr_t_batched_contract_finalize(m_tensor_ptr, nullptr);
  }

  vec<int> idx_speed() {

    auto map1 = this->map1_2d();
    auto map2 = this->map2_2d();
    map2.insert(map2.end(), map1.begin(), map1.end());

    return map2;
  }

  std::shared_ptr<tensor<N, T>> get_ptr() { return this->shared_from_this(); }
};

template <int N, typename T = double>
using shared_tensor = std::shared_ptr<tensor<N, T>>;

template <int N, typename T> class iterator_t {
private:
  void *m_iter_ptr;
  void *m_tensor_ptr;

  index<N> m_idx;
  int m_blk_n;
  int m_blk_p;
  std::array<int, N> m_size;
  std::array<int, N> m_offset;

public:
  iterator_t(tensor<N, T> &t_tensor)
      : m_iter_ptr(nullptr), m_tensor_ptr(t_tensor.m_tensor_ptr), m_blk_n(0),
        m_blk_p(0) {}

  ~iterator_t() {}

  void start() { c_dbcsr_t_iterator_start(&m_iter_ptr, m_tensor_ptr); }

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

  bool blocks_left() { return c_dbcsr_t_iterator_blocks_left(m_iter_ptr); }

  const index<N> &idx() { return m_idx; }

  const index<N> &size() { return m_size; }

  const index<N> &offset() { return m_offset; }

  int blk_n() { return m_blk_n; }

  int blk_p() { return m_blk_p; }
};

template <int N, typename T> void print(tensor<N, T> &t_in) {

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

} // namespace dbcsr
