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
# 42 "dbcsr_tensor.hpp"
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
# 226 "dbcsr_tensor.hpp"
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
    util::builder_type<std::string>::type c_name;
    util::builder_type<util::optional<dist_t<N> &>>::type c_set_dist;
    util::builder_type<util::optional<pgrid<N> &>>::type c_set_pgrid;
    util::builder_type<vec<int>>::type c_map1;
    util::builder_type<vec<int>>::type c_map2;
    util::builder_type<arrvec<int, N>>::type c_blk_sizes;
  public:
    _create_base &
    name(util::optional<util::base_type<std::string>::type> i_name) {
      c_name = std::move(i_name);
      return *this;
    }
    _create_base &
    set_dist(util::optional<util::base_type<util::optional<dist_t<N> &>>::type>
                 i_set_dist) {
      c_set_dist = std::move(i_set_dist);
      return *this;
    }
    _create_base &
    set_pgrid(util::optional<util::base_type<util::optional<pgrid<N> &>>::type>
                  i_set_pgrid) {
      c_set_pgrid = std::move(i_set_pgrid);
      return *this;
    }
    _create_base &map1(util::optional<util::base_type<vec<int>>::type> i_map1) {
      c_map1 = std::move(i_map1);
      return *this;
    }
    _create_base &map2(util::optional<util::base_type<vec<int>>::type> i_map2) {
      c_map2 = std::move(i_map2);
      return *this;
    }
    _create_base &blk_sizes(
        util::optional<util::base_type<arrvec<int, N>>::type> i_blk_sizes) {
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
  };

  template <int M = N, typename std::enable_if<${idim} $ == M, int>::type = 0>
  tensor(create_pack &&p) {

    void *dist_ptr = nullptr;
    std::shared_ptr<dist_t<M>> distn;

    if (p.p_set_dist) {

      dist_ptr = p.p_set_dist->m_dist_ptr;
      m_comm = p.p_set_dist->m_comm;

    } else {

      arrvec<int, M> distvecs;
      auto pgrid_dims = p.p_set_pgrid->dims();

      for (int i = 0; i != M; ++i) {
        distvecs[i] = default_dist(p.p_blk_sizes[i].size(), pgrid_dims[i],
                                   p.p_blk_sizes[i]);
      }

      distn = dist_t<M>::create()
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
    std::string p_name;
    util::optional<dist_t<N> &> p_ndist;
    util::optional<vec<int>> p_map1;
    util::optional<vec<int>> p_map2;
    tensor<N, T> &p_templet;
  };

  class create_template_base {
    typedef create_template_base _create_base;
  private:
    util::builder_type<std::string>::type c_name;
    util::builder_type<util::optional<dist_t<N> &>>::type c_ndist;
    util::builder_type<util::optional<vec<int>>>::type c_map1;
    util::builder_type<util::optional<vec<int>>>::type c_map2;
    util::builder_type<tensor<N, T> &>::type c_templet;
  public:
    _create_base &
    templet(util::optional<util::base_type<tensor<N, T> &>::type> i_templet) {
      c_templet = std::move(i_templet);
      return *this;
    }
    create_template_base(std::string i_name,
                         util::optional<dist_t<N> &> i_ndist,
                         util::optional<vec<int>> i_map1,
                         util::optional<vec<int>> i_map2)
        : c_name(i_name), c_ndist(i_ndist), c_map1(i_map1), c_map2(i_map2) {}
    std::shared_ptr<tensor> build() {
      if constexpr (!util::is_optional<tensor<N, T> &>::value) {
        if (!c_templet) {
          throw std::runtime_error("Parameter "
                                   "templet"
                                   " not present!");
        }
      }
      create_template_pack p = {
          util::evaluate<std::string>(c_name),
          util::evaluate<util::optional<dist_t<N> &>>(c_ndist),
          util::evaluate<util::optional<vec<int>>>(c_map1),
          util::evaluate<util::optional<vec<int>>>(c_map2),
          util::evaluate<tensor<N, T> &>(c_templet)};
      return std::make_shared<tensor>(std::move(p));
    }
  };
  template <typename... Types>
  static create_template_base create_template(Types &&... p) {
    return create_template_base(std::forward<Types>(p)...);
  };

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
  struct create_template_pack {
    std::string p_name;
    util::optional<dist_t<N> &> p_ndist;
    util::optional<vec<int>> p_map1;
    util::optional<vec<int>> p_map2;
    util::optional<std::string> p_name;
    util::optional<vec<int>> p_order;
  };

  class create_template_base {
    typedef create_template_base _create_base;
  private:
    util::builder_type<matrix<T> &>::type c_matrix_in;
    util::builder_type<util::optional<std::string>>::type c_name;
    util::builder_type<util::optional<vec<int>>>::type c_order;
  public:
    _create_base &
    name(util::optional<util::base_type<util::optional<std::string>>::type>
             i_name) {
      c_name = std::move(i_name);
      return *this;
    }
    _create_base &
    order(util::optional<util::base_type<util::optional<vec<int>>>::type>
              i_order) {
      c_order = std::move(i_order);
      return *this;
    }
    create_template_base(matrix<T> &i_matrix_in) : c_matrix_in(i_matrix_in) {}
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
      create_template_pack p = {
          util::evaluate<matrix<T> &>(c_matrix_in),
          util::evaluate<util::optional<std::string>>(c_name),
          util::evaluate<util::optional<vec<int>>>(c_order)};
      return std::make_shared<tensor>(std::move(p));
    }
  };
  template <typename... Types>
  static create_template_base create_template(Types &&... p) {
    return create_template_base(std::forward<Types>(p)...);
  };

  tensor(create_matrix_pack &&p) {

    m_comm = p.p_matrix_in.get_world().comm();

    c_dbcsr_t_create_matrix(p.p_matrix.m_matrix_ptr, &m_tensor_ptr,
                            (p.p_order) ? p.p_order->data() : nullptr,
                            (p.p_name) ? p.p_name->c_str() : nullptr);
  }
