#ifndef UTILS_OPTIONAL_HPP
#define UTILS_OPTIONAL_HPP

#include <cassert>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <type_traits>

namespace util {

static constexpr bool util_debug = false;

template <typename T>
struct has_const_iterator {
 private:
  typedef char one;
  typedef struct {
    char array[2];
  } two;

  template <typename C>
  static one test(typename C::const_iterator*);
  template <typename C>
  static two test(...);

 public:
  static const bool value = sizeof(test<T>(0)) == sizeof(one);
  typedef T type;
};

template <typename T>
struct has_begin_end {
  struct Dummy {
    typedef void const_iterator;
  };
  typedef
      typename std::conditional<has_const_iterator<T>::value, T, Dummy>::type
          TType;
  typedef typename TType::const_iterator iter;

  struct Fallback {
    iter begin() const;
    iter end() const;
  };
  struct Derived : TType, Fallback {
  };

  template <typename C, C>
  struct ChT;

  template <typename C>
  static char (&f(ChT<iter (Fallback::*)() const, &C::begin>*))[1];
  template <typename C>
  static char (&f(...))[2];
  template <typename C>
  static char (&g(ChT<iter (Fallback::*)() const, &C::end>*))[1];
  template <typename C>
  static char (&g(...))[2];

  static bool const beg_value = sizeof(f<Derived>(0)) == 2;
  static bool const end_value = sizeof(g<Derived>(0)) == 2;
};

template <typename T>
struct is_container {
  static const bool value = has_const_iterator<T>::value &&
      has_begin_end<T>::beg_value && has_begin_end<T>::end_value;
};

template <class T>
struct remove_all {
  typedef T type;
};
template <class T>
struct remove_all<T*> : remove_all<T> {
};
template <class T>
struct remove_all<T&> : remove_all<T> {
};
template <class T>
struct remove_all<T&&> : remove_all<T> {
};
template <class T>
struct remove_all<T const> : remove_all<T> {
};

struct nullopt_t {
  explicit constexpr nullopt_t()
  {
  }
};
inline constexpr nullopt_t nullopt{};

template <typename T>
class optional {
 public:
  typedef T value_type;
  typedef typename remove_all<T>::type base_type;
  typedef typename std::remove_reference<T>::type noref_type;

 protected:
  noref_type* _val = nullptr;

 public:
  optional() : _val(nullptr)
  {
    if constexpr (util_debug) {
      std::cout << "0" << std::endl;
      std::cout << "This is me: " << this << std::endl;
      std::cout << "INIT: " << &_val << " is null: " << _val << std::endl;
    }
  }

  template <
      class D = T,
      typename = typename std::enable_if<!std::is_reference<D>::value>::type>
  optional(const noref_type var)
  {
    if constexpr (util_debug) {
      std::cout << "1" << std::endl;
      std::cout << "TYPE: " << typeid(*_val).name() << std::endl;
      std::cout << "This is me: " << this << std::endl;
    }

    _val = new noref_type(var);
  }

  template <
      class D = T,
      typename = typename std::enable_if<std::is_reference<D>::value>::type>
  optional(noref_type& var)
  {
    if constexpr (util_debug)
      std::cout << "12" << std::endl;
    _val = &var;
  }

  template <
      class E,
      class D = T,
      typename = typename std::enable_if<
          !std::is_same<D, E>::value && !std::is_reference<D>::value>::type>
  optional(const E val)
  {
    if constexpr (util_debug)
      std::cout << "2" << std::endl;
    _val = new noref_type(val);
  }

  template <
      class D = T,
      typename = typename std::enable_if<!std::is_reference<D>::value>::type>
  optional(const std::optional<noref_type>& opt_val)
  {
    if constexpr (util_debug)
      std::cout << "3" << std::endl;
    _val = (opt_val) ? new noref_type(*opt_val) : nullptr;
  }

  template <
      class E,
      class D = T,
      typename = typename std::enable_if<
          !std::is_reference<D>::value && is_container<D>::value>::type>
  optional(std::initializer_list<E> list)
  {
    if constexpr (util_debug)
      std::cout << "4" << std::endl;
    _val = new noref_type(list);
  }

  template <
      class D = T,
      typename = typename std::enable_if<
          std::is_copy_constructible<D>::value &&
          !std::is_reference<D>::value>::type>
  void init_copy(const optional<D>& obj)
  {
    if constexpr (util_debug)
      std::cout << "5" << std::endl;
    this->_val = (obj._val) ? new noref_type(*obj._val) : nullptr;
  }

  template <class D = T, std::enable_if_t<std::is_reference<D>::value, int> = 0>
  void init_copy(const optional<D>& obj)
  {
    if constexpr (util_debug)
      std::cout << "6" << std::endl;
    this->_val = (obj._val) ? obj._val : nullptr;
  }

  template <
      class D = T,
      typename = typename std::enable_if<!std::is_reference<D>::value>::type>
  void init_move(optional&& obj)
  {
    if constexpr (util_debug)
      std::cout << "7" << std::endl;

    this->_val = (obj._val) ? obj._val : nullptr;
    obj._val = nullptr;
  }

  template <class D = T, std::enable_if_t<std::is_reference<D>::value, int> = 0>
  void init_move(optional&& obj)
  {
    if constexpr (util_debug)
      std::cout << "8" << std::endl;
    if (obj._val) {
      this->_val = obj._val;
      obj._val = nullptr;
    }
    else {
      this->_val = nullptr;
    }
  }

  template <
      class D = T,
      typename = typename std::enable_if<!std::is_reference<D>::value>::type>
  void assign_copy(const optional& obj)
  {
    if constexpr (util_debug)
      std::cout << "9" << std::endl;

    if (this == &obj)
      return;
    if (this->_val)
      delete this->_val;

    this->_val = (obj._val) ? new noref_type(*obj._val) : nullptr;
  }

  template <class D = T, std::enable_if_t<std::is_reference<D>::value, int> = 0>
  void assign_copy(const optional& obj)
  {
    if constexpr (util_debug)
      std::cout << "10" << std::endl;

    if (this == &obj)
      return;
    this->_val = nullptr;

    this->_val = (obj._val) ? obj._val : nullptr;
  }

  template <
      class D = T,
      typename = typename std::enable_if<!std::is_reference<D>::value>::type>
  void assign_move(optional&& obj)
  {
    if constexpr (util_debug)
      std::cout << "11" << std::endl;

    if (this == &obj)
      return;
    if (this->_val)
      delete this->_val;

    this->_val = (obj._val) ? obj._val : nullptr;
    obj._val = nullptr;
  }

  template <class D = T, std::enable_if_t<std::is_reference<D>::value, int> = 0>
  void assign_move(optional&& obj)
  {
    if constexpr (util_debug)
      std::cout << "12" << std::endl;

    if (this == &obj)
      return;

    this->_val = (obj._val) ? obj._val : nullptr;
    obj._val = nullptr;
  }

  template <
      class D = T,
      typename = typename std::enable_if<!std::is_reference<D>::value>::type>
  void destroy()
  {
    if constexpr (util_debug)
      std::cout << "D0" << std::endl;
    if (_val)
      delete _val;
    _val = nullptr;
  }

  template <class D = T, std::enable_if_t<std::is_reference<D>::value, int> = 0>
  void destroy()
  {
    if constexpr (util_debug)
      std::cout << "D1" << std::endl;
    _val = nullptr;
  }

  optional(const optional& obj)
  {
    this->init_copy(obj);
  }

  optional(optional&& obj)
  {
    this->init_move(std::forward<optional>(obj));
  }

  optional& operator=(const optional& obj)
  {
    this->assign_copy(obj);
    return *this;
  }

  optional& operator=(optional&& obj)
  {
    this->assign_move(std::forward<optional>(obj));
    return *this;
  }

  optional(nullopt_t no_arg) : _val(nullptr)
  {
    (void)no_arg;
    if constexpr (util_debug)
      std::cout << "13" << std::endl;
    if constexpr (util_debug)
      std::cout << "This is me: " << this << std::endl;
  }

  optional(std::nullopt_t no_arg) : _val(nullptr)
  {
    (void)no_arg;
    if constexpr (util_debug)
      std::cout << "14" << std::endl;
    if constexpr (util_debug)
      std::cout << "This is me: " << this << std::endl;
  }

  noref_type& operator*() const
  {
    return *_val;
  }

  noref_type* operator->() const
  {
    return _val;
  }

  operator bool() const
  {
    return (_val) ? true : false;
  }

  ~optional()
  {
    if constexpr (util_debug)
      std::cout << "Destructor for " << this << std::endl;

    this->destroy();

    if constexpr (util_debug)
      std::cout << "DELETED: " << &_val << " with " << _val << std::endl;
  }
};

}  // end namespace util

#endif
