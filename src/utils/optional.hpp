#ifndef UTILS_OPTIONAL_HPP
#define UTILS_OPTIONAL_HPP

#include <functional>
#include <type_traits>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <optional>

namespace util {
	
template<typename T>
struct has_const_iterator
{
private:
  typedef char                      one;
  typedef struct { char array[2]; } two;

  template<typename C> static one test(typename C::const_iterator*);
  template<typename C> static two  test(...);
public:
  static const bool value = sizeof(test<T>(0)) == sizeof(one);
  typedef T type;
};

template <typename T>
struct has_begin_end
{
  struct Dummy { typedef void const_iterator; };
  typedef typename std::conditional<has_const_iterator<T>::value, T, Dummy>::type TType;
  typedef typename TType::const_iterator iter;

  struct Fallback { iter begin() const; iter end() const; };
  struct Derived : TType, Fallback { };

  template<typename C, C> struct ChT;

  template<typename C> static char (&f(ChT<iter (Fallback::*)() const, &C::begin>*))[1];
  template<typename C> static char (&f(...))[2];
  template<typename C> static char (&g(ChT<iter (Fallback::*)() const, &C::end>*))[1];
  template<typename C> static char (&g(...))[2];

  static bool const beg_value = sizeof(f<Derived>(0)) == 2;
  static bool const end_value = sizeof(g<Derived>(0)) == 2;
};

template <typename T>
struct is_container
{
  static const bool value = has_const_iterator<T>::value &&
    has_begin_end<T>::beg_value && has_begin_end<T>::end_value;
};

template<class T> struct remove_all { typedef T type; };
template<class T> struct remove_all<T*> : remove_all<T> {};
template<class T> struct remove_all<T&> : remove_all<T> {};
template<class T> struct remove_all<T&&> : remove_all<T> {};
template<class T> struct remove_all<T const> : remove_all<T> {};

struct nullopt_t {
    explicit constexpr nullopt_t() {}
};
inline constexpr nullopt_t nullopt{};

template <typename T>
class optional {
public:

	typedef T value_type;
	typedef typename remove_all<T>::type base_type;
	typedef typename std::remove_reference<T>::type noref_type;
	
private:

	noref_type* _val;
	noref_type* _ref;

public:
	
	optional() : _val(nullptr), _ref(nullptr) {}
	
	template <class D = T, typename = typename std::enable_if<
		!std::is_reference<D>::value>::type>
	optional(const noref_type val) {
		_val = new noref_type(val);
		_ref = _val;
	}
	
	template <class E, class D = T, typename = typename std::enable_if<
		!std::is_reference<D>::value>::type>
	optional(const E val) {
		_val = new noref_type(val);
		_ref = _val;
	}
	
	template <class D = T, typename = typename std::enable_if<
		!std::is_reference<D>::value>::type>
	optional(const std::optional<T> opt_val) {
		if (opt_val) {
			_val = new noref_type(*opt_val);
			_ref = _val;
		} else {
			_val = nullptr;
			_ref = nullptr;
		}
	}
	
	template <class E, class D = T, typename = typename std::enable_if<
		!std::is_reference<D>::value &&
		is_container<D>::value
		>::type>
	optional(std::initializer_list<E> list) {
		_val = new noref_type(list);
		_ref = _val;
	}
	
	optional(nullopt_t no_arg) : _ref(nullptr), _val(nullptr) {}
	
	optional(std::nullopt_t no_arg) : _ref(nullptr), _val(nullptr) {}
	
	template <class D = T, typename = typename std::enable_if<
		std::is_reference<D>::value>::type>
	optional(noref_type& val) {
		_val = nullptr;
		_ref = &val;
	}
	
	noref_type& operator*() const {
		return *_ref;
	}
	
	noref_type* operator->() const {
		 return _ref;
	}
	
	operator bool () const {
		return (_ref) ? true : false;
	}
	
	~optional() {
		if (_val) delete _val;
	}
	
};
		
} // end namespace util

#endif
