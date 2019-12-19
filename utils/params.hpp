#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <functional>
#include <type_traits>
#include <cassert>
#include <initializer_list>

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
 
namespace params {

enum attribute0 { required_, optional_ };
enum attribute1 { reference_, value_ };

template <class T, attribute0 a0, attribute1 a1>
class attribute_wrapper {
private:
	
	T* m_val = nullptr;
	T* m_ref = nullptr;
	bool init;
	
public:

	template <attribute0 A0 = a0, 
		typename std::enable_if<A0 == required_, int>::type = 0>
	attribute_wrapper() = delete;
	
	template <attribute0 A0 = a0, 
		typename = typename std::enable_if<A0 == optional_, int>::type>
	attribute_wrapper() : init(false), m_val(nullptr), m_ref(nullptr) {}
	
	template <class D = T, attribute0 A0 = a0, attribute1 A1 = a1, 
		typename = typename std::enable_if<A1 == value_, int>::type>
	attribute_wrapper(D r) : m_val(new T(r)), m_ref(m_val), init(true) {}
	
	attribute_wrapper(attribute_wrapper& awrp) : init(false), m_val(nullptr), m_ref(nullptr) {
		std::cout << "Passing through!" << std::endl;
		if (awrp.m_val != nullptr) {
			this->m_val = new T(*awrp.m_val);
		}
		if (awrp.m_ref != nullptr) {
			this->m_ref = &*awrp.m_ref;
			this->init = true;
		} 
		std::cout << "Done" << std::endl;
	}
		
	template <typename D = T, attribute0 A0 = a0, attribute1 A1 = a1, 
		typename = typename std::enable_if<(A1 == value_) && (is_container<D>::value), int>::type>
	attribute_wrapper(std::initializer_list<typename D::value_type> r) 
		: m_val(new T), m_ref(m_val), init(true) {*m_val = r;}

	template <class D = T, attribute0 A0 = a0, attribute1 A1 = a1, 
		typename = typename std::enable_if<A1 == reference_, int>::type>
	attribute_wrapper(D& r) : m_ref(&r), init(true) {}
	
	attribute_wrapper operator=(T& in) = delete;	
	
	T& operator*() {
		assert(("Optional parameter not initialized.", init == true));
		return *m_ref;
	}
	
	T* operator->() const {
		 return m_ref;
	}
	
	operator bool () {
		return init;
	}
	
	~attribute_wrapper() { if (m_val) { delete m_val;} }
	
};

} // end namespace params

template <class T, params::attribute1 a1>
using required = params::attribute_wrapper<T,params::required_,a1>;

template <class T, params::attribute1 a1>
using optional = params::attribute_wrapper<T,params::optional_,a1>;

const auto val = params::value_;
const auto ref = params::reference_;

#endif
