#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <functional>
#include <type_traits>
#include <cassert>
#include <initializer_list>
#include <iostream>

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

/* A class spefically designed for Fortran-like parameter passing
 * It is a wrapper class which allows to use special attributes like
 * optional/required or reference/value
 * Mostly used in conjuction whith designated initilaizers in c++20
 */

namespace params {

enum attribute0 { required_, optional_ };
enum attribute1 { reference_, value_ };

template <class T, attribute0 a0, attribute1 a1>
class attribute_wrapper;

template <typename T> 
struct is_attribute_wrapper {
private:
		
	typedef std::true_type yes;
	typedef std::false_type no;
		
	template<typename U> static auto test(int) -> decltype(U::is_attribute_wrapper_ == true, yes());
	template<typename> static no test(...);
		
public:
 
	static constexpr bool value = std::is_same<decltype(test<T>(0)),yes>::value;
	
};
		
template <class T, attribute0 a0, attribute1 a1>
class attribute_wrapper {
private:	
	// if a1 = value, the class stores it own copy,
	// and then m_ref points to that value
	T* m_val = nullptr;
	T* m_ref = nullptr;
	bool init;

public:

	static const bool is_attribute_wrapper_ = true;
	
	template <typename D, attribute0 A0, attribute1 A1> 
	friend class attribute_wrapper;
	
	// delete for required, such that default initilization is disbaled
	// currently disabled because not working in C++20
	//template <attribute0 A0 = a0, 
	//	typename std::enable_if<A0 == required_, int>::type = 0>
	//attribute_wrapper() = delete;
	
	// optional can be default initialized
	//template <attribute0 A0 = a0, 
	//	typename = typename std::enable_if<A0 == optional_, int>::type>
	attribute_wrapper() : init(false), m_val(nullptr), m_ref(nullptr) { /*std::cout << "D2" << std::endl;*/ }
	
	// only enable for a1 = value
	template <class D = T, attribute0 A0 = a0, attribute1 A1 = a1, 
		typename = typename std::enable_if<A1 == value_ && !is_attribute_wrapper<D>::value, int>::type>
	attribute_wrapper(D r) : m_val(new T(r)), m_ref(m_val), init(true) { /*std::cout << typeid(r).name() << "D1" << std::endl;*/ }
	
	// copy constructor. Pay attention that references remain valid
	attribute_wrapper(const attribute_wrapper<T,a0,a1>& awrp) : init(false), m_val(nullptr), m_ref(nullptr) {
		//std::cout << "Passing through!" << std::endl;
		if (awrp.m_val != nullptr) {
			
			//int* i = (int*) awrp.m_val;
			
			//std::cout << *i << std::endl;
			
			//std::cout << "Pointer: " << awrp.m_val << std::endl;
			
			this->m_val = new T(*(awrp.m_val));
			
			this->m_ref = this->m_val;
			this->init = true;
		} else if (awrp.m_ref != nullptr) {
			//std::cout << "Doing that..:" << std::endl;
			this->m_ref = &*awrp.m_ref;
			this->init = true;
		} 
		
		//std::cout << "Done in copy..." << std::endl;
		
	}
	
	// move constructor
	attribute_wrapper(attribute_wrapper<T,a0,a1>&& awrp) : init(false), m_val(nullptr), m_ref(nullptr) {

		if (awrp.m_val != nullptr) {
			
			this->m_val = awrp.m_val;
			awrp.m_val = nullptr;
			
			this->m_ref = this->m_val;
			awrp.m_ref = nullptr;
			
			this->init = true;
			
		} else if (awrp.m_ref != nullptr) {
			
			this->m_ref = awrp.m_ref;
			awrp.m_ref = nullptr;
			this->init = true;
			
		} 
		
	}
	
	// convert a value type to a reference type. use with caution
	template <typename D = T, attribute0 A0 = a0, attribute1 A1 = a1, 
		typename = typename std::enable_if<(A1 != value_), int>::type>
	[[deprecated("Converting attribute value to attribute reference.")]] 
	attribute_wrapper(attribute_wrapper<T,a0,value_>& awrp) : init(false), m_val(nullptr), m_ref(nullptr) {
		//std::cout << "D3" << std::endl;
		
		if (awrp.m_val != nullptr) {
			this->m_val = new T(*awrp.m_val);
			this->m_ref = this->m_val;
			this->init = true;
		} else if (awrp.m_ref != nullptr) {
			this->m_ref = &*awrp.m_ref;
			this->init = true;
		} 
		//std::cout << "Done" << std::endl;
	}
	
	// a nice constructor to allow initializer lists
	template <typename D = T, attribute0 A0 = a0, attribute1 A1 = a1, 
		typename = typename std::enable_if<(A1 == value_) && (is_container<D>::value), int>::type>
	attribute_wrapper(std::initializer_list<typename D::value_type> r) 
		: m_val(new T(r)), m_ref(m_val), init(true) {/* *m_val = r; */}

	template <class D = T, attribute0 A0 = a0, attribute1 A1 = a1, 
		typename = typename std::enable_if<A1 == reference_ && !is_attribute_wrapper<D>::value, int>::type>
	attribute_wrapper(D& r) : m_ref(&r), init(true) { /*std::cout << "D4" << std::endl;*/ }
	
	attribute_wrapper& operator=(const attribute_wrapper<T,a0,a1>& in) {
		
		//std::cout << "making copy...:" << std::endl;
		
		if (this != &in) {
			
			if (m_val) delete m_val;
		
			if (in.m_val != nullptr) {
				this->m_val = new T(*in.m_val);
				this->m_ref = this->m_val;
				this->init = true;
				return *this;
			} else if (in.m_ref != nullptr) {
				this->m_ref = &*in.m_ref;
				this->init = true;
				return *this;
			} 
			
		}
		
		return *this;
		
		//std::cout << "Done copying" << std::endl; 
	}
	
	attribute_wrapper& operator=(attribute_wrapper<T,a0,a1>&& in) {
		
		if (m_val) delete m_val;
		
		//std::cout << "Also doing this..." << std::endl;
		
		if (in.m_val != nullptr) {
			
			//std::cout << "Going in here..." << std::endl;
			
			this->m_val = in.m_val;
			this->m_ref = m_val;
			this->init = true;
			
			in.m_val = nullptr;
			
		} else if (in.m_ref != nullptr) {
			this->m_ref = in.m_ref;
			this->init = true;
		}
		
		return *this;
		
	}
	
	T& operator*() const {
		//assert(("Optional parameter not initialized.", init == true));
		return *m_ref;
	}
	
	T* operator->() const {
		 //assert(("Optional parameter not initialized.", init == true));
		 return m_ref;
	}
	
	operator bool () const {
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
