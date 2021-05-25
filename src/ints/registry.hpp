#ifndef INTS_REGISTRY_H
#define INTS_REGISTRY_H

#include <any>
#include <map>
#include <dbcsr_tensor_ops.hpp>
#include <dbcsr_btensor.hpp>
#include "ints/screening.hpp"
#include <stdexcept>

namespace megalochem {

namespace ints {
	
class registry {
private:

	std::map<std::string,std::any> m_map;
	
	template <typename item>
	item get_item(std::string intname) {
		item out = nullptr;
		if (m_map.find(intname) != m_map.end()) {
			if (typeid(out).name() != m_map[intname].type().name()) {
				throw std::runtime_error("Registry: Bad cast.");
			}
			out = std::any_cast<item>(m_map[intname]);
		} else {
			throw std::runtime_error("Could not find " + intname + " in registry!");
		}
		return out;
	}
	
public:

	registry() {}
		
	template <typename T>
	void insert_matrix(std::string name, dbcsr::smatrix<T>& m) {
		std::any in = std::any(m);
		m_map[name] = in;
	}
	
	template <int N, typename T>
	void insert_tensor(std::string name, dbcsr::stensor<N,T>& t) {
		std::any in = std::any(t);
		m_map[name] = in;
	}
	
	template <int N, typename T>
	void insert_btensor(std::string name, dbcsr::sbtensor<N,T>& t) {
		std::any in = std::any(t);
		m_map[name] = in;
	}
	
	void insert_screener(std::string name, ints::shared_screener scr) {
		std::any in = std::any(scr);
		m_map[name] = in;
	}
	
	template <typename T>
	dbcsr::smatrix<T> get_matrix(std::string intname) {
		return get_item<dbcsr::smatrix<T>>(intname);
	}
	
	template <int N, typename T>
	dbcsr::stensor<N,T> get_tensor(std::string intname) {
		return get_item<dbcsr::stensor<N,T>>(intname);
	}
	
	template <int N, typename T>
	dbcsr::sbtensor<N,T> get_btensor(std::string intname) {
		return get_item<dbcsr::sbtensor<N,T>>(intname);
	}
	
	ints::shared_screener get_screener(std::string name) {
		return get_item<ints::shared_screener>(name);
	}
	
	void clear() {
		m_map.clear();
	}
	
	// clear registry for a molecule
	
	void clear(std::string name) {
		
		while (true) {
			
			auto iter = std::find_if(m_map.begin(),m_map.end(),
			[name](std::pair<std::string,std::any> p) {
				if (p.first.find(name) != std::string::npos) {
					return true;
				}
				return false;
			});
			
			if (iter == m_map.end()) break;
			
			m_map.erase(iter);
			
		}
			
	}
	
	void erase(std::string key) {
		m_map.erase(key);
	}
	
	~registry() {}
		
		
};

template <typename T> 
struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template <class KEY>
class key_registry {
private:

	std::array<std::any,static_cast<const int>(KEY::NUM_KEYS)> m_array;
	
	template <class KEY2>
	friend class key_registry;
	
public:

	key_registry() {}
		
	template <class M>
	typename std::enable_if<is_shared_ptr<M>::value == true, void>::type
	insert(KEY name, M& m) {
		int pos = static_cast<int>(name);
		std::any in = std::any(m);
		if (this->present(name)) {
			std::string msg = "Key nr. " + std::to_string(pos) 
				+ " of type " + typeid(name).name() + " already present!";
			throw std::runtime_error(msg);
		} else {
			m_array[pos] = in;
		}
	}
	
	template <class M>
	typename std::enable_if<is_shared_ptr<M>::value == true, M>::type
	get(KEY name) const {
		M out;
		int pos = static_cast<int>(name);
		if (this->present(name)) {
			if (typeid(out).name() != m_array[pos].type().name()) {
				auto badtype = std::string(typeid(out).name());
				auto goodtype = std::string(m_array[pos].type().name());
				throw std::runtime_error("Registry: Bad cast from " + badtype + " to " + goodtype);
			}
			out = std::any_cast<M>(m_array[pos]);
		} else {
			std::string msg = "Key nr. " + std::to_string(pos) 
				+ " of type " + typeid(name).name() + " not found!";
			throw std::runtime_error(msg);
		}
		return out;
	}
	
	void clear() {
		for (auto& a : m_array) a = nullptr;
	}
	
	void erase(KEY key) {
		int pos = static_cast<int>(key);
		m_array[pos].reset();
	}
	
	bool present(KEY key) const {
		int pos = static_cast<int>(key);
		return (m_array[pos].has_value()) ? true : false;
	}
	
	template <class REG, class KEY2>
	void add(REG registry_in, KEY2 old_key, KEY new_key) {
		int pos_old = static_cast<int>(old_key);
		int pos_new = static_cast<int>(new_key);
		m_array[pos_new] = registry_in.m_array[pos_old];
	}
	
	~key_registry() {}
		
};

} // namespace ints

} // namespace megalochem

#endif
