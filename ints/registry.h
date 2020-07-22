#ifndef INTS_REGISTRY_H
#define INTS_REGISTRY_H

#include <any>
#include <map>
#include <dbcsr_tensor_ops.hpp>
#include <dbcsr_btensor.hpp>
#include "ints/screening.h"

namespace ints {
	
class registry {
private:

	inline static std::map<std::string,std::any> m_map;
	
	template <typename item>
	item get_item(std::string intname) {
		item out = nullptr;
		if (m_map.find(intname) != m_map.end()) {
			if (typeid(out).name() != m_map[intname].type().name()) {
				throw std::runtime_error("Registry: Bad cast.");
			}
			out = std::any_cast<item>(m_map[intname]);
		}
		return out;
	}
	
public:

	registry() {}
	
	~registry() {}
	
	std::string map_to_string(vec<int>& map1, vec<int>& map2) {
		std::string out = "(";
		
		for (auto m : map1) { out += std::to_string(m); }
		out += "|";
		for (auto m : map2) { out += std::to_string(m); }
		out += ")";
		
		return out;
		
	}
	
	template <typename T>
	void insert_matrix(std::string name, dbcsr::smatrix<T>& m) {
		std::any in = std::any(m);
		m_map[name] = in;
	}
	
	template <int N, typename T>
	void insert_tensor(std::string name, dbcsr::stensor<N,T>& t) {
		std::any in = std::any(t);
		m_map[name] = in;
	};
	
	template <int N, typename T>
	void insert_btensor(std::string name, dbcsr::sbtensor<N,T>& t) {
		std::any in = std::any(t);
		m_map[name] = in;
	}; 
	
	void insert_screener(std::string name, shared_screener scr) {
		std::any in = std::any(scr);
		m_map[name] = in;
	}; 
	
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
	
	shared_screener get_screener(std::string name) {
		return get_item<shared_screener>(name);
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
		
		
};

} // end namespace

#endif
