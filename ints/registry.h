#ifndef INTS_REGISTRY_H
#define INTS_REGISTRY_H

#include <any>
#include <map>

namespace ints {
	
struct int_params {
	std::string molname; // e.g. "mol1"
	std::string bis; // e.g. "bb"
	std::string op; // e.g. "coulomb"
	int inv; // 0: not inverted 1: ^-1, 2: ^-1/2
};

bool operator<(int_params p1, int_params p2) {
	return p1.molname < p2.molname;
}

class registry {
private:

	std::map<int_params,std::any> m_map;
	
public:
	
	template <int N>
	void insert(int_params&& p, dbcsr::stensor<N>& t) {
		std::any in = std::any(t);
		m_map[p] = in;
	};
	
	template <int N>
	dbcsr::stensor<N> get(int_params&& p) {
		dbcsr::stensor<N> out;
		if (m_map.find(p) != m_map.end()) 
			return m_map[p];
		return out;
	}
	
	void clear() {
		m_map.clear();
	}
	
	// clear registry for a molecule
	void clear(std::string name) {
		for (auto& m : m_map) {
			if (m.first.molname == name) {
				m_map.erase(m.first);
			}
		}
	}
		
		
};

static registry INT_REGISTRY;

} // end namespace

#endif
