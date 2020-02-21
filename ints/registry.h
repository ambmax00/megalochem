#ifndef INTS_REGISTRY_H
#define INTS_REGISTRY_H

#include <any>
#include <map>

namespace ints {
	
struct int_params {
	std::string molname; // e.g. "mol1"
	std::string intname;
	
	bool operator==(const int_params p) const {
		return molname == p.molname && intname == p.intname;
	}
	
	bool operator<(const int_params p) const {
		return molname < p.molname || (molname == p.molname && intname < p.intname);
	}

};

class registry {
private:

	inline static std::map<int_params,std::any> m_map;
	
public:

	registry() {}
	
	~registry() {}
	
	template <int N>
	void insert(std::string molname, std::string intname, dbcsr::stensor<N>& t) {
		std::any in = std::any(t);
		int_params p{.molname = molname, .intname = intname};
		m_map[p] = in;
	};
	
	template <int N>
	dbcsr::stensor<N> get(std::string molname, std::string intname) {
		dbcsr::stensor<N> out;
		int_params p{.molname = molname, .intname = intname};
		if (m_map.find(p) != m_map.end()) 
			out = std::any_cast<dbcsr::stensor<N>>(m_map[p]);
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

} // end namespace

#endif
