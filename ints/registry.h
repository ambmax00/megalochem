#ifndef INTS_REGISTRY_H
#define INTS_REGISTRY_H

#include <any>
#include <map>
#include <dbcsr_tensor_ops.hpp>

namespace ints {
	
class registry {
private:

	inline static std::map<std::string,std::any> m_map;
	
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
	
	template <typename T>
	dbcsr::smatrix<T> get_matrix(std::string intname) {
		dbcsr::smatrix<T> out;
		if (m_map.find(intname) != m_map.end()) {
			if (typeid(out).name() != m_map[intname].type().name()) {
				throw std::runtime_error("Registry: Bas cast.");
			}
			out = std::any_cast<dbcsr::smatrix<T>>(m_map[intname]);
		}
		return out;
	}
	
	template <int N, typename T>
	dbcsr::stensor<N,T> get_tensor(std::string intname, bool reorder = false, bool move = false) {
			
		dbcsr::stensor<N,T> out;
		if (m_map.find(intname) != m_map.end()) {
			if (typeid(out).name() != m_map[intname].type().name()) {
				throw std::runtime_error("Registry: bad cast");
			}
			out = std::any_cast<dbcsr::stensor<N>>(m_map[intname]);
			return out;
		}
		
		if (reorder) {
			
			std::cout << "Reordering." << std::endl;
			
			std::vector<int> map1, map2;
			int iread = 0;
			
			for (std::string::reverse_iterator riter = intname.rbegin();
				riter != intname.rend(); ++riter) {
					
				if (*riter == ')') {
					iread = 1;
				} else if (*riter == '|') {
					iread = 2;
				} else if (*riter == '(') {
					iread = 0;
				} else if (iread == 1) {
					std::cout << "P1" << std::endl;
					map2.insert(map2.begin(),*riter - '0');
				} else if (iread == 2) {
					std::cout << "P2" << std::endl;
					map1.insert(map1.begin(),*riter - '0');
				}
				
			}
			
			std::string root(intname.begin(),intname.begin()+intname.size() - 3 - N);
			
			std::cout << "ROOT " << root << std::endl;
 			
			auto iter = std::find_if(m_map.begin(),m_map.end(),
				[root](std::pair<std::string,std::any> p) {
					if (p.first.find(root) != std::string::npos) {
						return true;
					}
					return false;
			});
			
			if (iter == m_map.end()) return out;
			
			std::cout << "MAPS:" << std::endl;
			for (auto i : map1) { std::cout << i << " ";}
			for (auto i : map2) { std::cout << i << " ";}
			
			auto data = this->get_tensor<N,T>(iter->first);
			std::string newname = root + this->map_to_string(map1,map2);
			
			out = dbcsr::make_stensor<N,T>(typename dbcsr::tensor<N,T>::create_template(*data)
				.name(newname).map1(map1).map2(map2));
			
			dbcsr::copy(*data,*out).move_data(move).perform();
			
			this->insert_tensor<N,T>(newname,out);
			if (move) m_map.erase(iter);
			
		}
		
		return out;
		
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
