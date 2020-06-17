#ifndef DESC_OPTIONS_H
#define DESC_OPTIONS_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <any>
#include <type_traits>

// a fancy "any" map
#include <iostream>

namespace desc {

template <typename T, typename U = void>
struct is_valid : std::false_type {};

template <typename T>
struct is_valid<T,typename std::enable_if<(
	std::is_same<T,int>::value 		||
	std::is_same<T,double>::value 	|| 
	std::is_same<T,bool>::value 	||
	//std::is_same<T,std::vector<typename T::value_type>>::value ||
	std::is_same<T,std::string>::value
)>::type> : std::true_type {};
	
class options  {
private:

	std::map<std::string, std::any> m_map;
	std::string m_prefix = "";
	
public:

	template <typename T>
	typename std::enable_if<is_valid<T>::value, void>::type
	set(std::string key, T in) {
		
		key = m_prefix + key;
		
		//if (m_map.find(key) != m_map.end()) throw std::runtime_error("Already in options: "+key);
		
		//std::cout << "SETTING: " << key << std::endl;
		
		std::any val = std::any(in);
		m_map[key] = val;
	}
	
	template <typename T>
	typename std::enable_if<is_valid<T>::value, T>::type
	get(std::string key, std::optional<T> opt = std::nullopt) {
		
		key = m_prefix + key;
		
		if (opt) {
			
			auto val = opt.value();
			if (m_map.find(key) == m_map.end()) {
				return val;
			}
			
		}
		
		if (m_map.find(key) == m_map.end())
			throw std::runtime_error("Could not find keyword: " + key);
		
		auto val = m_map[key];
		return std::any_cast<T>(val);
		
	}
	
	options() {}
	
	options(const options& opt) {
		m_map = opt.m_map;
		m_prefix = opt.m_prefix;
	}
	
	options& operator=(const options& opt) {
		
		if (this == &opt) return *this;
		
		m_map = opt.m_map;
		m_prefix = opt.m_prefix;
		return *this;
	}
		
	~options() {}
	
	options subtext(std::string root) {
		
		options out;
		out.m_map = this->m_map;
		out.m_prefix = m_prefix + root + "/";
		
		return out;
		
	}
		
	bool present(std::string name) {
		
		if (m_map.find(m_prefix + name) != m_map.end()) return true;
		return false;
		
	}
	
};

}

#endif
