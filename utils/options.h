#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <any>
#include <type_traits>

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

	std::map<std::string, std::any*> m_map;
	std::string m_prefix = "";
	
	options(options&) {};
	
public:

	template <typename T>
	typename std::enable_if<is_valid<T>::value, void>::type
	set(std::string key, T val) {
		
		key = m_prefix + key;
		
		std::any* valptr = new std::any(val);
		m_map[key] = valptr;
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
		
		auto valptr = m_map[key];
		return std::any_cast<T>(*valptr);
		
	}
	
	options() {}
	
	~options() {
		for (auto& ele : m_map) {
			delete ele.second;
		}
	}
	
	options subtext(std::string root) {
		
		options out;
		out.m_map = this->m_map;
		out.m_prefix = m_prefix + root + "/";
		
		return out;
		
	}
		
	
};

