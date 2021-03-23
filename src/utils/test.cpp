#include "../utils/ppdirs.hpp"
#include <vector>

int main() {

	util::optional<int&> t;

	util::optional<std::vector<std::vector<int>>> p0;
	typename util::builder_type<util::optional<std::vector<std::vector<int>>>>::type p1;


	
	std::cout << typeid(p0).name() << std::endl;
	std::cout << typeid(p1).name() << std::endl;
}
