#include "math/linalg/symmetrize.h"

namespace math {
	
dbcsr::stensor<2> symmetrize(dbcsr::stensor<2>& t_in, std::string name_in) {
	
	std::string name = name_in;
	
	dbcsr::pgrid<2> grid(t_in->comm());
	
	dbcsr::tensor<2> t_sym = dbcsr::tensor<2>::create().name(name).ngrid(grid)
		.map1({0}).map2({1}).blk_sizes(t_in->blk_sizes());

	dbcsr::copy(*t_in, t_sym).order({1,0}).perform();
	
	dbcsr::copy(*t_in, t_sym).sum(true).perform();
	
	t_sym.scale(0.5);
	
	//std::cout << "SYM: " << std::endl;
	//dbcsr::print(t_sym);
	
	return t_sym.get_stensor();
}

	
} // end namespace
