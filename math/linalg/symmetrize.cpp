#include "math/linalg/symmetrize.h"

namespace math {
	
dbcsr::tensor<2> symmetrize(dbcsr::tensor<2>& t_in, std::string name_in) {
	
	std::string name = /*(name_in == "") ? t_in.name() :*/ name_in;
	
	dbcsr::pgrid<2> grid({.comm = t_in.comm()});
	
	dbcsr::tensor<2> t_sym({.name = name, .pgridN = grid, .map1 = {0}, .map2 = {1}, .blk_sizes = t_in.blk_size()});

	dbcsr::copy<2>({.t_in = t_in, .t_out = t_sym, .order = {1,0}});
	
	dbcsr::copy<2>({.t_in = t_in, .t_out = t_sym, .sum = true});
	
	t_sym.scale(0.5);
	
	//std::cout << "SYM: " << std::endl;
	//dbcsr::print(t_sym);
	
	return t_sym;
}

	
} // end namespace
