#include "ints/gentran.h"

namespace ints {

dbcsr::stensor<3> transform3(tranp3&& p) {
	
	// frorm intermediary tensor
	//auto& in = *p.t_in;
	//auto& out = *p.t_out;
	//auto& c1 = *p.c_1;
	//auto& c2 = *p.c_2;
	std::cout << "B1" << std::endl;
	
	auto insizes = p.t_in->blk_size();
	std::cout << "B2" << std::endl;
	auto c1sizes = p.c_1->blk_size();
	std::cout << "B3" << std::endl;
	auto c2sizes = p.c_2->blk_size();
	std::cout << "B4" << std::endl;
	
	dbcsr::pgrid<3> grid3({.comm = p.t_in->comm()});
	
	dbcsr::tensor<3> HTI({.name = "HT", .pgridN = grid3,
		.map1 = {0,2}, .map2 = {1}, .blk_sizes = {insizes[0],c1sizes[1],insizes[2]}});
	
	// contract 1
	dbcsr::einsum<3,2,3>({.x = "XMN, Mi -> XiN", .t1 = *p.t_in, .t2 = *p.c_1, .t3 = HTI});
	
	dbcsr::stensor<3> out = dbcsr::make_stensor<3>({.name = *p.name, .pgridN = grid3,
		.map1 = {0}, .map2 = {1,2}, .blk_sizes = {insizes[0],c1sizes[1],c2sizes[1]}});
	
	// contract 2
	dbcsr::einsum<3,2,3>({.x = "XiN, Nj -> Xij", .t1 = HTI, .t2 = *p.c_2, .t3 = *out});
	
	HTI.destroy();
	
	dbcsr::print(*out);
	
	return out;
	
}
		
}
