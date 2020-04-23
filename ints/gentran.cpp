#include "ints/gentran.h"

namespace ints {

dbcsr::stensor<3> transform3(dbcsr::stensor<3>& d_xab, dbcsr::stensor<2>& ca, dbcsr::stensor<2>& cb, std::string name) {
	
	auto insizes = d_xab->blk_sizes();

	auto casizes = ca->blk_sizes();
	
	auto cbsizes = cb->blk_sizes();
	
	
	dbcsr::pgrid<3> grid3(d_xab->comm());
	arrvec<int,3> blksizes_hti = {insizes[0],casizes[1],insizes[2]};
	
	dbcsr::tensor<3> HTI = dbcsr::tensor<3>::create().name("HT").ngrid(grid3)
		.map1({0,2}).map2({1}).blk_sizes(blksizes_hti);
	
	// contract 1
	//dbcsr::einsum<3,2,3>({.x = "XMN, Mi -> XiN", .t1 = *p.t_in, .t2 = *p.c_1, .t3 = HTI});
	dbcsr::contract(*d_xab, *ca, HTI).perform("XMN, Mi -> XiN");
	
	arrvec<int,3> blksizes_full = {insizes[0],casizes[1],cbsizes[2]};
	dbcsr::stensor<3> out = dbcsr::make_stensor<3>(
		dbcsr::tensor<3>::create().name(name).ngrid(grid3)
		.map1({0}).map2({1,2}).blk_sizes(blksizes_full));
	
	// contract 2
	//dbcsr::einsum<3,2,3>({.x = "XiN, Nj -> Xij", .t1 = HTI, .t2 = *p.c_2, .t3 = *out});
	dbcsr::contract(HTI, *cb, *out).perform("XiN, Nj -> Xij");
	
	HTI.destroy();
	
	dbcsr::print(*out);
	
	return out;
	
}
		
}
