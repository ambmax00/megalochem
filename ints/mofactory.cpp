#include "ints/mofactory.h"

namespace ints {

/*
template <int N>
dbcsr::stensor<N> mofactory::compute(mofac_params&& p) {
	
	// get the integrals
	
	std::string aobas;
	switch(N) {
		case 2 : aobas = ""; //"bb";
		case 3 : aobas = "xbb";
		case 4 : aobas = ""; //"bbbb";
	}
	
	if (aobas == "") throw std::runtime_error("Not yet implemented.");
	
	auto aotensor = m_aofac.compute({.op = p.op, .bas = aobas, .name = std::string(*p.name + "AO"), .map1 = p.map1,
			.map2 = p.map2});
	
	auto metric = m_aofac.compute({.op = p.op, .bas = "xx", .name = std::string(*p.name + "AO"), .map1 = p.map1,
			.map2 = p.map2});
	
	if (!p.c1 || !p.c2) throw std::runtime_error("Mofactory: give both transformation matrices.");
	
	
	auto aoblksize = aotensor.blk_size();
	auto& c1 = *p.c1;
	auto c1blksize = c1.blk_size();
		
	// c1(mu|i) * (X|mu,nu) -> (X|i,nu)
	vec<vec<int>> blksizes = {aoblksize[0],c1blksize[1],aoblksize[2]};
	dbcsr::pgrid<N> grid{.comm = aotensor.comm()};

	auto trans_1 = std::make_stensor<N>({.name = "T1", .pgridN = grid, .map1 = {0}, .map2 = {1,2}, .blk_sizes = blksizes});
		
	dbcsr::contract<2,N,N>({.x = "Mi, XMN -> XiN", .t1 = c1, .t2 = *aotensor, .t3 = *trans_1});
		
	auto& c2 = *p.c2;
	auto c2blksize = c2.blk_size();
	vec<vec<int>> fullblksizes = {aoblksize[0],c1blksize[1],c2blksize[1]};
	
	auto trans_2 = std::make_stensor<N>({.name = *p.name, .pgridN = grid, .map1 = {0}, .map2 = {1,2}, .blk_sizes = blksizes});
	
	dbcsr::contract<2,N,N>({.x = "Nj, XiN -> Xij", .t1 = c2, .t2 = *trans_1, .t3 = *trans_2});
	
	trans_1->destroy();
	grid.destroy();
	
	return trans_2;
	
}
*/

dbcsr::stensor<3> mofactory::transform(std::string name, dbcsr::stensor<3>& t_ints, dbcsr::stensor<2>& c1, dbcsr::stensor<2>& c2) {
	
	auto aoblksize = t_ints->blk_size();
	auto c1blksize = c1->blk_size();
		
	// c1(mu|i) * (X|mu,nu) -> (X|i,nu)
	vec<vec<int>> blksizes = {aoblksize[0],c1blksize[1],aoblksize[2]};
	dbcsr::pgrid<N> grid{.comm = aotensor.comm()};

	auto trans_1 = std::make_stensor<N>({.name = "T1", .pgridN = grid, .map1 = {0}, .map2 = {1,2}, .blk_sizes = blksizes});
		
	dbcsr::contract<2,N,N>({.x = "Mi, XMN -> XiN", .t1 = *c1, .t2 = *t_ints, .t3 = *trans_1});
		
	auto c2blksize = c2->blk_size();
	vec<vec<int>> fullblksizes = {aoblksize[0],c1blksize[1],c2blksize[1]};
	
	auto trans_2 = std::make_stensor<N>({.name = name, .pgridN = grid, .map1 = {0}, .map2 = {1,2}, .blk_sizes = fullblksizes});
	
	dbcsr::contract<2,N,N>({.x = "Nj, XiN -> Xij", .t1 = *c2, .t2 = *trans_1, .t3 = *trans_2});
	
	trans_1->destroy();
	grid.destroy();
	
	return trans_2;
	
}

} // namespace 
