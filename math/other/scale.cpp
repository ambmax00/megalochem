#include "math/other/scale.h"

namespace math {

void scale(scale_params&& p) {
	
	auto& t = *p.t_in;
	auto& v = *p.v_in;
	
	std::vector<int> b(0);
	
	if (p.bounds) {
		if (p.bounds->size() != 2) 
			throw std::runtime_error("Scale: bounds needs to be size 2 (at the moment).");
		b = *p.bounds;
	}
	
	dbcsr::iterator<2> iter(t);
	
	auto blkoff = t.blk_offset();
	auto blksizes = t.blk_size();
	auto dims = t.nfull_tot();
	
	//std::cout << "ij: " << dims[0] << " " << dims[1] << std::endl;
	//std::cout << "vec: " << v.size() << std::endl;
	//std::cout << "bounds: " << b.size() << std::endl;
	
//	for (auto x : v) {
//		std::cout << x << " ";
//	} std::cout << std::endl;
	
	int nbound = dims[1];
	
	if (p.bounds) { nbound = b[1] - b[0] + 1; } 
	
	if (nbound > dims[1]) throw std::runtime_error("Scale: bounds too large.");
	if (nbound != v.size()) throw std::runtime_error("Scale: incompatible dimensions.");
	
	// make bounding functions
	std::function<bool(int i,vec<int>& b)> inbound;
	
	if (p.bounds) {
		inbound = [](int i,vec<int>& b) {
			if ((i >= b[0]) && (i <= b[1])) return true;
			return false;
		};
	} else {
		inbound = [](int i,vec<int>& b) { return true; };
	}
		
	
	//dbcsr::print(t);
	
	while (iter.blocks_left()) {
		
		iter.next();
		
		auto idx = iter.idx();
		auto blksize = iter.sizes();
		bool found = true;
		
		int off1 = blkoff[0][idx[0]];
		int off2 = blkoff[1][idx[1]];
		
		if (!inbound(off2,b)) continue;
		
		auto blk = t.get_block({.idx = idx, .blk_size = blksize, .found = found});
		
		// loop over columns
		for (int j = 0; j != blksize[1]; ++j) {
			if (!inbound(off2+j,b)) continue;
			int ncol = off2 + j;
			for (int i = 0; i != blksize[0]; ++i) {
				blk(i,j) *= v[ncol];
			}
		}
		
		t.put_block({.idx = idx, .blk = blk});
		
	}
	
	//dbcsr::print(t);

}	
	
} // end namspace
