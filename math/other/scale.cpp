#include "math/other/scale.h"

namespace math {

void scale(dbcsr::tensor<2>& t, std::vector<double>& v) {
	
	dbcsr::iterator<2> iter(t);
	
	auto blkoff = t.blk_offset();
	auto blksizes = t.blk_size();
	auto dims = t.nfull_tot();
	
	std::cout << "ij: " << dims[0] << " " << dims[1] << std::endl;
	std::cout << "vec: " << v.size() << std::endl;
	
	for (auto x : v) {
		std::cout << x << " ";
	} std::cout << std::endl;
	
	if (dims[1] != v.size()) throw std::runtime_error("Scale: incompatible dimensions.");
	
	dbcsr::print(t);
	
	while (iter.blocks_left()) {
		
		iter.next();
		
		auto idx = iter.idx();
		auto blksize = iter.sizes();
		bool found = true;
		
		int off1 = blkoff[0][idx[0]];
		int off2 = blkoff[1][idx[1]];
		
		auto blk = t.get_block({.idx = idx, .blk_size = blksize, .found = found});
		
		// loop over columns
		for (int j = 0; j != blksize[1]; ++j) {
			int ncol = off2 + j;
			for (int i = 0; i != blksize[0]; ++i) {
				blk(i,j) *= v[ncol];
			}
		}
		
		t.put_block({.idx = idx, .blk = blk});
		
	}
	
	dbcsr::print(t);

}	
	
} // end namspace
