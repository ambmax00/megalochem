#include "math/other/scale.h"

namespace math {

void scale(dbcsr::tensor<2>& t_in, std::vector<double>& v_in, std::optional<std::vector<int>> bounds) {
	
	auto& t = t_in;
	auto& v = v_in;
	
	std::vector<int> b(0);
	
	if (bounds) {
		if (bounds->size() != 2) 
			throw std::runtime_error("Scale: bounds needs to be size 2 (at the moment).");
		b = *bounds;
	}
	
	auto dims = t.nfull_total();
	
	int nbound = dims[1];
	
	if (bounds) { nbound = b[1] - b[0] + 1; } 
	
	if (nbound > dims[1]) throw std::runtime_error("Scale: bounds too large.");
	if (nbound != v.size()) throw std::runtime_error("Scale: incompatible dimensions.");
	
	// make bounding functions
	std::function<bool(int i,vec<int>& b)> inbound;
	
	if (bounds) {
		inbound = [](int i,vec<int>& b) {
			if ((i >= b[0]) && (i <= b[1])) return true;
			return false;
		};
	} else {
		inbound = [](int i,vec<int>& b) { return true; };
	}
		
	
	//dbcsr::print(t);
	
	#pragma omp parallel 
	{
		dbcsr::iterator<2> iter(t);
		iter.start();
	
		while (iter.blocks_left()) {
			
			iter.next();
			
			auto& idx = iter.idx();
			auto& blksize = iter.size();
			auto& blkoff = iter.offset();
			
			bool found = true;
			
			int off1 = blkoff[0];
			int off2 = blkoff[1];
			
			if (!inbound(off2,b)) continue;
			
			auto blk = t.get_block(idx, blksize, found);
			
			// loop over columns
			for (int j = 0; j != blksize[1]; ++j) {
				if (!inbound(off2+j,b)) continue;
				int ncol = off2 + j;
				for (int i = 0; i != blksize[0]; ++i) {
					blk(i,j) *= v[ncol];
				}
			}
			
			t.put_block(idx, blk);
			
		}
		
		t.finalize();
		iter.stop();
		
	}
	//dbcsr::print(t);

}	
	
} // end namspace
