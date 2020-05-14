#ifndef ADC_ADC_OPS_H
#define ADC_ADC_OPS_H

namespace adc {

// do t_iajb /= eo(i) + eo(j) - ev(a) - ev(b)
template <typename T>
void scale(dbcsr::tensor<4,T>& t, vec<T>& eo, vec<T>& ev, T co = -1.0, T cv = 1.0) {
	
	// scale it
#pragma omp parallel 
{
	
	dbcsr::iterator_t<4,T> iter4(t);
	iter4.start();
		
	while (iter4.blocks_left()) {
		
		iter4.next();
	
		auto& idx = iter4.idx();
		auto& off = iter4.offset();
		auto& blksize = iter4.size();
		
		bool found = false;
		
		auto blk = t.get_block(idx, blksize, found);
	
		if (!found) continue;
		
		int offi = off[0];
		int offa = off[1];
		int offj = off[2];
		int offb = off[3];
		
		std::cout << offi << " " << offa << " " << offj << " " << offb << std::endl;
		
		for (int i = 0; i != blksize[0]; ++i) {
		 T epsi = co * eo[offi+i];
		 for (int a = 0; a != blksize[1]; ++a) {
		  T epsa = cv * ev[offa+a];
		  for (int j = 0; j != blksize[2]; ++j) {
		   T epsj = co * eo[offj+j];
		   for (int b = 0; b != blksize[3]; ++b) {
			   T epsb = cv * ev[offb+b];
			   
			   blk(i,a,j,b) /= epsi + epsj + epsa + epsb;
			   
		}}}}
		
		t.put_block(idx, blk);
		
	}
	
	t.finalize();
	iter4.stop();
	
} // end omp
	
}

// t1(ikjl) = alpha * t1(ikjl) - t2(iljk)

template <typename T>
void antisym(dbcsr::tensor<4,T>& t, T alpha) {
	
	MPI_Comm comm = t.comm();
	
	dbcsr::tensor<4,T> ttran 
		= typename dbcsr::tensor<4,T>::create_template()
		.tensor_in(t).name("TRAN");
	dbcsr::copy(t, ttran).perform();
	
	auto off = t.blk_offsets();
	auto blksizes = t.blk_sizes();
	
	auto& s1 = blksizes[0];
	auto& s2 = blksizes[1];
	auto& s3 = blksizes[2];
	auto& s4 = blksizes[3];

	int myrank = -1;
	MPI_Comm_rank(comm,&myrank);
	
	for (int IX1 = 0; IX1 != s1.size(); ++IX1) {
	 int b1 = s1[IX1];
	 
	 for (int IX2 = 0; IX2 != s2.size(); ++IX2) {
		int b2 = s2[IX2];
		
		for (int IX3 = 0; IX3 != s3.size(); ++IX3) {
		int b3 = s3[IX3]; 
		
		for (int IX4 = 0; IX4 != s4.size(); ++IX4) {
			int b4 = s4[IX4];
			
			//std::cout << "BLOCK: " << I << " " << A << " " << J << " " << B << std::endl;
			auto off1 = off[0][IX1];
			auto off2 = off[1][IX2];
			auto off3 = off[2][IX3];
			auto off4 = off[3][IX4];
		
			dbcsr::idx4 idx_1 = {IX1,IX2,IX3,IX4};
			dbcsr::idx4 idx_2 = {IX1,IX4,IX3,IX2};
			
			int proc1 = -1;
			int proc2 = -1;
			
			proc1 = t.proc(idx_1);
			proc2 = ttran.proc(idx_2);
			
			bool found1(false), found2(false);

			arr<int,4> blk1size = {b1,b2,b3,b4};
			arr<int,4> blk2size = {b1,b4,b3,b2};
			
			dbcsr::block<4,T> blk1(blk1size), blk2(blk2size);
			
			if (myrank == proc1) 
				blk1 = t.get_block(idx_1, blk1size, found1);
			
			if (myrank == proc2) 
				blk2 = ttran.get_block(idx_2, blk2size, found2);
			//} else {
			//	found2 = found1;
			//}
			
			MPI_Bcast(&found1,1,MPI_C_BOOL,proc1,comm);
			MPI_Bcast(&found2,1,MPI_C_BOOL,proc2,comm);
			
			if (found1 == false && found2 == false) continue;
			//if (found1 == false) continue;
			
			if (proc1 != proc2 && (myrank == proc1 || myrank == proc2)) {
				// send it from 2 to 1
				MPI_Sendrecv(blk2.data(),blk2.ntot(),MPI_DOUBLE,proc1,1,blk2.data(),blk2.ntot(),
					MPI_DOUBLE,proc2,1,comm,MPI_STATUS_IGNORE);
					
			}
			
			if (myrank == proc1) {
			
				for (int i1 = 0; i1 != b1; ++i1) {
					for (int i2 = 0; i2 != b2; ++i2) {
						for (int i3 = 0; i3 != b3; ++i3) {
							for (int i4 = 0; i4 != b4; ++i4) {
								std::cout << i1+off1 << i2+off2 << i3+off3 << i4+off4 << " " << blk1(i1,i2,i3,i4);
								blk1(i1,i2,i3,i4) = alpha*blk1(i1,i2,i3,i4) - blk2(i1,i4,i3,i2);
								std::cout << " " << blk2(i1,i4,i3,i2) << std::endl;
							}
						}
					}
				}
				
				if (found1 == false) {
					t.reserve(arrvec<int,4>{
						vec<int>{IX1},vec<int>{IX2},
						vec<int>{IX3},vec<int>{IX4}});
				}
				
				t.put_block(idx_1, blk1);
				
			}
			
			MPI_Barrier(comm);
			
	   } // end loop B
	  } // end loop J
	 } // end loop A
	} // end loop I
	
}



} // end namespace

#endif
