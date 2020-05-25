 #include "math/linalg/piv_cd.h"
 #include <cmath>
 #include <limits>
 #include <algorithm>
 
#include "extern/scalapack.h"

namespace math {
 
void piv_cholesky_decomposition::compute(int nb) {
	
	// convert input mat to scalapack format
	
	int N = m_mat_in->nfullrows_total();
	double thresh = N * std::numeric_limits<double>::epsilon();
	int rank = 0;
	int iter = 0;
	int* iwork = new int[N];
	
	MPI_Comm comm = m_mat_in->get_world().comm();
	int myrank = m_mat_in->get_world().rank();
	int myprow = scalapack::global_grid.myprow();
	int mypcol = scalapack::global_grid.mypcol();
	
	int ori_proc = m_mat_in->proc(0,0);
	int ori_coord[2];
	
	if (myrank == ori_proc) {
		ori_coord[0] = scalapack::global_grid.myprow();
		ori_coord[1] = scalapack::global_grid.mypcol();
	}
	
	MPI_Bcast(&ori_coord[0],2,MPI_INT,ori_proc,comm);
		
	scalapack::distmat<double> U = dbcsr::matrix_to_scalapack(*m_mat_in, 
		m_mat_in->name() + "_scalapack", nb, nb, ori_coord[0], ori_coord[1]);
		
	U.print();
	
	// permutation vector
	scalapack::distmat<int> Pcol(N, 1, nb, nb, 0, 0);
	scalapack::distmat<int> Prow(1, N, nb, nb, 0, 0);
	
	std::vector<int> perms(N);
	for (size_t i = 0; i != N; ++i) { perms[i] = i+1; }
	// !!! WE ARE USING 1-INDEXING FOR PERMUTATIONS !!!
	
	
	// chol mat
	scalapack::distmat<double> L(N,N,nb,nb,0,0);
	
	// u_i
	scalapack::distmat<double> u_i(N,1,nb,nb,0,0);
	
	// get max diag element
	double max_U_diag_global = 0.0;
	for (int ix = 0; ix != N; ++ix) {
		
		max_U_diag_global = std::max(
			fabs(max_U_diag_global),
			fabs(U.get('A', ' ', ix, ix)));
			
	}
		
	std::cout << "MAX GLOBAL DIAG: " << max_U_diag_global << std::endl;	
	
	for (int I = 0; I != N-1; ++I) {
		
		// find index of maximum diagonal element
		
		std::cout << "LOOP: " << I << std::endl;
		
		int max_U_dd_i = 0;
		double max_U_dd = 0;
		
		for (int ix = I; ix != N; ++ix) {
	
			double ele_ix = U.get('A', ' ', ix, ix);
			
			if (fabs(max_U_dd) < fabs(ele_ix)) {
				max_U_dd = ele_ix;
				max_U_dd_i = ix;
			}
			
		}
		
		if (myrank == 0) std::cout << "MAX: " << max_U_dd << " @ " << max_U_dd_i << std::endl;
		
		for (int ix = 0; ix != N; ++ix) {
			Pcol.set(ix,0,ix+1);
			Prow.set(0,ix,ix+1);
		}
		
		Pcol.set(I,0,max_U_dd_i + 1);
		Prow.set(0,I,max_U_dd_i + 1);
		
		Pcol.print();
		
		auto desca = U.desc();
		
		c_pdlapiv('F', 'C', 'C', N, N, U.data(), 0, 0, &desca[0], Pcol.data(), 0, 0, Pcol.desc().data(), iwork);
		c_pdlapiv('F', 'R', 'R', N, N, U.data(), 0, 0, &desca[0], Prow.data(), 0, 0, Prow.desc().data(), iwork);
		
		U.print();
		
		// get first diagonal element of permuted matrix U_ii
		
		double U_II = U.get('A', ' ', I, I);
		
		if (myrank == 0) std::cout << "MAXDIAG: " << U_II << std::endl;
		
		if (U_II < 0.0) {
			throw std::runtime_error("Negative pivot element, cholesky decomposition not possible!");
		}
		
		std::cout << "THRESH: " << thresh * max_U_diag_global << std::endl;
		
		if (fabs(U_II) < fabs(thresh * max_U_diag_global)) {
			rank = I;
			std::cout << "FINISHED AT RANK :" << rank << std::endl;
			break;
		}
		
		std::swap(perms[I], perms[max_U_dd_i]);
		
		// // get rest of considered column, u
		
		c_pdgeadd('N', N-I-1, 1, 1.0, U.data(), I+1, I, U.desc().data(), 0.0, u_i.data(), I+1, 0, u_i.desc().data());
		
		u_i.print(); 
		
		// FORM Lpr
		L.set(I,I,sqrt(fabs(U_II)));
		
		c_pdgemm('N', 'T', N-I-1, N-I-1, 1, -1/U_II, u_i.data(), I+1, 0, u_i.desc().data(),
			u_i.data(), I+1, 0, u_i.desc().data(), 1.0, U.data(), I+1, I+1, U.desc().data());
			
		U.print();
		
		// copy u_i/U_II into L
		
		c_pdgeadd('N', N-I-1, 1, 1/sqrt(fabs(U_II)), u_i.data(), I+1, 0, u_i.desc().data(), 0.0, 
			L.data(), I+1, I, L.desc().data());
		
		std::cout << "L: " << std::endl;
		L.print();
		
		++iter;
		
	}
	
	if (iter == N-1) {
		
		double Unn = U.get('A', ' ', N-1, N-1);
		
		if (fabs(Unn) < max_U_diag_global * thresh) {
			rank = N - 1;
		} else {
			L.set(N-1,N-1,sqrt(Unn));
			rank = N;
		}
		
	}
	
	std::cout << "NUMERICAL RANK: " << rank << std::endl;
	
	// REORDER BACK
	
	for (auto x : perms) {
		std::cout << x << " ";
	} std::cout << std::endl;
	
	for (int i = 0; i != N; ++i) {
		Prow.set(0,i,perms[i]);
		Pcol.set(i,0,perms[i]);
	}
	
	Prow.print();
	
	c_pdlapiv('F', 'R', 'R', N, N, L.data(), 0, 0, L.desc().data(), Prow.data(), 0, 0, Prow.desc().data(), iwork);
	c_pdlapiv('F', 'C', 'C', N, N, L.data(), 0, 0, L.desc().data(), Pcol.data(), 0, 0, Pcol.desc().data(), iwork);
	
	std::cout << "LAST: " << std::endl;
	L.print();
	
	delete [] iwork;
	
	c_pdgemm('N', 'T', N, N, N, 1.0, L.data(), 0, 0, L.desc().data(),
			L.data(), 0, 0, L.desc().data(), 0.0, U.data(), 0, 0, U.desc().data());
			
	U.print();
	
	exit(0);
	
}
	
}

// copy L into U
 
// loop form 0 to N-1

	// get index for maximum diagonal element imax
	
	// permute columns i and imax
	
	// get first diagonal element of permuted matrix for current iteration Uii
	
	// get rest of considered column, u (j i+1 -> N-1
 
	 // abortion criterion!
 
	// FORM Lpr
	//Lpr[i*N+i] = sqrt(fabs(Uii));

    // get index of largest diagonal element in U~ for permutation P~ on u,
    // actual permutation on U~ will be performed on next loop iteration

    //Lpr[i*N+i] = sqrt(Uii);
    

    // update U by forming U~ - uuT/Lii

    //for (j = i+1; j < N; j++){
    //  for (k = i+1; k < N; k++){

     //   U[j*N+k] -= u[j]*u[k]/Uii;
	
	/*
    maxUjjtilde = 0.e0;

    for (j = i+1; j < N; j++){

      if (fabs(U[j*N+j]) >= maxUjjtilde){

         maxUjjtilde = U[j*N+j];
         j_maxUjjtilde = j;

      }
    }
                                                                           
    for (j = i+1; j < N; j++){

      Lpr[j*N+i] = u[j]/Lpr[i*N+i];

    }
    */
    
    //DO SOMETHING!!!

    // permute lower left block of Lpr according to P~

    //permute_rows(Lpr,N,0,i+1,i+1,j_maxUjjtilde);
           
 // } // end decomposition loop

/*
  // in case the algorithm has not stopped during the previous loop, 
  // i.e. if the iteration counter has reached its maximum value of N-1, implying that the matrix 
  // to be decomposed has rank greater than N-1,
  // the last element of the Cholesky factor has to be computed this way
  // if (it_counter == N-1 && fabs(U[(N-1)*N+(N-1)]) >= thresh){ 
  if (it_counter == N-1){
     
     //if (fabs(U[(N-1)*N+(N-1)]) >= thresh){
     if (fabs(U[(N-1)*N+(N-1)]) >= maxUii*thresh){
        Lpr[(N-1)*N+(N-1)] = sqrt(U[(N-1)*N+(N-1)]);
        num_rank = N;
        it_counter++;
     } else {
        num_rank = N-1;
     }
     cout << "numerical rank: " << N << endl;
  }
            
  // permute Lprime back to L using record of original order of diagonal elements
  // (ATTENTION: this destroys L's triangular form!!)

  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){

      L[order[i]*N+j] = Lpr[i*N+j]; 

    }
  }
 
 */


