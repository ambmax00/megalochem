#ifndef LAPLACE_MINIMAX_C
#define LAPLACE_MINIMAX_C

extern "C" {
	
	void c_laplace_minimax(double* errmax, double* xpnts, double* wghts,
							int* nlap, double ymin, double ymax,
                            int* mxiter, int* iprint, double* stepmx,
                            double* tolrng, double* tolpar, double* tolerr,
                            double* delta, double* afact, bool* do_rmsd,
                            bool* do_init, bool* do_nlap);
                            
}

#endif
