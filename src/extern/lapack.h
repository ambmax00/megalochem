#ifndef EXTERN_LAPACK_H
#define EXTERN_LAPACK_H

extern "C" {
	
	void dgesvd_(char* jobu, char* jobv, int* m, int* n, double* a,
		int* lda, double* s, double* u, int* ldu, double* vt, int* ldv, 
		double* work, int* lwork, int* info);
		
}
		
inline void c_dgesvd(char jobu, char jobv, int m, int n, double* a, 
	int lda, double* s, double* u, int ldu, double* vt, int ldv,
	double* work, int lwork, int* info) 
{
	dgesvd_(&jobu,&jobv,&m,&n,a,&lda,s,u,&ldu,vt,&ldv,work,&lwork,info);
}

#endif
