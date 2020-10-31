#ifndef EXTERN_LAPACK_H
#define EXTERN_LAPACK_H

extern "C" {
	
	void dgesvd_(char* jobu, char* jobv, int* m, int* n, double* a,
		int* lda, double* s, double* u, int* ldu, double* vt, int* ldv, 
		double* work, int* lwork, int* info);
	
	void dlatsqr_(int* m, int* n, int* mb, int* nb, double* a, int* lda,
		double* t, int* ldt, double* work, int* lwork, int* info);
		
	void dormqr_(char* side, char* trans, int* m, int* n, int* k, 
		double* a, int* lda, double* tau, double* c, int* ldc, 
		double* work, int* lwork, int* info);
	
	void dtrtrs_(char* uplo, char* trans, char* diag, int* n, int* nrhs,
		double* a, int* lda, double* b, int* ldb, int* info); 
	
	void dgeqr_(int* m, int* n, double* a, int* lda, double* t, 
		int* tsize, double* work, int* lwork, int* info);
	
	void dgels_(char* t, int* m, int* n, int* nrhs, double* a, int* lda,
		double* b, int* ldb, double* work, int* lwork, int* info);
	
}
		
inline void c_dgesvd(char jobu, char jobv, int m, int n, double* a, 
	int lda, double* s, double* u, int ldu, double* vt, int ldv,
	double* work, int lwork, int* info) 
{
	dgesvd_(&jobu,&jobv,&m,&n,a,&lda,s,u,&ldu,vt,&ldv,work,&lwork,info);
}

inline void c_dlatsqr(int m, int n, int mb, int nb, double* a, int lda,
	double* t, int ldt, double* work, int lwork, int* info)
{
	dlatsqr_(&m, &n, &mb, &nb, a, &lda, t, &ldt, work, &lwork, info);
}

inline void c_dormqr(char side, char trans, int m, int n, int k, 
	double* a, int lda, double* tau, double* c, int ldc, 
	double* work, int lwork, int* info)
{
	dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, 
		&lwork, info);
}

inline void c_dtrtrs(char uplo, char trans, char diag, int n, int nrhs,
	double* a, int lda, double* b, int ldb, int* info)
{ 
	dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, info);
} 

inline void c_dgeqr(int m, int n, double* a, int lda, double* t, 
	int tsize, double* work, int lwork, int* info)
{
	dgeqr_(&m, &n, a, &lda, t, &tsize, work, &lwork, info);
}

inline void c_dgels(char t, int m, int n, int nrhs, double* a, int lda,
	double* b, int ldb, double* work, int lwork, int* info)
{
	dgels_(&t, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
}

#endif
