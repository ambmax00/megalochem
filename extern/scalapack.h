#ifndef EXTERN_SCALAPACK_H
#define EXTERN_SCALAPACK_H

#include <iostream>

extern "C" {
	
	void blacs_pinfo_(int* mypnum, int* nprocs);
	void blacs_get_(int* icontxt, int* what, int* val);
	void blacs_gridinit_(int* icontxt, char* layout, int* nprow, int* npcol);
	void blacs_barrier_(int* icontxt, char* scope);
	void blacs_gridinfo_(int* icontxt, int* nprow, int* npcol, int* myprow, int* mypcol);
	void blacs_gridexit_(int* ctxt);
	void blacs_exit_(int* cont);
	
	int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);
	void descinit_(int* desc, int* m, int* n, int* mb, int* nb, int* irsrc, int* icsrc,
		int* ictxt, int* lld, int* info);
	int indxl2g_(int* indxloc, int* nb, int* iproc, int* isrcproc, int* nprocs);
	int indxg2l_(int* indxglob, int* nb, int* iproc, int* isrcproc, int* nprocs);
	int indxg2p_(int* indxglob, int* nb, int* iproc, int* isrcproc, int* nprocs);
	
	void pdsyev_(char* jobz, char* uplo, int* n, double* a, int* ia, int* ja,
		int* desca, double* w, double* z, int* iz, int* jz, int* descz, 
		double* work, int* lwork, int* info);
	
}

inline void c_blacs_pinfo(int* mypnum, int* nprocs)
{
	blacs_pinfo_(mypnum, nprocs);
};
inline void c_blacs_get(int icontxt, int what, int* val) 
{
	blacs_get_(&icontxt, &what, val);
};
inline void c_blacs_gridinit(int icontxt, char layout, int nprow, int npcol) 
{
	blacs_gridinit_(&icontxt, &layout, &nprow, &npcol);
};
inline void c_blacs_barrier(int icontxt, char scope) 
{
	blacs_barrier_(&icontxt, &scope);
};
inline void c_blacs_gridinfo(int icontxt, int* nprow, int* npcol, int* myprow, int* mypcol)
{
	blacs_gridinfo_(&icontxt, nprow, npcol, myprow, mypcol);
};
inline void c_blacs_gridexit(int ctxt) 
{
	blacs_gridexit_(&ctxt);
};
inline void c_blacs_exit(int comt) 
{
	blacs_exit_(&comt);
};

inline int c_indxl2g(int indxloc, int nb, int iproc, int isrcproc, int nprocs)
{
	int f_indxloc = indxloc + 1;
	return indxl2g_(&f_indxloc, &nb, &iproc, &isrcproc, &nprocs) - 1;
};
inline int c_indxg2l(int indxglob, int nb, int iproc, int isrcproc, int nprocs)
{
	int f_indxglob = indxglob + 1;
	return indxg2l_(&f_indxglob, &nb, &iproc, &isrcproc, &nprocs) - 1;
};
inline int c_indxg2p(int indxglob, int nb, int iproc, int isrcproc, int nprocs)
{
	int f_indxglob = indxglob + 1;
	return indxg2p_(&f_indxglob, &nb, &iproc, &isrcproc, &nprocs);
};

inline int c_numroc(int n, int nb, int iproc, int isrcproc, int nprocs)
{
	return numroc_(&n, &nb, &iproc, &isrcproc, &nprocs);
};
inline void c_descinit(int* desc, int m, int n, int mb, int nb, int irsrc, int icsrc,
		int ictxt, int lld, int* info)
{
	descinit_(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, info);
};

inline void c_pdsyev(char jobz, char uplo, int n, double* a, int ia, int ja,
		int* desca, double* w, double* z, int iz, int jz, int* descz, 
		double* work, int lwork, int* info)
{
	int f_ia = ia + 1;
	int f_ja = ja + 1;
	int f_iz = iz + 1;
	int f_jz = jz + 1;
	pdsyev_(&jobz,&uplo,&n,a,&f_ia,&f_ja,desca,w,z,&f_iz,&f_jz,descz,work,&lwork,info);
};

namespace scalapack {
	
class grid {
private: 

	int m_ctxt;
	int m_nprow, m_npcol;
	int m_myprow, m_mypcol;
	int m_mypnum;
	int m_nprocs;

public:

	grid(int nprow, int npcol, char layout = 'R') {

		// determine rank and size of processor set
		c_blacs_pinfo(&m_mypnum, &m_nprocs);
		
		// get the internal defualt context
		c_blacs_get(0, 0, &m_ctxt);
		
		// set up a proc grid for the proc set
		c_blacs_gridinit(m_ctxt,layout,nprow,npcol);
		
		c_blacs_barrier(m_ctxt,'A');
		
		c_blacs_gridinfo(m_ctxt, &m_nprow, &m_npcol, &m_myprow, &m_mypcol);
		
	}
	
	~grid() {}
	
	void release() {
		c_blacs_gridexit(m_ctxt);
	}
	
	int ctx() { return m_ctxt; }
	int nprow() { return m_nprow; }
	int npcol() { return m_npcol; }
	int myprow() { return m_myprow; }
	int mypcol() { return m_mypcol; }
	int mypnum() { return m_mypnum; }
	int nprocs() { return m_nprocs; }
	
};

template <typename T>
class distmat {
private:
	
	T* m_data;
	int m_nrowstot;
	int m_ncolstot;
	int m_nrowsloc;
	int m_ncolsloc;
	
	int m_rowblk_size;
	int m_colblk_size;
	
	grid m_grid;
	
	int m_desc[9]; // matrix descriptor
	
public:

	distmat(grid& igrid, int nrows, int ncols, int rowblksize, int colblksize) :
		m_grid(igrid), m_nrowstot(nrows), m_ncolstot(ncols), 
		m_rowblk_size(rowblksize), m_colblk_size(colblksize) 
	{
		
		m_nrowsloc = c_numroc(m_nrowstot, m_rowblk_size, m_grid.myprow(), 0, m_grid.nprow());
		m_ncolsloc = c_numroc(m_ncolstot, m_colblk_size, m_grid.mypcol(), 0, m_grid.npcol());
		
		int info = 0;
		c_descinit(&m_desc[0],m_nrowstot,m_ncolstot,m_rowblk_size,
					m_colblk_size,0,0,m_grid.ctx(),m_nrowsloc,&info);
					
		m_data = new T[m_nrowsloc*m_ncolsloc];
		
	}
	
	inline T& local_access(int i, int j) {
		return m_data[i + m_nrowsloc*j];
	}
	
	inline T& global_access(int iglob, int jglob) {
		int iloc = c_indxg2l(iglob, m_rowblk_size, 0, 0, m_grid.nprow());
		int jloc = c_indxg2l(jglob, m_colblk_size, 0, 0, m_grid.npcol());
		return m_data[iloc + m_nrowsloc*jloc];
	}
	
	int iglob(int iloc) {
		return c_indxl2g(iloc,m_rowblk_size,m_grid.myprow(),0,m_grid.nprow());
	}
	
	int jglob(int jloc) {
		return c_indxl2g(jloc,m_colblk_size,m_grid.mypcol(),0,m_grid.npcol());
	}
	
	~distmat() {
		if (m_data != nullptr) delete [] m_data;
	}
	
	int nfull_loc() {
		return m_nrowsloc*m_ncolsloc;
	}
	
	T* data() { return m_data; }
	
	int* desc() { return &m_desc[0]; }
	
	void print() {
		for (int p = 0; p != m_grid.nprocs(); ++p) {
			if (m_grid.mypnum() == p) {
				std::cout << "RANK: " << p << std::endl;
				for (int i = 0; i != m_nrowsloc; ++i) {
					for (int j = 0; j != m_ncolsloc; ++j) {
						std::cout << local_access(i,j) << " ";
					} std::cout << std::endl;
				}
			}
			c_blacs_barrier(m_grid.ctx(),'A');
		}
	}
	
};

} // end namespace
		
#endif
		
