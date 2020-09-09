MODULE c_laplace

 USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_double, c_bool, c_int
 IMPLICIT NONE
 
#include "consts.h"
#include "laplace_minimax.h"

CONTAINS

 SUBROUTINE c_laplace_minimax(errmax,xpnts,wghts,nlap,ymin,ymax,&
                           mxiter,iprint,stepmx,tolrng,tolpar,tolerr,&
                           delta,afact,do_rmsd,do_init,do_nlap) &
    BIND(C,name="c_laplace_minimax")
  
    REAL(kind=c_double), INTENT(OUT) :: errmax
    INTEGER(kind=c_int), INTENT(INOUT) :: nlap
    REAL(kind=c_double), DIMENSION(nlap), INTENT(INOUT) :: xpnts, wghts
    REAL(kind=c_double), INTENT(IN), VALUE :: ymin, ymax
    
    INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: mxiter, iprint
    LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: do_rmsd, do_init, do_nlap
    REAL(kind=c_double), INTENT(IN), OPTIONAL :: stepmx, tolrng, tolpar, delta, afact, tolerr
    
    LOGICAL, ALLOCATABLE :: f_do_rmsd, f_do_init, f_do_nlap
    
    IF (PRESENT(do_rmsd)) THEN
       ALLOCATE(f_do_rmsd)
       f_do_rmsd = do_rmsd
    ENDIF
    
    IF (PRESENT(do_init)) THEN
       ALLOCATE(f_do_init)
       f_do_init = do_init
    ENDIF
    
    IF (PRESENT(do_nlap)) THEN
       ALLOCATE(f_do_nlap)
       f_do_nlap = do_nlap
    ENDIF

    CALL laplace_minimax(errmax,xpnts,wghts,nlap,ymin,ymax,&
                           mxiter,iprint,stepmx,tolrng,tolpar,tolerr,&
                           delta,afact,f_do_rmsd,f_do_init,f_do_nlap)
                           
    IF (PRESENT(do_rmsd)) THEN
       DEALLOCATE(f_do_rmsd)
    ENDIF
    
    IF (PRESENT(do_init)) THEN
       DEALLOCATE(f_do_init)
    ENDIF
    
    IF (PRESENT(do_nlap)) THEN
       DEALLOCATE(f_do_nlap)
    ENDIF
    
    flush (6)

 END SUBROUTINE

END MODULE
