set (sources
	orthogonalizer.h
	orthogonalizer.cpp
	piv_cd.h
	piv_cd.cpp	
)

add_library(chem_math_linalg ${sources})
target_link_libraries(
	chem_math_linalg
	chem_math_solvers
        chem_tensor
        ${SCALAPACK_LIB}	
)
