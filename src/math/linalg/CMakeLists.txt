set (sources
	orthogonalizer.h
	orthogonalizer.cpp
	piv_cd.h
	piv_cd.cpp
	LLT.h
	LLT.cpp
	SVD.h
	SVD.cpp	
)

add_library(chem_math_linalg ${sources})

target_link_libraries(
	chem_math_linalg
	chem_math_solvers
        chem_dbcsrx
	SCALAPACK::SCALAPACK
	Python::Python
	Python::Module
	Python::NumPy
)
