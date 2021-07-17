set (sources
	orthogonalizer.hpp
	orthogonalizer.cpp
	newton_schulz.hpp
	newton_schulz.cpp
	piv_cd.hpp
	piv_cd.cpp
	LLT.hpp
	LLT.cpp
	SVD.hpp
	SVD.cpp	
)

add_library(chem_math_linalg ${sources})

target_link_libraries(
	chem_math_linalg
	chem_math_solvers
        chem_dbcsrx
	SCALAPACK::SCALAPACK
)
