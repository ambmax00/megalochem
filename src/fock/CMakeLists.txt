set(SOURCES
	jkcommon.cpp
	jkexact.cpp
	jkdf.cpp
	jkdf_robust.cpp
)

set(CPP_SOURCES
	jkbuilder.hpp
)

add_cpp_sources(fock ${CPP_SOURCES})

add_library(chem_fock ${SOURCES})

add_dependencies(chem_fock cpp_fock)

target_link_libraries(
	chem_fock
	chem_desc
	chem_ints
	chem_math_linalg
	chem_math_solvers
	chem_dbcsrx
	${LAPACK_LIBRARIES}
)
