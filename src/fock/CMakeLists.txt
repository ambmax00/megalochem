set(SOURCES
	jkcommon.cpp
	jkexact.cpp
	jkdf.cpp
	jkdf_robust.cpp
)

set(FYPP_SOURCES
	jkbuilder.hpp.fypp
)

add_fypp_sources(HEADERS GEN_DIR chem_fock_fypp ${FYPP_SOURCES})

add_library(chem_fock ${SOURCES})
add_dependencies(chem_fock chem_fock_fypp)

target_link_libraries(
	chem_fock
	chem_desc
	chem_ints
	chem_math_linalg
	chem_math_solvers
	chem_dbcsrx
	${LAPACK_LIBRARIES}
)
