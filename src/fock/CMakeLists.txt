set(SOURCES
	fockmod.hpp
	fock_builder.cpp
	jkbuilder.hpp
	jkcommon.cpp
	jkexact.cpp
	jkdf.cpp
	jkdf_robust.cpp
)

add_library(chem_fock ${SOURCES})

target_link_libraries(
	chem_fock
	chem_desc
	chem_ints
	chem_math_linalg
	chem_math_solvers
	chem_dbcsrx
	${LAPACK_LIBRARIES}
)
