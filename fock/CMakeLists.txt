set(SOURCES
	fockmod.h
	fock_builder.cpp
	jkbuilder.h
	jkcommon.cpp
	jkexact.cpp
	jkdf.cpp
	jkpari.cpp
)

add_library(chem_fock ${SOURCES})

target_link_libraries(
	chem_fock
	chem_desc
	chem_ints
	chem_math_linalg
	chem_math_solvers
	chem_dbcsrx
)
