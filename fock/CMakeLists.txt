set(SOURCES
	fockmod.h
	fock_builder.cpp
	jkbuilder.h
	jkbuilder.cpp
)

add_library(chem_fock ${SOURCES})

target_link_libraries(
	chem_fock
	chem_desc
	chem_math_solvers
	chem_tensor
)
