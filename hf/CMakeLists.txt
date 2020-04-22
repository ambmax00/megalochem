set(SOURCES
	hfmod.h
	hfscf.cpp
	hfguess.cpp
	hfdiagfock.cpp
	fock_builder.h
	fock_builder.cpp
)

add_library(chem_hf ${SOURCES})

target_link_libraries(
	chem_hf
	chem_ints
	chem_desc
	chem_math_linalg
	chem_math_other
)
