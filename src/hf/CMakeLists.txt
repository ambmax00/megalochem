set(SOURCES
	hfmod.hpp
	hfscf.cpp
	hfguess.cpp
	hfdiagfock.cpp
)

set(CPP_SOURCES
	hfmod.hpp
)

add_cpp_sources(hf ${CPP_SOURCES})

add_library(chem_hf ${SOURCES})

add_dependencies(chem_hf cpp_hf)

target_link_libraries(
	chem_hf
	chem_ints
	chem_desc
	chem_math_linalg
	chem_math_solvers
	chem_fock
)
