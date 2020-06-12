set(SOURCES
	mpmod.h
	mp_compute.cpp
)

add_library(chem_mp ${SOURCES})

target_link_libraries(
	chem_mp
	chem_ints
	chem_tensor
	chem_desc
	chem_math_solvers
	chem_math_laplace
)
