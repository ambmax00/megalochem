set(SOURCES
	mpmod.hpp
	mp_compute.cpp
	z_builder.hpp
	z_common.cpp
	z_full.cpp
	z_mem.cpp
)

add_library(chem_mp ${SOURCES})

target_link_libraries(
	chem_mp
	chem_ints
	chem_dbcsrx
	chem_desc
	chem_math_solvers
	chem_math_laplace
)
