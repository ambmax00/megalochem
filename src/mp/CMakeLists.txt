set(SOURCES
	mpmod.hpp
	mp_compute.cpp
	z_common.cpp
	z_full.cpp
	z_mem.cpp
)

set(FYPP_SOURCES
        z_builder.hpp.fypp
)

add_fypp_sources(HEADERS chem_mp_fypp ${FYPP_SOURCES})

add_library(chem_mp ${SOURCES})
add_dependencies(chem_mp chem_mp_fypp)

target_link_libraries(
	chem_mp
	chem_ints
	chem_dbcsrx
	chem_desc
	chem_math_solvers
	chem_math_laplace
)
