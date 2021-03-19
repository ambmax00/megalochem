set(SOURCES
	mpmod.hpp
	mp_compute.cpp
	z_common.cpp
	z_full.cpp
	z_mem.cpp
)

set(CPP_SOURCES
        z_builder.hpp
)

add_cpp_sources(chem_mp_cpp ${CPP_SOURCES})

add_library(chem_mp ${SOURCES})

target_link_libraries(
	chem_mp
	chem_ints
	chem_dbcsrx
	chem_desc
	chem_math_solvers
	chem_math_laplace
)
