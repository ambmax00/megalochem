set(SOURCES
	moprintmod.cpp
	locorb.hpp
	locorb_chol.cpp
	locorb_boys.cpp
	locorb_pao.cpp
)

set(CPP_SOURCES
	moprintmod.hpp
)

add_cpp_sources(locorb ${CPP_SOURCES})

add_library(chem_locorb ${SOURCES})

add_dependencies(chem_locorb cpp_locorb)

target_link_libraries(
	chem_locorb
	chem_ints
	chem_desc
	chem_math_linalg
	chem_dbcsrx
)
