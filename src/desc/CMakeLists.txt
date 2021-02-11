set(BASIS_ROOT ${CMAKE_SOURCE_DIR}/basis)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/basis.cpp
	${CMAKE_CURRENT_BINARY_DIR}/basis.cpp @ONLY
)

set(SOURCES
	basis.hpp
	${CMAKE_CURRENT_BINARY_DIR}/basis.cpp
	molecule.cpp
)

set(FYPP_SOURCES
	molecule.hpp.fypp
)

add_fypp_sources(HEADERS chem_desc_fypp ${FYPP_SOURCES})

add_library(chem_desc ${SOURCES})
add_dependencies(chem_desc chem_desc_fypp)

target_link_libraries(
	chem_desc
        chem_io	
	chem_dbcsrx
)
