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

set(CPP_SOURCES
	molecule.hpp
)

add_cpp_sources(desc ${CPP_SOURCES})

add_library(chem_desc ${SOURCES})

add_dependencies(chem_desc cpp_desc)

target_link_libraries(
	chem_desc
        chem_io	
	chem_dbcsrx
)
