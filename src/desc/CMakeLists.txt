set(BASIS_ROOT ${CMAKE_SOURCE_DIR}/basis)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/basis.cpp
	${CMAKE_CURRENT_BINARY_DIR}/basis.cpp @ONLY
)

set(SOURCES
	basis.h
	molecule.h
	${CMAKE_CURRENT_BINARY_DIR}/basis.cpp
	molecule.cpp
)

add_library(chem_desc ${SOURCES})
target_link_libraries(
	chem_desc
        chem_io	
	chem_dbcsrx
)
