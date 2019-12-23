set(SOURCES
	basis.h
	molecule.h
	basis.cpp
	molecule.cpp
)

add_library(chem_desc ${SOURCES})
target_link_libraries(chem_desc Libint2::int2)
