set(SOURCES
	basis.h
	molecule.h
	basis.cpp
	molecule.cpp
	io.h
	io.cpp
)

add_library(chem_desc ${SOURCES})
target_link_libraries(chem_desc 
	chem_dbcsrx
	Libint2::int2)
