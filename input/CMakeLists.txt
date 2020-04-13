set (SOURCES
	reader.h
	reader.cpp
)

add_library(chem_input ${SOURCES})

target_link_libraries(chem_input 
	Libint2::int2
	chem_desc
	chem_tensor
)
