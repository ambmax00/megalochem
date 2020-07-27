set (SOURCES
	reader.h
	reader.cpp
	io.h
	io.cpp
)

add_library(chem_io ${SOURCES})

target_link_libraries(chem_io 
	Libint2::int2
	chem_desc
	chem_dbcsrx
	Eigen3::Eigen
)
