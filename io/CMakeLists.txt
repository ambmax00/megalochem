set (SOURCES
	reader.h
	reader.cpp
	io.h
	io.cpp
)

add_library(chem_io ${SOURCES})

target_include_directories(chem_io
	PUBLIC
	${LIBINT2_INCLUDE_DIRS}
)

target_link_libraries(chem_io 
	Libint2::cxx
	chem_desc
	chem_dbcsrx
	Eigen3::Eigen
)
