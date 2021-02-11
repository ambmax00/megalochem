set (SOURCES
	parser.h
	parser.cpp
	io.h
	io.cpp
)

add_library(chem_io ${SOURCES})

target_link_libraries(chem_io 
	chem_desc
	chem_dbcsrx
	Eigen3::Eigen
	${PYTHON_LIBRARIES}
)
