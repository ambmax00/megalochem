set (SOURCES
	reader.h
	reader.cpp
	io.h
	io.cpp
	data_handler.h
	data_handler.cpp
)

add_library(chem_io ${SOURCES})

target_include_directories(chem_io
	PUBLIC
	${HDF5_INCLUDE_DIRS}
)

target_link_libraries(chem_io 
	chem_desc
	chem_dbcsrx
	Eigen3::Eigen
	${HDF5_LIBRARIES}
	${PYTHON_LIBRARIES}
)
