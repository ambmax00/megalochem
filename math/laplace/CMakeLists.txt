set (sources
	lap_func.cpp
	laplace.cpp
	laplace.h
	lap_data.cpp
	lap_data.h
	lap_paraopt.cpp
	lap_root.cpp
)

add_library(chem_math_laplace ${sources})
find_library(quadmath NAMES quadmath.a)
target_link_libraries(
	chem_math_laplace 
	MPI::MPI_CXX
	Eigen3::Eigen
	quadmath
)

target_include_directories(
	chem_math_laplace
	PUBLIC 
	${CMAKE_SOURCE_DIR}
)

