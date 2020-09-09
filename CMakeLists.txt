cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(chem)
enable_language(Fortran CXX)

set(CMAKE_BUILD_TYPE Debug)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

find_package(Eigen3 REQUIRED NO_MODULE)

#get_target_property(dir Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

include_directories(${dir})

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(DBCSR 2.0.0 CONFIG REQUIRED)
find_package(PkgConfig)
pkg_check_modules(deps REQUIRED IMPORTED_TARGET GLOBAL libxsmmf)
find_package(Libint2 2.6.0 CONFIG REQUIRED)
find_package(SCALAPACK REQUIRED)

get_target_property(DBCSR_INCLUDE_DIRS DBCSR::dbcsr INTERFACE_INCLUDE_DIRECTORIES)
include_directories(
	${DBCSR_INCLUDE_DIRS} 
	${MPI_CXX_HEADER_DIR} 
	${LIBINT2_INCLUDE_DIRS} 
	${CMAKE_CURRENT_BINARY_DIR} 
	${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(dbcsrx)
add_subdirectory(math)
add_subdirectory(io)
add_subdirectory(desc)
add_subdirectory(ints)
add_subdirectory(fock)
add_subdirectory(hf)
add_subdirectory(mp)
add_subdirectory(adc)

add_executable(
	chem 
	main.cpp 
)

target_link_libraries(
	chem
	PUBLIC
	chem_dbcsrx
	chem_io
	#	chem_math_laplace
	MPI::MPI_CXX	
	chem_hf
	chem_mp
	#chem_desc
	chem_ints
	chem_math_linalg
	#	chem_math_solvers
	chem_adc
)
