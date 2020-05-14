cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(chem)
enable_language(Fortran CXX)

set(CMAKE_BUILD_TYPE Debug)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

find_package(Eigen3 REQUIRED NO_MODULE)

get_target_property(dir Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

include_directories(${dir})

find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)
find_package(MPI REQUIRED)
find_package(DBCSR 2.0.0 CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(Libint2 2.6.0 CONFIG REQUIRED)

get_target_property(DBCSR_INCLUDE_DIRS DBCSR::dbcsr INTERFACE_INCLUDE_DIRECTORIES)
include_directories(
	${DBCSR_INCLUDE_DIRS} ${MPI_CXX_HEADER_DIR} ${Boost_INCLUDE_DIRS}
	${LIBINT2_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(tensor)
#add_subdirectory(math)
#add_subdirectory(desc)
#add_subdirectory(input)
#add_subdirectory(ints)
#add_subdirectory(hf)
#add_subdirectory(adc)
#add_subdirectory(mp2)

add_executable(
	chem 
	main.cpp 
)

target_link_libraries(
	chem
	PUBLIC
	chem_tensor
	#	chem_input
	#	chem_math_laplace
	MPI::MPI_CXX	
	#chem_hf
	#chem_desc
	#chem_ints
	#chem_adc
)
