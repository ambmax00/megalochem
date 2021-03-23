cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(chem)
enable_language(Fortran CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules)

include(cpp_sources)

find_package(Eigen3 REQUIRED NO_MODULE)

#get_target_property(dir Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

include_directories(${dir})

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(DBCSR 2.1.0 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter)
#find_package(HDF5 REQUIRED)
find_package(PkgConfig)
pkg_check_modules(deps REQUIRED IMPORTED_TARGET GLOBAL libxsmmf)
find_package(SCALAPACK REQUIRED)

find_package(QUADMATH)

find_library(LIBCINT_LIBRARIES 
	NAMES libcint.a 
	PATHS ${LIBCINT_DIR}/lib/
	REQUIRED
)

set(LIBCINT_INCLUDE_DIRS ${LIBCINT_DIR}/include/)
message(STATUS "Found libcint: ${LIBCINT_LIBRARIES}")
message(STATUS "Include directories: ${LIBCINT_INCLUDE_DIRS}")

get_target_property(DBCSR_INCLUDE_DIRS DBCSR::dbcsr INTERFACE_INCLUDE_DIRECTORIES)
include_directories(
	${DBCSR_INCLUDE_DIRS} 
	${MPI_CXX_HEADER_DIR} 
)

add_subdirectory(src)
add_dependencies(chem cpp_target)

target_link_libraries(
	chem
	PUBLIC
	git
	chem_dbcsrx
	chem_io
	MPI::MPI_CXX	
	chem_hf
	chem_mp
	chem_adc
)

target_link_libraries(
        chem_test
        PUBLIC
	git
        chem_dbcsrx
        chem_io
        MPI::MPI_CXX
	chem_hf
	chem_mp
	chem_adc
)

