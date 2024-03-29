cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(chem)
enable_language(Fortran CXX)
enable_testing()

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -pedantic)

message(STATUS "Compiling with CXX flags ${CMAKE_CXX_FLAGS}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules)

include(cpp_sources)

find_package(Eigen3 REQUIRED NO_MODULE)

#get_target_property(dir Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(DBCSR 2.1.0 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter)
find_package(HDF5 REQUIRED)
find_package(PkgConfig)
pkg_check_modules(LIBXSMMF REQUIRED IMPORTED_TARGET GLOBAL libxsmmf)
pkg_check_modules(LIBXSMMEXT REQUIRED IMPORTED_TARGET GLOBAL libxsmmext)
find_package(SCALAPACK REQUIRED)

find_package(QUADMATH)

if (NOT LIBCINT_DIR)
	set(LIBCINT_DIR /opt/libcint)
endif()

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
add_subdirectory(test)

target_link_libraries(
	chem
	PUBLIC
	git
	megalochem
)

target_link_libraries(
        chem_test
        PUBLIC
	git
	megalochem
)
