cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(chem)
enable_language(Fortran CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

find_package(Eigen3 REQUIRED NO_MODULE)

#get_target_property(dir Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

include_directories(${dir})

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(DBCSR 2.0.0 CONFIG REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PkgConfig)
pkg_check_modules(deps REQUIRED IMPORTED_TARGET GLOBAL libxsmmf)
find_package(SCALAPACK REQUIRED)

if(${USE_LIBCINT})
	message(STATUS "Building with libcint")
	find_library(LIBCINT_LIBRARIES 
		NAMES libcint.a 
		PATHS ${LIBCINT_DIR}/lib/
		REQUIRED
	)
	set(LIBCINT_INCLUDE_DIRS ${LIBCINT_DIR}/include/)
	message(STATUS "Found libcint: ${LIBCINT_LIBRARIES}")
        message(STATUS "Include directories: ${LIBCINT_INCLUDE_DIRS}")
else()
	message(STATUS "Building with libint2")
	find_package(Libint2 2.6.0 CONFIG REQUIRED)
endif()

get_target_property(DBCSR_INCLUDE_DIRS DBCSR::dbcsr INTERFACE_INCLUDE_DIRECTORIES)
include_directories(
	${DBCSR_INCLUDE_DIRS} 
	${MPI_CXX_HEADER_DIR} 
	${PYTHON_INCLUDE_DIRS}
	#${CMAKE_CURRENT_BINARY_DIR} 
	#${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(src)

target_link_libraries(
	chem
	PUBLIC
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
        chem_dbcsrx
        chem_io
        MPI::MPI_CXX
	chem_hf
	chem_mp
	chem_adc
)

