cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(chem)
enable_language(Fortran CXX)

set(CMAKE_BUILD_TYPE Debug)
set (CMAKE_CXX_STANDARD 17)

#include(cmake/modules/AddMPQCLibrary.cmake)
#set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

#list(APPEND CMAKE_MODULE_PATH /export/home/mambrois/libint2/gcc/lib/cmake/libint2)
#list(APPEND CMAKE_MODULE_PATH /export/home/mambrois/ctx/gnu/share/cmake/ctx)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(MPI REQUIRED)
find_package(Boost 1.66 REQUIRED)
find_package(DBCSR 2.0.0 CONFIG REQUIRED)

#set(libint2_DIR /export/home/mambrois/libint2/gcc/lib/cmake/libint2)
#find_package(libint2 CONFIG)


#set(ctx_DIR /export/home/mambrois/ctx/gnu/share/cmake/ctx)
#find_package(ctx REQUIRED)

#find_package(TBB REQUIRED)

#set(CTX_INCLUDE_DIR /export/home/mambrois/ctx/gnu/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

add_subdirectory(math)
#add_subdirectory(utils)
#add_subdirectory(mpqc)
#add_subdirectory(input)
#add_subdirectory(aofactory)
#add_subdirectory(hf)
#add_subdirectory(mp2)

add_executable(
	chem 
	main.cpp 
)

target_link_libraries(
	chem
	chem_math_laplace
	MPI::MPI_CXX	
	DBCSR::dbcsr_c
)
