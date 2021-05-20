set(LAPLACE_INSTALL ${CMAKE_INSTALL_PREFIX})
set(LAPLACE_ROOT ${CMAKE_CURRENT_BINARY_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
	set(INTEL_CXX "DEFINED")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	set(GNU_CXX "DEFINED")
endif()

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/quadmath.hpp.in
	${CMAKE_CURRENT_BINARY_DIR}/quadmath.hpp @ONLY
)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/minimax_rddata.cpp
	${CMAKE_CURRENT_BINARY_DIR}/minimax_rddata.cpp @ONLY
)

set(SOURCES
	minimax.cpp
	${CMAKE_CURRENT_BINARY_DIR}/minimax_rddata.cpp
	minimax_maehly.cpp
	minimax_paraopt.cpp
)

configure_file(
        init_para.txt
        ${CMAKE_CURRENT_BINARY_DIR}/init_para.txt
        COPYONLY
)

add_library(chem_math_laplace ${SOURCES})

if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
	set(LAP_FLAGS "-Qoption,cpp,--extended_float_type")
	set_target_properties(
                chem_math_laplace
                PROPERTIES
		COMPILE_FLAGS ${LAP_FLAGS}
        )
endif()

target_link_libraries(
	chem_math_laplace 
	Eigen3::Eigen
	quadmath
)

