add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dbcsr.hpp
    COMMAND fypp -F ${CMAKE_CURRENT_SOURCE_DIR}/dbcsr.fpp ${CMAKE_CURRENT_BINARY_DIR}/dbcsr.hpp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dbcsr.fpp
)

add_custom_target(fypped ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dbcsr.hpp)

add_library(chem_tensor INTERFACE)
add_dependencies(chem_tensor fypped)

target_link_libraries(
	chem_tensor 
	INTERFACE
	Eigen3::Eigen
	MPI::MPI_CXX
        DBCSR::dbcsr_c
)

target_include_directories(
	chem_tensor
	INTERFACE 
	${CMAKE_CURRENT_BINARY_DIR}
)

