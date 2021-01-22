set(FYPP_SOURCES
	dbcsr_common.hpp.fypp
	dbcsr_matrix.hpp.fypp
	dbcsr_tensor.hpp.fypp
	dbcsr_matrix_ops.hpp.fypp
	dbcsr_tensor_ops.hpp.fypp
	fortran_utils.hpp.fypp
	fortran_utils.f.fypp
)       

add_fypp_sources(DBCSRX_SRCS dbcsrx_fypp ${FYPP_SOURCES})

set(FORTRAN_SOURCES
	${CMAKE_CURRENT_BINARY_DIR}/fortran_utils.f90
)

add_library(fortran_utils ${FORTRAN_SOURCES})
add_library(fortran_utils_c INTERFACE)

target_link_libraries(
        fortran_utils_c
        INTERFACE
        fortran_utils
)

add_library(chem_dbcsrx INTERFACE)
add_dependencies(chem_dbcsrx dbcsrx_fypp)

target_link_libraries(
	chem_dbcsrx
	INTERFACE
	fortran_utils_c
	Eigen3::Eigen
	MPI::MPI_CXX
	OpenMP::OpenMP_CXX
	PkgConfig::deps
        DBCSR::dbcsr_c
	${SCALAPACK_LIBRARIES}
)

target_include_directories(
	chem_dbcsrx
	INTERFACE 
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
)

