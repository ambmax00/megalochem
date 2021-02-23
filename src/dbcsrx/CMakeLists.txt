set(FYPP_SOURCES
	dbcsr_common.hpp.fypp
	dbcsr_matrix.hpp.fypp
	dbcsr_tensor.hpp.fypp
	dbcsr_btensor.hpp.fypp
	dbcsr_matrix_ops.hpp.fypp
	dbcsr_tensor_ops.hpp.fypp
	fortran_utils.hpp.fypp
)

set(FYPP_F_SOURCES
	fortran_utils.f.fypp
)       

add_fypp_sources(DBCSRX_SRCS DBCSRX_DIR dbcsrx_fypp ${FYPP_SOURCES})
add_fypp_sources(DBCSRX_F_SRCS DBCSRX_F_DIR dbcsrx_f_fypp ${FYPP_F_SOURCES})

add_library(fortran_utils ${DBCSRX_F_SRCS})
add_library(fortran_utils_c INTERFACE)

target_link_libraries(
        fortran_utils_c
        INTERFACE
        fortran_utils
)

add_library(chem_dbcsrx INTERFACE)
add_dependencies(chem_dbcsrx dbcsrx_fypp dbcsrx_f_fypp)

target_link_libraries(
	chem_dbcsrx
	INTERFACE
	fortran_utils_c
	Eigen3::Eigen
	MPI::MPI_CXX
	OpenMP::OpenMP_CXX
	PkgConfig::deps
        DBCSR::dbcsr_c
	SCALAPACK::SCALAPACK
)

target_include_directories(
	chem_dbcsrx
	INTERFACE 
	${DBCSRX_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
)

