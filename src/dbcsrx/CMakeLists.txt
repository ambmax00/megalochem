set(CPP_SOURCES
	dbcsr_common.hpp
	dbcsr_matrix.hpp
	dbcsr_tensor.hpp
	dbcsr_matrix_ops.hpp
	dbcsr_tensor_ops.hpp
	dbcsr_btensor.hpp
)

add_cpp_sources(dbcsrx_cpp ${CPP_SOURCES})

add_library(chem_dbcsrx INTERFACE)

add_dependencies(chem_dbcsrx cpp_dbcsrx)

target_link_libraries(
	chem_dbcsrx
	INTERFACE
	Eigen3::Eigen
	MPI::MPI_CXX
	OpenMP::OpenMP_CXX
	PkgConfig::LIBXSMMF
	PkgConfig::LIBXSMMEXT
        DBCSR::dbcsr_c
	SCALAPACK::SCALAPACK
)

target_include_directories(
	chem_dbcsrx
	INTERFACE 
	${CMAKE_CURRENT_SOURCE_DIR}
)

