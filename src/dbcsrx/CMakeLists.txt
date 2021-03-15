add_library(chem_dbcsrx INTERFACE)

target_link_libraries(
	chem_dbcsrx
	INTERFACE
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
	${CMAKE_CURRENT_SOURCE_DIR}
)

