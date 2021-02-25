include(CheckFunctionExists)
include(CheckSymbolExists)
include(CheckLibraryExists)

set(SCALAPACK_FOUND true)

# check if we have Intel
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS QUIET)

if (BLAS_FOUND)
	message(STATUS "MKL environment detected")
else()
	unset(BLA_VENDOR)
	find_package(BLAS REQUIRED)
endif()

find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)

message(STATUS "BLAS VENDOR: ${BLA_VENDOR}")

if ("${BLA_VENDOR}" MATCHES "Intel.*") 

	if (NOT MPI_VENDOR) 
		message(FATAL_ERROR "Please specify MPI_VENDOR (openmpi/intelmpi)")
	endif()	

	set(MKLROOT $ENV{MKLROOT})
	set(MKL_ROOT $ENV{MKL_ROOT})
	
	if ("${LAPACK_LIBRARIES}" MATCHES ".*intel64_lin.*") 
		set(IDIR "intel64_lin")
	else()
		set(IDIR "intel64")
	endif()
		
	find_library(
		SCALAPACK_LIBRARIES
		NAMES 
		libmkl_scalapack_lp64.so
		PATHS
		${MKLROOT}/lib/${IDIR}/
		${MKL_ROOT}/lib/${IDIR}/
		${SCALAPACK_DIR}/lib/
		REQUIRED
	)
	
	find_library(
		BLACS_LIBRARIES
		NAMES
		"libmkl_blacs_${MPI_VENDOR}_lp64.so"
		PATHS
		${MKLROOT}/lib/${IDIR}/
		${MKL_ROOT}/lib/${IDIR}/
		${SCALAPACK_DIR}/lib/
                REQUIRED
	)

	message(STATUS "Found SCALAPACK: ${SCALAPACK_LIBRARIES}")
	message(STATUS "Found BLACS: ${BLACS_LIBRARIES}")

	add_library(SCALAPACK::SCALAPACK INTERFACE IMPORTED)
	set_target_properties(
                SCALAPACK::SCALAPACK
                PROPERTIES
                INTERFACE_LINK_LIBRARIES
		"${SCALAPACK_LIBRARIES};${BLACS_LIBRARIES};${LAPACK_LIBRARIES};${MPI_Fortran_LIBRARIES}"
        )
else()

	find_library(
		SCALAPACK_LIBRARIES
		NAMES
		libscalapack.a
		PATHS
		"${SCALAPACK_DIR}"
		/usr/bin/lib
		REQUIRED
	)

	message(STATUS "Found SCALAPACK: ${SCALAPACK_LIBRARIES}")

	add_library(SCALAPACK::SCALAPACK INTERFACE IMPORTED)
        
	set_target_properties(
		SCALAPACK::SCALAPACK
		PROPERTIES
		INTERFACE_LINK_LIBRARIES
		"${SCALAPACK_LIBRARIES};${LAPACK_LIBRARIES};${MPI_Fortran_LIBRARIES};gfortran"
        )

endif()

list(APPEND CMAKE_REQUIRED_LIBRARIES SCALAPACK::SCALAPACK)

check_function_exists(blacs_gridinit_ HAS_BLACS)
check_function_exists(pdgemm_ HAS_PDGEMM)

if (NOT HAS_BLACS) 
	message(FATAL_ERROR 
		"Could not find blacs_init_. \
If you are using Reference-ScaLAPACK, \
make sure you are using a version (>=2.0) with BLACS embedded."
	)
endif()

if (NOT HAS_PDGEMM)
	message(FATAL_ERROR
		"Could not find pdgemm_."
	)
endif()

set(SCALAPACK_FOUND true)
