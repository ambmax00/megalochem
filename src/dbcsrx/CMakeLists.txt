set(FYPP_SOURCES
	dbcsr_common.HPP
	dbcsr_matrix.HPP
	dbcsr_tensor.HPP
	dbcsr_matrix_ops.HPP
	dbcsr_tensor_ops.HPP
	fortran_utils.HPP
	fortran_utils.F
)       

set(FYPP_OUT)

foreach (file ${FYPP_SOURCES})	

	get_filename_component(file "${file}" ABSOLUTE)
	get_filename_component(rootname "${file}" NAME_WE)
	get_filename_component(extension "${file}" LAST_EXT)

	if(${extension} STREQUAL ".HPP")
		set(new_extension ".hpp")
		set(of "${CMAKE_CURRENT_BINARY_DIR}/${rootname}${new_extension}")
		add_custom_command(
          	  OUTPUT ${of}
          	  COMMAND fypp -F "${file}" "${of}"
        	  DEPENDS "${file}"
	        )
	elseif(${extension} STREQUAL ".F")
		set(new_extension ".f90")
		set(of "${CMAKE_CURRENT_BINARY_DIR}/${rootname}${new_extension}")
		add_custom_command(
                  OUTPUT ${of}
                  COMMAND fypp "${file}" "${of}"
                  DEPENDS "${file}"
                )
	else()
		message(FATAL_ERROR "Unknown file extension.")
	endif()

	list(APPEND FYPP_OUT "${of}")

endforeach()

add_custom_target(fypped ALL DEPENDS ${FYPP_OUT})

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
add_dependencies(chem_dbcsrx fypped)

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

