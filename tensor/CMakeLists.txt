set(FYPP_SOURCES
       dbcsr_common.tpp
       dbcsr_matrix.tpp
       dbcsr_tensor.tpp
       dbcsr_matrix_ops.tpp
       dbcsr_tensor_ops.tpp
)       

set(FYPP_OUT)

foreach (file ${FYPP_SOURCES})	

	get_filename_component(file "${file}" ABSOLUTE)
	get_filename_component(rootname "${file}" NAME_WE)

	set(of "${CMAKE_CURRENT_BINARY_DIR}/${rootname}.hpp")

	add_custom_command(
	    OUTPUT ${of}
	    COMMAND fypp -F "${file}" "${of}"
	    DEPENDS "${file}"
	)

	list(APPEND FYPP_OUT "${of}")

endforeach()

add_custom_target(fypped ALL DEPENDS ${FYPP_OUT})

add_library(chem_tensor INTERFACE)
add_dependencies(chem_tensor fypped)

target_link_libraries(
	chem_tensor 
	INTERFACE
	Eigen3::Eigen
	MPI::MPI_CXX
	OpenMP::OpenMP_CXX
	PkgConfig::deps
        DBCSR::dbcsr_c
	${SCALAPACK_LIBRARIES}
)

target_include_directories(
	chem_tensor
	INTERFACE 
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
)

