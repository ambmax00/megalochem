set(FYPP_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/utils)
set(FYPP_DEFINITIONS 
	"${FYPP_INCLUDE_DIR}/megalochem.fypp"
)

set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")

if (NOT EXISTS "${GENERATED_DIR}")	
	file(MAKE_DIRECTORY "${GENERATED_DIR}")
endif()

function (ADD_FYPP_SOURCES OUTVAR OUTDIR FYPP_TARGET) 
	
	set(outfiles)

	foreach (file ${ARGN})
		
		get_filename_component(file "${file}" ABSOLUTE)
		get_filename_component(abs_path "${file}" DIRECTORY)

		file(RELATIVE_PATH rel_path "${CMAKE_SOURCE_DIR}" "${abs_path}")
		set(GENERATED_SUBDIR "${GENERATED_DIR}/${rel_path}") 
		if (NOT EXISTS "${GENERATED_SUBDIR}") 
			file(MAKE_DIRECTORY "${GENERATED_SUBDIR}")
		endif()

		get_filename_component(root "${file}" NAME_WE)
		get_filename_component(extension "${file}" EXT)

		if (${extension} STREQUAL ".hpp.fypp")
			set(new_extension ".hpp")
			set(of "${GENERATED_SUBDIR}/${root}${new_extension}")
			add_custom_command(
				OUTPUT ${of}
				COMMAND "${Python_EXECUTABLE}" -m fypp -I "${FYPP_INCLUDE_DIR}" -DCPLUSPLUS=0 -F "${file}" "${of}"
				DEPENDS "${file}" "${FYPP_DEFINITIONS}"
                	)
        		list(APPEND outfiles "${of}")
		elseif(${extension} STREQUAL ".f.fypp")
			set(new_extension ".f90")
			set(of "${GENERATED_SUBDIR}/${root}${new_extension}")
			add_custom_command(
                  		OUTPUT ${of}
				COMMAND "${Python_EXECUTABLE}" -m fypp -I "${FYPP_INCLUDE_DIR}" -DCPLUSPLUS=1 "${file}" "${of}"
				DEPENDS "${file}" "${FYPP_DEFINTIONS}"
                	)
			list(APPEND outfiles "${of}")
        	else()
                	message(FATAL_ERROR "Unknown file extension.")
        	endif()

	endforeach()

	add_custom_target("${FYPP_TARGET}" ALL DEPENDS "${outfiles}")
	
	set(${OUTVAR} ${outfiles} PARENT_SCOPE)
	set(${OUTDIR} ${GENERATED_SUBDIR} PARENT_SCOPE)

endfunction()
