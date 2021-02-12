function (ADD_FYPP_SOURCES OUTVAR FYPP_TARGET) 
	set(outfiles)

	foreach (file ${ARGN})

		get_filename_component(file "${file}" ABSOLUTE)
		get_filename_component(root "${file}" NAME_WE)
		get_filename_component(extension "${file}" EXT)

		if (${extension} STREQUAL ".hpp.fypp")
			set(new_extension ".hpp")
			set(of "${CMAKE_CURRENT_BINARY_DIR}/${root}${new_extension}")
			add_custom_command(
				OUTPUT ${of}
				COMMAND "${Python_EXECUTABLE}" -m fypp -I "${FYPP_INCLUDE_DIR}" -DCPLUSPLUS=0 -F "${file}" "${of}"
                  		DEPENDS "${file}"
                	)
        		list(APPEND outfiles "${of}")
		elseif(${extension} STREQUAL ".f.fypp")
			set(new_extension ".f90")
			set(of "${CMAKE_CURRENT_BINARY_DIR}/${root}${new_extension}")
			add_custom_command(
                  		OUTPUT ${of}
				COMMAND "${Python_EXECUTABLE}" -m fypp -I "${FYPP_INCLUDE_DIR}" "${file}" "${of}"
                  		DEPENDS "${file}"
                	)
			list(APPEND outfiles "${of}")
        	else()
                	message(FATAL_ERROR "Unknown file extension.")
        	endif()

	endforeach()

	add_custom_target("${FYPP_TARGET}" ALL DEPENDS "${outfiles}")
	set(OUTVAR "${outfiles}")

endfunction()
