set(CPP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src")
set(CPP_DEFINITIONS "${CPP_INCLUDE_DIR}/utils/ppdirs.hpp")
set(CPP_GENERATED_DIR "${CMAKE_BINARY_DIR}/cpp_generated")

if (NOT EXISTS "${CPP_GENERATED_DIR}")
	file(MAKE_DIRECTORY "${CPP_GENERATED_DIR}")
endif()

add_custom_target(cpp_target)

function (ADD_CPP_SOURCES OUTVAR)

        set(outfiles)

        foreach (file ${ARGN})

                get_filename_component(file "${file}" ABSOLUTE)
                get_filename_component(abs_path "${file}" DIRECTORY)

                file(RELATIVE_PATH rel_path "${CMAKE_SOURCE_DIR}" "${abs_path}")
		set(CPP_GENERATED_SUBDIR "${CPP_GENERATED_DIR}/${rel_path}")
		if (NOT EXISTS "${CPP_GENERATED_SUBDIR}")
			file(MAKE_DIRECTORY "${CPP_GENERATED_SUBDIR}")
                endif()

                get_filename_component(root "${file}" NAME_WE)
                get_filename_component(extension "${file}" EXT)

                set(new_extension ".hpp")
		set(of "${CPP_GENERATED_SUBDIR}/${root}${extension}")
                add_custom_command(
                        OUTPUT ${of}
                        COMMAND "${CMAKE_CXX_COMPILER}" -I ${CPP_INCLUDE_DIR} -E -DTEST_MACRO ${file} | clang-format > ${of}
                        DEPENDS "${file}" "${CPP_DEFINITIONS}"
                )
                list(APPEND outfiles "${of}")

        endforeach()

	add_custom_target("cpp_${OUTVAR}" ALL DEPENDS "${outfiles}")
	add_dependencies(cpp_target "cpp_${OUTVAR}")

endfunction()
