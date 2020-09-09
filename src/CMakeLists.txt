include_directories(
	${CMAKE_CURRENT_BINARY_DIR} 
	${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(dbcsrx)
add_subdirectory(math)
add_subdirectory(io)
add_subdirectory(desc)
add_subdirectory(ints)
add_subdirectory(fock)
add_subdirectory(hf)
add_subdirectory(mp)
add_subdirectory(adc)

add_executable(
        chem
        main.cpp
)

set_target_properties(
	chem 
	PROPERTIES RUNTIME_OUTPUT_DIRECTORY 
	"${CMAKE_BINARY_DIR}"
)
