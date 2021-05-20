include_directories(
	${GENERATED_DIR}/src 
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
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
add_subdirectory(locorb)

set(PRE_CONFIGURE_FILE "git.hpp.in") 
set(POST_CONFIGURE_FILE "${CMAKE_CURRENT_BINARY_DIR}/git.hpp")

include(../cmake/modules/git_watcher.cmake)

add_library(git INTERFACE)
target_include_directories(
	git
	INTERFACE
	"${CMAKE_CURRENT_BINARY_DIR}"
)
add_dependencies(git check_git)

set(MEGALOCHEM_SRC
	megalochem_driver.hpp
	megalochem_driver.cpp
)

add_library(megalochem ${MEGALOCHEM_SRC})

target_link_libraries(
	megalochem
	MPI::MPI_CXX
	chem_dbcsrx
        chem_io
        chem_hf
        chem_mp
        chem_adc
)

add_executable(
        chem
        main.cpp
)

add_executable(
	chem_test
	main.cpp
)

set_target_properties(
        chem
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/bin"
)

target_compile_definitions(
	chem_test
	PUBLIC
	DO_TESTING
)

set_target_properties(
	chem_test
	PROPERTIES RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/tests"
)
