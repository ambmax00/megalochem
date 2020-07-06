include(ExternalProject)

set(minimax_prefix ${CMAKE_CURRENT_BINARY_DIR}/laplace_minimax)

ExternalProject_Add(laplace_minimax_build
	PREFIX ${minimax_prefix}
	GIT_REPOSITORY https://github.com/ambmax00/laplace-minimax.git
	INSTALL_COMMAND ""
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/laplace_minimax
)

add_library(laplace_minimax INTERFACE)
add_dependencies(laplace_minimax laplace_minimax_build)

target_link_libraries(laplace_minimax 
	INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/laplace_minimax/liblaplace-minimax.a
)
