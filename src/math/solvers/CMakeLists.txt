set (sources
	hermitian_eigen_solver.cpp
)

add_library(chem_math_solvers ${sources})

target_link_libraries(
    chem_math_solvers
    chem_dbcsrx
    ${SCALAPACK_LIBRARIES}
)

