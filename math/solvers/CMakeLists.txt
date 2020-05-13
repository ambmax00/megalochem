set (sources
	hermitian_eigen_solver.cpp
)

add_library(chem_math_solvers ${sources})

target_link_libraries(
	chem_math_solvers
        chem_tensor	
)

if (SCALAPACK_LIB) 
	target_link_libraries(chem_math_solvers ${SCALAPACK_LIB})
endif()
