set (sources
	orthogonalizer.h
	orthogonalizer.cpp	
)

add_library(chem_math_linalg ${sources})
target_link_libraries(
	chem_math_linalg 
	Eigen3::Eigen
)
