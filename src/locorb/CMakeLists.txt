set(SOURCES
	locorb.hpp
	locorb_chol.cpp
	locorb_boys.cpp
	locorb_pao.cpp
)

add_library(chem_locorb ${SOURCES})

target_link_libraries(
	chem_locorb
	chem_ints
	chem_desc
	chem_math_linalg
	chem_dbcsrx
	Python::Python
	Python::Module
	Python::NumPy
)
