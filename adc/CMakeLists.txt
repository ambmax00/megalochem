set(SOURCES
	adcmod.h
	adc_diag.cpp
	adc_compute.cpp
	adc_init.cpp
	#adc_amplitudes.cpp
)

add_library(chem_adc ${SOURCES})

target_link_libraries(
	chem_adc
	PRIVATE
	chem_ints
	chem_desc
	chem_math_solvers
	chem_math_linalg
	chem_math_laplace
)
