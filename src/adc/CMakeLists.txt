set(SOURCES
	adcmod.hpp
	adc_diag.cpp
	adc_compute.cpp
	adc_init.cpp
	adc_mvp_common.cpp
	adc_mvp_aoriadc1.cpp
	adc_mvp_aoriadc2.cpp
)

set(CPP_SOURCES
        adc_mvp.hpp
	adcmod.hpp
)

add_cpp_sources(chem_adc ${CPP_SOURCES})

add_library(chem_adc ${SOURCES})

add_dependencies(chem_adc cpp_chem_adc)

target_link_libraries(
	chem_adc
	PRIVATE
	chem_ints
	chem_desc
	chem_math_solvers
	chem_math_linalg
	chem_math_laplace
	chem_mp
	chem_fock
	chem_locorb
)
