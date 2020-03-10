set(SOURCES
	adcmod.h
	adc_compute.cpp
	adc_init.cpp
)

add_library(chem_adc ${SOURCES})

target_link_libraries(
	chem_adc
	PRIVATE
	chem_ints
	chem_desc
)
