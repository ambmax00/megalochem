add_library(chem_io INTERFACE)

target_link_libraries(
	chem_io 
	INTERFACE
	${HDF5_LIBRARIES}
)
