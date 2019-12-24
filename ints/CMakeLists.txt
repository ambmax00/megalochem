set(SOURCES
	aofactory.h
	aofactory.cpp
)

add_library(chem_ints ${SOURCES})

target_include_directories(
	chem_ints
	PUBLIC
	${TBB_INCLUDE_DIRS}
)

target_link_libraries(
	chem_ints
	chem_desc
	${TBB_LIBRARIES}
)
