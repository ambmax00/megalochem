set(SOURCES
	aofactory.h
	aofactory.cpp
	integrals.h
	integrals.cpp
)

add_library(chem_ints ${SOURCES})

target_include_directories(
	chem_ints
	PUBLIC
	${TBB_INCLUDE_DIRS}
	${LIBINT2_INCLUDE_DIRS}
)

target_link_libraries(
	chem_ints
	chem_desc
	OpenMP::OpenMP_CXX
	Libint2::cxx
	${TBB_LIBRARIES}
)
