set(SOURCES
	aofactory.h
	aofactory.cpp
	integrals.h
	integrals.cpp
	#gentran.h
	#gentran.cpp
	screening.h
	screening.cpp
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
	chem_dbcsrx
	#chem_math_linalg
	OpenMP::OpenMP_CXX
	Libint2::cxx
	${TBB_LIBRARIES}
)
