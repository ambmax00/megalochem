set(SOURCES
	aofactory.hpp
	aofactory.cpp
	integrals.hpp
	integrals.cpp
	screening.hpp
	screening.cpp
	fitting.hpp
	fitting.cpp
	fitting_pari.cpp
	fitting_qr.cpp
	aoloader.cpp
)

set(FYPP_SOURCES
        aoloader.hpp.fypp
)

add_fypp_sources(HEADERS GEN_DIR chem_ints_fypp ${FYPP_SOURCES})

add_library(chem_ints ${SOURCES})
add_dependencies(chem_ints chem_ints_fypp)

set_source_files_properties(fitting_qr.cpp PROPERTIES COMPILE_FLAGS -fno-math-errno)

target_include_directories(
	chem_ints
	PUBLIC
	${LIBCINT_INCLUDE_DIRS}
)
target_link_libraries(
	chem_ints
	${LIBCINT_LIBRARIES}
       	chem_desc
        chem_dbcsrx
        OpenMP::OpenMP_CXX
)

