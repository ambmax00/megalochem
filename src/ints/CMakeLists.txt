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

set(CPP_SOURCES
        aoloader.hpp
)

add_cpp_sources(chem_ints_cpp ${CPP_SOURCES})

add_library(chem_ints ${SOURCES})

set_source_files_properties(fitting_qr.cpp PROPERTIES COMPILE_FLAGS -fno-math-errno)

target_include_directories(
	chem_ints
	PUBLIC
	${LIBCINT_INCLUDE_DIRS}
)
target_link_libraries(
	chem_ints
	${LIBCINT_LIBRARIES}
	quadmath
       	chem_desc
        chem_dbcsrx
        OpenMP::OpenMP_CXX
)

