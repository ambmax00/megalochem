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
	aoloader.hpp
	aoloader.cpp
)

set_source_files_properties(fitting_qr.cpp PROPERTIES COMPILE_FLAGS -fno-math-errno)

add_library(chem_ints ${SOURCES})

if (${USE_LIBCINT})
	target_compile_definitions(chem_ints PUBLIC USE_LIBCINT)
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
else()
	target_compile_definitions(chem_ints PUBLIC USE_LIBINT)
	target_include_directories(
                chem_ints
                PUBLIC
                ${LIBINT2_INCLUDE_DIRS}
        )
        target_link_libraries(
                chem_ints
		Libint2::cxx
	        chem_desc
	        chem_dbcsrx
	        OpenMP::OpenMP_CXX
        )
endif()

