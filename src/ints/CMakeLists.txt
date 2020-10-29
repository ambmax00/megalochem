set(SOURCES
	aofactory.h
	aofactory.cpp
	integrals.h
	integrals.cpp
	screening.h
	screening.cpp
	fitting.h
	fitting.cpp
	aoloader.h
	aoloader.cpp
)

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

