set(LAPLACE_INSTALL ${CMAKE_INSTALL_PREFIX})
set(LAPLACE_ROOT ${CMAKE_CURRENT_BINARY_DIR})

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/src/lap_rddata.F90
	${CMAKE_CURRENT_BINARY_DIR}/src/lap_rddata.F90 @ONLY
)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/src/lap_rderror.F90
	${CMAKE_CURRENT_BINARY_DIR}/src/lap_rderror.F90 @ONLY
)

set(SOURCES
    inc/consts.h
    inc/laplace_minimax.h
    src/dd128_arithmetics.F90
    src/dd128_linalg.F90
    ${CMAKE_CURRENT_BINARY_DIR}/src/lap_rderror.F90
    ${CMAKE_CURRENT_BINARY_DIR}/src/lap_rddata.F90
    src/lap_paraopt.F90
    src/lap_maehly.F90
    src/lap_rmsd.F90
    src/lap_numlap.F90
    src/laplace_minimax.F90
    src/laplace_minimax_c.F
)

configure_file(
	data/init_error.txt
	${CMAKE_CURRENT_BINARY_DIR}/data/init_error.txt
	COPYONLY)

configure_file(
        data/init_para.txt
        ${CMAKE_CURRENT_BINARY_DIR}/data/init_para.txt
        COPYONLY)

add_library(chem_math_laplace ${SOURCES})

# special compiler flags
if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
	target_compile_options(
        	chem_math_laplace
        	PRIVATE
        	-march=x86-64 -H -ffree-form
	)
endif()
	
	
if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
	target_compile_options(
                chem_math_laplace
                PRIVATE
                -march=x86-64 -free -assume protect_parens -nogen-interfaces
        )
endif()

target_include_directories(
	chem_math_laplace
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/inc
)
