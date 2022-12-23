include_directories(
	${GENERATED_DIR}/src 
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/src
)

add_executable(testHf test_hf.cpp)

target_link_libraries(testHf megalochem)

add_test(
  NAME TestHf
  COMMAND mpirun -np 2 ${CMAKE_CURRENT_BINARY_DIR}/testHf
)
