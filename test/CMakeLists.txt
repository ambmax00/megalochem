include_directories(
	${GENERATED_DIR}/src 
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/src
)

add_executable(test_h2o test_h2o.cpp)

target_link_libraries(test_h2o megalochem)

add_test(
  NAME Test_h2o
  COMMAND mpirun -np 2 ${CMAKE_CURRENT_BINARY_DIR}/test_h2o
)
