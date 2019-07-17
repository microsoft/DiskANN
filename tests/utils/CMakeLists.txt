set(CMAKE_CXX_STANDARD 14)


add_executable(test_partitioning test_partitioning.cpp)
target_link_libraries(test_partitioning ${PROJECT_NAME} -ltcmalloc)

add_executable(calc_recall calc_recall.cpp)
add_executable(opt_disk_graph opt_disk_graph.cpp)
add_executable(fvecs_to_bin fvecs_to_bin.cpp)
add_executable(ivecs_to_bin ivecs_to_bin.cpp)
add_executable(nsg_bfs nsg_bfs.cpp)

# formatter
add_custom_command(TARGET calc_recall PRE_BUILD COMMAND clang-format-4.0 -i ../../../include/*.h ../../../include/dll/*.h ../../../src/*.cpp  ../../../tests/*.cpp ../../../src/dll/*.cpp ../../../tests/utils/*.cpp)
