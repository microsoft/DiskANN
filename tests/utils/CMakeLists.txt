set(CMAKE_CXX_STANDARD 14)


add_executable(truncate_rnsg truncate_rnsg.cpp)
target_link_libraries(truncate_rnsg ${PROJECT_NAME} -ltcmalloc)

add_executable(calc_recall calc_recall.cpp)
add_executable(opt_disk_graph opt_disk_graph.cpp)
add_executable(fvecs_to_bin fvecs_to_bin.cpp)
add_executable(ivecs_to_bin ivecs_to_bin.cpp)
add_executable(int8_to_float int8_to_float.cpp)
add_executable(uint32_to_uint8 uint32_to_uint8.cpp)
add_executable(nsg_bfs nsg_bfs.cpp)
add_executable(gen_random_slice gen_random_slice.cpp)


# formatter
add_custom_command(TARGET calc_recall PRE_BUILD COMMAND clang-format-4.0 -i ../../../include/*.h ../../../include/dll/*.h ../../../src/*.cpp  ../../../tests/*.cpp ../../../src/dll/*.cpp ../../../tests/utils/*.cpp)
