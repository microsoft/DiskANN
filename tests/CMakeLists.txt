set(CMAKE_CXX_STANDARD 14)

add_executable(test_nsg_incr_index test_nsg_incr_index.cpp)
target_link_libraries(test_nsg_incr_index ${PROJECT_NAME} -ltcmalloc)

#add_executable(test_pq test_pq.cpp)
#target_link_libraries(test_pq ${PROJECT_NAME} -ltcmalloc)

#add_executable(test_pq_from_pivots test_pq_from_pivots.cpp)
#target_link_libraries(test_pq_from_pivots ${PROJECT_NAME} -ltcmalloc)

#add_executable(test_partitioning test_partitioning.cpp)
#target_link_libraries(test_partitioning ${PROJECT_NAME} -ltcmalloc)

add_executable(build_in_memory_index build_in_memory_index.cpp)
target_link_libraries(build_in_memory_index ${PROJECT_NAME} -ltcmalloc)

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME} -ltcmalloc)


add_executable(search_in_memory_index search_in_memory_index.cpp)
target_link_libraries(search_in_memory_index ${PROJECT_NAME})

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME} aio)

add_executable(calc_recall calc_recall.cpp)
add_executable(opt_disk_graph opt_disk_graph.cpp)
add_executable(fvecs_to_bin fvecs_to_bin.cpp)
add_executable(ivecs_to_bin ivecs_to_bin.cpp)
add_executable(nsg_bfs nsg_bfs.cpp)
add_executable(rename_merge_shards rename_merge_shards.cpp)

# formatter
add_custom_command(TARGET calc_recall PRE_BUILD COMMAND clang-format-4.0 -i ../../include/*.h ../../src/*.cpp ../../tests/*.cpp)
