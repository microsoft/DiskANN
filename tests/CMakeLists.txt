set(CMAKE_CXX_STANDARD 14)

add_executable(test_nsg_incr_index test_nsg_incr_index.cpp)
target_link_libraries(test_nsg_incr_index ${PROJECT_NAME} -ltcmalloc)

#add_executable(test_pq test_pq.cpp)
#target_link_libraries(test_pq ${PROJECT_NAME} -ltcmalloc)

#add_executable(test_pq_from_pivots test_pq_from_pivots.cpp)
#target_link_libraries(test_pq_from_pivots ${PROJECT_NAME} -ltcmalloc)

#add_executable(test_partitioning test_partitioning.cpp)
#target_link_libraries(test_partitioning ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_sweep_index test_nsg_sweep_index.cpp)
target_link_libraries(test_nsg_sweep_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})


add_executable(test_build test_build.cpp)
target_link_libraries(test_build ${PROJECT_NAME})

add_executable(test_search test_search.cpp)
target_link_libraries(test_search ${PROJECT_NAME} aio)


add_executable(calc_recall calc_recall.cpp)

add_executable(opt_disk_graph opt_disk_graph.cpp)

add_executable(fvecs_to_bin fvecs_to_bin.cpp)
add_executable(ivecs_to_bin ivecs_to_bin.cpp)
add_executable(nsg_bfs nsg_bfs.cpp)
add_executable(rename_merge_shards rename_merge_shards.cpp)

# formatter
add_custom_command(TARGET calc_recall PRE_BUILD COMMAND clang-format-4.0 -i ../../include/*.h ../../src/*.cpp ../../tests/*.cpp)
