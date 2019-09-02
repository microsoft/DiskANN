set(CMAKE_CXX_STANDARD 14)

add_executable(test_nsg_incr_index test_nsg_incr_index.cpp)
target_link_libraries(test_nsg_incr_index ${PROJECT_NAME} -ltcmalloc)


add_executable(build_in_memory_index build_in_memory_index.cpp)
target_link_libraries(build_in_memory_index ${PROJECT_NAME} -ltcmalloc)

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME} -ltcmalloc)

add_executable(search_in_memory_index search_in_memory_index.cpp)
target_link_libraries(search_in_memory_index ${PROJECT_NAME})

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME} aio)

add_executable(create_visited_cache create_visited_cache.cpp)
target_link_libraries(create_visited_cache ${PROJECT_NAME} aio)

add_executable(generate_pq generate_pq.cpp)
target_link_libraries(generate_pq ${PROJECT_NAME})

add_executable(partition_data partition_data.cpp)
target_link_libraries(partition_data ${PROJECT_NAME} -ltcmalloc)

add_executable(merge_shards merge_shards.cpp)
add_executable(create_disk_layout create_disk_layout.cpp)

# formatter
