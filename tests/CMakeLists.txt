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

add_executable(rename_merge_shards rename_merge_shards.cpp)

if(MSVC)
	add_executable(test_nsg_dll test_nsg_dll.cpp)
	target_link_options(test_nsg_dll PRIVATE /MACHINE:x64)
	target_link_libraries(test_nsg_dll debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_lib.lib)
	target_link_libraries(test_nsg_dll optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_lib.lib)
endif()
# formatter
