# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(search_memory_index search_memory_index.cpp)
target_link_libraries(search_memory_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} ${DISKANN_ASYNC_LIB} Boost::program_options)

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(range_search_disk_index range_search_disk_index.cpp)
target_link_libraries(range_search_disk_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

#add_executable(test_incremental_index test_incremental_index.cpp
#     ${PROJECT_SOURCE_DIR}/src/aux_utils.cpp )
#if(MSVC)
# 	target_link_options(test_incremental_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
# 	target_link_libraries(test_incremental_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
# 	target_link_libraries(test_incremental_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
#else() 
# 	target_link_libraries(test_incremental_index ${PROJECT_NAME} -ltcmalloc Boost::program_options)
#endif()

add_executable(test_streaming_scenario test_streaming_scenario.cpp)
if(MSVC)
	target_link_options(test_streaming_scenario PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(test_streaming_scenario debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/gperftools/x64/Release-Patch/libtcmalloc_minimal.lib)
	target_link_libraries(test_streaming_scenario optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/gperftools/x64/Release-Patch/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(test_streaming_scenario ${PROJECT_NAME} -ltcmalloc Boost::program_options)
endif()

add_executable(search_memory_index_dynamic search_memory_index_dynamic.cpp)
if(MSVC)
	target_link_options(search_memory_index_dynamic PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_memory_index_dynamic debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/gperftools/x64/Release-Patch/libtcmalloc_minimal.lib)
	target_link_libraries(search_memory_index_dynamic optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/gperftools/x64/Release-Patch/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(search_memory_index_dynamic ${PROJECT_NAME} -ltcmalloc Boost::program_options)
endif()
