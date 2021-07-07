# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

add_executable(lazy_reinsert_batch lazy_reinsert_batch.cpp )
if(MSVC)
	target_link_options(lazy_reinsert_batch PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(lazy_reinsert_batch debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(lazy_reinsert_batch optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(lazy_reinsert_batch ${PROJECT_NAME} -ltcmalloc)
endif()

add_executable(build_memory_index build_memory_index.cpp )
if(MSVC)
	target_link_options(build_memory_index PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_memory_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_memory_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(build_memory_index ${PROJECT_NAME} -ltcmalloc)
endif()

add_executable(search_memory_index search_memory_index.cpp )
if(MSVC)
	target_link_options(search_memory_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_memory_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
	target_link_libraries(search_memory_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
else() 
	target_link_libraries(search_memory_index ${PROJECT_NAME} aio -ltcmalloc)
endif()

add_executable(build_disk_index build_disk_index.cpp )
if(MSVC)
	target_link_options(build_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(build_disk_index ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(search_disk_index search_disk_index.cpp
    ${PROJECT_SOURCE_DIR}/src/aux_utils.cpp )
if(MSVC)
	target_link_options(search_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
	target_link_libraries(search_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
else() 
	target_link_libraries(search_disk_index ${PROJECT_NAME} aio -ltcmalloc)
endif()

set(CMAKE_CXX_STANDARD 14)

add_executable(test_concurr_merge_insert test_concurr_merge_insert.cpp)
if(MSVC)

	target_link_options(test_concurr_merge_insert PRIVATE /MACHINE:x64)
	target_link_libraries(test_concurr_merge_insert debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(test_concurr_merge_insert optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
	target_link_libraries(test_concurr_merge_insert ${PROJECT_NAME} aio)
endif()

add_executable(test_incremental_index test_incremental_index.cpp)
if(MSVC)

	target_link_options(test_incremental_index PRIVATE /MACHINE:x64)
	target_link_libraries(test_incremental_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(test_incremental_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
    target_link_libraries(test_incremental_index ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(build_stream_merger_disk_index build_stream_merger_disk_index.cpp)
if(MSVC)
	target_link_options(build_stream_merger_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_stream_merger_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_stream_merger_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(build_stream_merger_disk_index ${PROJECT_NAME} -ltcmalloc aio)
endif()

if (MSVC)
add_executable(test_incr_intf test_incr_intf.cpp)
	target_link_options(test_incr_intf PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(test_incr_intf debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(test_incr_intf optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
endif()
# formatter
