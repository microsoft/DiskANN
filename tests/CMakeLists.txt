set(CMAKE_CXX_STANDARD 14)

add_executable(test_nsg_incr_index test_nsg_incr_index.cpp )
if(MSVC)
	target_link_options(test_nsg_incr_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(test_nsg_incr_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(test_nsg_incr_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)

else() 
	target_link_libraries(test_nsg_incr_index ${PROJECT_NAME} -ltcmalloc)
endif()

add_executable(build_memory_index build_memory_index.cpp )
if(MSVC)
	target_link_options(build_memory_index PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_memory_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_memory_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else() 
	target_link_libraries(build_memory_index ${PROJECT_NAME} -ltcmalloc)
endif()

add_executable(search_memory_index search_memory_index.cpp )
if(MSVC)
	target_link_options(search_memory_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_memory_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib)
	target_link_libraries(search_memory_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib)
else() 
	target_link_libraries(search_memory_index ${PROJECT_NAME} aio -ltcmalloc)
endif()

# formatter
