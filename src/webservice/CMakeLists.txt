#if(MSVC)
#	add_executable(diskann_service disk_main.cpp disk_index_search.cpp disk_server.cpp)

#	target_link_options(diskann_service PRIVATE /MACHINE:x64)
#	target_link_libraries(diskann_service debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_lib.lib)
#	target_link_libraries(diskann_service optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_lib.lib)
#else()
#	message("Currently, there is no support for building the service on Linux/Unix systems.")
#endif()
