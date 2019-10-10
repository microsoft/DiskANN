#if(MSVC)
#	add_executable(nsg_service disk_main.cpp disk_nsg_search.cpp disk_server.cpp)

#	target_link_options(nsg_service PRIVATE /MACHINE:x64)
#	target_link_libraries(nsg_service debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_lib.lib)
#	target_link_libraries(nsg_service optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_lib.lib)
#else()
#	message("Currently, there is no support for building the service on Linux/Unix systems.")
#endif()
