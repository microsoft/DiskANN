# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

add_executable(inmem_server inmem_server.cpp)
if(MSVC)
	target_link_options(inmem_server PRIVATE /MACHINE:x64)
	target_link_libraries(inmem_server debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
	target_link_libraries(inmem_server optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
else() 
	target_link_libraries(inmem_server ${PROJECT_NAME} aio -ltcmalloc -lboost_system -lcrypto -lssl -lcpprest)
endif()

add_executable(ssd_server ssd_server.cpp)
if(MSVC)
	target_link_options(ssd_server PRIVATE /MACHINE:x64)
	target_link_libraries(ssd_server debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
	target_link_libraries(ssd_server optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
else() 
	target_link_libraries(ssd_server ${PROJECT_NAME} aio -ltcmalloc -lboost_system -lcrypto -lssl -lcpprest)
endif()

add_executable(client client.cpp)
if(MSVC)
	target_link_options(client PRIVATE /MACHINE:x64)
	target_link_libraries(client debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
	target_link_libraries(client optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
else() 
	target_link_libraries(client ${PROJECT_NAME} -lboost_system -lcrypto -lssl -lcpprest)
endif()