add_library(nsg_lib SHARED dllmain.cpp nsg_interface.cpp ../partition_and_pq.cpp ../pq_flash_index_nsg.cpp ../windows_aligned_file_reader.cpp ../memory_mapper.cpp ../index_nsg.cpp ../math_utils.cpp)
if (MSVC)
	add_definitions(-D_USRDLL -D_WINDLL)
	add_compile_options(/MD)
	target_link_options(nsg_lib PRIVATE /DLL /MACHINE:X64)
	target_link_options(nsg_lib PRIVATE $<$<CONFIG:Debug>:/IMPLIB:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_lib.lib>
										$<$<CONFIG:Release>:/IMPLIB:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_lib.lib>
					   )
endif()


