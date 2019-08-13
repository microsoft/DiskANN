add_library(nsg_lib SHARED dllmain.cpp nsg_interface.cpp ../partition_and_pq.cpp ../pq_flash_index_nsg.cpp ../windows_aligned_file_reader.cpp ../memory_mapper.cpp)


add_definitions(-D_USRDLL -D_WINDLL)
add_compile_options(/MD)
add_link_options(/DLL)
