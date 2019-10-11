if (MSVC)
	add_library(diskpriority_io STATIC DiskPriorityIO.cpp)
endif()

add_library(nsg_dll SHARED dllmain.cpp nsg_interface.cpp bing_aligned_file_reader.cpp ../partition_and_pq.cpp ../pq_flash_index_nsg.cpp ../windows_aligned_file_reader.cpp ../memory_mapper.cpp ../index_nsg.cpp ../math_utils.cpp)
if (MSVC)
	add_definitions(-D_USRDLL -D_WINDLL)
	add_compile_options(/MD)
	target_link_options(nsg_dll PRIVATE /DLL /MACHINE:X64)
	target_link_options(nsg_dll PRIVATE $<$<CONFIG:Debug>:/IMPLIB:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/nsg_dll.lib>
										$<$<CONFIG:Release>:/IMPLIB:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/nsg_dll.lib>
					   )
	target_link_libraries(nsg_dll debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskpriority_io.lib)
	target_link_libraries(nsg_dll optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskpriority_io.lib)

					

	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${INTEL_ROOT}redist/intel64/compiler/libiomp5md.dll "$<$<CONFIG:debug>:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}>$<$<CONFIG:release>:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}>" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${INTEL_ROOT}redist/intel64/mkl/mkl_avx.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${INTEL_ROOT}redist/intel64/mkl/mkl_avx2.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${INTEL_ROOT}redist/intel64/mkl/mkl_avx512.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${INTEL_ROOT}redist/intel64/mkl/mkl_core.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${INTEL_ROOT}redist/intel64/mkl/mkl_def.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${INTEL_ROOT}redist/intel64/mkl/mkl_intel_thread.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${INTEL_ROOT}redist/intel64/mkl/mkl_rt.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/dependencies/windows/dll/boost_date_time-vc141-mt-x64-1_70.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/dependencies/windows/dll/cpprest_2_10.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/dependencies/windows/dll/LIBEAY32.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/dependencies/windows/dll/SSLEAY32.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )
	add_custom_command(TARGET 
							nsg_dll 
							POST_BUILD 
							COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/dependencies/windows/dll/zlib1.dll "$<$<CONFIG:debug>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}\">$<$<CONFIG:release>:\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}\">" )

endif()



