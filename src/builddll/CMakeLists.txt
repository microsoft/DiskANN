#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT                        license.
# Dedicated cmake for tcmalloc index build

add_definitions(-DRELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)

set(DISKANN_DLL_TCMALLOC_LINK_OPTIONS libtcmalloc_minimal_for_dll)
set(DISKANN_DLL_NAME "diskann_build_dll")

if (DISKANN_USE_STATIC_LIB)
    add_library(${DISKANN_DLL_NAME} STATIC ../abstract_data_store.cpp ../partition.cpp ../pq.cpp ../pq_flash_index.cpp ../logger.cpp ../utils.cpp
        ../windows_aligned_file_reader.cpp ../distance.cpp ../memory_mapper.cpp ../index.cpp 
        ../in_mem_data_store.cpp ../in_mem_graph_store.cpp ../math_utils.cpp ../disk_utils.cpp ../filter_utils.cpp
        ../ann_exception.cpp ../natural_number_set.cpp ../natural_number_map.cpp ../scratch.cpp ../index_factory.cpp ../abstract_index.cpp)
else()
    add_library(${DISKANN_DLL_NAME} SHARED dllmain.cpp ../abstract_data_store.cpp ../partition.cpp ../pq.cpp ../pq_flash_index.cpp ../logger.cpp ../utils.cpp
        ../windows_aligned_file_reader.cpp ../distance.cpp ../memory_mapper.cpp ../index.cpp 
        ../in_mem_data_store.cpp ../in_mem_graph_store.cpp ../math_utils.cpp ../disk_utils.cpp ../filter_utils.cpp
        ../ann_exception.cpp ../natural_number_set.cpp ../natural_number_map.cpp ../scratch.cpp ../index_factory.cpp ../abstract_index.cpp)
endif()

set(TARGET_DIR "$<$<CONFIG:Debug>:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}>$<$<CONFIG:Release>:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}>")

if (DISKANN_USE_STATIC_LIB)
    target_compile_definitions(${DISKANN_DLL_NAME} PRIVATE _NODLL)
else()
    set(DISKANN_DLL_IMPLIB "${TARGET_DIR}/${DISKANN_DLL_NAME}.lib")
    target_compile_definitions(${DISKANN_DLL_NAME} PRIVATE _USRDLL _WINDLL)
endif()

target_compile_options(${DISKANN_DLL_NAME} PRIVATE /GL)
target_include_directories(${DISKANN_DLL_NAME} PRIVATE ${DISKANN_MKL_INCLUDE_DIRECTORIES})

if (NOT DEFINED DISKANN_USE_STATIC_LIB)
    target_link_options(${DISKANN_DLL_NAME} PRIVATE /DLL /IMPLIB:${DISKANN_DLL_IMPLIB} /LTCG)
endif()

target_link_libraries(${DISKANN_DLL_NAME} PRIVATE ${DISKANN_MKL_LINK_LIBRARIES})
target_link_libraries(${DISKANN_DLL_NAME} PRIVATE synchronization.lib)

target_link_libraries(${DISKANN_DLL_NAME} PUBLIC ${DISKANN_DLL_TCMALLOC_LINK_OPTIONS})

# Copy OpenMP DLL and PDB.
set(RUNTIME_FILES_TO_COPY ${OPENMP_WINDOWS_RUNTIME_FILES} ${TCMALLOC_WINDOWS_RUNTIME_FILES})

foreach(RUNTIME_FILE ${RUNTIME_FILES_TO_COPY})
    add_custom_command(TARGET ${DISKANN_DLL_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${RUNTIME_FILE}" "${TARGET_DIR}")
endforeach()