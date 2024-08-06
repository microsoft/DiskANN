#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT                        license.

if (DISKANN_USE_STATIC_LIB)
    add_library(${PROJECT_NAME} STATIC ../abstract_data_store.cpp ../partition.cpp ../pq.cpp ../pq_flash_index.cpp ../logger.cpp ../utils.cpp
        ../windows_aligned_file_reader.cpp ../distance.cpp ../memory_mapper.cpp ../index.cpp 
        ../in_mem_data_store.cpp ../in_mem_graph_store.cpp ../math_utils.cpp ../disk_utils.cpp ../filter_utils.cpp
        ../ann_exception.cpp ../natural_number_set.cpp ../natural_number_map.cpp ../scratch.cpp ../index_factory.cpp ../abstract_index.cpp)

    # tcmalloc verison
    add_library(${PROJECT_NAME}_build STATIC ../abstract_data_store.cpp ../partition.cpp ../pq.cpp ../pq_flash_index.cpp ../logger.cpp ../utils.cpp
        ../windows_aligned_file_reader.cpp ../distance.cpp ../memory_mapper.cpp ../index.cpp ../../gperftools/src/gperftools/malloc_extension.h 
        ../in_mem_data_store.cpp ../in_mem_graph_store.cpp ../math_utils.cpp ../disk_utils.cpp ../filter_utils.cpp
        ../ann_exception.cpp ../natural_number_set.cpp ../natural_number_map.cpp ../scratch.cpp ../index_factory.cpp ../abstract_index.cpp)
else()
    add_library(${PROJECT_NAME} SHARED dllmain.cpp ../abstract_data_store.cpp ../partition.cpp ../pq.cpp ../pq_flash_index.cpp ../logger.cpp ../utils.cpp
        ../windows_aligned_file_reader.cpp ../distance.cpp ../memory_mapper.cpp ../index.cpp 
        ../in_mem_data_store.cpp ../in_mem_graph_store.cpp ../math_utils.cpp ../disk_utils.cpp ../filter_utils.cpp
        ../ann_exception.cpp ../natural_number_set.cpp ../natural_number_map.cpp ../scratch.cpp ../index_factory.cpp ../abstract_index.cpp)
    
    # tcmalloc version
    add_library(${PROJECT_NAME}_build SHARED dllmain.cpp ../abstract_data_store.cpp ../partition.cpp ../pq.cpp ../pq_flash_index.cpp ../logger.cpp ../utils.cpp
        ../windows_aligned_file_reader.cpp ../distance.cpp ../memory_mapper.cpp ../index.cpp ../../gperftools/src/gperftools/malloc_extension.h 
        ../in_mem_data_store.cpp ../in_mem_graph_store.cpp ../math_utils.cpp ../disk_utils.cpp ../filter_utils.cpp
        ../ann_exception.cpp ../natural_number_set.cpp ../natural_number_map.cpp ../scratch.cpp ../index_factory.cpp ../abstract_index.cpp)
endif()

set(TARGET_DIR "$<$<CONFIG:Debug>:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}>$<$<CONFIG:Release>:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}>")

if (DISKANN_USE_STATIC_LIB)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _NODLL)

    target_compile_definitions(${PROJECT_NAME}_build PRIVATE _NODLL DISKANN_BUILD RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
else()
    set(DISKANN_DLL_IMPLIB "${TARGET_DIR}/${PROJECT_NAME}.lib")
    target_compile_definitions(${PROJECT_NAME} PRIVATE _USRDLL _WINDLL)

    set(DISKANN_DLL_IMPLIB_BUILD "${TARGET_DIR}/${PROJECT_NAME}_build.lib")
    target_compile_definitions(${PROJECT_NAME}_build PRIVATE _USRDLL _WINDLL DISKANN_BUILD RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
endif()

target_compile_options(${PROJECT_NAME} PRIVATE /GL)
target_include_directories(${PROJECT_NAME} PRIVATE ${DISKANN_MKL_INCLUDE_DIRECTORIES})

target_compile_options(${PROJECT_NAME}_build PRIVATE /GL)
target_include_directories(${PROJECT_NAME}_build PRIVATE ${DISKANN_MKL_INCLUDE_DIRECTORIES} ../../gperftools/src/)


if (NOT DEFINED DISKANN_USE_STATIC_LIB)
    target_link_options(${PROJECT_NAME} PRIVATE /DLL /IMPLIB:${DISKANN_DLL_IMPLIB} /LTCG)
    target_link_options(${PROJECT_NAME}_build PRIVATE /DLL /IMPLIB:${DISKANN_DLL_IMPLIB_BUILD} /LTCG)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${DISKANN_MKL_LINK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE synchronization.lib)

target_link_libraries(${PROJECT_NAME}_build PRIVATE ${DISKANN_MKL_LINK_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_build PRIVATE synchronization.lib)
#This is the crux of the build dll
target_link_libraries(${PROJECT_NAME}_build PUBLIC ${DISKANN_DLL_TCMALLOC_LINK_OPTIONS})
target_link_libraries(${PROJECT_NAME}_build PRIVATE ${TCMALLOC_LINK_LIBRARY})
set_target_properties(${PROJECT_NAME}_build PROPERTIES LINK_FLAGS /INCLUDE:_tcmalloc)

# Copy OpenMP DLL and PDB.
set(RUNTIME_FILES_TO_COPY ${OPENMP_WINDOWS_RUNTIME_FILES} ${TCMALLOC_WINDOWS_RUNTIME_FILES})

foreach(RUNTIME_FILE ${RUNTIME_FILES_TO_COPY})
    add_custom_command(TARGET ${PROJECT_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${RUNTIME_FILE}" "${TARGET_DIR}")
    add_custom_command(TARGET ${PROJECT_NAME}_build
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${RUNTIME_FILE}" "${TARGET_DIR}")

endforeach()