# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

add_library(${PROJECT_NAME} SHARED dllmain.cpp ../partition_and_pq.cpp ../pq_flash_index.cpp ../logger.cpp ../utils.cpp 
    ../windows_aligned_file_reader.cpp ../distance.cpp ../memory_mapper.cpp ../index.cpp ../math_utils.cpp ../aux_utils.cpp
    ../ann_exception.cpp ../natural_number_set.cpp ../natural_number_map.cpp)

set(TARGET_DIR "$<$<CONFIG:Debug>:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}>$<$<CONFIG:Release>:${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}>")
set(DISKANN_DLL_IMPLIB "${TARGET_DIR}/${PROJECT_NAME}.lib")

target_compile_definitions(${PROJECT_NAME} PRIVATE _USRDLL _WINDLL)
target_compile_options(${PROJECT_NAME} PRIVATE /MD /GL)
target_include_directories(${PROJECT_NAME} PRIVATE ${DISKANN_MKL_INCLUDE_DIRECTORIES})

target_link_options(${PROJECT_NAME} PRIVATE /DLL /IMPLIB:${DISKANN_DLL_IMPLIB} /LTCG)
target_link_libraries(${PROJECT_NAME} PRIVATE ${DISKANN_MKL_LINK_LIBRARIES})

if (DISKANN_DLL_TCMALLOC_LINK_OPTIONS)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${DISKANN_DLL_TCMALLOC_LINK_OPTIONS})
endif()

# Copy OpenMP DLL and PDB.
set(RUNTIME_FILES_TO_COPY ${OPENMP_WINDOWS_RUNTIME_FILES} ${TCMALLOC_WINDOWS_RUNTIME_FILES})

foreach(RUNTIME_FILE ${RUNTIME_FILES_TO_COPY})
    add_custom_command(TARGET ${PROJECT_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${RUNTIME_FILE}" "${TARGET_DIR}")
endforeach()
