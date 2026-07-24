#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT                        license.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if(MSVC)
    add_subdirectory(dll)

    # Static-library variant of the DiskANN core, built for the unit tests so
    # they can link the internal symbols directly instead of going through the
    # DLL's export table. Compiled with DISKANN_STATIC_LIB (PUBLIC, so consumers
    # see it too) which makes DISKANN_DLLEXPORT a no-op -- see
    # include/windows_customizations.h. This lets internal-only types drop their
    # DISKANN_DLLEXPORT annotations without breaking the tests.
    set(DISKANN_STATIC_SOURCES
        abstract_data_store.cpp abstract_index.cpp ann_exception.cpp
        color_helper.cpp color_info.cpp disk_utils.cpp distance.cpp
        filter_match_proxy.cpp filter_utils.cpp
        in_mem_data_store.cpp in_mem_graph_reformat_store.cpp in_mem_graph_store.cpp
        in_mem_reorder_data_store.cpp in_mem_static_graph_reformat.cpp in_mem_static_graph_store.cpp
        index.cpp index_factory.cpp integer_label_vector.cpp
        label_bitmask.cpp label_helper.cpp logger.cpp
        math_utils.cpp memory_mapper.cpp natural_number_map.cpp natural_number_set.cpp
        neighbor_list.cpp partition.cpp pq.cpp pq_data_store.cpp pq_flash_index.cpp
        pq_l2_distance.cpp scratch.cpp unified_index.cpp unified_index_builder.cpp unified_index_io.cpp
        unified_label_data.cpp unified_node_store.cpp
        unified_index_base.cpp unified_index_memory.cpp unified_index_ssd.cpp
        utils.cpp windows_aligned_file_reader.cpp)

    add_library(${PROJECT_NAME}_s STATIC ${DISKANN_STATIC_SOURCES})
    target_compile_definitions(${PROJECT_NAME}_s PUBLIC DISKANN_STATIC_LIB)
    # index.cpp exceeds the COFF section limit without /GL (which the DLL uses);
    # /bigobj lifts it for the static build.
    target_compile_options(${PROJECT_NAME}_s PRIVATE /bigobj)
    target_include_directories(${PROJECT_NAME}_s PRIVATE ${DISKANN_MKL_INCLUDE_DIRECTORIES})
    # MKL + synchronization.lib are PRIVATE to the DLL; the static lib exposes
    # them PUBLICly so anything linking diskann_s (the tests) inherits them.
    target_link_libraries(${PROJECT_NAME}_s PUBLIC ${DISKANN_MKL_LINK_LIBRARIES} synchronization.lib)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES abstract_data_store.cpp abstract_index.cpp ann_exception.cpp
        color_helper.cpp color_info.cpp disk_utils.cpp distance.cpp
        filter_match_proxy.cpp filter_utils.cpp
        in_mem_data_store.cpp in_mem_graph_reformat_store.cpp in_mem_graph_store.cpp
        in_mem_reorder_data_store.cpp in_mem_static_graph_reformat.cpp in_mem_static_graph_store.cpp
        index.cpp index_factory.cpp integer_label_vector.cpp
        label_bitmask.cpp label_helper.cpp linux_aligned_file_reader.cpp logger.cpp
        math_utils.cpp memory_mapper.cpp natural_number_map.cpp natural_number_set.cpp
        neighbor_list.cpp partition.cpp pq.cpp pq_data_store.cpp pq_flash_index.cpp
        pq_l2_distance.cpp scratch.cpp unified_index.cpp unified_index_builder.cpp unified_index_io.cpp
        unified_label_data.cpp unified_node_store.cpp
        unified_index_base.cpp unified_index_memory.cpp unified_index_ssd.cpp utils.cpp)
    if (RESTAPI)
        list(APPEND CPP_SOURCES restapi/search_wrapper.cpp restapi/server.cpp)
    endif()
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()

if (NOT MSVC)
    install(TARGETS ${PROJECT_NAME} LIBRARY)
endif()
