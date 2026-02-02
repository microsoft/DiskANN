#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT                        license.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if(MSVC)
    add_subdirectory(dll)
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
        pq_l2_distance.cpp scratch.cpp utils.cpp)
    if (RESTAPI)
        list(APPEND CPP_SOURCES restapi/search_wrapper.cpp restapi/server.cpp)
    endif()
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()

if (NOT MSVC)
    install(TARGETS ${PROJECT_NAME} LIBRARY)
endif()
