#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT                        license.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES abstract_data_store.cpp ann_exception.cpp disk_utils.cpp 
        distance.cpp index.cpp in_mem_graph_store.cpp in_mem_data_store.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        in_mem_data_store.cpp in_mem_graph_store.cpp
        natural_number_set.cpp memory_mapper.cpp partition.cpp pq.cpp
        pq_flash_index.cpp scratch.cpp logger.cpp utils.cpp filter_utils.cpp index_factory.cpp abstract_index.cpp pq_l2_distance.cpp pq_data_store.cpp)
    list(APPEND CPP_SOURCES rabitq.cpp)
    list(APPEND CPP_SOURCES bf16_simd_kernels.cpp)
    list(APPEND CPP_SOURCES bf16_amx_kernels.cpp)

    include(CheckCXXCompilerFlag)

    # AVX-512 BF16 kernels: compile-time optional, runtime dispatched.
    # - DISKANN_AVX512BF16 (default ON): enable kernels when compiler supports required flags.
    # - DISKANN_FORCE_AVX512BF16 (default OFF): require kernels; fail configure if unsupported.
    option(DISKANN_AVX512BF16 "Enable AVX-512 BF16 kernels when supported by compiler" ON)
    option(DISKANN_FORCE_AVX512BF16 "Force AVX-512 BF16 kernels; fail if unsupported" OFF)

    check_cxx_compiler_flag("-mavx512bf16" DISKANN_HAS_MAVX512BF16)
    check_cxx_compiler_flag("-mavx512f" DISKANN_HAS_MAVX512F)

    set(DISKANN_ENABLE_AVX512BF16_KERNELS OFF)
    if (DISKANN_FORCE_AVX512BF16)
        if (NOT (DISKANN_HAS_MAVX512BF16 AND DISKANN_HAS_MAVX512F))
            message(FATAL_ERROR "DISKANN_FORCE_AVX512BF16=ON but compiler does not support -mavx512bf16 and -mavx512f")
        endif()
        set(DISKANN_ENABLE_AVX512BF16_KERNELS ON)
    elseif (DISKANN_AVX512BF16)
        if (DISKANN_HAS_MAVX512BF16 AND DISKANN_HAS_MAVX512F)
            set(DISKANN_ENABLE_AVX512BF16_KERNELS ON)
        endif()
    endif()

    if (DISKANN_ENABLE_AVX512BF16_KERNELS)
        set_source_files_properties(bf16_simd_kernels.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512bf16")
    endif()

    # AMX BF16 kernels: compile-time optional, runtime dispatched.
    # - DISKANN_AMXBF16 (default OFF): enable kernels when compiler supports required flags.
    # - DISKANN_FORCE_AMXBF16 (default OFF): require kernels; fail configure if unsupported.
    option(DISKANN_AMXBF16 "Enable AMX BF16 kernels when supported by compiler" OFF)
    option(DISKANN_FORCE_AMXBF16 "Force AMX BF16 kernels; fail if unsupported" OFF)

    check_cxx_compiler_flag("-mamx-tile" DISKANN_HAS_MAMX_TILE)
    check_cxx_compiler_flag("-mamx-bf16" DISKANN_HAS_MAMX_BF16)
    # Some builds may set global -march=native, which can implicitly enable AMX.
    # When DISKANN_AMXBF16=OFF, we still want to avoid emitting any AMX instructions.
    check_cxx_compiler_flag("-mno-amx-tile" DISKANN_HAS_MNO_AMX_TILE)
    check_cxx_compiler_flag("-mno-amx-bf16" DISKANN_HAS_MNO_AMX_BF16)

    set(DISKANN_ENABLE_AMXBF16_KERNELS OFF)
    if (DISKANN_FORCE_AMXBF16)
        if (NOT (DISKANN_HAS_MAMX_TILE AND DISKANN_HAS_MAMX_BF16))
            message(FATAL_ERROR "DISKANN_FORCE_AMXBF16=ON but compiler does not support -mamx-tile and -mamx-bf16")
        endif()
        set(DISKANN_ENABLE_AMXBF16_KERNELS ON)
    elseif (DISKANN_AMXBF16)
        if (DISKANN_HAS_MAMX_TILE AND DISKANN_HAS_MAMX_BF16)
            set(DISKANN_ENABLE_AMXBF16_KERNELS ON)
        endif()
    endif()

    if (DISKANN_ENABLE_AMXBF16_KERNELS)
        set_source_files_properties(bf16_amx_kernels.cpp PROPERTIES
            COMPILE_OPTIONS "-mamx-tile;-mamx-bf16")
    else()
        # Explicitly disable AMX for this translation unit to prevent accidental
        # enablement via global compiler flags such as -march=native.
        if (DISKANN_HAS_MNO_AMX_TILE AND DISKANN_HAS_MNO_AMX_BF16)
            set_source_files_properties(bf16_amx_kernels.cpp PROPERTIES
                COMPILE_OPTIONS "-mno-amx-tile;-mno-amx-bf16")
        endif()
    endif()
    if (RESTAPI)
        list(APPEND CPP_SOURCES restapi/search_wrapper.cpp restapi/server.cpp)
    endif()
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()

if (NOT MSVC)
    install(TARGETS ${PROJECT_NAME} LIBRARY)
endif()
