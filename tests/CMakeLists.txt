# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

find_package(Boost COMPONENTS unit_test_framework)

# For Windows, fall back to nuget version if find_package didn't find it.
if (MSVC AND NOT Boost_FOUND)
    set(DISKANN_BOOST_INCLUDE "${DISKANN_MSVC_PACKAGES}/boost/lib/native/include")
    # Multi-threaded static library.
    set(UNIT_TEST_FRAMEWORK_LIB_PATTERN "${DISKANN_MSVC_PACKAGES}/boost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}/lib/native/libboost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}-mt-x64-*.lib")
    file(GLOB DISKANN_BOOST_UNIT_TEST_FRAMEWORK_LIB ${UNIT_TEST_FRAMEWORK_LIB_PATTERN})

    set(UNIT_TEST_FRAMEWORK_DLIB_PATTERN "${DISKANN_MSVC_PACKAGES}/boost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}/lib/native/libboost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}-mt-gd-x64-*.lib")
    file(GLOB DISKANN_BOOST_UNIT_TEST_FRAMEWORK_DLIB ${UNIT_TEST_FRAMEWORK_DLIB_PATTERN})

    if (EXISTS ${DISKANN_BOOST_INCLUDE} AND EXISTS ${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_LIB} AND EXISTS ${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_DLIB})
        set(Boost_FOUND ON)
        set(Boost_INCLUDE_DIR ${DISKANN_BOOST_INCLUDE})
        add_library(Boost::unit_test_framework STATIC IMPORTED)
        set_target_properties(Boost::unit_test_framework PROPERTIES IMPORTED_LOCATION_RELEASE "${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_LIB}")
        set_target_properties(Boost::unit_test_framework PROPERTIES IMPORTED_LOCATION_DEBUG "${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_DLIB}")
        message(STATUS "Falling back to using Boost from the nuget package")
    else()
        message(WARNING "Couldn't find Boost. Was looking for ${DISKANN_BOOST_INCLUDE} and ${UNIT_TEST_FRAMEWORK_LIB_PATTERN}")
    endif()
endif()

if (NOT Boost_FOUND)
    message(FATAL_ERROR "Couldn't find Boost dependency")
endif()


set(DISKANN_UNIT_TEST_SOURCES main.cpp index_write_parameters_builder_tests.cpp unified_index_tests.cpp)

# Link the tests against the static DiskANN core (diskann_s) rather than the
# DLL, so internal-only symbols don't need to be exported from the DLL just to
# satisfy the test linker. diskann_s carries DISKANN_STATIC_LIB (making
# DISKANN_DLLEXPORT a no-op) plus MKL + synchronization.lib transitively.
add_executable(${PROJECT_NAME}_unit_tests ${DISKANN_UNIT_TEST_SOURCES})
target_link_libraries(${PROJECT_NAME}_unit_tests ${PROJECT_NAME}_s ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::unit_test_framework)

# The static test exe still needs libiomp5md.dll / libtcmalloc_minimal.dll at
# runtime; those are copied to the output dir by the DLL target's POST_BUILD,
# so make sure the DLL is built first.
add_dependencies(${PROJECT_NAME}_unit_tests ${PROJECT_NAME})

add_test(NAME ${PROJECT_NAME}_unit_tests COMMAND ${PROJECT_NAME}_unit_tests)

# insert_fast lives entirely in the neighbor.h header, and its lower_bound
# dispatch is selected at compile time by USE_AVX512 / USE_AVX2 (else scalar
# binary search). To cover all three code paths we build the same test source
# three times with different SIMD defines. These targets only need the header,
# so they link Boost alone -- not the DiskANN library (which is compiled with a
# single project-wide arch and cannot have its defines overridden per target).
# Each variant is "name|space-separated defines|arch flag" (fields may be empty).
set(INSERT_FAST_SIMD_VARIANTS
    "avx512|USE_AVX512 USE_AVX2|/arch:AVX512"
    "avx2|USE_AVX2|/arch:AVX2"
    "scalar||")

foreach(variant ${INSERT_FAST_SIMD_VARIANTS})
    string(REPLACE "|" ";" variant_fields "${variant}")
    list(GET variant_fields 0 v_name)
    list(GET variant_fields 1 v_defs)
    list(GET variant_fields 2 v_arch)

    set(v_target ${PROJECT_NAME}_insert_fast_${v_name})
    add_executable(${v_target} main.cpp neighbor_insert_fast_tests.cpp)
    target_include_directories(${v_target} PRIVATE ${PROJECT_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
    target_link_libraries(${v_target} Boost::unit_test_framework)

    # On Windows the nuget Boost uses auto-link pragmas that request the
    # versioned .lib by bare name, so its lib dir must be on the link path.
    if(MSVC AND DEFINED DISKANN_MSVC_PACKAGES)
        target_link_directories(${v_target} PRIVATE
            "${DISKANN_MSVC_PACKAGES}/boost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}/lib/native")
    endif()

    if(v_defs)
        separate_arguments(v_defs_list NATIVE_COMMAND "${v_defs}")
        target_compile_definitions(${v_target} PRIVATE ${v_defs_list})
    endif()

    if(v_arch)
        if(MSVC)
            target_compile_options(${v_target} PRIVATE ${v_arch})
        else()
            # GCC/Clang: translate MSVC /arch flags to -m equivalents.
            if(v_arch STREQUAL "/arch:AVX512")
                target_compile_options(${v_target} PRIVATE -mavx512f -mavx512bw -mavx2 -mfma)
            elseif(v_arch STREQUAL "/arch:AVX2")
                target_compile_options(${v_target} PRIVATE -mavx2 -mfma)
            endif()
        endif()
    endif()

    add_test(NAME ${v_target} COMMAND ${v_target})
endforeach()

