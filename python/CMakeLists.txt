# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

execute_process(COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

execute_process(COMMAND python -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE _numpy_include
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)

# pybind11_add_module(diskannpy MODULE src/diskann_bindings.cpp)
# the following is fairly synonymous with pybind11_add_module, but we need more target_link_libraries
# see https://pybind11.readthedocs.io/en/latest/compiling.html#advanced-interface-library-targets for more details
add_library(diskannpy MODULE src/diskann_bindings.cpp)

target_link_libraries(
        diskannpy
        PRIVATE
        pybind11::module
        pybind11::lto
        pybind11::windows_extras
        ${PROJECT_NAME}
        ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS}
        ${DISKANN_ASYNC_LIB}
)

pybind11_extension(diskannpy)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(diskannpy)
endif()

set_target_properties(diskannpy PROPERTIES CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden")

target_compile_definitions(diskannpy PRIVATE VERSION_INFO="0.2.0")