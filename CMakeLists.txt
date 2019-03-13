cmake_minimum_required(VERSION 2.8)

project(efanna2e)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_STANDARD 14)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -mtune=native -ftree-vectorize")
add_compile_options(-lboost -DINFO -march=native -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -Wfatal-errors)

add_subdirectory(src)
add_subdirectory(tests)
