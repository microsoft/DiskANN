cmake_minimum_required(VERSION 2.8)

project(rand-nsg)
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


#MKL Config
set(INTEL_ROOT /opt/intel/compilers_and_libraries/linux)
set(MKL_ROOT ${INTEL_ROOT}/mkl)
add_definitions(-DMKL_ILP64)
add_compile_options(-m64 -Wl,--no-as-needed)
include_directories(include ${INTEL_ROOT}/include ${MKL_ROOT}/include)
link_directories(${INTEL_ROOT}/lib/intel64 ${MKL_ROOT}/lib/intel64)
link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)

set(ENV{TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD} 500000000000)

 set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined")
 #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG -march=native -mtune=native -ftree-vectorize")
add_compile_options(-march=native -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_ACCELERATED_PQ -DUSE_AVX2)
#add_compile_options(-march=native -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_ACCELERATED_PQ)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)
