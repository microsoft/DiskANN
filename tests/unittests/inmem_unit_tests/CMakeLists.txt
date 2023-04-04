cmake_minimum_required(VERSION 3.15)
project(inmem_unit_tests)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


enable_testing()

add_executable(
  inmem_unit_tests
  inmem_unit_tests.cpp
)
target_link_libraries(
  inmem_unit_tests
  GTest::gtest_main
  ${DISKANN_ASYNC_LIB} 
  ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} 
  Boost::program_options
)


include(GoogleTest)
gtest_discover_tests(inmem_unit_tests)

