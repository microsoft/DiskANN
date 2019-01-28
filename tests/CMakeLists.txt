set(CMAKE_CXX_STANDARD 14)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

add_executable(calc_recall calc_recall.cpp)
add_executable(gen_rand gen_rand.cpp)
add_executable(gen_grid gen_grid.cpp)
add_executable(opt_disk_graph opt_disk_graph.cpp)
add_executable(read_fvecs read_fvecs.cpp)
