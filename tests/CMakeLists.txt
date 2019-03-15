set(CMAKE_CXX_STANDARD 14)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_twolevel_index test_nsg_twolevel_index.cpp)
target_link_libraries(test_nsg_twolevel_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_twolevel_search test_nsg_twolevel_search.cpp)
target_link_libraries(test_nsg_twolevel_search ${PROJECT_NAME} -ltcmalloc)

add_executable(truncate_efanna truncate_efanna.cpp)
target_link_libraries(truncate_efanna ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_flash_nsg_search test_flash_nsg_search.cpp)
target_link_libraries(test_flash_nsg_search ${PROJECT_NAME} aio)

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

add_executable(calc_recall calc_recall.cpp)
add_executable(gen_rand gen_rand.cpp)
add_executable(gen_grid gen_grid.cpp)
add_executable(opt_disk_graph opt_disk_graph.cpp)
add_executable(verify_disk_opt verify_disk_opt.cpp)
add_executable(dedup_fvecs dedup_fvecs.cpp)
add_executable(dedup_int8_bin dedup_int8_bin.cpp)
add_executable(quantize quantize.cpp)
add_executable(iops_eater iops_eater.cpp)
target_link_libraries(iops_eater ${PROJECT_NAME} -ltcmalloc)
target_link_libraries(iops_eater ${PROJECT_NAME} aio)
add_executable(read_fvecs read_fvecs.cpp)
add_executable(nsg_bfs nsg_bfs.cpp)

# formatter
add_custom_command(TARGET calc_recall PRE_BUILD COMMAND clang-format-4.0 -i ../../include/efanna2e/*.h ../../src/*.cpp ../../tests/*.cpp)