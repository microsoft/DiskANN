set(CMAKE_CXX_STANDARD 14)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_sweep_index test_nsg_sweep_index.cpp)
target_link_libraries(test_nsg_sweep_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_twolevel_index test_nsg_twolevel_index.cpp)
target_link_libraries(test_nsg_twolevel_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_iterated_index test_nsg_iterated_index.cpp)
target_link_libraries(test_nsg_iterated_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_twolevel_search test_nsg_twolevel_search.cpp)
target_link_libraries(test_nsg_twolevel_search ${PROJECT_NAME} -ltcmalloc)

add_executable(truncate_efanna truncate_efanna.cpp)
target_link_libraries(truncate_efanna ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_flash_nsg_search test_flash_nsg_search.cpp)
target_link_libraries(test_flash_nsg_search ${PROJECT_NAME} aio)

add_executable(test_pq_flash_nsg_search test_pq_flash_nsg_search.cpp)
target_link_libraries(test_pq_flash_nsg_search ${PROJECT_NAME} aio)

add_executable(test_ivfpq_flash_nsg_search test_ivfpq_flash_nsg_search.cpp)
target_link_libraries(test_ivfpq_flash_nsg_search ${PROJECT_NAME} aio)

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)


add_executable(calc_recall calc_recall.cpp)
add_executable(gen_rand gen_rand.cpp)
add_executable(gen_grid gen_grid.cpp)
add_executable(opt_disk_graph opt_disk_graph.cpp)
add_executable(dedup_fvecs dedup_fvecs.cpp)
add_executable(embed_knn embed_knn.cpp)
add_executable(dedup_int8_bin dedup_int8_bin.cpp)
add_executable(fvecs_to_bin fvecs_to_bin.cpp)
add_executable(ivecs_to_bin_u8 ivecs_to_bin_u8.cpp)
add_executable(quantize quantize.cpp)
add_executable(read_fvecs read_fvecs.cpp)
add_executable(nsg_bfs nsg_bfs.cpp)
add_executable(rename_merge_shards rename_merge_shards.cpp)

# formatter
add_custom_command(TARGET calc_recall PRE_BUILD COMMAND clang-format-4.0 -i ../../include/efanna2e/*.h ../../src/*.cpp ../../tests/*.cpp)
