
#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include "neighbor.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "util.h"

template<typename T>
void gen_random_slice(T *base_data, size_t points_num, size_t dim,
                      const char *outputfile, size_t slice_size);

template <typename T>
int partition(const char *base_file, const char *train_file, size_t num_centers,
              size_t max_k_means_reps, const char *prefix_dir, size_t k_base);


template<typename T>
int generate_pq_pivots(std::string train_file_path, size_t num_centers,
                       size_t num_pq_chunks, size_t max_k_means_reps,
                       std::string pq_pivots_path);

template<typename T>
int generate_pq_data_from_pivots(const T *base_data, size_t num_points,
                                 size_t dim, size_t num_centers,
                                 size_t      num_pq_chunks,
                                 std::string pq_pivots_path,
                                 std::string pq_compressed_vectors_path);
