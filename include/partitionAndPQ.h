
#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include "efanna2e/index.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "tsl/robin_set.h"
#include "efanna2e/util.h"

void gen_random_slice(float *base_data, size_t points_num, size_t dim,
                      const char *outputfile, size_t slice_size);
int partition(const char *base_file, const char *train_file, size_t num_centers,
              size_t max_k_means_reps, const char *prefix_dir, size_t k_base);
int generate_pq_pivots(const char *train_file, size_t num_centers,
                       size_t num_chunks, size_t max_k_means_reps,
                       const char *working_prefix_file);
int generate_pq_data_from_pivots(const char *base_file, size_t num_centers,
                                 size_t      num_chunks,
                                 const char *working_prefix_file);
