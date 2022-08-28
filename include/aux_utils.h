// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <algorithm>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#ifdef _WINDOWS
#include <Windows.h>
typedef HANDLE FileHandle;
#else
#include <unistd.h>
typedef int FileHandle;
#endif

#include "cached_io.h"
#include "common_includes.h"
#include "tsl/robin_set.h"

#include "utils.h"
#include "windows_customizations.h"

namespace diskann {
  const size_t   MAX_PQ_TRAINING_SET_SIZE = 256000;
  const size_t   MAX_SAMPLE_POINTS_FOR_WARMUP = 100000;
  const double   PQ_TRAINING_SET_FRACTION = 0.1;
  const double   SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
  const double   THRESHOLD_FOR_CACHING_IN_GB = 1.0;
  const uint32_t NUM_NODES_TO_CACHE = 250000;
  const uint32_t WARMUP_L = 20;
  const uint32_t NUM_KMEANS_REPS = 12;

  template<typename T>
  class PQFlashIndex;

  DISKANN_DLLEXPORT double get_memory_budget(const std::string &mem_budget_str);
  DISKANN_DLLEXPORT double get_memory_budget(double search_ram_budget_in_gb);
  DISKANN_DLLEXPORT void   add_new_file_to_single_index(std::string index_file,
                                                        std::string new_file);

  DISKANN_DLLEXPORT size_t calculate_num_pq_chunks(double final_index_ram_limit,
                                                   size_t points_num,
                                                   uint32_t dim);

  DISKANN_DLLEXPORT double calculate_recall(
      unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
      unsigned *our_results, unsigned dim_or, unsigned recall_at);

  DISKANN_DLLEXPORT double calculate_recall(
      unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
      unsigned *our_results, unsigned dim_or, unsigned recall_at,
      const tsl::robin_set<unsigned> &active_tags);

  DISKANN_DLLEXPORT double calculate_range_search_recall(
      unsigned num_queries, std::vector<std::vector<_u32>> &groundtruth,
      std::vector<std::vector<_u32>> &our_results);

  DISKANN_DLLEXPORT void read_idmap(const std::string &    fname,
                                    std::vector<unsigned> &ivecs);

#ifdef EXEC_ENV_OLS
  template<typename T>
  DISKANN_DLLEXPORT T *load_warmup(MemoryMappedFiles &files,
                                   const std::string &cache_warmup_file,
                                   uint64_t &warmup_num, uint64_t warmup_dim,
                                   uint64_t warmup_aligned_dim);
#else
  template<typename T>
  DISKANN_DLLEXPORT T *load_warmup(const std::string &cache_warmup_file,
                                   uint64_t &warmup_num, uint64_t warmup_dim,
                                   uint64_t warmup_aligned_dim);
#endif

  DISKANN_DLLEXPORT int merge_shards(const std::string &vamana_prefix,
                                     const std::string &vamana_suffix,
                                     const std::string &idmaps_prefix,
                                     const std::string &idmaps_suffix,
                                     const _u64 nshards, unsigned max_degree,
                                     const std::string &output_vamana,
                                     const std::string &medoids_file);

  template<typename T>
  DISKANN_DLLEXPORT std::string preprocess_base_file(
      const std::string &infile, const std::string &indexPrefix,
      diskann::Metric &distMetric);

  template<typename T>
  DISKANN_DLLEXPORT int build_merged_vamana_index(
      std::string base_file, diskann::Metric _compareMetric, unsigned L,
      unsigned R, double sampling_rate, double ram_budget,
      std::string mem_index_path, std::string medoids_file,
      std::string centroids_file);

  template<typename T>
  DISKANN_DLLEXPORT uint32_t optimize_beamwidth(
      std::unique_ptr<diskann::PQFlashIndex<T>> &_pFlashIndex, T *tuning_sample,
      _u64 tuning_sample_num, _u64 tuning_sample_aligned_dim, uint32_t L,
      uint32_t nthreads, uint32_t start_bw = 2);

  template<typename T>
  DISKANN_DLLEXPORT int build_disk_index(const char *    dataFilePath,
                                         const char *    indexFilePath,
                                         const char *    indexBuildParameters,
                                         diskann::Metric _compareMetric,
                                         bool            use_opq = false);

  template<typename T>
  DISKANN_DLLEXPORT void create_disk_layout(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file,
      const std::string reorder_data_file = std::string(""));

}  // namespace diskann
