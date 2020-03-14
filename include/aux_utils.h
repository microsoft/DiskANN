
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

#include "aligned_dtor.h"
#include "cached_io.h"
#include "common_includes.h"
#include "utils.h"
#include "windows_customizations.h"

namespace diskann {

  template<typename T>
  class PQFlashIndex;

  DISKANN_DLLEXPORT double calculate_recall(
      unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
      unsigned *our_results, unsigned dim_or, unsigned recall_at);

  DISKANN_DLLEXPORT void read_idmap(const std::string &    fname,
                                    std::vector<unsigned> &ivecs);

  DISKANN_DLLEXPORT int merge_shards(
      const std::string &nsg_prefix, const std::string &nsg_suffix,
      const std::string &idmaps_prefix, const std::string &idmaps_suffix,
      const _u64 nshards, const std::string &output_nsg, unsigned max_degree);

  template<typename T>
  DISKANN_DLLEXPORT int build_merged_vamana_index(
      std::string base_file, diskann::Metric _compareMetric, unsigned L,
      unsigned R, double sampling_rate, double ram_budget,
      std::string mem_index_path);

  template<typename T>
  DISKANN_DLLEXPORT uint32_t
  optimize_beamwidth(diskann::PQFlashIndex<T> &_pFlashIndex, T *tuning_sample,
                     _u64 tuning_sample_num, _u64 tuning_sample_aligned_dim,
                     uint32_t L, uint32_t start_bw = 2);

}  // namespace diskann
