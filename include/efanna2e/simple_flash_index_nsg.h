#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include "aligned_file_reader.h"
#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "util.h"

namespace NSG {
  template<typename T, typename NhoodType>
  class SimpleFlashNSG {
   public:

    SimpleFlashNSG(Distance *dist_cmp);
    ~SimpleFlashNSG();

    // load data, but obtain handle to nsg file
    void load(const char *data_file, const char *nsg_file);

    // implemented
    void cache_bfs_levels(_u64 nlevels);

    // implemented
    std::pair<int, int> beam_search(const float *query, const _u64 k_search,
                                    const _u64 l_search, _u32 *indices,
                                    const _u64  beam_width,
                                    QueryStats *stats = nullptr);

    // implemented
    std::pair<int, int> cached_beam_search(const float *query,
                                           const _u64 k_search,
                                           const _u64 l_search, _u32 *indices,
                                           const _u64  beam_width,
                                           QueryStats *stats = nullptr);
    AlignedFileReader reader;

    // index info
    _u64 *node_offsets = nullptr;
    _u64 *node_sizes = nullptr;

    // cache adjacency list for K-levels
    std::vector<_u32> *nbrs_cache = nullptr;

    // cache coords for K+1 levels
    float **coords_cache = nullptr;

    // data statics
    _u64  n_base = 0;
    _u64  data_dim = 0;
    float scale_factor = 1.0f;
    _u64  aligned_dim = 0;

    // distance comparator
    Distance *dist_cmp;

    // medoid/start info
    _u64      medoid = 0;
    NhoodType medoid_nhood;
  };
}
