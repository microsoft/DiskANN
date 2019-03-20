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
  class FlashNSG {
   public:
    FlashNSG(Distance *dist_cmp);
    ~FlashNSG();

    // load data, but obtain handle to nsg file
    void load(const char *data_bin, const char *nsg_file);

    // NOTE:: not implemented
    void cache_bfs_levels(_u64 nlevels);

    // implemented
    std::pair<int, int> beam_search(const float *query, const _u64 k_search,
                                    const _u64 l_search, _u32 *indices,
                                    const _u64  beam_width,
                                    QueryStats *stats = nullptr);

    // not implemented
    std::pair<int, int> cached_beam_search(const float *query,
                                           const _u64 k_search,
                                           const _u64 l_search, _u32 *indices,
                                           const _u64  beam_width,
                                           QueryStats *stats = nullptr);
    AlignedFileReader reader;

    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

    // data info
    _u64    n_base = 0;
    _u64    data_dim = 0;
    _u64    aligned_dim = 0;
    int8_t *data = nullptr;

    // distance comparator
    Distance *dist_cmp;

    // medoid/start info
    _u64 medoid = 0;
    std::pair<_u64, unsigned *> medoid_nhood;

    // cache
    tsl::robin_map<_u64, std::pair<_u64, unsigned *>> nhood_cache;
  };
}
