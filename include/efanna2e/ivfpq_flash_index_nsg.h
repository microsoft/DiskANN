#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include "aligned_file_reader.h"
#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "pq_table.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "util.h"

#define MAX_N_CMPS 16384
#define SECTOR_LEN 4096
#define MAX_N_SECTOR_READS 16

namespace NSG {
  struct QueryScratch {
    _s8 *coord_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_CMPS * data_dim]
    _u64 coord_idx = 0;            // index of next [data_dim] scratch to use

    char *sector_scratch =
        nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

    float *aligned_scratch = nullptr;  // MUST BE AT LEAST [aligned_dim]
  };

  class IVFPQFlashNSG {
   public:
    IVFPQFlashNSG(Distance *dist_cmp, IVFPQTable *ivfpq_table);
    ~IVFPQFlashNSG();

    // obtain handle to nsg file
    void load(const char *nsg_file, const _u64 npts, const _u64 data_dim);

    // implemented
    void cache_bfs_levels(_u64 nlevels);

    // IGNORED -- possibly WRONG
    std::pair<int, int> beam_search(const float *query, const _u64 k_search,
                                    const _u64 l_search, _u32 *indices,
                                    const _u64  beam_width,
                                    QueryStats *stats = nullptr);

    // implemented -- CORRECT
    std::pair<int, int> cached_beam_search(const float *query,
                                           const _u64 k_search,
                                           const _u64 l_search, _u32 *indices,
                                           const _u64    beam_width,
                                           QueryStats *  stats = nullptr,
                                           QueryScratch *scratch = nullptr);
    AlignedFileReader reader;

    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

    // data info
    _u64 n_base = 0;
    _u64 data_dim = 0;
    _u64 aligned_dim = 0;

    // PQ converter (must be initialized outside)
    IVFPQTable *ivfpq_table = nullptr;
    // distance comparator
    Distance *dist_cmp = nullptr;

    // medoid/start info
    _u64 medoid = 0;
    std::pair<_u64, unsigned *> medoid_nhood;

    // nhood_cache
    unsigned *nhood_cache_buf = nullptr;
    tsl::robin_map<_u64, std::pair<_u64, unsigned *>> nhood_cache;

    // coord_cache
    _s8 *coord_cache_buf = nullptr;
    tsl::robin_map<_u64, _s8 *> coord_cache;
  };
}