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
#define MAX_N_SECTOR_READS 16

namespace NSG {
  class PQFlashNSG {
   public:
    PQFlashNSG(Distance *dist_cmp);
    ~PQFlashNSG();

    // load data, but obtain handle to nsg file
    void load(const char *data_bin, const char *nsg_file,
              const char *pq_tables_bin, const _u64 chunk_size,
              const _u64 n_chunks, const _u64 data_dim);

    // NOTE:: implemented
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
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

    // data info
    _u64 n_base = 0;
    _u64 data_dim = 0;
    _u64 aligned_dim = 0;

    // PQ data
    // n_chunks = # of chunks ndims is split into
    // data: _u8 * n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    _u8 *              data = nullptr;
    _u64               chunk_size;
    _u64               n_chunks;
    FixedChunkPQTable *pq_table;
    // distance comparator
    Distance *dist_cmp;

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