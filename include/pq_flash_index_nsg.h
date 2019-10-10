#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "windows_customizations.h"

#define MAX_N_CMPS 16384
#define SECTOR_LEN 4096
#define MAX_N_SECTOR_READS 16

namespace NSG {
  template<typename T>
  struct QueryScratch {
    T *  coord_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_CMPS * data_dim]
    _u64 coord_idx = 0;            // index of next [data_dim] scratch to use

    char *sector_scratch =
        nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

    float *aligned_scratch = nullptr;  // MUST BE AT LEAST [aligned_dim]
    float *aligned_pqtable_dist_scratch =
        nullptr;                            // MUST BE AT LEAST [256 * NCHUNKS]
    float *aligned_dist_scratch = nullptr;  // MUST BE AT LEAST NSG MAX_DEGREE
    _u8 *  aligned_pq_coord_scratch =
        nullptr;  // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]

    void reset() {
      coord_idx = 0;
      sector_idx = 0;
    }
  };

  template<typename T>
  struct ThreadData {
    QueryScratch<T> scratch;
    IOContext       ctx;
  };

  template<typename T>
  class PQFlashNSG {
   public:
    NSGDLLEXPORT PQFlashNSG();
    NSGDLLEXPORT ~PQFlashNSG();

    // load compressed data, and obtains the handle to the disk-resident index
    NSGDLLEXPORT int load(uint32_t num_threads, const char *pq_centroids_bin,
                          const char *compressed_data_bin,
                          const char *disk_index_file,
                          const char *medoids_file = {0});

    NSGDLLEXPORT void create_disk_layout(const std::string base_file,
                                         const std::string mem_index_file,
                                         const std::string output_file);
    NSGDLLEXPORT void cache_visited_nodes(_u64 *node_list, _u64 num_nodes);
    NSGDLLEXPORT void load_cache_from_file(std::string cache_bin);
    NSGDLLEXPORT void cache_bfs_levels(_u64 nlevels);

    NSGDLLEXPORT void set_cache_create_flag();

    NSGDLLEXPORT void save_cached_nodes(_u64        num_nodes,
                                        std::string cache_file_path);

    // setting up thread-specific data
    NSGDLLEXPORT void setup_thread_data(_u64 nthreads);
    NSGDLLEXPORT void destroy_thread_data();

    // implemented
    NSGDLLEXPORT void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, QueryStats *stats = nullptr,
        Distance<T> *output_dist_func = nullptr);
    AlignedFileReader *reader;

    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

    // data info
    _u64 num_points = 0;
    _u64 data_dim = 0;
    _u64 aligned_dim = 0;

    std::vector<std::pair<_u64, _u32>> node_visit_counter;
    bool create_visit_cache;

    // PQ data
    // n_chunks = # of chunks ndims is split into
    // data: _u8 * n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    _u8 *                data = nullptr;
    _u64                 chunk_size;
    _u64                 n_chunks;
    FixedChunkPQTable<T> pq_table;

    // distance comparator
    Distance<T> *dist_cmp;

    // medoid/start info
    uint32_t *medoids;
    std::vector<std::pair<_u64, unsigned *>> medoid_nhoods;
    size_t num_medoids;
    T *    medoid_full_precs;

    // nhood_cache
    unsigned *nhood_cache_buf = nullptr;
    tsl::robin_map<_u64, std::pair<_u64, unsigned *>> nhood_cache;

    // coord_cache
    T *coord_cache_buf = nullptr;
    tsl::robin_map<_u64, T *> coord_cache;

    // thread-specific scratch
    ConcurrentQueue<ThreadData<T>> thread_data;
    _u64                           max_nthreads;
    bool                           load_flag = false;
  };
}  // namespace NSG
