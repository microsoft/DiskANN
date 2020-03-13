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
#define MAX_N_SECTOR_READS 128

namespace diskann {
  template<typename T>
  struct QueryScratch {
    T *  coord_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_CMPS * data_dim]
    _u64 coord_idx = 0;            // index of next [data_dim] scratch to use

    char *sector_scratch =
        nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

    float *aligned_scratch = nullptr;  // MUST BE AT LEAST [aligned_dim]
    float *aligned_pqtable_dist_scratch =
        nullptr;  // MUST BE AT LEAST [256 * NCHUNKS]
    float *aligned_dist_scratch =
        nullptr;  // MUST BE AT LEAST diskann MAX_DEGREE
    _u8 *aligned_pq_coord_scratch =
        nullptr;  // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]
    T *    aligned_query_T = nullptr;
    float *aligned_query_float = nullptr;

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
  class PQFlashIndex {
   public:
    DISKANN_DLLEXPORT PQFlashIndex();
    DISKANN_DLLEXPORT ~PQFlashIndex();

    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int load(uint32_t    num_threads,
                               const char *pq_centroids_bin,
                               const char *compressed_data_bin,
                               const char *disk_index_file);

    DISKANN_DLLEXPORT void load_entry_points(
        const std::string entry_points_file, const std::string centroids_file);

    DISKANN_DLLEXPORT void cache_medoid_nhoods();

    DISKANN_DLLEXPORT void create_disk_layout(const std::string base_file,
                                              const std::string mem_index_file,
                                              const std::string output_file);
    DISKANN_DLLEXPORT void load_cache_from_file(std::string cache_bin);

    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &node_list);

    DISKANN_DLLEXPORT void cache_bfs_levels(_u64 num_nodes_to_cache,
                                            std::vector<uint32_t> &node_list);

    DISKANN_DLLEXPORT void set_cache_create_flag();

    DISKANN_DLLEXPORT void save_cached_nodes(_u64        num_nodes,
                                             std::string cache_file_path);

    // setting up thread-specific data
    DISKANN_DLLEXPORT void setup_thread_data(_u64 nthreads);
    DISKANN_DLLEXPORT void destroy_thread_data();

    // implemented
    DISKANN_DLLEXPORT void cached_beam_search(
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

    std::string disk_index_file;
    std::vector<std::pair<_u32, _u32>> node_visit_counter;
    bool create_visit_cache = false;

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
    Distance<T> *    dist_cmp = nullptr;
    Distance<float> *dist_cmp_float = nullptr;

    // medoid/start info
    uint32_t *medoids =
        nullptr;         // by default it is just one entry point of graph, we
                         // can optionally have multiple starting points
    size_t num_medoids;  // by default it is set to 1
    float *centroid_data =
        nullptr;  // by default, it is empty. If there are multiple
                  // centroids, we pick the medoid corresponding to the
                  // closest centroid as the starting point of search
    bool using_default_medoid_data = true;

    std::vector<std::pair<_u32, unsigned *>> medoid_nhoods;

    // nhood_cache
    unsigned *nhood_cache_buf = nullptr;
    tsl::robin_map<_u32, std::pair<_u32, _u32 *>> nhood_cache;

    // coord_cache
    T *coord_cache_buf = nullptr;
    tsl::robin_map<_u32, T *> coord_cache;

    // thread-specific scratch
    ConcurrentQueue<ThreadData<T>> thread_data;
    _u64                           max_nthreads;
    bool                           load_flag = false;
  };
}  // namespace diskann
