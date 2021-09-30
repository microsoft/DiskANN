// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "utils.h"
#include "windows_customizations.h"

#define MAX_N_CMPS 16384
#define SECTOR_LEN 4096
#define MAX_N_SECTOR_READS 128
#define MAX_PQ_CHUNKS 100

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
    // Gopal. Adapting to the new Bing interface. Since the DiskPriorityIO is
    // now a singleton, we have to take it in the DiskANNInterface and
    // pass it around. Since I don't want to pollute this interface with Bing
    // classes, this class takes an AlignedFileReader object that can be
    // created the way we need. Linux will create a simple AlignedFileReader
    // and pass it. Regular Windows code should create a BingFileReader using
    // the DiskPriorityIOInterface class, and for running on XTS, create a
    // BingFileReader
    // using the object passed by the XTS environment.
    // Freeing the reader object is now the client's (DiskANNInterface's)
    // responsibility.
    DISKANN_DLLEXPORT PQFlashIndex(
        std::shared_ptr<AlignedFileReader> &fileReader,
        diskann::Metric                     metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT ~PQFlashIndex();

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files,
                               uint32_t num_threads, const char *pq_prefix,
                               const char *disk_index_file);
#else
    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *pq_prefix,
                               const char *disk_index_file);
#endif

    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &node_list);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(
        MemoryMappedFiles &files, std::string sample_bin, _u64 l_search,
        _u64 beamwidth, _u64 num_nodes_to_cache, uint32_t nthreads,
        std::vector<uint32_t> &node_list);
#else
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(
        std::string sample_bin, _u64 l_search, _u64 beamwidth,
        _u64 num_nodes_to_cache, uint32_t num_threads,
        std::vector<uint32_t> &node_list);
#endif

    DISKANN_DLLEXPORT void cache_bfs_levels(_u64 num_nodes_to_cache,
                                            std::vector<uint32_t> &node_list);

    DISKANN_DLLEXPORT void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, QueryStats *stats = nullptr);


  DISKANN_DLLEXPORT _u32 range_search(const T *query1, const double range,
                                     const _u64          min_l_search,
                                     const _u64          max_l_search,
                                     std::vector<_u64> & indices,
                                     std::vector<float> &distances,
                                     const _u64          min_beam_width,
                                     QueryStats *        stats = nullptr);

    DISKANN_DLLEXPORT Metric get_metric();

    std::shared_ptr<AlignedFileReader> &reader;

   protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(_u64 nthreads);
    DISKANN_DLLEXPORT void destroy_thread_data();

   private:
    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

    diskann::Metric metric = diskann::Metric::L2;

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    float max_base_norm = 0;

    // data info
    _u64 num_points = 0;
    _u64 data_dim = 0;
    _u64 disk_data_dim = 0;  // will be different from data_dim only if we use
                             // PQ for disk data (very large dimensionality)
    _u64 aligned_dim = 0;
    _u64 disk_bytes_per_point = 0;

    std::string                        disk_index_file;
    std::vector<std::pair<_u32, _u32>> node_visit_counter;

    // PQ data
    // n_chunks = # of chunks ndims is split into
    // data: _u8 * n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    _u8 *             data = nullptr;
    _u64              n_chunks;
    FixedChunkPQTable pq_table;

    // distance comparator
    Distance<T> *    dist_cmp = nullptr;
    Distance<float> *dist_cmp_float = nullptr;

    // for very large datasets: we use PQ even for the disk resident index
    bool              use_disk_index_pq = false;
    _u64              disk_pq_n_chunks;
    FixedChunkPQTable disk_pq_table;

    // medoid/start info
    uint32_t *medoids =
        nullptr;         // by default it is just one entry point of graph, we
                         // can optionally have multiple starting points
    size_t num_medoids;  // by default it is set to 1
    float *centroid_data =
        nullptr;  // by default, it is empty. If there are multiple
                  // centroids, we pick the medoid corresponding to the
                  // closest centroid as the starting point of search

    // nhood_cache
    unsigned *                                    nhood_cache_buf = nullptr;
    tsl::robin_map<_u32, std::pair<_u32, _u32 *>> nhood_cache;

    // coord_cache
    T *                       coord_cache_buf = nullptr;
    tsl::robin_map<_u32, T *> coord_cache;

    // thread-specific scratch
    ConcurrentQueue<ThreadData<T>> thread_data;
    _u64                           max_nthreads;
    bool                           load_flag = false;
    bool                           count_visited_nodes = false;

#ifdef EXEC_ENV_OLS
    // Set to a larger value than the actual header to accommodate
    // any additions we make to the header. This is an outer limit
    // on how big the header can be.
    static const int HEADER_SIZE = 256;
    char *           getHeaderBytes();
#endif
  };
}  // namespace diskann
