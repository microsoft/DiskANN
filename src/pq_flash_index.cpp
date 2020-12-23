// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "logger.h"
#include "pq_flash_index.h"
#include <malloc.h>
#include "percentile_stats.h"

#include <omp.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iterator>
#include <thread>
#include "distance.h"
#include "exceptions.h"
#include "parameters.h"
#include "timer.h"
#include "utils.h"

#include "tsl/robin_set.h"

#ifdef _WINDOWS
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#else
#include "linux_aligned_file_reader.h"
#endif

#define SECTOR_LEN 4096

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

// sector # on disk where node_id is present
#define NODE_SECTOR_NO(node_id) (((_u64)(node_id)) / nnodes_per_sector + 1)

// obtains region of sector containing node
#define OFFSET_TO_NODE(sector_buf, node_id) \
  ((char *) sector_buf + (((_u64) node_id) % nnodes_per_sector) * max_node_len)

// offset into sector where node_id's nhood starts
#define NODE_SECTOR_OFFSET(sector_buf, node_id) \
  ((char *) sector_buf +                        \
   ((((_u64) node_id) % nnodes_per_sector) * max_node_len))

// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + data_dim * sizeof(T))

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *) (node_buf)

namespace {
  void aggregate_coords(const unsigned *ids, const _u64 n_ids,
                        const _u8 *all_coords, const _u64 ndims, _u8 *out) {
    for (_u64 i = 0; i < n_ids; i++) {
      memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(_u8));
    }
  }

  void pq_dist_lookup(const _u8 *pq_ids, const _u64 n_pts,
                      const _u64 pq_nchunks, const float *pq_dists,
                      float *dists_out) {
    _mm_prefetch((char *) dists_out, _MM_HINT_T0);
    _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
      const float *chunk_dists = pq_dists + 256 * chunk;
      if (chunk < pq_nchunks - 1) {
        _mm_prefetch((char *) (chunk_dists + 256), _MM_HINT_T0);
      }
      for (_u64 idx = 0; idx < n_pts; idx++) {
        _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
        dists_out[idx] += chunk_dists[pq_centerid];
      }
    }
  }
}  // namespace

namespace diskann {
  template<>
  PQFlashIndex<_u8>::PQFlashIndex(
      std::shared_ptr<AlignedFileReader> &fileReader)
      : reader(fileReader) {
    diskann::cout
        << "dist_cmp function for _u8 uses slow implementation."
           " Please contact gopalsr@microsoft.com if you need an AVX/AVX2"
           " implementation."
        << std::endl;
    // TODO: No AVX2/AVX implementation available for uint8.
    this->dist_cmp = new DistanceL2UInt8();
    if (Avx2SupportedCPU) {
      diskann::cout << "Using AVX2 dist_cmp_float function." << std::endl;
      this->dist_cmp_float = new DistanceL2();
    } else if (AvxSupportedCPU) {
      diskann::cout << "Using AVX dist_cmp_float function" << std::endl;
      this->dist_cmp_float = new AVXDistanceL2Float();
    } else {
      diskann::cout << "No AVX/AVX2 support. Using Slow dist_cmp_float function"
                    << std::endl;
      this->dist_cmp_float = new SlowDistanceL2Float();
    }
  }

  template<>
  PQFlashIndex<_s8>::PQFlashIndex(
      std::shared_ptr<AlignedFileReader> &fileReader)
      : reader(fileReader) {
    if (Avx2SupportedCPU) {
      diskann::cout << "Using AVX2 function for dist_cmp and dist_cmp_float"
                    << std::endl;
      this->dist_cmp = new DistanceL2Int8();
      this->dist_cmp_float = new DistanceL2();
    } else if (AvxSupportedCPU) {
      diskann::cout << "No AVX2 support. Switching to AVX routines for "
                       "dist_cmp, dist_cmp_float."
                    << std::endl;
      this->dist_cmp = new AVXDistanceL2Int8();
      this->dist_cmp_float = new AVXDistanceL2Float();
    } else {
      diskann::cout << "No AVX/AVX2 support. Switching to slow routines for "
                       "dist_cmp, dist_cmp_float"
                    << std::endl;
      this->dist_cmp = new SlowDistanceL2Int<int8_t>();
      this->dist_cmp_float = new SlowDistanceL2Float();
    }
  }

  template<>
  PQFlashIndex<float>::PQFlashIndex(
      std::shared_ptr<AlignedFileReader> &fileReader)
      : reader(fileReader) {
    if (Avx2SupportedCPU) {
      diskann::cout << "Using AVX2 functions for dist_cmp and dist_cmp_float"
                    << std::endl;
      this->dist_cmp = new DistanceL2();
      this->dist_cmp_float = new DistanceL2();
    } else if (AvxSupportedCPU) {
      diskann::cout << "No AVX2 support. Switching to AVX functions for "
                       "dist_cmp and dist_cmp_float."
                    << std::endl;
      this->dist_cmp = new AVXDistanceL2Float();
      this->dist_cmp_float = new AVXDistanceL2Float();
    } else {
      diskann::cout << "No AVX/AVX2 support. Switching to slow implementations "
                       "for dist_cmp and dist_cmp_float"
                    << std::endl;
      this->dist_cmp = new AVXDistanceL2Float();
      this->dist_cmp_float = new AVXDistanceL2Float();
    }
  }

  template<typename T>
  PQFlashIndex<T>::~PQFlashIndex() {
#ifndef EXEC_ENV_OLS
    if (data != nullptr) {
      delete[] data;
    }
#endif

    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    // delete backing bufs for nhood and coord cache
    if (nhood_cache_buf != nullptr) {
      delete[] nhood_cache_buf;
      diskann::aligned_free(coord_cache_buf);
    }

    delete this->dist_cmp;
    delete this->dist_cmp_float;
    if (load_flag) {
      this->destroy_thread_data();
      reader->close();
      // delete reader; //not deleting reader because it is now passed by ref.
    }
  }

  template<typename T>
  void PQFlashIndex<T>::setup_thread_data(_u64 nthreads) {
    diskann::cout << "Setting up thread-specific contexts for nthreads: "
                  << nthreads << std::endl;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int) nthreads)
    for (_s64 thread = 0; thread < (_s64) nthreads; thread++) {
#pragma omp critical
      {
        this->reader->register_thread();
        IOContext &ctx = this->reader->get_ctx();
        // diskann::cout << "ctx: " << ctx << "\n";
        QueryScratch<T> scratch;
        _u64 coord_alloc_size = ROUND_UP(MAX_N_CMPS * this->aligned_dim, 256);
        diskann::alloc_aligned((void **) &scratch.coord_scratch,
                               coord_alloc_size, 256);
        // scratch.coord_scratch = new T[MAX_N_CMPS * this->aligned_dim];
        // //Gopal. Commenting out the reallocation!
        diskann::alloc_aligned((void **) &scratch.sector_scratch,
                               MAX_N_SECTOR_READS * SECTOR_LEN, SECTOR_LEN);
        diskann::alloc_aligned((void **) &scratch.aligned_scratch,
                               256 * sizeof(float), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_pq_coord_scratch,
                               25600 * sizeof(_u8), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_pqtable_dist_scratch,
                               25600 * sizeof(float), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_dist_scratch,
                               512 * sizeof(float), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_query_T,
                               this->aligned_dim * sizeof(T), 8 * sizeof(T));
        diskann::alloc_aligned((void **) &scratch.aligned_query_float,
                               this->aligned_dim * sizeof(float),
                               8 * sizeof(float));

        memset(scratch.aligned_scratch, 0, 256 * sizeof(float));
        memset(scratch.coord_scratch, 0, MAX_N_CMPS * this->aligned_dim);
        memset(scratch.aligned_query_T, 0, this->aligned_dim * sizeof(T));
        memset(scratch.aligned_query_float, 0,
               this->aligned_dim * sizeof(float));

        ThreadData<T> data;
        data.ctx = ctx;
        data.scratch = scratch;
        this->thread_data.push(data);
      }
    }
    load_flag = true;
  }

  template<typename T>
  void PQFlashIndex<T>::destroy_thread_data() {
    diskann::cout << "Clearing scratch" << std::endl;
    assert(this->thread_data.size() == this->max_nthreads);
    while (this->thread_data.size() > 0) {
      ThreadData<T> data = this->thread_data.pop();
      while (data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
      }
      auto &scratch = data.scratch;
      diskann::aligned_free((void *) scratch.coord_scratch);
      diskann::aligned_free((void *) scratch.sector_scratch);
      diskann::aligned_free((void *) scratch.aligned_scratch);
      diskann::aligned_free((void *) scratch.aligned_pq_coord_scratch);
      diskann::aligned_free((void *) scratch.aligned_pqtable_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_query_float);
      diskann::aligned_free((void *) scratch.aligned_query_T);
    }
  }

  template<typename T>
  void PQFlashIndex<T>::load_cache_list(std::vector<uint32_t> &node_list) {
    diskann::cout << "Loading the cache list into memory.." << std::flush;
    _u64 num_cached_nodes = node_list.size();

    // borrow thread data
    ThreadData<T> this_thread_data = this->thread_data.pop();
    while (this_thread_data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      this_thread_data = this->thread_data.pop();
    }

    IOContext &ctx = this_thread_data.ctx;

    nhood_cache_buf = new unsigned[num_cached_nodes * (max_degree + 1)];
    memset(nhood_cache_buf, 0, num_cached_nodes * (max_degree + 1));

    _u64 coord_cache_buf_len = num_cached_nodes * aligned_dim;
    diskann::alloc_aligned((void **) &coord_cache_buf,
                           coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);

    for (_u64 block = 0; block < num_blocks; block++) {
      _u64 start_idx = block * BLOCK_SIZE;
      _u64 end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);
      std::vector<AlignedRead> read_reqs;
      std::vector<std::pair<_u32, char *>> nhoods;
      for (_u64 node_idx = start_idx; node_idx < end_idx; node_idx++) {
        AlignedRead read;
        char *      buf = nullptr;
        alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
        nhoods.push_back(std::make_pair(node_list[node_idx], buf));
        read.len = SECTOR_LEN;
        read.buf = buf;
        read.offset = NODE_SECTOR_NO(node_list[node_idx]) * SECTOR_LEN;
        read_reqs.push_back(read);
      }

      reader->read(read_reqs, ctx);

      _u64 node_idx = start_idx;
      for (auto &nhood : nhoods) {
        char *node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
        T *   node_coords = OFFSET_TO_NODE_COORDS(node_buf);
        T *   cached_coords = coord_cache_buf + node_idx * aligned_dim;
        memcpy(cached_coords, node_coords, data_dim * sizeof(T));
        coord_cache.insert(std::make_pair(nhood.first, cached_coords));

        // insert node nhood into nhood_cache
        unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);
        auto      nnbrs = *node_nhood;
        unsigned *nbrs = node_nhood + 1;
        // diskann::cout << "CACHE: nnbrs = " << nnbrs << "\n";
        std::pair<_u32, unsigned *> cnhood;
        cnhood.first = nnbrs;
        cnhood.second = nhood_cache_buf + node_idx * (max_degree + 1);
        memcpy(cnhood.second, nbrs, nnbrs * sizeof(unsigned));
        nhood_cache.insert(std::make_pair(nhood.first, cnhood));
        aligned_free(nhood.second);
        node_idx++;
      }
    }
    // return thread data
    this->thread_data.push(this_thread_data);
    diskann::cout << "..done." << std::endl;
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  void PQFlashIndex<T>::generate_cache_list_from_sample_queries(
      MemoryMappedFiles &files, std::string sample_bin, _u64 l_search,
      _u64 beamwidth, _u64 num_nodes_to_cache, uint32_t nthreads,
      std::vector<uint32_t> &node_list) {
#else
  template<typename T>
  void PQFlashIndex<T>::generate_cache_list_from_sample_queries(
      std::string sample_bin, _u64 l_search, _u64 beamwidth,
      _u64 num_nodes_to_cache, uint32_t nthreads,
      std::vector<uint32_t> &node_list) {
#endif
    this->count_visited_nodes = true;
    this->node_visit_counter.clear();
    this->node_visit_counter.resize(this->num_points);
    for (_u32 i = 0; i < node_visit_counter.size(); i++) {
      this->node_visit_counter[i].first = i;
      this->node_visit_counter[i].second = 0;
    }

    _u64 sample_num, sample_dim, sample_aligned_dim;
    T *  samples;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(sample_bin)) {
      diskann::load_aligned_bin<T>(files, sample_bin, samples, sample_num,
                                   sample_dim, sample_aligned_dim);
    }
#else
    if (file_exists(sample_bin)) {
      diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim,
                                   sample_aligned_dim);
    }
#endif
    else {
      diskann::cerr << "Sample bin file not found. Not generating cache."
                    << std::endl;
      return;
    }

    std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
    std::vector<float>    tmp_result_dists(sample_num, 0);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (_s64 i = 0; i < (int64_t) sample_num; i++) {
      cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search,
                         tmp_result_ids_64.data() + (i * 1),
                         tmp_result_dists.data() + (i * 1), beamwidth);
    }

    std::sort(this->node_visit_counter.begin(), node_visit_counter.end(),
              [](std::pair<_u32, _u32> &left, std::pair<_u32, _u32> &right) {
                return left.second > right.second;
              });
    node_list.clear();
    node_list.shrink_to_fit();
    node_list.reserve(num_nodes_to_cache);
    for (_u64 i = 0; i < num_nodes_to_cache; i++) {
      node_list.push_back(this->node_visit_counter[i].first);
    }
    this->count_visited_nodes = false;

    diskann::aligned_free(samples);
  }

  template<typename T>
  void PQFlashIndex<T>::cache_bfs_levels(_u64 num_nodes_to_cache,
                                         std::vector<uint32_t> &node_list) {
    // Gopal. random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937       urng(rng());

    node_list.clear();

    // borrow thread data
    ThreadData<T> this_thread_data = this->thread_data.pop();
    while (this_thread_data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      this_thread_data = this->thread_data.pop();
    }

    IOContext &ctx = this_thread_data.ctx;

    std::unique_ptr<tsl::robin_set<unsigned>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<unsigned>>();
    prev_level = std::make_unique<tsl::robin_set<unsigned>>();

    for (_u64 miter = 0; miter < num_medoids; miter++) {
      cur_level->insert(medoids[miter]);
    }

    _u64     lvl = 1;
    uint64_t prev_node_list_size = 0;
    while ((node_list.size() + cur_level->size() < num_nodes_to_cache) &&
           cur_level->size() != 0) {
      // swap prev_level and cur_level
      std::swap(prev_level, cur_level);
      // clear cur_level
      cur_level->clear();

      std::vector<unsigned> nodes_to_expand;

      for (const unsigned &id : *prev_level) {
        if (std::find(node_list.begin(), node_list.end(), id) !=
            node_list.end()) {
          continue;
        }
        node_list.push_back(id);
        nodes_to_expand.push_back(id);
      }

      // Gopal. random_shuffle() is deprecated.
      std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);

      diskann::cout << "Level: " << lvl << std::flush;
      bool finish_flag = false;

      uint64_t BLOCK_SIZE = 1024;
      uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
      for (size_t block = 0; block < nblocks && !finish_flag; block++) {
        diskann::cout << "." << std::flush;
        size_t start = block * BLOCK_SIZE;
        size_t end =
            (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());
        std::vector<AlignedRead> read_reqs;
        std::vector<std::pair<_u32, char *>> nhoods;
        for (size_t cur_pt = start; cur_pt < end; cur_pt++) {
          char *buf = nullptr;
          alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
          nhoods.push_back(std::make_pair(nodes_to_expand[cur_pt], buf));
          AlignedRead read;
          read.len = SECTOR_LEN;
          read.buf = buf;
          read.offset = NODE_SECTOR_NO(nodes_to_expand[cur_pt]) * SECTOR_LEN;
          read_reqs.push_back(read);
        }
        // issue read requests
        reader->read(read_reqs, ctx);
        // process each nhood buf
        for (auto &nhood : nhoods) {
          // insert node coord into coord_cache
          char *    node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
          unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);
          _u64      nnbrs = (_u64) *node_nhood;
          unsigned *nbrs = node_nhood + 1;
          // explore next level
          for (_u64 j = 0; j < nnbrs && !finish_flag; j++) {
            if (std::find(node_list.begin(), node_list.end(), nbrs[j]) ==
                node_list.end()) {
              cur_level->insert(nbrs[j]);
            }
            if (cur_level->size() + node_list.size() >= num_nodes_to_cache) {
              finish_flag = true;
            }
          }
          aligned_free(nhood.second);
        }
      }

      diskann::cout << ". #nodes: " << node_list.size() - prev_node_list_size
                    << ", #nodes thus far: " << node_list.size() << std::endl;
      prev_node_list_size = node_list.size();
      lvl++;
    }

    std::vector<uint32_t> cur_level_node_list;
    for (const unsigned &p : *cur_level)
      cur_level_node_list.push_back(p);

    // Gopal. random_shuffle() is deprecated
    std::shuffle(cur_level_node_list.begin(), cur_level_node_list.end(), urng);
    size_t residual = num_nodes_to_cache - node_list.size();

    for (size_t i = 0; i < (std::min)(residual, cur_level_node_list.size());
         i++)
      node_list.push_back(cur_level_node_list[i]);

    diskann::cout << "Level: " << lvl << std::flush;
    diskann::cout << ". #nodes: " << node_list.size() - prev_node_list_size
                  << ", #nodes thus far: " << node_list.size() << std::endl;

    // return thread data
    this->thread_data.push(this_thread_data);
  }

  template<typename T>
  void PQFlashIndex<T>::use_medoids_data_as_centroids() {
    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    alloc_aligned(((void **) &centroid_data),
                  num_medoids * aligned_dim * sizeof(float), 32);
    std::memset(centroid_data, 0, num_medoids * aligned_dim * sizeof(float));

    // borrow ctx
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    IOContext &ctx = data.ctx;
    diskann::cout << "Loading centroid data from medoids vector data of "
                  << num_medoids << " medoid(s)" << std::endl;
    for (uint64_t cur_m = 0; cur_m < num_medoids; cur_m++) {
      auto medoid = medoids[cur_m];
      // read medoid nhood
      char *medoid_buf = nullptr;
      alloc_aligned((void **) &medoid_buf, SECTOR_LEN, SECTOR_LEN);
      std::vector<AlignedRead> medoid_read(1);
      medoid_read[0].len = SECTOR_LEN;
      medoid_read[0].buf = medoid_buf;
      medoid_read[0].offset = NODE_SECTOR_NO(medoid) * SECTOR_LEN;
      reader->read(medoid_read, ctx);

      // all data about medoid
      char *medoid_node_buf = OFFSET_TO_NODE(medoid_buf, medoid);

      // add medoid coords to `coord_cache`
      T *medoid_coords = new T[data_dim];
      T *medoid_disk_coords = OFFSET_TO_NODE_COORDS(medoid_node_buf);
      memcpy(medoid_coords, medoid_disk_coords, data_dim * sizeof(T));

      for (uint32_t i = 0; i < data_dim; i++)
        centroid_data[cur_m * aligned_dim + i] = medoid_coords[i];

      aligned_free(medoid_buf);
    }

    // return ctx
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  int PQFlashIndex<T>::load(MemoryMappedFiles &files, uint32_t num_threads,
                            const char *pq_prefix,
                            const char *disk_index_file) {
#else
  template<typename T>
  int PQFlashIndex<T>::load(uint32_t num_threads, const char *pq_prefix,
                            const char *disk_index_file) {
#endif
    std::string pq_table_bin = std::string(pq_prefix) + "_pivots.bin";
    std::string pq_compressed_vectors =
        std::string(pq_prefix) + "_compressed.bin";
    std::string medoids_file = std::string(disk_index_file) + "_medoids.bin";
    std::string centroids_file =
        std::string(disk_index_file) + "_centroids.bin";

    size_t pq_file_dim, pq_file_num_centroids;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim);
#endif

    this->disk_index_file = std::string(disk_index_file);

    if (pq_file_num_centroids != 256) {
      diskann::cout << "Error. Number of PQ centroids is not 256. Exitting."
                    << std::endl;
      return -1;
    }

    this->data_dim = pq_file_dim;
    this->aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
#ifdef EXEC_ENV_OLS
    diskann::load_bin<_u8>(files, pq_compressed_vectors, this->data, npts_u64,
                           nchunks_u64);
#else
    diskann::load_bin<_u8>(pq_compressed_vectors, this->data, npts_u64,
                           nchunks_u64);
#endif

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;

#ifdef EXEC_ENV_OLS
    pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

    diskann::cout
        << "Loaded PQ centroids and in-memory compressed vectors. #points: "
        << num_points << " #dim: " << data_dim
        << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks
        << std::endl;

// read index metadata
#ifdef EXEC_ENV_OLS
    // This is a bit tricky. We have to read the header from the
    // disk_index_file. But  this is
    // now exclusively a preserve of the DiskPriorityIO class. So, we need to
    // estimate how many
    // bytes are needed to store the header and read in that many using our
    // 'standard' aligned
    // file reader approach.
    reader->open(disk_index_file);
    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

    char *                   bytes = getHeaderBytes();
    ContentBuf               buf(bytes, HEADER_SIZE);
    std::basic_istream<char> index_metadata(&buf);
#else
    std::ifstream index_metadata(disk_index_file, std::ios::binary);
#endif

    size_t actual_index_size = get_file_size(disk_index_file);
    size_t expected_file_size;
    READ_U64(index_metadata, expected_file_size);
    if (actual_index_size != expected_file_size) {
      diskann::cout << "File size mismatch for " << disk_index_file
                    << " (size: " << actual_index_size << ")"
                    << " with meta-data size: " << expected_file_size
                    << std::endl;
      return -1;
    }

    _u64 disk_nnodes;
    READ_U64(index_metadata, disk_nnodes);
    if (disk_nnodes != num_points) {
      diskann::cout << "Mismatch in #points for compressed data file and disk "
                       "index file: "
                    << disk_nnodes << " vs " << num_points << std::endl;
      return -1;
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, max_node_len);
    READ_U64(index_metadata, nnodes_per_sector);
    max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;

    diskann::cout << "Disk-Index File Meta-data: ";
    diskann::cout << "# nodes per sector: " << nnodes_per_sector;
    diskann::cout << ", max node len (bytes): " << max_node_len;
    diskann::cout << ", max node degree: " << max_degree << std::endl;

#ifdef EXEC_ENV_OLS
    delete[] bytes;
#else
    index_metadata.close();
#endif

#ifndef EXEC_ENV_OLS
    // open AlignedFileReader handle to index_file
    std::string index_fname(disk_index_file);
    reader->open(index_fname);
    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

#endif

#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file)) {
      size_t tmp_dim;
      diskann::load_bin<uint32_t>(files, medoids_file, medoids, num_medoids,
                                  tmp_dim);
#else
    if (file_exists(medoids_file)) {
      size_t tmp_dim;
      diskann::load_bin<uint32_t>(medoids_file, medoids, num_medoids, tmp_dim);
#endif

      if (tmp_dim != 1) {
        std::stringstream stream;
        stream << "Error loading medoids file. Expected bin format of m times "
                  "1 vector of uint32_t."
               << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
#ifdef EXEC_ENV_OLS
      if (!files.fileExists(centroids_file)) {
#else
      if (!file_exists(centroids_file)) {
#endif
        diskann::cout
            << "Centroid data file not found. Using corresponding vectors "
               "for the medoids "
            << std::endl;
        use_medoids_data_as_centroids();
      } else {
        size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
        diskann::load_aligned_bin<float>(files, centroids_file, centroid_data,
                                         num_centroids, tmp_dim,
                                         aligned_tmp_dim);
#else
        diskann::load_aligned_bin<float>(centroids_file, centroid_data,
                                         num_centroids, tmp_dim,
                                         aligned_tmp_dim);
#endif
        if (aligned_tmp_dim != aligned_dim || num_centroids != num_medoids) {
          std::stringstream stream;
          stream << "Error loading centroids data file. Expected bin format of "
                    "m times data_dim vector of float, where m is number of "
                    "medoids "
                    "in medoids file."
                 << std::endl;
          throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
        }
      }
    } else {
      num_medoids = 1;
      medoids = new uint32_t[1];
      medoids[0] = (_u32)(medoid_id_on_file);
      use_medoids_data_as_centroids();
    }

    diskann::cout << "done.." << std::endl;
    return 0;
  }

#ifdef USE_BING_INFRA
  bool getNextCompletedRequest(const IOContext &ctx, size_t size,
                               int &completedIndex) {
    bool waitsRemaining = false;
    for (int i = 0; i < size; i++) {
      auto ithStatus = (*ctx.m_pRequestsStatus)[i];
      if (ithStatus == IOContext::Status::READ_SUCCESS) {
        completedIndex = i;
        return true;
      } else if (ithStatus == IOContext::Status::READ_WAIT) {
        waitsRemaining = true;
      }
    }
    completedIndex = -1;
    return waitsRemaining;
  }
#endif

  template<typename T>
  void PQFlashIndex<T>::cached_beam_search(const T *query1, const _u64 k_search,
                                           const _u64 l_search, _u64 *indices,
                                           float *      distances,
                                           const _u64   beam_width,
                                           QueryStats * stats,
                                           Distance<T> *output_dist_func) {
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    for (uint32_t i = 0; i < this->data_dim; i++) {
      data.scratch.aligned_query_float[i] = query1[i];
    }
    memcpy(data.scratch.aligned_query_T, query1, this->data_dim * sizeof(T));
    const T *    query = data.scratch.aligned_query_T;
    const float *query_float = data.scratch.aligned_query_float;

    IOContext &ctx = data.ctx;
    auto       query_scratch = &(data.scratch);

    // reset query
    query_scratch->reset();

    // scratch space to compute distances between FP32 Query and INT8 data
    float *scratch = query_scratch->aligned_scratch;
    _mm_prefetch((char *) scratch, _MM_HINT_T0);

    // pointers to buffers for data
    T *   data_buf = query_scratch->coord_scratch;
    _u64 &data_buf_idx = query_scratch->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](
        const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };
    Timer                 query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset(l_search + 1);
    tsl::robin_set<_u64>  visited(4096);

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);
    tsl::robin_map<_u64, T *> fp_coords;

    _u32                        best_medoid = 0;
    float                       best_dist = (std::numeric_limits<float>::max)();
    std::vector<SimpleNeighbor> medoid_dists;
    for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
      float cur_expanded_dist = dist_cmp_float->compare(
          query_float, centroid_data + aligned_dim * cur_m,
          (unsigned) aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = medoids[cur_m];
        best_dist = cur_expanded_dist;
      }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset[0].id = best_medoid;
    retset[0].distance = dist_scratch[0];
    retset[0].flag = true;
    visited.insert(best_medoid);

    unsigned cur_list_size = 1;

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    std::vector<AlignedRead> frontier_read_reqs;
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;

    while (k < cur_list_size) {
      auto nk = cur_list_size;

      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;

      // find new beam
      // WAS: _u64 marker = k - 1;
      _u32 marker = k;
      _u32 num_seen = 0;

      /*
        bool marker_set = false;
        diskann::cout << "hop " << hops << ": ";
        for (_u32 i = 0; i < cur_list_size; i++) {
          diskann::cout << retset[i].id << "( " << retset[i].distance;
          if (retset[i].flag && !marker_set) {
            diskann::cout << ",*)  ";
            marker_set = true;
          } else
            diskann::cout << ")  ";
        }
        diskann::cout << std::endl;
  */
      while (marker < cur_list_size && frontier.size() < beam_width &&
             num_seen < beam_width + 2) {
        if (retset[marker].flag) {
          num_seen++;
          auto iter = nhood_cache.find(retset[marker].id);
          if (iter != nhood_cache.end()) {
            cached_nhoods.push_back(
                std::make_pair(retset[marker].id, iter->second));
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
          } else {
            frontier.push_back(retset[marker].id);
          }
          retset[marker].flag = false;
          if (this->count_visited_nodes) {
            reinterpret_cast<std::atomic<_u32> &>(
                this->node_visit_counter[retset[marker].id].second)
                .fetch_add(1);
          }
        }
        marker++;
      }

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto id = frontier[i];
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(
              NODE_SECTOR_NO(((size_t) id)) * SECTOR_LEN, SECTOR_LEN,
              fnhood.second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        io_timer.reset();
#ifdef USE_BING_INFRA
        reader->read(frontier_read_reqs, ctx, true);  // async reader windows.
#else
        reader->read(frontier_read_reqs, ctx);  // synchronous IO linux
#endif
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }
      }

      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        auto  global_cache_iter = coord_cache.find(cached_nhood.first);
        T *   node_fp_coords_copy = global_cache_iter->second;
        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                                    (unsigned) aligned_dim);
        full_retset.push_back(
            Neighbor((unsigned) cached_nhood.first, cur_expanded_dist, true));

        _u64      nnbrs = cached_nhood.second.first;
        unsigned *node_nbrs = cached_nhood.second.second;

        // compute node_nbrs <-> query dists in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += nnbrs;
          stats->cpu_us += cpu_timer.elapsed();
        }

        // process prefetched nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
            float dist = dist_scratch[m];
            // diskann::cout << "cmp: " << id << ", dist: " << dist <<
            // std::endl; std::cerr << "dist: " << dist << std::endl;
            if (dist >= retset[cur_list_size - 1].distance &&
                (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            auto     r = InsertIntoPool(
                retset.data(), cur_list_size,
                nn);  // Return position in sorted list where nn inserted.
            if (cur_list_size < l_search)
              ++cur_list_size;
            if (r < nk)
              nk = r;  // nk logs the best position in the retset that was
            // updated
            // due to neighbors of n.
          }
        }
      }
#ifdef USE_BING_INFRA
      // process each frontier nhood - compute distances to unvisited nodes
      int completedIndex = -1;
      // If we issued read requests and if a read is complete or there are reads
      // in wait
      // state, then enter the while loop.
      while (frontier_read_reqs.size() > 0 &&
             getNextCompletedRequest(ctx, frontier_read_reqs.size(),
                                     completedIndex)) {
        if (completedIndex == -1) {  // all reads are waiting
          continue;
        }
        auto &frontier_nhood = frontier_nhoods[completedIndex];
        (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else
      for (auto &frontier_nhood : frontier_nhoods) {
#endif
        char *node_disk_buf =
            OFFSET_TO_NODE(frontier_nhood.second, frontier_nhood.first);
        unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
        _u64      nnbrs = (_u64)(*node_buf);
        T *       node_fp_coords = OFFSET_TO_NODE_COORDS(node_disk_buf);
        assert(data_buf_idx < MAX_N_CMPS);

        T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
        data_buf_idx++;
        memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(T));

        float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                                    (unsigned) aligned_dim);
        full_retset.push_back(
            Neighbor(frontier_nhood.first, cur_expanded_dist, true));

        unsigned *node_nbrs = (node_buf + 1);
        // compute node_nbrs <-> query dist in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += nnbrs;
          stats->cpu_us += cpu_timer.elapsed();
        }

        cpu_timer.reset();
        // process prefetch-ed nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            cmps++;
            float dist = dist_scratch[m];
            // diskann::cout << "cmp: " << id << ", dist: " << dist <<
            // std::endl;
            // diskann::cout << "dist: " << dist << std::endl;
            if (stats != nullptr) {
              stats->n_cmps++;
            }
            if (dist >= retset[cur_list_size - 1].distance &&
                (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            auto     r = InsertIntoPool(
                retset.data(), cur_list_size,
                nn);  // Return position in sorted list where nn inserted.
            if (cur_list_size < l_search)
              ++cur_list_size;
            if (r < nk)
              nk = r;  // nk logs the best position in the retset that was
                       // updated
                       // due to neighbors of n.
          }
        }

        if (stats != nullptr) {
          stats->cpu_us += cpu_timer.elapsed();
        }
      }

      // update best inserted position
      //

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
    }
    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // copy k_search values
    for (_u64 i = 0; i < k_search; i++) {
      indices[i] = full_retset[i].id;
      if (distances != nullptr) {
        distances[i] = full_retset[i].distance;
      }
    }

    this->thread_data.push(data);
    this->thread_data.push_notify_all();

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  char *PQFlashIndex<T>::getHeaderBytes() {
    IOContext & ctx = reader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    reader->read(readReqs, ctx, false);

    return (char *) readReq.buf;
  }
#endif

  // instantiations
  template class PQFlashIndex<_u8>;
  template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;

}  // namespace diskann
