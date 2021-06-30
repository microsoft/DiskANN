// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "logger.h"
#include "aux_utils.h"
#include "cached_io.h"
#include "index.h"
#include "mkl.h"
#include "omp.h"
#include "partition_and_pq.h"
#include "percentile_stats.h"
#include "pq_flash_index.h"
#include "utils.h"

#include "pq_flash_index.h"
#include "tsl/robin_set.h"
#include "utils.h"

#define NUM_KMEANS 15

namespace diskann {

  void add_new_file_to_single_index(std::string index_file,
                                    std::string new_file) {
    std::unique_ptr<_u64[]> metadata;
    _u64                    nr, nc;
    diskann::load_bin<_u64>(index_file, metadata, nr, nc, 0);
    if (nc != 1) {
      std::stringstream stream;
      stream << "Error, index file specified does not have correct metadata. "
             << std::endl;
      throw diskann::ANNException(stream.str(), -1);
    }
    size_t          index_ending_offset = metadata[nr - 1];
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ofstream writer(index_file, read_blk_size, index_ending_offset);
    _u64            check_file_size = get_file_size(index_file);
    if (check_file_size != index_ending_offset) {
      std::stringstream stream;
      stream << "Error, index file specified does not have correct metadata "
                "(last entry must match the filesize). "
             << std::endl;
      throw diskann::ANNException(stream.str(), -1);
    }

    cached_ifstream reader(new_file, read_blk_size);
    size_t          fsize = reader.get_file_size();
    if (fsize == 0) {
      std::stringstream stream;
      stream << "Error, new file specified is empty. Not appending. "
             << std::endl;
      throw diskann::ANNException(stream.str(), -1);
    }

    size_t num_blocks = DIV_ROUND_UP(fsize, read_blk_size);
    char * dump = new char[read_blk_size];
    for (_u64 i = 0; i < num_blocks; i++) {
      size_t cur_block_size = read_blk_size > fsize - (i * read_blk_size)
                                  ? fsize - (i * read_blk_size)
                                  : read_blk_size;
      reader.read(dump, cur_block_size);
      writer.write(dump, cur_block_size);
    }
    reader.close();
    writer.close();

    delete[] dump;
    std::vector<_u64> new_meta;
    for (_u64 i = 0; i < nr; i++)
      new_meta.push_back(metadata[i]);
    new_meta.push_back(metadata[nr - 1] + fsize);

    diskann::save_bin<_u64>(index_file, new_meta.data(), new_meta.size(), 1, 0);
  }

  double get_memory_budget(double search_ram_budget) {
    double final_index_ram_limit = search_ram_budget;
    if (search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB >
        THRESHOLD_FOR_CACHING_IN_GB) {  // slack for space used by cached
                                        // nodes
      final_index_ram_limit = search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB;
    }
    return final_index_ram_limit * 1024 * 1024 * 1024;
  }

  double get_memory_budget(const std::string &mem_budget_str) {
    double search_ram_budget = atof(mem_budget_str.c_str());
    return get_memory_budget(search_ram_budget);
  }

  size_t calculate_num_pq_chunks(double final_index_ram_limit,
                                 size_t points_num, uint32_t dim) {
    size_t num_pq_chunks =
        (size_t)(std::floor)(_u64(final_index_ram_limit / (double) points_num));

    diskann::cout << "Calculated num_pq_chunks :" << num_pq_chunks << std::endl;
    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks =
        num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

    diskann::cout << "Compressing " << dim << "-dimensional data into "
                  << num_pq_chunks << " bytes per vector." << std::endl;
    return num_pq_chunks;
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std,
                          float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or,
                          unsigned recall_at) {
    double             total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t    tie_breaker = recall_at;
      if (gs_dist != nullptr) {
        float *gt_dist_vec = gs_dist + dim_gs * i;
        tie_breaker = recall_at - 1;
        while (tie_breaker < dim_gs &&
               gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);

      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std,
                          float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or,
                          unsigned                        recall_at,
                          const tsl::robin_set<unsigned> &active_tags) {
    double             total_recall = 0;
    std::set<unsigned> gt, res;
    bool               printed = false;
    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t    tie_breaker = recall_at;
      unsigned  active_points_count = 0;
      unsigned  cur_counter = 0;
      while (active_points_count < recall_at && cur_counter < dim_gs) {
        if (active_tags.find(*(gt_vec + cur_counter)) != active_tags.end()) {
          active_points_count++;
        }
        cur_counter++;
      }
      if (active_tags.empty())
        cur_counter = recall_at;

      if ((active_points_count < recall_at && !active_tags.empty()) &&
          !printed) {
        diskann::cout << "Warning: Couldn't find enough closest neighbors "
                      << active_points_count << "/" << recall_at
                      << " from "
                         "truthset for query # "
                      << i << ". Will result in under-reported value of recall."
                      << std::endl;
        printed = true;
      }
      if (gs_dist != nullptr) {
        tie_breaker = cur_counter - 1;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs &&
               gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);
      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return ((double) (total_recall / (num_queries))) *
           ((double) (100.0 / recall_at));
  }

  template<typename T>
  T *generateRandomWarmup(uint64_t warmup_num, uint64_t warmup_dim,
                          uint64_t warmup_aligned_dim) {
    T *warmup = nullptr;
    warmup_num = 100000;
    diskann::cout << "Generating random warmup file with dim " << warmup_dim
                  << " and aligned dim " << warmup_aligned_dim << std::flush;
    diskann::alloc_aligned(((void **) &warmup),
                           warmup_num * warmup_aligned_dim * sizeof(T),
                           8 * sizeof(T));
    std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(-128, 127);
    for (uint32_t i = 0; i < warmup_num; i++) {
      for (uint32_t d = 0; d < warmup_dim; d++) {
        warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
      }
    }
    diskann::cout << "..done" << std::endl;
    return warmup;
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  T *load_warmup(MemoryMappedFiles &files, const std::string &cache_warmup_file,
                 uint64_t &warmup_num, uint64_t warmup_dim,
                 uint64_t warmup_aligned_dim) {
    T *      warmup = nullptr;
    uint64_t file_dim, file_aligned_dim;

    if (files.fileExists(cache_warmup_file)) {
      diskann::load_aligned_bin<T>(files, cache_warmup_file, warmup, warmup_num,
                                   file_dim, file_aligned_dim);
      diskann::cout << "In the warmup file: " << cache_warmup_file
                    << " File dim: " << file_dim
                    << " File aligned dim: " << file_aligned_dim
                    << " Expected dim: " << warmup_dim
                    << " Expected aligned dim: " << warmup_aligned_dim
                    << std::endl;
      if (file_dim != warmup_dim || file_aligned_dim != warmup_aligned_dim) {
        std::stringstream stream;
        stream << "Mismatched dimensions in sample file. file_dim = "
               << file_dim << " file_aligned_dim: " << file_aligned_dim
               << " index_dim: " << warmup_dim
               << " index_aligned_dim: " << warmup_aligned_dim << std::endl;
        throw diskann::ANNException(stream.str(), -1);
      }
    } else {
      warmup =
          generateRandomWarmup<T>(warmup_num, warmup_dim, warmup_aligned_dim);
    }
    return warmup;
  }
#endif

  template<typename T>
  T *load_warmup(const std::string &cache_warmup_file, uint64_t &warmup_num,
                 uint64_t warmup_dim, uint64_t warmup_aligned_dim) {
    T *      warmup = nullptr;
    uint64_t file_dim, file_aligned_dim;

    if (file_exists(cache_warmup_file)) {
      diskann::load_aligned_bin<T>(cache_warmup_file, warmup, warmup_num,
                                   file_dim, file_aligned_dim);
      if (file_dim != warmup_dim || file_aligned_dim != warmup_aligned_dim) {
        std::stringstream stream;
        stream << "Mismatched dimensions in sample file. file_dim = "
               << file_dim << " file_aligned_dim: " << file_aligned_dim
               << " index_dim: " << warmup_dim
               << " index_aligned_dim: " << warmup_aligned_dim << std::endl;
        throw diskann::ANNException(stream.str(), -1);
      }
    } else {
      warmup =
          generateRandomWarmup<T>(warmup_num, warmup_dim, warmup_aligned_dim);
    }
    return warmup;
  }

  /***************************************************
      Support for Merging Many Vamana Indices
   ***************************************************/

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
    uint32_t      npts32, dim;
    size_t        actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *) &npts32, sizeof(uint32_t));
    reader.read((char *) &dim, sizeof(uint32_t));
    if (dim != 1 || actual_file_size != ((size_t) npts32) * sizeof(uint32_t) +
                                            2 * sizeof(uint32_t)) {
      std::stringstream stream;
      stream << "Error reading idmap file. Check if the file is bin file with "
                "1 dimensional data. Actual: "
             << actual_file_size
             << ", expected: " << (size_t) npts32 + 2 * sizeof(uint32_t)
             << std::endl;

      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    ivecs.resize(npts32);
    reader.read((char *) ivecs.data(), ((size_t) npts32) * sizeof(uint32_t));
    reader.close();
  }

  int merge_shards(const std::string &vamana_prefix,
                   const std::string &vamana_suffix,
                   const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards,
                   unsigned max_degree, const std::string &output_vamana,
                   const std::string &medoids_file) {
    // Read ID maps
    std::vector<std::string>           vamana_names(nshards);
    std::vector<std::vector<unsigned>> idmaps(nshards);
    for (_u64 shard = 0; shard < nshards; shard++) {
      vamana_names[shard] =
          vamana_prefix + std::to_string(shard) + vamana_suffix;
      read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix,
                 idmaps[shard]);
    }

    // find max node id
    _u64 nnodes = 0;
    _u64 nelems = 0;
    for (auto &idmap : idmaps) {
      for (auto &id : idmap) {
        nnodes = std::max(nnodes, (_u64) id);
      }
      nelems += idmap.size();
    }
    nnodes++;
    diskann::cout << "# nodes: " << nnodes << ", max. degree: " << max_degree
                  << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<unsigned, unsigned>> node_shard;
    node_shard.reserve(nelems);
    for (_u64 shard = 0; shard < nshards; shard++) {
      diskann::cout << "Creating inverse map -- shard #" << shard << std::endl;
      for (_u64 idx = 0; idx < idmaps[shard].size(); idx++) {
        _u64 node_id = idmaps[shard][idx];
        node_shard.push_back(std::make_pair((_u32) node_id, (_u32) shard));
      }
    }
    std::sort(node_shard.begin(), node_shard.end(),
              [](const auto &left, const auto &right) {
                return left.first < right.first || (left.first == right.first &&
                                                    left.second < right.second);
              });
    diskann::cout << "Finished computing node -> shards map" << std::endl;

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (_u64 i = 0; i < nshards; i++) {
      vamana_readers[i].open(vamana_names[i], 1024 * 1048576);
      size_t expected_file_size;
      vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
    }

    size_t merged_index_size = 24;
    size_t merged_index_frozen = 0;
    // create cached vamana writers
    cached_ofstream diskann_writer(output_vamana, 1024 * 1048576);
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    unsigned output_width = max_degree;
    unsigned max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(unsigned) bytes
    for (auto &reader : vamana_readers) {
      unsigned input_width;
      reader.read((char *) &input_width, sizeof(unsigned));
      max_input_width =
          input_width > max_input_width ? input_width : max_input_width;
    }

    diskann::cout << "Max input width: " << max_input_width
                  << ", output width: " << output_width << std::endl;

    diskann_writer.write((char *) &output_width, sizeof(unsigned));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    _u32          nshards_u32 = (_u32) nshards;
    _u32          one_val = 1;
    medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *) &one_val, sizeof(uint32_t));

    _u64 vamana_index_frozen = 0;
    for (_u64 shard = 0; shard < nshards; shard++) {
      unsigned medoid;
      // read medoid
      vamana_readers[shard].read((char *) &medoid, sizeof(unsigned));
      vamana_readers[shard].read((char *) &vamana_index_frozen, sizeof(_u64));
      assert(vamana_index_frozen == false);
      // rename medoid
      medoid = idmaps[shard][medoid];

      medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      // write renamed medoid
      if (shard == (nshards - 1))  //--> uncomment if running hierarchical
        diskann_writer.write((char *) &medoid, sizeof(unsigned));
    }
    diskann_writer.write((char *) &merged_index_frozen, sizeof(_u64));
    medoid_writer.close();

    diskann::cout << "Starting merge" << std::endl;

    // random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937       urng(rng());

    std::vector<bool>     nhood_set(nnodes, 0);
    std::vector<unsigned> final_nhood;

    unsigned nnbrs = 0, shard_nnbrs = 0;
    unsigned cur_id = 0;
    for (const auto &id_shard : node_shard) {
      unsigned node_id = id_shard.first;
      unsigned shard_id = id_shard.second;
      if (cur_id < node_id) {
        // random_shuffle() is deprecated.
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        nnbrs =
            (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
        // write into merged ofstream
        diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
        diskann_writer.write((char *) final_nhood.data(),
                             nnbrs * sizeof(unsigned));
        merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
        if (cur_id % 499999 == 1) {
          diskann::cout << "." << std::flush;
        }
        cur_id = node_id;
        nnbrs = 0;
        for (auto &p : final_nhood)
          nhood_set[p] = 0;
        final_nhood.clear();
      }
      // read from shard_id ifstream
      vamana_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<unsigned> shard_nhood(shard_nnbrs);
      vamana_readers[shard_id].read((char *) shard_nhood.data(),
                                    shard_nnbrs * sizeof(unsigned));

      // rename nodes
      for (_u64 j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0) {
          nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
          final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
        }
      }
    }

    // random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
    // write into merged ofstream
    diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
    diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
    merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
    for (auto &p : final_nhood)
      nhood_set[p] = 0;
    final_nhood.clear();

    diskann::cout << "Expected size: " << merged_index_size << std::endl;

    diskann_writer.reset();
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    diskann::cout << "Finished merge" << std::endl;
    return 0;
  }

  template<typename T>
  int build_merged_vamana_index(
      std::string base_file, diskann::Metric _compareMetric,
      bool single_file_index, unsigned L, unsigned R, double sampling_rate,
      double ram_budget, std::string mem_index_path, std::string medoids_file,
      std::string centroids_file, const char *tag_file) {
    size_t base_num, base_dim;
    diskann::get_bin_metadata(base_file, base_num, base_dim);

    double full_index_ram =
        ESTIMATE_RAM_USAGE(base_num, base_dim, sizeof(T), R);
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      diskann::cout << "Full index fits in RAM, building in one shot"
                    << std::endl;
      diskann::Parameters paras;
      paras.Set<unsigned>("L", (unsigned) L);
      paras.Set<unsigned>("R", (unsigned) R);
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<unsigned>("num_rnds", 2);
      paras.Set<bool>("saturate_graph", 1);  // was 0 earlier.
      paras.Set<std::string>("save_path", mem_index_path);

      bool tags_enabled;
      if (tag_file == nullptr)
        tags_enabled = false;
      else
        tags_enabled = true;

      std::unique_ptr<diskann::Index<T>> _pvamanaIndex =
          std::unique_ptr<diskann::Index<T>>(
              new diskann::Index<T>(_compareMetric, base_dim, base_num, false,
                                    single_file_index, tags_enabled));
      if (tags_enabled)
        _pvamanaIndex->build(base_file.c_str(), base_num, paras, tag_file);
      else
        _pvamanaIndex->build(base_file.c_str(), base_num, paras);

      _pvamanaIndex->save(mem_index_path.c_str());
      std::remove(medoids_file.c_str());
      std::remove(centroids_file.c_str());
      return 0;
    }

    if (single_file_index || tag_file != nullptr) {
      diskann::cout << "Cannot build merged index if single_file_index is "
                       "required or if tags are specified. Please contact "
                       "rakri@microsoft.com if this is required"
                    << std::endl;
      return 1;
    }

    std::string merged_index_prefix = mem_index_path + "_tempFiles";
    int         num_parts =
        partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget,
                                     2 * R / 3, merged_index_prefix, 2);

    std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
    std::rename(cur_centroid_filepath.c_str(), centroids_file.c_str());

    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_index_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

      diskann::Parameters paras;
      paras.Set<unsigned>("L", L);
      paras.Set<unsigned>("R", (2 * (R / 3)));
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<unsigned>("num_rnds", 2);
      paras.Set<bool>("saturate_graph", 0);
      paras.Set<std::string>("save_path", shard_index_file);

      _u64 shard_base_dim, shard_base_pts;
      get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
      std::unique_ptr<diskann::Index<T>> _pvamanaIndex =
          std::unique_ptr<diskann::Index<T>>(new diskann::Index<T>(
              _compareMetric, shard_base_dim, shard_base_pts, false,
              single_file_index));  // TODO: Single?
      _pvamanaIndex->build(shard_base_file.c_str(), shard_base_pts, paras);
      _pvamanaIndex->save(shard_index_file.c_str());
    }

    diskann::merge_shards(merged_index_prefix + "_subshard-", "_mem.index",
                          merged_index_prefix + "_subshard-", "_ids_uint32.bin",
                          num_parts, R, mem_index_path, medoids_file);

    // delete tempFiles
    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_id_file = merged_index_prefix + "_subshard-" +
                                  std::to_string(p) + "_ids_uint32.bin";
      std::string shard_index_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
      // Required if Index.cpp thinks we are building a multi-file index.
      std::string shard_index_file_data = shard_index_file + ".data";

      std::remove(shard_base_file.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_index_file.c_str());
      std::remove(shard_index_file_data.c_str());
    }
    return 0;
  }

  // General purpose support for DiskANN interface
  //
  //

  // optimizes the beamwidth to maximize QPS for a given L_search subject to
  // 99.9 latency not blowing up
  template<typename T, typename TagT>
  uint32_t optimize_beamwidth(
      std::unique_ptr<diskann::PQFlashIndex<T, TagT>> &pFlashIndex,
      T *tuning_sample, _u64 tuning_sample_num, _u64 tuning_sample_aligned_dim,
      uint32_t L, uint32_t nthreads, uint32_t start_bw) {
    uint32_t cur_bw = start_bw;
    double   max_qps = 0;
    uint32_t best_bw = start_bw;
    bool     stop_flag = false;

    while (!stop_flag) {
      std::vector<TagT>    tuning_sample_result_tags(tuning_sample_num, 0);
      std::vector<float>   tuning_sample_result_dists(tuning_sample_num, 0);
      diskann::QueryStats *stats = new diskann::QueryStats[tuning_sample_num];

      auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
      for (_s64 i = 0; i < (int64_t) tuning_sample_num; i++) {
        pFlashIndex->cached_beam_search(
            tuning_sample + (i * tuning_sample_aligned_dim), 1, L,
            tuning_sample_result_tags.data() + (i * 1),
            tuning_sample_result_dists.data() + (i * 1), cur_bw, stats + i);
      }
      auto e = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = e - s;
      double                        qps =
          (1.0f * (float) tuning_sample_num) / (1.0f * (float) diff.count());

      double lat_999 = diskann::get_percentile_stats(
          stats, tuning_sample_num, 0.999f,
          [](const diskann::QueryStats &stats) { return stats.total_us; });

      double mean_latency = diskann::get_mean_stats(
          stats, tuning_sample_num,
          [](const diskann::QueryStats &stats) { return stats.total_us; });

      if (qps > max_qps && lat_999 < (15000) + mean_latency * 2) {
        max_qps = qps;
        best_bw = cur_bw;
        cur_bw = (uint32_t)(std::ceil)((float) cur_bw * 1.1f);
      } else {
        stop_flag = true;
      }
      if (cur_bw > 64)
        stop_flag = true;

      delete[] stats;
    }
    return best_bw;
  }

  // if single_index format is true, we assume that the entire mem index is in
  // mem_index_file, and the entire disk index will be in output_file.
  template<typename T, typename TagT>
  void create_disk_layout(const std::string &mem_index_file,
                          const std::string &base_file,
                          const std::string &tag_file,
                          const std::string &pq_pivots_file,
                          const std::string &pq_vectors_file,
                          bool               single_file_index,
                          const std::string &output_file) {
    unsigned npts, ndims;

    // amount to read or write in one shot
    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;
    cached_ifstream base_reader;
    std::ifstream   vamana_reader;
    _u64            base_offset = 0, vamana_offset = 0, tags_offset = 0;
    bool            tags_enabled = false;

    if (single_file_index) {
      _u64                    nr, nc;
      std::unique_ptr<_u64[]> offsets;
      diskann::load_bin<_u64>(mem_index_file, offsets, nr, nc);
      if (nr != Index<T, TagT>::METADATA_ROWS && nc != 1) {
        std::stringstream stream;
        stream
            << "Vamana Single Index file size does not meet meta-data criteria."
            << std::endl;

        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
      vamana_offset = offsets[0];
      base_offset = offsets[1];
      tags_offset = offsets[2];
      tags_enabled = tags_offset != offsets[3];
      vamana_reader.open(mem_index_file, std::ios::binary);
      vamana_reader.seekg(vamana_offset, vamana_reader.beg);
      base_reader.open(mem_index_file, read_blk_size, base_offset);
    } else {
      base_reader.open(base_file, read_blk_size);
      vamana_reader.open(mem_index_file, std::ios::binary);
      tags_enabled = tag_file != "";
    }

    base_reader.read((char *) &npts, sizeof(uint32_t));
    base_reader.read((char *) &ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    // create cached reader + writer
    //    size_t          actual_file_size = get_file_size(mem_index_file);
    std::remove(output_file.c_str());
    cached_ofstream diskann_writer;
    diskann_writer.open(output_file, write_blk_size);

    // metadata: width, medoid
    unsigned width_u32, medoid_u32;
    size_t   index_file_size;

    vamana_reader.read((char *) &index_file_size, sizeof(uint64_t));

    _u64 vamana_frozen_num = false, vamana_frozen_loc = 0;
    vamana_reader.read((char *) &width_u32, sizeof(unsigned));
    vamana_reader.read((char *) &medoid_u32, sizeof(unsigned));
    vamana_reader.read((char *) &vamana_frozen_num, sizeof(_u64));
    // compute
    _u64 medoid, max_node_len, nnodes_per_sector;
    npts_64 = (_u64) npts;
    medoid = (_u64) medoid_u32;
    if (vamana_frozen_num == 1)
      vamana_frozen_loc = medoid;
    max_node_len =
        (((_u64) width_u32 + 1) * sizeof(unsigned)) + (ndims_64 * sizeof(T));
    nnodes_per_sector = SECTOR_LEN / max_node_len;

    diskann::cout << "medoid: " << medoid << "B" << std::endl;
    diskann::cout << "max_node_len: " << max_node_len << "B" << std::endl;
    diskann::cout << "nnodes_per_sector: " << nnodes_per_sector << "B"
                  << std::endl;

    // SECTOR_LEN buffer for each sector
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
    std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
    unsigned &nnbrs = *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T));
    unsigned *nhood_buf =
        (unsigned *) (node_buf.get() + (ndims_64 * sizeof(T)) +
                      sizeof(unsigned));

    // number of sectors (1 for meta data)
    _u64 n_sectors = ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector;
    _u64 disk_index_file_size = (n_sectors + 1) * SECTOR_LEN;

    std::vector<_u64> output_file_meta;
    output_file_meta.push_back(npts_64);
    output_file_meta.push_back(ndims_64);
    output_file_meta.push_back(medoid);
    output_file_meta.push_back(max_node_len);
    output_file_meta.push_back(nnodes_per_sector);
    output_file_meta.push_back(vamana_frozen_num);
    output_file_meta.push_back(vamana_frozen_loc);
    output_file_meta.push_back(disk_index_file_size);

    diskann_writer.write(sector_buf.get(), SECTOR_LEN);  // write out the empty
                                                         // first sector, will
                                                         // be populated at the
                                                         // end.

    std::unique_ptr<T[]> cur_node_coords = std::make_unique<T[]>(ndims_64);
    diskann::cout << "# sectors: " << n_sectors << std::endl;
    _u64 cur_node_id = 0;
    for (_u64 sector = 0; sector < n_sectors; sector++) {
      if (sector % 100000 == 0) {
        diskann::cout << "Sector #" << sector << "written" << std::endl;
      }
      memset(sector_buf.get(), 0, SECTOR_LEN);
      for (_u64 sector_node_id = 0;
           sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
           sector_node_id++) {
        memset(node_buf.get(), 0, max_node_len);
        // read cur node's nnbrs
        vamana_reader.read((char *) &nnbrs, sizeof(unsigned));

        // sanity checks on nnbrs
        if (nnbrs == 0) {
          diskann::cout << "ERROR. Found point with no out-neighbors; Point#: "
                        << cur_node_id << std::endl;
          exit(-1);
        }

        // read node's nhood
        vamana_reader.read((char *) nhood_buf,
                           (std::min)(nnbrs, width_u32) * sizeof(unsigned));
        if (nnbrs > width_u32) {
          vamana_reader.seekg((nnbrs - width_u32) * sizeof(unsigned),
                              vamana_reader.cur);
        }

        // write coords of node first
        //  T *node_coords = data + ((_u64) ndims_64 * cur_node_id);
        base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
        memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

        // write nnbrs
        *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T)) =
            (std::min)(nnbrs, width_u32);

        // write nhood next
        memcpy(node_buf.get() + ndims_64 * sizeof(T) + sizeof(unsigned),
               nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(unsigned));

        // get offset into sector_buf
        char *sector_node_buf =
            sector_buf.get() + (sector_node_id * max_node_len);

        // copy node buf into sector_node_buf
        memcpy(sector_node_buf, node_buf.get(), max_node_len);
        cur_node_id++;
      }
      // flush sector to disk
      diskann_writer.write(sector_buf.get(), SECTOR_LEN);
    }
    diskann_writer.close();
    size_t tag_bytes_written = 0;

    // frozen point implies dynamic index which must have tags
    if (vamana_frozen_num > 0) {
      std::unique_ptr<TagT[]> mem_index_tags;
      size_t                  nr, nc;
      if (single_file_index)
        diskann::load_bin<TagT>(mem_index_file, mem_index_tags, nr, nc,
                                tags_offset);
      else
        diskann::load_bin<TagT>(tag_file, mem_index_tags, nr, nc, tags_offset);

      if (nr != npts_64 && nc != 1) {
        std::stringstream stream;
        stream << "Error loading tags file. File dims are " << nr << ", " << nc
               << ", but expecting " << npts_64
               << " tags in 1 dimension (bin format)." << std::endl;

        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }

      if (single_file_index) {
        tag_bytes_written = diskann::save_bin<TagT>(
            output_file, mem_index_tags.get(), nr, nc,
            output_file_meta[output_file_meta.size() - 1]);
      } else {
        diskann::save_bin<TagT>(output_file + std::string(".tags"),
                                mem_index_tags.get(), nr, nc);
      }
    } else {
      if (tags_enabled) {
        std::unique_ptr<TagT[]> mem_index_tags;
        size_t                  nr, nc;
        if (single_file_index) {
          diskann::load_bin<TagT>(mem_index_file, mem_index_tags, nr, nc,
                                  tags_offset);
        } else {
          if (!file_exists(tag_file)) {
            diskann::cout << "Static vamana index, tag file " << tag_file
                          << "does not exist. Exiting...." << std::endl;
            exit(-1);
          }

          diskann::load_bin<TagT>(tag_file, mem_index_tags, nr, nc,
                                  tags_offset);
        }

        if (nr != npts_64 && nc != 1) {
          std::stringstream stream;
          stream << "Error loading tags file. File dims are " << nr << ", "
                 << nc << ", but expecting " << npts_64
                 << " tags in 1 dimension (bin format)." << std::endl;

          throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
        }

        if (single_file_index) {
          tag_bytes_written = diskann::save_bin<TagT>(
              output_file, mem_index_tags.get(), nr, nc,
              output_file_meta[output_file_meta.size() - 1]);
        } else {
          diskann::save_bin<TagT>(output_file + std::string(".tags"),
                                  mem_index_tags.get(), nr, nc);
        }
      }
    }

    output_file_meta.push_back(output_file_meta[output_file_meta.size() - 1] +
                               tag_bytes_written);
    diskann::save_bin<_u64>(output_file, output_file_meta.data(),
                            output_file_meta.size(), 1, 0);

    if (single_file_index) {
      add_new_file_to_single_index(output_file, pq_pivots_file);
      add_new_file_to_single_index(output_file, pq_vectors_file);
    }
    diskann::cout << "Output file written." << std::endl;
  }

  template<typename T, typename TagT>
  bool build_disk_index(const char *dataPath, const char *indexFilePath,
                        const char *    indexBuildParameters,
                        diskann::Metric _compareMetric, bool single_file_index,
                        const char *tag_file) {
    std::stringstream parser;
    parser << std::string(indexBuildParameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() != 5 && param_list.size() != 6) {
      diskann::cout
          << "Correct usage of parameters is: R (max degree)"
             " L (indexing list size, should be >= R) "
             " B (RAM limit of final index in GB) "
             " M (memory limit while indexing in GB)"
             " T (number of threads for indexing) "
             " [C (compression ratio for PQ. Overrides parameter value B)] "
          << std::endl;
      return false;
    }

    std::string dataFilePath(dataPath);
    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path =
        index_prefix_path + "_pq_compressed.bin";
    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";
    std::string medoids_path = disk_index_path + "_medoids.bin";
    std::string centroids_path = disk_index_path + "_centroids.bin";
    std::string sample_base_prefix = index_prefix_path + "_sample";

    unsigned R = (unsigned) atoi(param_list[0].c_str());
    unsigned L = (unsigned) atoi(param_list[1].c_str());

    double final_index_ram_limit = get_memory_budget(param_list[2]);
    if (final_index_ram_limit <= 0) {
      diskann::cerr << "Insufficient memory budget (or string was not in right "
                       "format). Should be > 0."
                    << std::endl;
      return false;
    }
    double indexing_ram_budget = (float) atof(param_list[3].c_str());
    if (indexing_ram_budget <= 0) {
      diskann::cerr << "Not building index. Please provide more RAM budget"
                    << std::endl;
      return false;
    }
    _u32 num_threads = (_u32) atoi(param_list[4].c_str());

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
      mkl_set_num_threads(num_threads);
    }

    diskann::cout << "Starting index build: R=" << R << " L=" << L
                  << " Query RAM budget: " << final_index_ram_limit
                  << " Indexing RAM budget: " << indexing_ram_budget
                  << " T: " << num_threads << " Final index will be in "
                  << (single_file_index ? "single file" : "multiple files")
                  << std::endl;

    std::string normalized_file_path = dataFilePath;
    if (_compareMetric == diskann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        diskann::cout << "Cosine metric chosen. Normalizing vectors and "
                         "changing distance to L2 to boost accuracy."
                      << std::endl;

        normalized_file_path =
            std::string(indexFilePath) + "_data.normalized.bin";
        normalize_data_file(dataFilePath, normalized_file_path);
        _compareMetric = diskann::Metric::L2;
      } else {
        diskann::cerr << "WARNING: Cannot normalize integral data types."
                      << " Using cosine distance with integer data types may "
                         "result in poor recall."
                      << " Consider using L2 distance with integral data types."
                      << std::endl;
      }
    }

    auto s = std::chrono::high_resolution_clock::now();

    size_t points_num, dim;

    diskann::get_bin_metadata(normalized_file_path, points_num, dim);
    auto training_set_size =
        PQ_TRAINING_SET_FRACTION * points_num > MAX_PQ_TRAINING_SET_SIZE
            ? MAX_PQ_TRAINING_SET_SIZE
            : (_u32) std::round(PQ_TRAINING_SET_FRACTION * points_num);
    training_set_size = (training_set_size == 0) ? 1 : training_set_size;
    diskann::cout << "(Normalized, if required) file : " << normalized_file_path
                  << " has: " << points_num
                  << " points. Changing training set size to "
                  << training_set_size << " points" << std::endl;

    size_t num_pq_chunks =
        calculate_num_pq_chunks(final_index_ram_limit, points_num, dim);

    size_t train_size, train_dim;
    float *train_data;

    auto   start = std::chrono::high_resolution_clock::now();
    double p_val = ((double) training_set_size / (double) points_num);
    // generates random sample and sets it to train_data and updates train_size
    gen_random_slice<T>(normalized_file_path, p_val, train_data, train_size,
                        train_dim);

    diskann::cout << "Generating PQ pivots with training data of size: "
                  << train_size << " num PQ chunks: " << num_pq_chunks
                  << std::endl;
    generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256,
                       (uint32_t) num_pq_chunks, NUM_KMEANS, pq_pivots_path);
    auto end = std::chrono::high_resolution_clock::now();

    diskann::cout << "Pivots generated in "
                  << std::chrono::duration<double>(end - start).count() << "s."
                  << std::endl;
    start = std::chrono::high_resolution_clock::now();
    generate_pq_data_from_pivots<T>(normalized_file_path, 256,
                                    (uint32_t) num_pq_chunks, pq_pivots_path,
                                    pq_compressed_vectors_path);
    delete[] train_data;
    train_data = nullptr;
    end = std::chrono::high_resolution_clock::now();
    diskann::cout << "Compressed data generated and written in: "
                  << std::chrono::duration<double>(end - start).count() << "s."
                  << std::endl;
    start = std::chrono::high_resolution_clock::now();
    diskann::build_merged_vamana_index<T>(
        normalized_file_path, _compareMetric, single_file_index, L, R, p_val,
        indexing_ram_budget, mem_index_path, medoids_path, centroids_path,
        tag_file);
    end = std::chrono::high_resolution_clock::now();
    diskann::cout << "Vamana index built in: "
                  << std::chrono::duration<double>(end - start).count() << "s."
                  << std::endl;

    if (tag_file == nullptr) {
      diskann::create_disk_layout<T, TagT>(
          mem_index_path, normalized_file_path, "", pq_pivots_path,
          pq_compressed_vectors_path, single_file_index, disk_index_path);
    } else {
      std::string tag_filename = std::string(tag_file);
      diskann::create_disk_layout<T, TagT>(
          mem_index_path, normalized_file_path, tag_filename, pq_pivots_path,
          pq_compressed_vectors_path, single_file_index, disk_index_path);
    }

    double ten_percent_points = std::ceil(points_num * 0.1);
    double num_sample_points = ten_percent_points > MAX_SAMPLE_POINTS_FOR_WARMUP
                                   ? MAX_SAMPLE_POINTS_FOR_WARMUP
                                   : ten_percent_points;
    double sample_sampling_rate = num_sample_points / points_num;
    diskann::cout << "Generating warmup file with " << num_sample_points
                  << " points using a sampling rate of: "
                  << sample_sampling_rate << std::endl;
    gen_random_slice<T>(normalized_file_path, sample_base_prefix,
                        sample_sampling_rate);

    diskann::cout << "Deleting memory index file: " << mem_index_path
                  << std::endl;
    std::remove(mem_index_path.c_str());
    // TODO: This is poor design. The decision to add the ".data" prefix
    // is taken by build_vamana_index. So, we shouldn't repeate it here.
    // Checking to see if we can merge the data and index into one file.
    std::remove((mem_index_path + ".data").c_str());
    if (normalized_file_path != dataFilePath) {
      // then we created a normalized vector file. Delete it.
      diskann::cout << "Deleting normalized vector file: "
                    << normalized_file_path << std::endl;
      std::remove(normalized_file_path.c_str());
    }

    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    diskann::cout << "Indexing time: " << diff.count() << std::endl;
    return true;
  }

  template DISKANN_DLLEXPORT void create_disk_layout<int8_t, uint32_t>(
      const std::string &mem_index_file, const std::string &base_file,
      const std::string &tag_file, const std::string &pq_pivots_file,
      const std::string &pq_compressed_vectors_file, bool single_file_index,
      const std::string &output_file);
  template DISKANN_DLLEXPORT void create_disk_layout<uint8_t, uint32_t>(
      const std::string &mem_index_file, const std::string &base_file,
      const std::string &tag_file, const std::string &pq_pivots_file,
      const std::string &pq_compressed_vectors_file, bool single_file_index,
      const std::string &output_file);
  template DISKANN_DLLEXPORT void create_disk_layout<float, uint32_t>(
      const std::string &mem_index_file, const std::string &base_file,
      const std::string &tag_file, const std::string &pq_pivots_file,
      const std::string &pq_compressed_vectors_file, bool single_file_index,
      const std::string &output_file);
  template DISKANN_DLLEXPORT void create_disk_layout<int8_t, uint64_t>(
      const std::string &mem_index_file, const std::string &base_file,
      const std::string &tag_file, const std::string &pq_pivots_file,
      const std::string &pq_compressed_vectors_file, bool single_file_index,
      const std::string &output_file);
  template DISKANN_DLLEXPORT void create_disk_layout<uint8_t, uint64_t>(
      const std::string &mem_index_file, const std::string &base_file,
      const std::string &tag_file, const std::string &pq_pivots_file,
      const std::string &pq_compressed_vectors_file, bool single_file_index,
      const std::string &output_file);
  template DISKANN_DLLEXPORT void create_disk_layout<float, uint64_t>(
      const std::string &mem_index_file, const std::string &base_file,
      const std::string &tag_file, const std::string &pq_pivots_file,
      const std::string &pq_compressed_vectors_file, bool single_file_index,
      const std::string &output_file);

  template DISKANN_DLLEXPORT int8_t *load_warmup<int8_t>(
      const std::string &cache_warmup_file, uint64_t &warmup_num,
      uint64_t warmup_dim, uint64_t warmup_aligned_dim);
  template DISKANN_DLLEXPORT uint8_t *load_warmup<uint8_t>(
      const std::string &cache_warmup_file, uint64_t &warmup_num,
      uint64_t warmup_dim, uint64_t warmup_aligned_dim);
  template DISKANN_DLLEXPORT float *load_warmup<float>(
      const std::string &cache_warmup_file, uint64_t &warmup_num,
      uint64_t warmup_dim, uint64_t warmup_aligned_dim);

#ifdef EXEC_ENV_OLS
  template DISKANN_DLLEXPORT int8_t *load_warmup<int8_t>(
      MemoryMappedFiles &files, const std::string &cache_warmup_file,
      uint64_t &warmup_num, uint64_t warmup_dim, uint64_t warmup_aligned_dim);
  template DISKANN_DLLEXPORT uint8_t *load_warmup<uint8_t>(
      MemoryMappedFiles &files, const std::string &cache_warmup_file,
      uint64_t &warmup_num, uint64_t warmup_dim, uint64_t warmup_aligned_dim);
  template DISKANN_DLLEXPORT float *load_warmup<float>(
      MemoryMappedFiles &files, const std::string &cache_warmup_file,
      uint64_t &warmup_num, uint64_t warmup_dim, uint64_t warmup_aligned_dim);
#endif

  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<int8_t, uint32_t>(
      std::unique_ptr<diskann::PQFlashIndex<int8_t, uint32_t>> &pFlashIndex,
      int8_t *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);
  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<uint8_t, uint32_t>(
      std::unique_ptr<diskann::PQFlashIndex<uint8_t, uint32_t>> &pFlashIndex,
      uint8_t *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);
  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<float, uint32_t>(
      std::unique_ptr<diskann::PQFlashIndex<float, uint32_t>> &pFlashIndex,
      float *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);

  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<int8_t, uint64_t>(
      std::unique_ptr<diskann::PQFlashIndex<int8_t, uint64_t>> &pFlashIndex,
      int8_t *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);
  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<uint8_t, uint64_t>(
      std::unique_ptr<diskann::PQFlashIndex<uint8_t, uint64_t>> &pFlashIndex,
      uint8_t *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);
  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<float, uint64_t>(
      std::unique_ptr<diskann::PQFlashIndex<float, uint64_t>> &pFlashIndex,
      float *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);

  template DISKANN_DLLEXPORT bool build_disk_index<int8_t, uint32_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric,
      bool singleFileIndex, const char *tag_file);
  template DISKANN_DLLEXPORT bool build_disk_index<uint8_t, uint32_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric,
      bool singleFileIndex, const char *tag_file);
  template DISKANN_DLLEXPORT bool build_disk_index<float, uint32_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric,
      bool singleFileIndex, const char *tag_file);
  template DISKANN_DLLEXPORT bool build_disk_index<int8_t, uint64_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric,
      bool singleFileIndex, const char *tag_file);
  template DISKANN_DLLEXPORT bool build_disk_index<uint8_t, uint64_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric,
      bool singleFileIndex, const char *tag_file);
  template DISKANN_DLLEXPORT bool build_disk_index<float, uint64_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric,
      bool singleFileIndex, const char *tag_file);

  template DISKANN_DLLEXPORT int build_merged_vamana_index<int8_t>(
      std::string base_file, diskann::Metric _compareMetric,
      bool single_file_index, unsigned L, unsigned R, double sampling_rate,
      double ram_budget, std::string mem_index_path, std::string medoids_path,
      std::string centroids_file, const char *tag_file);
  template DISKANN_DLLEXPORT int build_merged_vamana_index<float>(
      std::string base_file, diskann::Metric _compareMetric,
      bool single_file_index, unsigned L, unsigned R, double sampling_rate,
      double ram_budget, std::string mem_index_path, std::string medoids_path,
      std::string centroids_file, const char *tag_file);
  template DISKANN_DLLEXPORT int build_merged_vamana_index<uint8_t>(
      std::string base_file, diskann::Metric _compareMetric,
      bool single_file_index, unsigned L, unsigned R, double sampling_rate,
      double ram_budget, std::string mem_index_path, std::string medoids_path,
      std::string centroids_file, const char *tag_file);
};  // namespace diskann
