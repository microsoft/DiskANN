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

namespace diskann {

  double get_memory_budget(const std::string &mem_budget_str) {
    double mem_ram_budget = atof(mem_budget_str.c_str());
    double final_index_ram_limit = mem_ram_budget;
    if (mem_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB >
        THRESHOLD_FOR_CACHING_IN_GB) {  // slack for space used by cached
                                        // nodes
      final_index_ram_limit = mem_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB;
    }
    return final_index_ram_limit * 1024 * 1024 * 1024;
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
        tie_breaker = recall_at - 1;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs &&
               gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec,
                 res_vec + recall_at);  // change to recall_at for recall k@k or
                                        // dim_or for k@dim_or
      unsigned cur_recall = 0;
      for (auto &v : gt) {
        if (res.find(v) != res.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
  }

  double calculate_range_search_recall(unsigned num_queries, std::vector<std::vector<_u32>> &groundtruth,
                          std::vector<std::vector<_u32>> &our_results) {
    double             total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();

      gt.insert(groundtruth[i].begin(), groundtruth[i].end());
      res.insert(our_results[i].begin(), our_results[i].end()); 
      unsigned cur_recall = 0;
      for (auto &v : gt) {
        if (res.find(v) != res.end()) {
          cur_recall++;
        }
      }
      if (gt.size() != 0)
      total_recall += ((100.0*cur_recall)/gt.size());
      else
      total_recall += 100;
    }
    return total_recall / (num_queries);
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
      size_t actual_file_size = get_file_size(vamana_names[i]);
      size_t expected_file_size;
      vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
      if (actual_file_size != expected_file_size) {
        std::stringstream stream;
        stream << "Error in Vamana Index file " << vamana_names[i]
               << " Actual file size: " << actual_file_size
               << " does not match expected file size: " << expected_file_size
               << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
    }

    size_t merged_index_size = 16;
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

    for (_u64 shard = 0; shard < nshards; shard++) {
      unsigned medoid;
      // read medoid
      vamana_readers[shard].read((char *) &medoid, sizeof(unsigned));
      // rename medoid
      medoid = idmaps[shard][medoid];

      medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      // write renamed medoid
      if (shard == (nshards - 1))  //--> uncomment if running hierarchical
        diskann_writer.write((char *) &medoid, sizeof(unsigned));
    }
    medoid_writer.close();

    diskann::cout << "Starting merge" << std::endl;

    // Gopal. random_shuffle() is deprecated.
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
        // Gopal. random_shuffle() is deprecated.
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

    // Gopal. random_shuffle() is deprecated.
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
  int build_merged_vamana_index(std::string     base_file,
                                diskann::Metric compareMetric, unsigned L,
                                unsigned R, double sampling_rate,
                                double ram_budget, std::string mem_index_path,
                                std::string medoids_file,
                                std::string centroids_file) {
    size_t base_num, base_dim;
    diskann::get_bin_metadata(base_file, base_num, base_dim);

    double full_index_ram =
        ESTIMATE_RAM_USAGE(base_num, base_dim, sizeof(T), R);
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      diskann::cout << "Full index fits in RAM budget, should consume at most "
                    << full_index_ram / (1024 * 1024 * 1024)
                    << "GiBs, so building in one shot" << std::endl;
      diskann::Parameters paras;
      paras.Set<unsigned>("L", (unsigned) L);
      paras.Set<unsigned>("R", (unsigned) R);
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<unsigned>("num_rnds", 2);
      paras.Set<bool>("saturate_graph", 1);
      paras.Set<std::string>("save_path", mem_index_path);

      std::unique_ptr<diskann::Index<T>> _pvamanaIndex =
          std::unique_ptr<diskann::Index<T>>(
              new diskann::Index<T>(compareMetric, base_file.c_str()));
      _pvamanaIndex->build(paras);
      _pvamanaIndex->save(mem_index_path.c_str());
      std::remove(medoids_file.c_str());
      std::remove(centroids_file.c_str());
      return 0;
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

      std::string shard_ids_file = merged_index_prefix + "_subshard-" +
                                   std::to_string(p) + "_ids_uint32.bin";

      retrieve_shard_data_from_ids<T>(base_file, shard_ids_file,
                                      shard_base_file);

      std::string shard_index_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

      diskann::Parameters paras;
      paras.Set<unsigned>("L", L);
      paras.Set<unsigned>("R", (2 * (R / 3)));
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<unsigned>("num_rnds", 2);
      paras.Set<bool>("saturate_graph", 1);
      paras.Set<std::string>("save_path", shard_index_file);

      std::unique_ptr<diskann::Index<T>> _pvamanaIndex =
          std::unique_ptr<diskann::Index<T>>(
              new diskann::Index<T>(compareMetric, shard_base_file.c_str()));
      _pvamanaIndex->build(paras);
      _pvamanaIndex->save(shard_index_file.c_str());
      std::remove(shard_base_file.c_str());
      //      wait_for_keystroke();
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
      std::remove(shard_base_file.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_index_file.c_str());
    }
    return 0;
  }

  // General purpose support for DiskANN interface
  //
  //

  // optimizes the beamwidth to maximize QPS for a given L_search subject to
  // 99.9 latency not blowing up
  template<typename T>
  uint32_t optimize_beamwidth(
      std::unique_ptr<diskann::PQFlashIndex<T>> &pFlashIndex, T *tuning_sample,
      _u64 tuning_sample_num, _u64 tuning_sample_aligned_dim, uint32_t L,
      uint32_t nthreads, uint32_t start_bw) {
    uint32_t cur_bw = start_bw;
    double   max_qps = 0;
    uint32_t best_bw = start_bw;
    bool     stop_flag = false;

    while (!stop_flag) {
      std::vector<uint64_t> tuning_sample_result_ids_64(tuning_sample_num, 0);
      std::vector<float>    tuning_sample_result_dists(tuning_sample_num, 0);
      diskann::QueryStats * stats = new diskann::QueryStats[tuning_sample_num];

      auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
      for (_s64 i = 0; i < (int64_t) tuning_sample_num; i++) {
        pFlashIndex->cached_beam_search(
            tuning_sample + (i * tuning_sample_aligned_dim), 1, L,
            tuning_sample_result_ids_64.data() + (i * 1),
            tuning_sample_result_dists.data() + (i * 1), cur_bw, stats + i);
      }
      auto e = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = e - s;
      double qps = (1.0f * tuning_sample_num) / (1.0f * diff.count());

      double lat_999 = diskann::get_percentile_stats(
          stats, tuning_sample_num, 0.999,
          [](const diskann::QueryStats &stats) { return stats.total_us; });

      double mean_latency = diskann::get_mean_stats(
          stats, tuning_sample_num,
          [](const diskann::QueryStats &stats) { return stats.total_us; });

      if (qps > max_qps && lat_999 < (15000) + mean_latency * 2) {
        max_qps = qps;
        best_bw = cur_bw;
        cur_bw = (uint32_t)(std::ceil)((float) cur_bw * 1.1);
      } else {
        stop_flag = true;
      }
      if (cur_bw > 64)
        stop_flag = true;

      delete[] stats;
    }
    return best_bw;
  }

  template<typename T>
  void create_disk_layout(const std::string base_file,
                          const std::string mem_index_file,
                          const std::string output_file) {
    unsigned npts, ndims;

    // amount to read or write in one shot
    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;
    cached_ifstream base_reader(base_file, read_blk_size);
    base_reader.read((char *) &npts, sizeof(uint32_t));
    base_reader.read((char *) &ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    // create cached reader + writer
    size_t          actual_file_size = get_file_size(mem_index_file);
    cached_ifstream vamana_reader(mem_index_file, read_blk_size);
    cached_ofstream diskann_writer(output_file, write_blk_size);

    // metadata: width, medoid
    unsigned width_u32, medoid_u32;
    size_t   index_file_size;

    vamana_reader.read((char *) &index_file_size, sizeof(uint64_t));
    if (index_file_size != actual_file_size) {
      std::stringstream stream;
      stream << "Vamana Index file size does not match expected size per "
                "meta-data."
             << " file size from file: " << index_file_size
             << " actual file size: " << actual_file_size << std::endl;

      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    vamana_reader.read((char *) &width_u32, sizeof(unsigned));
    vamana_reader.read((char *) &medoid_u32, sizeof(unsigned));

    // compute
    _u64 medoid, max_node_len, nnodes_per_sector;
    npts_64 = (_u64) npts;
    medoid = (_u64) medoid_u32;
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
    // write first sector with metadata
    *(_u64 *) (sector_buf.get() + 0 * sizeof(_u64)) = disk_index_file_size;
    *(_u64 *) (sector_buf.get() + 1 * sizeof(_u64)) = npts_64;
    *(_u64 *) (sector_buf.get() + 2 * sizeof(_u64)) = medoid;
    *(_u64 *) (sector_buf.get() + 3 * sizeof(_u64)) = max_node_len;
    *(_u64 *) (sector_buf.get() + 4 * sizeof(_u64)) = nnodes_per_sector;
    diskann_writer.write(sector_buf.get(), SECTOR_LEN);

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
        assert(nnbrs > 0);
        assert(nnbrs <= width_u32);

        // read node's nhood
        vamana_reader.read((char *) nhood_buf, nnbrs * sizeof(unsigned));

        // write coords of node first
        //  T *node_coords = data + ((_u64) ndims_64 * cur_node_id);
        base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
        memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

        // write nnbrs
        *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T)) = nnbrs;

        // write nhood next
        memcpy(node_buf.get() + ndims_64 * sizeof(T) + sizeof(unsigned),
               nhood_buf, nnbrs * sizeof(unsigned));

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
    diskann::cout << "Output file written." << std::endl;
  }

  template<typename T>
  bool build_disk_index(const char *dataFilePath, const char *indexFilePath,
                        const char *    indexBuildParameters,
                        diskann::Metric compareMetric) {
    std::stringstream parser;
    parser << std::string(indexBuildParameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() != 5 && param_list.size() != 6) {
      diskann::cout
          << "Correct usage of parameters is R (max degree) "
             "L (indexing list size, better if >= R) B (RAM limit of final "
             "index in "
             "GB) M (memory limit while indexing) T (number of threads for "
             "indexing) B' (PQ bytes for disk index: optional parameter for "
             "very large dimensional data)"
          << std::endl;
      return false;
    }

    if (!std::is_same<T, float>::value &&
        compareMetric == diskann::Metric::INNER_PRODUCT) {
      std::stringstream stream;
      stream << "DiskANN currently only supports floating point data for Max "
                "Inner Product Search. "
             << std::endl;
      throw diskann::ANNException(stream.str(), -1);
    }

    _u32 disk_pq_dims = 0;
    bool use_disk_pq = false;

    // if there is a 6th parameter, it means we compress the disk index vectors
    // also using PQ data (for very large dimensionality data). If the provided
    // parameter is 0, it means we store full vectors.
    if (param_list.size() == 6) {
      disk_pq_dims = atoi(param_list[5].c_str());
      use_disk_pq = true;
      if (disk_pq_dims == 0)
        use_disk_pq = false;
    }

    std::string base_file(dataFilePath);
    std::string data_file_to_use = base_file;
    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path =
        index_prefix_path + "_pq_compressed.bin";
    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";
    std::string medoids_path = disk_index_path + "_medoids.bin";
    std::string centroids_path = disk_index_path + "_centroids.bin";
    std::string sample_base_prefix = index_prefix_path + "_sample";
    std::string disk_pq_pivots_path =
        index_prefix_path +
        "_disk.index_pq_pivots.bin";  // optional if disk index is also storing
                                      // pq data
    std::string disk_pq_compressed_vectors_path =  // optional if disk index is
                                                   // also storing pq data
        index_prefix_path + "_disk.index_pq_compressed.bin";

    // output a new base file which contains extra dimension with sqrt(1 -
    // ||x||^2/M^2) for every x, M is max norm of all points. Extra space on
    // disk needed!
    if (compareMetric == diskann::Metric::INNER_PRODUCT) {
      std::cout << "Using Inner Product search, so need to pre-process base "
                   "data into temp file. Please ensure there is additional "
                   "(n*(d+1)*4) bytes for storing pre-processed base vectors, "
                   "apart from the intermin indices and final index."
                << std::endl;
      std::string prepped_base = index_prefix_path + "_prepped_base.bin";
      data_file_to_use = prepped_base;
      float max_norm_of_base =
          diskann::prepare_base_for_inner_products<T>(base_file, prepped_base);
      std::string norm_file = disk_index_path + "_max_base_norm.bin";
      diskann::save_bin<float>(norm_file, &max_norm_of_base, 1, 1);
    }

    unsigned R = (unsigned) atoi(param_list[0].c_str());
    unsigned L = (unsigned) atoi(param_list[1].c_str());

    double final_index_ram_limit = get_memory_budget(param_list[2]);
    if (final_index_ram_limit <= 0) {
      std::cerr << "Insufficient memory budget (or string was not in right "
                   "format). Should be > 0."
                << std::endl;
      return false;
    }
    double indexing_ram_budget = (float) atof(param_list[3].c_str());
    if (indexing_ram_budget <= 0) {
      std::cerr << "Not building index. Please provide more RAM budget"
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
                  << " T: " << num_threads << std::endl;

    auto s = std::chrono::high_resolution_clock::now();

    size_t points_num, dim;

    diskann::get_bin_metadata(data_file_to_use.c_str(), points_num, dim);

    size_t num_pq_chunks =
        (size_t)(std::floor)(_u64(final_index_ram_limit / points_num));

    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks =
        num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

    diskann::cout << "Compressing " << dim << "-dimensional data into "
                  << num_pq_chunks << " bytes per vector." << std::endl;

    size_t train_size, train_dim;
    float *train_data;

    double p_val = ((double) TRAINING_SET_SIZE / (double) points_num);
    // generates random sample and sets it to train_data and updates
    // train_size
    gen_random_slice<T>(data_file_to_use.c_str(), p_val, train_data, train_size,
                        train_dim);

    if (use_disk_pq) {
      if (disk_pq_dims > dim)
        disk_pq_dims = dim;

      std::cout << "Compressing base for disk-PQ into " << disk_pq_dims
                << " chunks " << std::endl;
      generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256,
                         (uint32_t) disk_pq_dims, NUM_KMEANS_REPS,
                         disk_pq_pivots_path, false);
      if (compareMetric == diskann::Metric::INNER_PRODUCT)
        generate_pq_data_from_pivots<float>(
            data_file_to_use.c_str(), 256, (uint32_t) disk_pq_dims,
            disk_pq_pivots_path, disk_pq_compressed_vectors_path);
      else
        generate_pq_data_from_pivots<T>(
            data_file_to_use.c_str(), 256, (uint32_t) disk_pq_dims,
            disk_pq_pivots_path, disk_pq_compressed_vectors_path);
    }
    diskann::cout << "Training data loaded of size " << train_size << std::endl;

    // don't translate data to make zero mean for PQ compression. We must not
    // translate for inner product search.
    bool make_zero_mean = true;
    if (compareMetric == diskann::Metric::INNER_PRODUCT)
      make_zero_mean = false;

    generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256,
                       (uint32_t) num_pq_chunks, NUM_KMEANS_REPS,
                       pq_pivots_path, make_zero_mean);

    generate_pq_data_from_pivots<T>(data_file_to_use.c_str(), 256,
                                    (uint32_t) num_pq_chunks, pq_pivots_path,
                                    pq_compressed_vectors_path);

    delete[] train_data;

    train_data = nullptr;
    MallocExtension::instance()->ReleaseFreeMemory();

    diskann::build_merged_vamana_index<T>(
        data_file_to_use.c_str(), diskann::Metric::L2, L, R, p_val,
        indexing_ram_budget, mem_index_path, medoids_path, centroids_path);

    if (!use_disk_pq) {
      diskann::create_disk_layout<T>(data_file_to_use.c_str(), mem_index_path,
                                     disk_index_path);
    } else
      diskann::create_disk_layout<_u8>(disk_pq_compressed_vectors_path,
                                       mem_index_path, disk_index_path);

    double sample_sampling_rate = (150000.0 / points_num);
    gen_random_slice<T>(data_file_to_use.c_str(), sample_base_prefix,
                        sample_sampling_rate);

    std::remove(mem_index_path.c_str());
    if (use_disk_pq)
      std::remove(disk_pq_compressed_vectors_path.c_str());

    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    diskann::cout << "Indexing time: " << diff.count() << std::endl;

    return true;
  }

  template DISKANN_DLLEXPORT void create_disk_layout<int8_t>(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file);

  template DISKANN_DLLEXPORT void create_disk_layout<uint8_t>(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file);
  template DISKANN_DLLEXPORT void create_disk_layout<float>(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file);

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

  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<int8_t>(
      std::unique_ptr<diskann::PQFlashIndex<int8_t>> &pFlashIndex,
      int8_t *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);
  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<uint8_t>(
      std::unique_ptr<diskann::PQFlashIndex<uint8_t>> &pFlashIndex,
      uint8_t *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);
  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<float>(
      std::unique_ptr<diskann::PQFlashIndex<float>> &pFlashIndex,
      float *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);

  template DISKANN_DLLEXPORT bool build_disk_index<int8_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric compareMetric);
  template DISKANN_DLLEXPORT bool build_disk_index<uint8_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric compareMetric);
  template DISKANN_DLLEXPORT bool build_disk_index<float>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric compareMetric);

  template DISKANN_DLLEXPORT int build_merged_vamana_index<int8_t>(
      std::string base_file, diskann::Metric compareMetric, unsigned L,
      unsigned R, double sampling_rate, double ram_budget,
      std::string mem_index_path, std::string medoids_path,
      std::string centroids_file);
  template DISKANN_DLLEXPORT int build_merged_vamana_index<float>(
      std::string base_file, diskann::Metric compareMetric, unsigned L,
      unsigned R, double sampling_rate, double ram_budget,
      std::string mem_index_path, std::string medoids_path,
      std::string centroids_file);
  template DISKANN_DLLEXPORT int build_merged_vamana_index<uint8_t>(
      std::string base_file, diskann::Metric compareMetric, unsigned L,
      unsigned R, double sampling_rate, double ram_budget,
      std::string mem_index_path, std::string medoids_path,
      std::string centroids_file);
};  // namespace diskann
