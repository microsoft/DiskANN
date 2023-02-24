// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
  diskann::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(8) << percentiles[s] << "%";
  }
  diskann::cout << std::endl;
  diskann::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(9) << results[s];
  }
  diskann::cout << std::endl;
}

template <typename T1, typename T2>
void apply_max(T1& x, T2& y) {
    if (x < y) x = y;
}

template<typename T>
int search_disk_index_sharded(
    diskann::Metric& metric, const std::string& index_group_path_prefix,
    const unsigned num_shards,
    const std::string& result_output_prefix, const std::string& query_file,
    std::string& gt_file, const unsigned num_threads, const unsigned recall_at,
    const unsigned beamwidth, const unsigned num_nodes_to_cache,
    const _u32 search_io_limit, const std::vector<unsigned>& Lvec,
    const bool use_reorder_data, const std::string& mode, unsigned num_closest_shards) {
  diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
  else
    diskann::cout << " beamwidth: " << beamwidth << std::flush;
  if (search_io_limit == std::numeric_limits<_u32>::max())
    diskann::cout << "." << std::endl;
  else
    diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

  // load query bin
  T*        query = nullptr;
  unsigned* gt_ids = nullptr;
  float*    gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  // assign queries to shards
  std::vector<std::vector<size_t>> query_ids_for_shard(num_shards);
  if (mode == "all") {
    for (unsigned shard_id = 0; shard_id < num_shards; ++shard_id) {
      // all queries for all shards
      for (size_t query_id = 0; query_id < query_num; ++query_id) {
        query_ids_for_shard[shard_id].push_back(query_id);
      }
    }
  } else if (mode == "kmeans") {
    const std::string centroids_file =
        index_group_path_prefix + "_centroids.bin";
    size_t num_centroids, dim_centroids;
    std::unique_ptr<float[]> centroids;
    diskann::load_bin<float>(centroids_file, centroids, num_centroids,
                             dim_centroids);
    if (num_centroids != num_shards) {
      diskann::cout << "number of centroids not equal to the number of shards"
                    << std::endl;
      return -1;
    }
    if (dim_centroids != query_dim) {
      diskann::cout << "dimension of centroids file not equal to the dimension "
                       "of query file"
                    << std::endl;
      return -1;
    }
    std::unique_ptr<float[]> query_float;
    diskann::convert_types<T, float>(query, query_float.get(), query_num,
                                     query_dim);
    std::unique_ptr<uint32_t[]> closest_centers_ivf =
        std::make_unique<uint32_t[]>(query_num * num_closest_shards);
    math_utils::compute_closest_centers(
        query_float.get(), query_num, query_dim, centroids.get(), num_centroids,
        (size_t) num_closest_shards, closest_centers_ivf.get(), query_ids_for_shard.data());
    // linking of previous line fails under Windows for me :(
    diskann::cout << "number of queries per shard:";
    for (const auto& it : query_ids_for_shard) {
      diskann::cout << " " << it.size();
    }
    diskann::cout << std::endl;
  }

  bool calc_recall_flag = false;
  if (gt_file != std::string("null") && gt_file != std::string("NULL") &&
      file_exists(gt_file)) {
    diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      diskann::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
      return -1;
    }
    calc_recall_flag = true;
  }

  std::vector<std::vector<std::vector<std::pair<float, uint32_t>>>> global_query_result_topK(Lvec.size(),
      std::vector<std::vector<std::pair<float, uint32_t>>>(query_num));
  // global_query_result_topK[test_id][query_id] contains the current top
  // (usually at most K = recall_at) results, as pairs <dist, global_id>,
  // aggregated over (already processed) shards

  // global stats
  std::vector<double> global_time_spent(Lvec.size(), 0.0); // to compute global QPS
  std::vector<std::vector<double>> latency_max_per_shard(Lvec.size(), std::vector<double>(query_num, 0.0));
  std::vector<std::vector<double>> global_n_ios(Lvec.size(), std::vector<double>(query_num, 0.0));
  std::vector<std::vector<double>> global_cpu_us(Lvec.size(), std::vector<double>(query_num, 0.0));

  for (unsigned shard_id = 0; shard_id < num_shards; ++shard_id) {

      std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
      reader.reset(new WindowsAlignedFileReader());
#else
      reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
      reader.reset(new LinuxAlignedFileReader());
#endif

      std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex(
          new diskann::PQFlashIndex<T>(reader, metric));

      const std::string index_path_prefix = index_group_path_prefix + "_subshard-" + std::to_string(shard_id);
      int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

      if (res != 0) {
          return res;
      }

      const std::string local_id_to_global_id_file = index_path_prefix + "_ids_uint32.bin";
      std::vector<unsigned> local_id_to_global_id;
      diskann::read_idmap(local_id_to_global_id_file, local_id_to_global_id);
      // TODO check if local_id_to_global_id_num == number of points in index shard?

      // cache bfs levels
      std::vector<uint32_t> node_list;
      diskann::cout << "Caching " << num_nodes_to_cache
          << " BFS nodes around medoid(s)" << std::endl;
      //_pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
      std::string warmup_query_file = index_path_prefix + "_sample_data.bin";
      if (num_nodes_to_cache > 0)
          _pFlashIndex->generate_cache_list_from_sample_queries(
              warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list);
      _pFlashIndex->load_cache_list(node_list);
      node_list.clear();
      node_list.shrink_to_fit();

      omp_set_num_threads(num_threads);

      uint64_t warmup_L = 20;
      uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
      T* warmup = nullptr;

      if (WARMUP) {
          if (file_exists(warmup_query_file)) {
              diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num,
                  warmup_dim, warmup_aligned_dim);
          } else {
              warmup_num = (std::min)((_u32)150000, (_u32)15000 * num_threads);
              warmup_dim = query_dim;
              warmup_aligned_dim = query_aligned_dim;
              diskann::alloc_aligned(((void**)&warmup),
                  warmup_num * warmup_aligned_dim * sizeof(T),
                  8 * sizeof(T));
              std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
              std::random_device              rd;
              std::mt19937                    gen(rd());
              std::uniform_int_distribution<> dis(-128, 127);
              for (uint32_t i = 0; i < warmup_num; i++) {
                  for (uint32_t d = 0; d < warmup_dim; d++) {
                      warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                  }
              }
          }
          diskann::cout << "Warming up index... " << std::flush;
          std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
          std::vector<float>    warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
          for (_s64 i = 0; i < (int64_t)warmup_num; i++) {
              _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1,
                  warmup_L,
                  warmup_result_ids_64.data() + (i * 1),
                  warmup_result_dists.data() + (i * 1), 4);
          }
          diskann::cout << "..done" << std::endl;
      }

      diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
      diskann::cout.precision(2);

      std::string recall_string = "Recall@" + std::to_string(recall_at);
      diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth"
          << std::setw(16) << "QPS" << std::setw(16) << "QPS/thread" << std::setw(16) << "Mean Latency"
          << std::setw(16) << "99.9 Latency" << std::setw(16)
          << "Mean IOs" << std::setw(16) << "CPU (s)";
      //if (calc_recall_flag) {
      //    diskann::cout << std::setw(16) << recall_string << std::endl;
      //} else
          diskann::cout << std::endl;
      diskann::cout
          << "==============================================================="
          "======================================================="
          << std::endl;

      std::vector<std::vector<uint32_t>> shard_query_result_local_ids(Lvec.size());
      std::vector<std::vector<uint32_t>> shard_query_result_global_ids(Lvec.size());
      std::vector<std::vector<float>>    shard_query_result_dists(Lvec.size());

      uint32_t optimized_beamwidth = 2;

      const size_t query_num_this_shard = query_ids_for_shard[shard_id].size();

      for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
          _u64 L = Lvec[test_id];

          // we DO NOT ignore L < K in the sharded version

          const unsigned minKL = L < recall_at ? L : recall_at;

          if (beamwidth <= 0) {
              diskann::cout << "Tuning beamwidth.." << std::endl;
              optimized_beamwidth =
                  optimize_beamwidth(_pFlashIndex, warmup, warmup_num,
                      warmup_aligned_dim, L, optimized_beamwidth);
          } else
              optimized_beamwidth = beamwidth;

          shard_query_result_local_ids[test_id].resize(minKL *
                                                       query_num_this_shard);
          shard_query_result_global_ids[test_id].resize(minKL *
                                                        query_num_this_shard);
          shard_query_result_dists[test_id].resize(minKL *
                                                   query_num_this_shard);

          auto local_stats =
              std::make_unique<diskann::QueryStats[]>(query_num_this_shard);

          std::vector<uint64_t> shard_query_result_local_ids_64(
              minKL * query_num_this_shard);
          auto                  s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
          for (_s64 j = 0; j < (_s64)query_num_this_shard; ++j) {
              const _s64 i = query_ids_for_shard[shard_id][j];
              _pFlashIndex->cached_beam_search(
                  query + (i * query_aligned_dim), minKL, L,
                  shard_query_result_local_ids_64.data() + (j * minKL),
                  shard_query_result_dists[test_id].data() + (j * minKL),
                  optimized_beamwidth, search_io_limit, use_reorder_data, local_stats.get() + j);
          }
          auto                          e = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> diff = e - s;
          const float local_qps = (1.0 * query_num_this_shard) / (1.0 * diff.count());
          const float local_qps_per_thread = local_qps / num_threads;

          diskann::convert_types<uint64_t, uint32_t>(shard_query_result_local_ids_64.data(),
              shard_query_result_local_ids[test_id].data(),
              query_num_this_shard, minKL);

          // renumber shard_query_result_local_ids to shard_query_result_global_ids
          for (_s64 j = 0; j < (int64_t)query_num_this_shard * minKL; ++j) {
              shard_query_result_global_ids[test_id][j] =
                  local_id_to_global_id[shard_query_result_local_ids[test_id][j]];
          }

          // copy shard_query_result_ids[test_id] to global_query_result_topK[test_id]
          for (size_t j = 0; j < query_num_this_shard; ++j) {
              const size_t i = query_ids_for_shard[shard_id][j];
              for (unsigned k = 0; k < minKL; ++k) {
                  global_query_result_topK[test_id][i].emplace_back(
                      shard_query_result_dists[test_id][j * minKL + k],
                      shard_query_result_global_ids[test_id][j * minKL + k]
                  );
              }
              // sort global_query_result_topK[test_id] and leave only best K points
              std::sort(global_query_result_topK[test_id][i].begin(),
                  global_query_result_topK[test_id][i].end());
              if (global_query_result_topK[test_id][i].size() > recall_at) {
                  global_query_result_topK[test_id][i].erase(
                      global_query_result_topK[test_id][i].begin() + recall_at,
                      global_query_result_topK[test_id][i].end()
                  );
              }
          }

          // aggregate local into global stats
          global_time_spent[test_id] += diff.count();
          for (size_t j = 0; j < query_num_this_shard; ++j) {
              const size_t i = query_ids_for_shard[shard_id][j];
              apply_max(latency_max_per_shard[test_id][i], local_stats[j].total_us);
              global_n_ios[test_id][i] += local_stats[j].n_ios;
              global_cpu_us[test_id][i] += local_stats[j].cpu_us;
          }

          auto local_mean_latency = diskann::get_mean_stats<float>(
              local_stats.get(), query_num_this_shard,
              [](const diskann::QueryStats& stats) { return stats.total_us; });

          auto local_latency_999 = diskann::get_percentile_stats<float>(
              local_stats.get(), query_num_this_shard, 0.999,
              [](const diskann::QueryStats& stats) { return stats.total_us; });

          auto local_mean_ios = diskann::get_mean_stats<unsigned>(
              local_stats.get(), query_num_this_shard,
              [](const diskann::QueryStats& stats) { return stats.n_ios; });

          auto local_mean_cpuus = diskann::get_mean_stats<float>(
              local_stats.get(), query_num_this_shard,
              [](const diskann::QueryStats& stats) { return stats.cpu_us; });

          /*
          float shard_recall = 0;
          if (calc_recall_flag) {
              std::vector<uint32_t> resids; // = shard_query_result_global_ids[test_id];
              // `resids` is shard_query_result_ids[test_id] but filled up with junk:
              // if L < recall_at, then minKL = L < recall_at = K,
              // which could be an issue for `calculate_recall`,
              // so we fill up the vector with extra junk elements
              if (L < recall_at) {
                  // TODO if wanting to use this, fill up resids with some junk
              }

              shard_recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                  resids.data(),
                  recall_at, recall_at);
          }
          */

          diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth
              << std::setw(16) << local_qps
              << std::setw(16) << local_qps_per_thread
              << std::setw(16) << local_mean_latency
              << std::setw(16) << local_latency_999
              << std::setw(16) << local_mean_ios
              << std::setw(16) << local_mean_cpuus;
          //if (calc_recall_flag) {
          //    diskann::cout << std::setw(16) << shard_recall << std::endl;
          //}
          //else
              diskann::cout << std::endl;
      } // end loop over L-values

      if (warmup != nullptr)
          diskann::aligned_free(warmup);

  } // end loop over shards


  // now compute and display aggregate statistics over all shards
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  diskann::cout << std::setw(6) << "L"
      //<< std::setw(12) << "Beamwidth"
      << std::setw(16) << "QPS"
      << std::setw(16) << "QPS/thread"
      << std::setw(16) << "Mean Latency"
      << std::setw(16) << "99.9 Latency"
      << std::setw(16) << "Mean IOs"
      << std::setw(16) << "CPU (s)";
  if (calc_recall_flag) {
      diskann::cout << std::setw(16) << recall_string << std::endl;
  } else {
      diskann::cout << std::endl;
  }
  diskann::cout
      << "==============================================================="
      "======================================================================="
      << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
      _u64 L = Lvec[test_id];

      const float global_qps = (1.0 * query_num) / (1.0 * global_time_spent[test_id]);
      const float global_qps_per_thread = global_qps / num_threads;
      const double global_mean_latency = std::accumulate(latency_max_per_shard[test_id].begin(),
          latency_max_per_shard[test_id].end(), 0.0) / query_num;
      std::sort(latency_max_per_shard[test_id].begin(), latency_max_per_shard[test_id].end());
      const double global_latency_999 = latency_max_per_shard[test_id][(uint64_t)(0.999 * query_num)];
      const double global_mean_ios = std::accumulate(global_n_ios[test_id].begin(), global_n_ios[test_id].end(), 0.0) / query_num;
      const double global_mean_cpuus = std::accumulate(global_cpu_us[test_id].begin(), global_cpu_us[test_id].end(), 0.0) / query_num;

      float global_recall = 0;
      if (calc_recall_flag) {
          if (num_shards * L < recall_at) {
              continue; // ignore this L
              // TODO: handle this better?
          }

          std::vector<uint32_t> global_query_result_topK_ids;
          for (_s64 i = 0; i < (int64_t)query_num; i++) {
              if (global_query_result_topK[test_id][i].size() != recall_at) {
                  diskann::cout << "implementation error?" << std::endl;
                  return -1;
              }
              for (const std::pair<float, uint32_t>& p : global_query_result_topK[test_id][i]) {
                  global_query_result_topK_ids.push_back(p.second);
              }
          }

          global_recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
              global_query_result_topK_ids.data(),
              recall_at, recall_at);
      }

      diskann::cout << std::setw(6) << L
          //<< std::setw(12) << optimized_beamwidth
          << std::setw(16) << global_qps
          << std::setw(16) << global_qps_per_thread
          << std::setw(16) << global_mean_latency
          << std::setw(16) << global_latency_999
          << std::setw(16) << global_mean_ios
          << std::setw(16) << global_mean_cpuus;
      if (calc_recall_flag) {
          diskann::cout << std::setw(16) << global_recall << std::endl;
      }
  }

  diskann::aligned_free(query);
  return 0;
}

// differences between this and the base search_disk_index script:
// * index_path_prefix -> index_group_path_prefix (will be appended with: _subshard-1.bin etc.)
// * num_shards
// no fail_if_recall_below
// results are not saved (result_path can be given but will be ignored)

int main(int argc, char** argv) {
  std::string data_type, dist_fn, index_group_path_prefix, result_path_prefix,
      query_file, gt_file, mode;
  unsigned              num_threads, K, W, num_nodes_to_cache, search_io_limit, num_shards, num_closest_shards;
  std::vector<unsigned> Lvec;
  bool                  use_reorder_data = false;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips/fast_l2>");
    desc.add_options()("index_group_path_prefix",
                       po::value<std::string>(&index_group_path_prefix)->required(),
                       "Path prefix to the index shards");
    desc.add_options()("result_path",
                       po::value<std::string>(&result_path_prefix)->default_value(std::string("null")),
                       "Path prefix for saving results of the queries");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
    desc.add_options()(
        "gt_file",
        po::value<std::string>(&gt_file)->default_value(std::string("null")),
        "ground truth file for the queryset");
    desc.add_options()("num_shards", po::value<uint32_t>(&num_shards)->required(),
        "Number of index shards");
    desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
        "Number of neighbors to be returned");
    desc.add_options()("search_list,L",
                       po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                       "List of L values of search");
    desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                       "Beamwidth for search. Set 0 to optimize internally.");
    desc.add_options()(
        "num_nodes_to_cache",
        po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
        "Beamwidth for search");
    desc.add_options()("search_io_limit",
                       po::value<uint32_t>(&search_io_limit)
                           ->default_value(std::numeric_limits<_u32>::max()),
                       "Max #IOs for search");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("use_reorder_data",
                       po::bool_switch()->default_value(false),
                       "Include full precision data in the index. Use only in "
                       "conjuction with compressed data on SSD.");
    desc.add_options()(
        "mode",
        po::value<std::string>(&mode)->default_value(std::string("all")),
        "Mode of execution: which shards to ask each query: all (default) / "
        "kmeans");
    desc.add_options()(
        "num_closest_shards",
        po::value<unsigned>(&num_closest_shards)->default_value(0),
        "Number of closest shards to ask (in kmeans mode)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
    if (vm["use_reorder_data"].as<bool>())
      use_reorder_data = true;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if (dist_fn == std::string("mips")) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product/Cosine are supported."
              << std::endl;
    return -1;
  }

  if (mode != "all" && mode != "kmeans") {
    std::cout << "mode should be kmeans or all" << std::endl;
    return -1;
  }

  if (mode == "kmeans" && num_closest_shards <= 0) {
    std::cout
        << "when mode=kmeans, num_closest_shards must be given and positive"
        << std::endl;
  }

  if ((data_type != std::string("float")) &&
      (metric == diskann::Metric::INNER_PRODUCT)) {
    std::cout << "Currently support only floating point data for Inner Product."
              << std::endl;
    return -1;
  }

  if (use_reorder_data && data_type != std::string("float")) {
    std::cout << "Error: Reorder data for reordering currently only "
                 "supported for float data type."
              << std::endl;
    return -1;
  }

  try {
    if (data_type == std::string("float"))
      return search_disk_index_sharded<float>(
          metric, index_group_path_prefix, num_shards, result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
          use_reorder_data, mode, num_closest_shards);
    else if (data_type == std::string("int8"))
      return search_disk_index_sharded<int8_t>(
          metric, index_group_path_prefix, num_shards, result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec, use_reorder_data, mode, num_closest_shards);
    else if (data_type == std::string("uint8"))
      return search_disk_index_sharded<uint8_t>(
          metric, index_group_path_prefix, num_shards, result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec, use_reorder_data, mode, num_closest_shards);
    else {
      std::cerr << "Unsupported data type. Use float or int8 or uint8"
                << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index search failed." << std::endl;
    return -1;
  }
}
