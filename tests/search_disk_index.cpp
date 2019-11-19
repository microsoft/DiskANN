//#include <distances.h>
//#include <indexing.h>
#include <index.h>
#include <math_utils.h>
#include <omp.h>
#include <pq_flash_index.h>
#include <string.h>
#include <time.h>
#include <atomic>
#include <cstring>
#include <iomanip>
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

template<typename T>
int search_disk_index(int argc, char** argv) {
  // load query bin
  T*                query = nullptr;
  size_t            query_num, query_dim, query_aligned_dim;
  std::vector<_u64> Lvec;

  std::string pq_centroids_file(argv[2]);
  std::string compressed_data_file(argv[3]);
  std::string disk_index_file(argv[4]);
  std::string medoids_file(argv[5]);
  std::string cached_list_file(argv[6]);
  std::string query_bin(argv[7]);
  _u64        recall_at = std::atoi(argv[8]);
  _u32        num_threads = std::atoi(argv[9]);
  _u32        beam_width = std::atoi(argv[10]);
  std::string result_output_prefix(argv[11]);

  for (int ctr = 12; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at"
              << std::endl;
    return -1;
  }
  _u32 cache_nlevels = 3;

  std::cout << "Search parameters: #threads: " << num_threads
            << ", beamwidth: " << beam_width << std::endl;

  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);

  bool use_cache_list = false;
  if (file_exists(cached_list_file))
    use_cache_list = true;

  diskann::PQFlashIndex<T> _pFlashIndex;

  int res = _pFlashIndex.load(num_threads, pq_centroids_file.c_str(),
                              compressed_data_file.c_str(),
                              disk_index_file.c_str(), medoids_file.c_str());
  if (res != 0) {
    return res;
  }
  // cache bfs levels
  if (use_cache_list) {
    std::cout << "Caching nodes from bin_file " << cached_list_file
              << std::endl;
    _pFlashIndex.load_cache_from_file(cached_list_file);
  } else {
    std::cout << "Caching BFS levels " << cache_nlevels << " around medoid(s)"
              << std::endl;
    _pFlashIndex.cache_bfs_levels(cache_nlevels);
  }

  omp_set_num_threads(num_threads);

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::cout << std::setw(8) << "Ls" << std::setw(16) << "Avg Latency"
            << std::setw(16) << "99 Latency" << std::setw(16) << "Avg Disk I/Os"
            << std::endl;
  std::cout << "======================================="
               "============"
               "======="
            << std::endl;
  //    _u32*  query_res = new _u32[recall_at * query_num];
  //    float* query_dists = new float[recall_at * query_num];
  std::vector<std::vector<uint64_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];
    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);

    diskann::QueryStats* stats = new diskann::QueryStats[query_num];

    diskann::Timer timer;
// std::cout<<"aligned dim: " << _pFlashIndex->aligned_dim<<std::endl;

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      _pFlashIndex.cached_beam_search(
          query + (i * query_aligned_dim), recall_at, L,
          query_result_ids[test_id].data() + (i * recall_at),
          query_result_dists[test_id].data() + (i * recall_at), beam_width,
          stats + i);
    }

    float mean_latency = diskann::get_percentile_stats(
        stats, query_num, 0.5,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float latency_99 = diskann::get_percentile_stats(
        stats, query_num, 0.99,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float mean_io = diskann::get_percentile_stats(
        stats, query_num, 0.5,
        [](const diskann::QueryStats& stats) { return stats.n_ios; });

    std::cout << std::setw(8) << L << std::setw(16) << mean_latency
              << std::setw(16) << latency_99 << std::setw(16) << mean_io
              << std::endl;
  }

  std::cout << "Done searching. Now saving results " << std::endl;
  _u64      test_id = 0;
  uint32_t* results_u32 = new unsigned[recall_at * query_num];
  for (auto L : Lvec) {
    diskann::convert_types<uint64_t, uint32_t>(
        query_result_ids[test_id].data(), results_u32, query_num, recall_at);
    std::string cur_result_path =
        result_output_prefix + std::to_string(L) + "_idx_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, results_u32, query_num, recall_at);
    //    cur_result_path =
    //        result_output_prefix + std::to_string(L) + "_dist_float.bin";
    //    diskann::save_bin<float>(cur_result_path,
    //    query_result_dists[test_id].data(),
    //                         query_num, recall_at);
    test_id++;
  }
  delete[] results_u32;
  diskann::aligned_free(query);
  return 0;
}

int main(int argc, char** argv) {
  if (argc <= 12) {
    std::cout << "Usage: " << argv[0]
              << " <index_type[float/int8/uint8]>  <pq_centroids_bin> "
                 "<compressed_data_bin> <disk_index_path>  "
                 "<medoids_bin (use \"null\" if none)> <cache_list_bin (use "
                 "\"null\" for none)> "
                 "<query_bin> "
                 "<recall@> <num_threads> <beam_width> <result_output_prefix> "
                 "<L1> <L2> ... "
              << std::endl;
    exit(-1);
  }
  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8"
              << std::endl;
}
