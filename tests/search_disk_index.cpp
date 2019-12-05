//#include <distances.h>
//#include <indexing.h>
#include <index.h>
#include <math_utils.h>
#include <omp.h>
#include <pq_flash_index.h>
#include <string.h>
#include <time.h>
#include <atomic>
#include <set>
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

float calc_recall_set(unsigned num_queries, unsigned* gold_std, unsigned dim_gs,
                      unsigned* our_results, unsigned dim_or,
                      unsigned recall_at, unsigned subset_size) {
  unsigned           total_recall = 0;
  std::set<unsigned> gt, res;

  for (size_t i = 0; i < num_queries; i++) {
    gt.clear();
    res.clear();
    unsigned* gt_vec = gold_std + dim_gs * i;
    unsigned* res_vec = our_results + dim_or * i;
    gt.insert(gt_vec, gt_vec + recall_at);
    res.insert(res_vec, res_vec + subset_size);
    unsigned cur_recall = 0;
    for (auto& v : gt) {
      if (res.find(v) != res.end()) {
        cur_recall++;
      }
    }
    // std::cout << " idx: " << i << ", interesection: " << cur_recall << "\n";
    total_recall += cur_recall;
  }
  return ((float) total_recall) / ((float) num_queries) *
         (100.0 / ((float) recall_at));
}

template<typename T>
int search_disk_index(int argc, char** argv) {
  // load query bin
  T*                query = nullptr;
  unsigned*         gt_ids = nullptr;
  size_t            query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  std::vector<_u64> Lvec;

  std::string pq_centroids_file(argv[2]);
  std::string compressed_data_file(argv[3]);
  std::string disk_index_file(argv[4]);
  std::string medoids_file(argv[5]);
  std::string centroid_data_file(argv[6]);
  std::string cached_list_file(argv[7]);
  std::string query_bin(argv[8]);
  std::string gt_ids_bin(argv[9]);
  _u64        recall_at = std::atoi(argv[10]);
  _u32        num_threads = std::atoi(argv[11]);
  _u32        beam_width = std::atoi(argv[12]);
  std::string result_output_prefix(argv[13]);

  bool calc_recall_flag = false;

  for (int ctr = 14; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at"
              << std::endl;
    return -1;
  }
  _u32 cache_nlevels = 0;

  std::cout << "Search parameters: #threads: " << num_threads
            << ", beamwidth: " << beam_width << std::endl;

  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);

  if (file_exists(gt_ids_bin)) {
    diskann::load_bin<unsigned>(gt_ids_bin, gt_ids, gt_num, gt_dim);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data"
                << std::endl;
    }
    calc_recall_flag = true;
  }
  bool use_cache_list = false;
  if (file_exists(cached_list_file))
    use_cache_list = true;

  diskann::PQFlashIndex<T> _pFlashIndex;

  int res =
      _pFlashIndex.load(num_threads, pq_centroids_file.c_str(),
                        compressed_data_file.c_str(), disk_index_file.c_str());

  _pFlashIndex.load_entry_points(medoids_file, centroid_data_file);
  _pFlashIndex.cache_medoid_nhoods();

  /*      std::string inner_data_file = disk_index_file + "_inner_data.bin";
    std::string inner_remap_file = disk_index_file + "_inner_remap.bin";
    std::string inner_index_file = disk_index_file + "_inner_mem.index";
    if (file_exists(inner_data_file) && file_exists(inner_remap_file) &&
        file_exists(inner_index_file))
      _pFlashIndex.load_inner_index(inner_data_file.c_str(),
                                    inner_index_file.c_str(),
                                    inner_remap_file.c_str());
  */
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

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(16) << "Avg Latency"
            << std::setw(16) << "99 Latency" << std::setw(16) << "Avg Disk I/Os"
            << std::setw(12) << recall_string << std::endl;
  std::cout << "==================================================="
               "============"
               "======="
            << std::endl;
  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];
    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);

    diskann::QueryStats* stats = new diskann::QueryStats[query_num];

    diskann::Timer timer;

    std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
#pragma omp               parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      _pFlashIndex.cached_beam_search(
          query + (i * query_aligned_dim), recall_at, L,
          query_result_ids_64.data() + (i * recall_at),
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

    diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(),
                                               query_result_ids[test_id].data(),
                                               query_num, recall_at);

    float recall = 0;
    if (calc_recall_flag)
      recall = calc_recall_set(query_num, gt_ids, gt_dim,
                               query_result_ids[test_id].data(), recall_at,
                               recall_at, recall_at);

    std::cout << std::setw(4) << L << std::setw(16) << mean_latency
              << std::setw(16) << latency_99 << std::setw(16) << mean_io
              << std::setw(12) << recall << std::endl;
  }

  std::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  for (auto L : Lvec) {
    std::string cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, query_result_ids[test_id++].data(),
                            query_num, recall_at);
  }
  diskann::aligned_free(query);
  return 0;
}

int main(int argc, char** argv) {
  if (argc <= 14) {
    std::cout << "Usage: " << argv[0]
              << " <index_type[float/int8/uint8]>  <pq_centroids_bin> "
                 "<compressed_data_bin> <disk_index_path> "
                 "<medoids_bin (use \"null\" if none)> "
                 "<centroids_vectors_float> (use \"null\" if medoids file is "
                 "null) <cache_list_bin (use "
                 "\"null\" for none)> "
                 "<query_bin> <groundtruth_bin> (use \"null\" for none) "
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
