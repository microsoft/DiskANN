// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "v2/index_merger.h"
#include "v2/merge_insert.h"

#include <mutex>
#include <numeric>
#include <random>
#include <omp.h>
#include <cstring>
#include <ctime>
#include <timer.h>
#include <iomanip>
#include <atomic>

#include "aux_utils.h"
#include "utils.h"
#include "math_utils.h"
#include "partition_and_pq.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <pthread.h>
#include <sched.h>

#define NUM_INSERT_THREADS 2
#define NUM_DELETE_THREADS 1
#define NUM_SEARCH_THREADS 6
// random number generator
std::random_device dev;
std::mt19937       rng(dev());

tsl::robin_map<std::string, uint32_t> params;
float                                 mem_alpha, merge_alpha;
uint32_t              medoid_id = std::numeric_limits<uint32_t>::max();
std::atomic_bool      _insertions_done(true);
std::atomic_bool      _del_done(true);
std::vector<uint32_t> Lvec;
std::future<void>     delete_future;
std::future<void>     insert_future;
std::future<void>     merge_future;
diskann::Timer        global_timer;
std::string           all_points_file;
bool                  save_index_as_one_file;
std::string           TMP_FOLDER;

template<typename T, typename TagT = uint32_t>
void seed_iter(tsl::robin_set<uint32_t> &active_set,
               tsl::robin_set<uint32_t> &inactive_set,
               const std::string &       inserted_points_file,
               const std::string &       inserted_tags_file,
               tsl::robin_set<TagT> &    deleted_tags) {
  const uint32_t insert_count = params[std::string("insert_count")];
  const uint32_t delete_count = params[std::string("delete_count")];
  const uint32_t ndims = params[std::string("ndims")];
  std::cout << "ITER: start = " << active_set.size() << ", "
            << inactive_set.size() << "\n";

  // pick `delete_count` tags
  std::vector<uint32_t> active_vec(active_set.begin(), active_set.end());
  std::shuffle(active_vec.begin(), active_vec.end(), rng);
  std::vector<uint32_t> delete_vec;
  if (active_vec.size() < delete_count)
    delete_vec.insert(delete_vec.end(), active_vec.begin(), active_vec.end());
  else
    delete_vec.insert(delete_vec.end(), active_vec.begin(),
                      active_vec.begin() + delete_count);
  for (auto iter : delete_vec)
    deleted_tags.insert(iter);
  active_set.clear();
  active_set.insert(active_vec.begin() + delete_vec.size(), active_vec.end());
  std::cout << "ITER: DELETE - " << delete_vec.size() << " IDs\n";
  // pick `insert_count` tags
  std::vector<uint32_t> inactive_vec(inactive_set.begin(), inactive_set.end());
  std::shuffle(inactive_vec.begin(), inactive_vec.end(), rng);
  std::vector<uint32_t> insert_vec;
  if (inactive_vec.size() < insert_count)
    insert_vec.insert(insert_vec.end(), inactive_vec.begin(),
                      inactive_vec.end());
  else
    insert_vec.insert(insert_vec.end(), inactive_vec.begin(),
                      inactive_vec.begin() + insert_count);
  inactive_set.clear();

  std::cout << "ITER: INSERT - " << insert_vec.size() << " IDs in "
            << inserted_tags_file << "\n";
  inactive_set.insert(inactive_vec.begin() + insert_vec.size(),
                      inactive_vec.end());
  std::sort(insert_vec.begin(), insert_vec.end());
  TagT *tag_data = new TagT[insert_vec.size()];
  for (size_t i = 0; i < insert_vec.size(); i++)
    tag_data[i] = insert_vec[i];
  diskann::save_bin<TagT>(inserted_tags_file, tag_data, insert_vec.size(), 1);
  delete[] tag_data;

  // use ifstream reader to load node coordinates
  std::ifstream base_reader;
  base_reader.open(::all_points_file, std::ios::binary | std::ios::ate);

  base_reader.seekg(2 * sizeof(uint32_t), std::ios::beg);

  std::ofstream inserted_points_writer(inserted_points_file, std::ios::binary);
  T *new_pts = new T[(uint32_t) insert_vec.size() * (uint32_t) ndims];
  for (uint64_t idx = 0; idx < insert_vec.size(); idx++) {
    uint32_t actual_idx = insert_vec[idx];
    T *      point = new T[ndims];
    base_reader.seekg(
        (2 * sizeof(uint32_t) + actual_idx * (uint64_t) ndims * sizeof(T)),
        std::ios::beg);
    base_reader.read((char *) point, ((uint64_t) ndims) * sizeof(T));
    T *dest_ptr = new_pts + idx * (uint64_t) ndims;
    std::memcpy(dest_ptr, point, ndims * sizeof(T));
    delete[] point;
  }
  base_reader.close();

  uint32_t npts_u32 = (uint32_t) insert_vec.size();
  uint32_t ndims_u32 = ndims;
  inserted_points_writer.write((char *) &npts_u32, sizeof(uint32_t));
  inserted_points_writer.write((char *) &ndims_u32, sizeof(uint32_t));
  inserted_points_writer.write(
      (char *) new_pts,
      (uint64_t) insert_vec.size() * (uint64_t) ndims * sizeof(T));
  inserted_points_writer.close();
  delete[] new_pts;

  // balance tags
  inactive_set.insert(delete_vec.begin(), delete_vec.end());
  active_set.insert(insert_vec.begin(), insert_vec.end());

  diskann::cout << "ITER: end = " << active_set.size() << ", "
                << inactive_set.size() << "\n";
#ifndef _WINDOWS
  std::cout << "ITER: end = " << active_set.size() << ", "
            << inactive_set.size() << "\n";
  malloc_stats();
#endif
}

float compute_active_recall(const uint32_t *result_tags,
                            const uint32_t  result_count,
                            const uint32_t *gs_tags, const uint64_t gs_count,
                            const tsl::robin_set<uint32_t> &inactive_set) {
  tsl::robin_set<uint32_t> active_gs;
  for (uint32_t i = 0; i < gs_count && active_gs.size() < result_count; i++) {
    auto iter = inactive_set.find(gs_tags[i]);
    if (iter == inactive_set.end()) {
      active_gs.insert(gs_tags[i]);
    }
  }
  uint32_t match = 0;
  for (uint32_t i = 0; i < result_count; i++) {
    match += (active_gs.find(result_tags[i]) != active_gs.end());
  }
  return ((float) match / (float) result_count) * 100;
}

template<typename T, typename TagT = uint32_t>
void search_disk_index(const std::string &             index_prefix_path,
                       const tsl::robin_set<uint32_t> &inactive_tags,
                       const std::string &             query_path,
                       const std::string &             gs_path) {
  std::string pq_prefix = index_prefix_path + "_pq";
  std::string disk_index_file = index_prefix_path + "_disk.index";
  std::string warmup_query_file = index_prefix_path + "_sample_data.bin";
  uint32_t    beamwidth = params[std::string("beam_width")];
  uint32_t    num_threads = 60;
  std::string query_bin = query_path;
  std::string truthset_bin = gs_path;
  uint64_t    recall_at = params[std::string("recall_k")];
  uint64_t    search_L = ::Lvec[0];
  // hold data
  T *       query = nullptr;
  unsigned *gt_ids = nullptr;
  uint32_t *gt_tags = nullptr;
  float *   gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;

  // load query + truthset
  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);
  diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim);
  if (gt_num != query_num) {
    std::cout << "Error. Mismatch in number of queries and ground truth data"
              << std::endl;
  }

  // load PQ Flash Index
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<T, uint32_t>> _pFlashIndex(
      new diskann::PQFlashIndex<T, uint32_t>(diskann::Metric::L2, reader,
                                             ::save_index_as_one_file, true));
  int res = _pFlashIndex->load(num_threads, pq_prefix.c_str(),
                               disk_index_file.c_str());
  if (res != 0) {
    std::cerr << "Failed to load index.\n";
    exit(-1);
  }

  // prep for search
  std::vector<uint32_t> query_result_ids;
  std::vector<uint32_t> query_result_tags;
  std::vector<float>    query_result_dists;
  query_result_ids.resize(recall_at * query_num);
  query_result_dists.resize(recall_at * query_num);
  query_result_tags.resize(recall_at * query_num);
  diskann::QueryStats * stats = new diskann::QueryStats[query_num];
  std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
#pragma omp parallel for schedule(dynamic, 1)  // num_threads(1)
  for (_s64 i = 0; i < (int64_t) query_num; i++) {
    _pFlashIndex->cached_beam_search(
        query + (i * query_aligned_dim), recall_at, search_L,
        query_result_ids_64.data() + (i * recall_at),
        query_result_dists.data() + (i * recall_at), beamwidth, stats + i,
        query_result_tags.data() + (i * recall_at));
  }

  // compute mean recall, IOs
  float mean_recall = 0.0f;
  for (uint32_t i = 0; i < query_num; i++) {
    auto *result_tags = query_result_tags.data() + (i * recall_at);
    auto *gs_tags = gt_tags + (i * gt_dim);
    float query_recall = compute_active_recall(
        result_tags, (uint32_t) recall_at, gs_tags, gt_dim, inactive_tags);
    mean_recall += query_recall;
  }
  mean_recall /= query_num;

  float mean_ios = (float) diskann::get_mean_stats(
      stats, query_num,
      [](const diskann::QueryStats &stats) { return stats.n_ios; });
  std::cout << "PQFlashIndex :: recall-" << recall_at << "@" << recall_at
            << ": " << mean_recall << ", mean IOs: " << mean_ios << "\n";
  diskann::aligned_free(query);
  delete[] stats;
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
}

template<typename T, typename TagT = uint32_t>
void search_kernel(diskann::MergeInsert<T> &       merge_insert,
                   const tsl::robin_set<uint32_t> &active_tags,
                   const std::string query_path, const std::string gs_path,
                   bool print_stats = false) {
  std::string query_bin = query_path;
  std::string truthset_bin = gs_path;
  uint64_t    recall_at = params[std::string("recall_k")];

  // hold data
  T *       query = nullptr;
  unsigned *gt_ids = nullptr;
  uint32_t *gt_tags = nullptr;
  float *   gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;

  // load query + truthset
  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);
  std::cout << "Loaded query" << std::endl;
  diskann::load_truthset(gs_path, gt_ids, gt_dists, gt_num, gt_dim);
  std::cout << "Loaded gt" << std::endl;
  if (gt_num != query_num) {
    std::cout << "Error. Mismatch in number of queries and ground truth data"
              << std::endl;
  }

  if (print_stats) {
    std::string recall_string = "SS-Recall@" + std::to_string(recall_at);
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(18) << "Mean Latency (ms)" << std::setw(12)
              << "90 Latency" << std::setw(12) << "95 Latency" << std::setw(12)
              << "99 Latency" << std::setw(12) << "99.9 Latency"
              << std::setw(12) << recall_string << std::setw(12)
              << "Mean disk IOs" << std::endl;

    std::cout

        << "==============================================================="
           "==============="
        << std::endl;
  } else {
    std::string recall_string = "Recall@" + std::to_string(recall_at);
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(18) << "Mean Latency (ms)" << std::setw(12)
              << "90 Latency" << std::setw(12) << "95 Latency" << std::setw(12)
              << "99 Latency" << std::setw(12) << "99.9 Latency"
              << std::setw(12) << recall_string << std::setw(12)
              << "Mean disk IOs" << std::endl;
    std::cout
        << "==============================================================="
           "==============="
        << std::endl;
  }

  // prep for search
  std::vector<uint32_t> query_result_ids;
  std::vector<uint32_t> query_result_tags;
  std::vector<float>    query_result_dists;
  query_result_ids.resize(recall_at * query_num);
  query_result_dists.resize(recall_at * query_num);
  query_result_tags.resize(recall_at * query_num);
  std::vector<uint32_t> query_result_ids_32(recall_at * query_num);

  for (size_t test_id = 0; test_id < ::Lvec.size(); test_id++) {
    diskann::QueryStats *stats = new diskann::QueryStats[query_num];
    uint32_t             L = Lvec[test_id];
    std::vector<double>  latency_stats(query_num, 0);
    auto                 s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      merge_insert.search_sync(query + (i * query_aligned_dim), recall_at, L,
                               (query_result_tags.data() + (i * recall_at)),
                               query_result_dists.data() + (i * recall_at),
                               stats + i);
      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000;
      //      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (float) (((double) query_num) / diff.count());
    // compute mean recall, IOs
    float mean_recall = 0.0f;
    mean_recall = diskann::calculate_recall(
        (unsigned) query_num, gt_ids, gt_dists, (unsigned) gt_dim,
        query_result_tags.data(), (unsigned) recall_at, (unsigned) recall_at,
        active_tags);
    //    mean_recall /= (float) query_num;
    float mean_ios = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats &stats) { return stats.n_ios; });
    std::sort(latency_stats.begin(), latency_stats.end());
    std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
              << ((float) std::accumulate(latency_stats.begin(),
                                          latency_stats.end(), 0)) /
                     (float) query_num
              << std::setw(12)
              << (float) latency_stats[(_u64)(0.90 * ((double) query_num))]
              << std::setw(12)
              << (float) latency_stats[(_u64)(0.95 * ((double) query_num))]
              << std::setw(12)
              << (float) latency_stats[(_u64)(0.99 * ((double) query_num))]
              << std::setw(12)
              << (float) latency_stats[(_u64)(0.999 * ((double) query_num))]
              << std::setw(12) << mean_recall << std::setw(12) << mean_ios
              << std::endl;
    delete[] stats;
  }
  diskann::aligned_free(query);
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
}

template<typename T, typename TagT = uint32_t>
void insertion_kernel(diskann::MergeInsert<T> &merge_insert,
                      std::string mem_pts_file, std::string mem_tags_file) {
  if (::_insertions_done.load()) {
    std::cout << "Insertions_done is true at the beginning of insertion kernel"
              << std::endl;
    exit(-1);
  }
  T *    data_insert = nullptr;
  size_t npts, ndim, aligned_dim;
  diskann::load_aligned_bin<T>(mem_pts_file, data_insert, npts, ndim,
                               aligned_dim);
  size_t tag_num, tag_dim;
  TagT * tag_data;
  diskann::load_bin<TagT>(mem_tags_file, tag_data, tag_num, tag_dim);
  if (tag_num != npts) {
    std::cout << "In insertion_kernel(), number of tags loaded is not equal to "
                 "number of points loaded. Exiting....."
              << std::endl;
    exit(-1);
  }
  _s64                i;
  std::vector<double> insert_latencies(npts, 0);
  diskann::Timer      timer;
#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (i = 0; i < (_s64) npts; i++) {
    diskann::Timer insert_timer;
    if (merge_insert.insert(data_insert + i * aligned_dim, tag_data[i]) == 0) {
      insert_latencies[i] = ((double) insert_timer.elapsed());
    } else {
      std::cout << "Point " << i << "could not be inserted." << std::endl;
    }
    if ((i % 1000000 == 0) && (i > 0))
      std::cout << "Inserted another 1M points" << std::endl;
  }
  std::cout << "Mem index insertion time : " << timer.elapsed() / 1000 << " ms"
            << std::endl
            << "10th percentile insertion time : "
            << insert_latencies[(size_t)(0.10 * ((double) npts))] << " microsec"
            << std::endl
            << "50th percentile insertion time : "
            << insert_latencies[(size_t)(0.5 * ((double) npts))] << " microsec"
            << "90th percentile insertion time : "
            << insert_latencies[(size_t)(0.90 * ((double) npts))] << " microsec"
            << std::endl;
  ::_insertions_done.store(true);
  delete[] data_insert;
  delete[] tag_data;
}
template<typename T, typename TagT = uint32_t>
void deletion_kernel(diskann::MergeInsert<T> &merge_insert,
                     tsl::robin_set<uint32_t> del_tags) {
  if (::_del_done.load()) {
    std::cout << "_del_done is already true" << std::endl;
    exit(-1);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  diskann::Timer timer;
  for (auto iter : del_tags) {
    merge_insert.lazy_delete(iter);
  }
  std::cout << "Deletion time : " << timer.elapsed() / 1000 << " ms"
            << std::endl;
  ::_del_done.store(true);
}

template<typename T>
void merge_kernel(diskann::MergeInsert<T> &merge_insert) {
  merge_insert.final_merge();
}

template<typename T, typename TagT = uint32_t>
void run_iter(diskann::MergeInsert<T> & merge_insert,
              const std::string &       mem_prefix,
              tsl::robin_set<uint32_t> &active_set,
              tsl::robin_set<uint32_t> &inactive_set,
              const std::string query_file, const std::string gs_file,
              T *data) {
  // files for mem-DiskANN
  std::string mem_pts_file = mem_prefix + ".data_orig";
  std::string mem_tags_file = mem_prefix + ".tags_orig";
  std::this_thread::sleep_for(std::chrono::seconds(10));

  ::merge_future =
      std::async(std::launch::async, merge_kernel<T>, std::ref(merge_insert));

  while (!(::_insertions_done.load() && ::_del_done.load())) {
    /*    std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                  << " seconds " << std::endl;
        search_kernel<T>(merge_insert, inactive_set, query_file, gs_file);*/
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  }

  if (::_insertions_done.load() && ::_del_done.load()) {
    ::_insertions_done.store(false);
    ::_del_done.store(false);

    /*    std::cout << "Searching all indices" << std::endl;
        std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                  << " seconds " << std::endl;
        search_kernel<T>(merge_insert, inactive_set, query_file, gs_file, true);
        */

    std::cout << "ITER: Seeding iteration"
              << "\n";
    // seed the iteration
    tsl::robin_set<uint32_t> deleted_tags;
    seed_iter<T, TagT>(active_set, inactive_set, mem_pts_file, mem_tags_file,
                       deleted_tags);
    ::delete_future = std::async(std::launch::async, deletion_kernel<T, TagT>,
                                 std::ref(merge_insert), deleted_tags);
    ::insert_future =
        std::async(std::launch::async, insertion_kernel<T>,
                   std::ref(merge_insert), mem_pts_file, mem_tags_file);
  }

  std::future_status merge_status;
  do {
    merge_status = ::merge_future.wait_for(std::chrono::milliseconds(1));
    /*    std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                  << " seconds " << std::endl;
        search_kernel<T>(merge_insert, inactive_set, query_file, gs_file);
        */
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  } while ((merge_status != std::future_status::ready));
}

template<typename T, typename TagT = uint32_t>
void run_iter(diskann::MergeInsert<T> &merge_insert,
              const std::string &base_prefix, const std::string &merge_prefix,
              const std::string &       mem_prefix,
              tsl::robin_set<uint32_t> &active_set,
              tsl::robin_set<uint32_t> &inactive_set,
              const std::string &query_file, const std::string &gs_file,
              diskann::Distance<T> *dist_cmp) {
  // files for mem-DiskANN
  std::string mem_pts_file = mem_prefix + ".data_orig";
  std::string mem_tags_file = mem_prefix + ".tags_orig";
  if (::_insertions_done.load() && ::_del_done.load()) {
    ::_insertions_done.store(false);
    ::_del_done.store(false);

    /*    std::cout << "Searching all indices" << std::endl;
        std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                  << " seconds " << std::endl;
        search_kernel<T>(merge_insert, inactive_set, query_file, gs_file, true);
        */
    std::cout << "ITER: Seeding iteration"
              << "\n";
    // seed the iteration
    tsl::robin_set<uint32_t> deleted_tags;
    seed_iter<T, TagT>(active_set, inactive_set, mem_pts_file, mem_tags_file,
                       deleted_tags);
    ::delete_future = std::async(std::launch::async, deletion_kernel<T, TagT>,
                                 std::ref(merge_insert), deleted_tags);
    ::insert_future =
        std::async(std::launch::async, insertion_kernel<T>,
                   std::ref(merge_insert), mem_pts_file, mem_tags_file);
  }
  std::future_status insert_status, delete_status;
  do {
    insert_status = ::insert_future.wait_for(std::chrono::milliseconds(1));
    delete_status = ::delete_future.wait_for(std::chrono::milliseconds(1));
    std::this_thread::sleep_for(std::chrono::seconds(60));
  } while ((insert_status != std::future_status::ready) ||
           (delete_status != std::future_status::ready));

  ::merge_future =
      std::async(std::launch::async, merge_kernel<T>, std::ref(merge_insert));

  std::future_status merge_status;
  do {
    merge_status = ::merge_future.wait_for(std::chrono::milliseconds(1));
    /*  std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                << " seconds " << std::endl;
        search_kernel<T>(merge_insert, inactive_set, query_file, gs_file);
        */
    //    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  } while ((merge_status != std::future_status::ready));
}

template<typename T, typename TagT = uint32_t>
void run_all_iters(std::string base_prefix, std::string merge_prefix,
                   const std::string mem_prefix, const std::string data_file,
                   const std::string active_tags_file,
                   const std::string query_file, const std::string gs_file,
                   diskann::Distance<T> *dist_cmp) {
  // load all data points
  std::unique_ptr<T[]> data;
  uint64_t             npts = 0, ndims = 0;
  diskann::load_bin<T>(data_file, data, npts, ndims);
  std::cout << "Loaded base bin" << std::endl;
  params[std::string("ndims")] = (uint32_t) ndims;

  uint32_t n_iters = params["n_iters"];
  // load active tags
  tsl::robin_set<uint32_t> active_tags;
  TagT *                   tag_data;
  size_t                   tag_num, tag_dim;
  if (::save_index_as_one_file) {
    uint64_t *metadata;
    size_t    nr, nc;
    diskann::load_bin<uint64_t>(active_tags_file, metadata, nr, nc);
    diskann::load_bin<TagT>(active_tags_file, tag_data, tag_num, tag_dim,
                            metadata[7]);
  } else {
    diskann::load_bin<TagT>(active_tags_file, tag_data, tag_num, tag_dim);
  }

  size_t tags_loaded = 0;
  size_t del_tags_found = 0;
  active_tags.reserve(tag_num);
  for (size_t i = 0; i < tag_num; i++) {
    if (tag_data[i] != std::numeric_limits<uint32_t>::max()) {
      active_tags.insert(tag_data[i]);
      tags_loaded++;
    } else {
      if (del_tags_found < 5)
        std::cout << "Driver file found invalid tag in active tag file : "
                  << tag_data[i] << std::endl;
      del_tags_found++;
    }
  }
  std::cout << "Loaded " << tags_loaded << " tags" << std::endl;
  delete[] tag_data;
  std::cout << del_tags_found
            << " deleted/invalid tags found in active tags file" << std::endl;
  // read medoid ID from base_prefix
  std::ifstream disk_reader(base_prefix + "_disk.index", std::ios::binary);
  disk_reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
  disk_reader.seekg(2 * sizeof(uint64_t), std::ios::cur);
  uint64_t medoid = std::numeric_limits<uint64_t>::max();
  disk_reader.read((char *) &medoid, sizeof(uint64_t));
  std::cout << "Detected medoid = " << medoid
            << " ==> excluding from insert/deletes.\n";
  ::medoid_id = (uint32_t) medoid;

  // generate inactive tags
  tsl::robin_set<uint32_t> inactive_tags;
  inactive_tags.reserve(npts - tag_num);
  for (uint32_t i = 0; i < npts; i++) {
    auto iter = active_tags.find(i);
    if (iter == active_tags.end()) {
      inactive_tags.insert(i);
    }
  }
  std::cout << "Inactive tags : " << inactive_tags.size() << std::endl;
  // remove medoid from active_set
  active_tags.erase(::medoid_id);

  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", params[std::string("mem_l_index")]);
  paras.Set<unsigned>("R_mem", params[std::string("range")]);
  paras.Set<float>("alpha_mem", ::mem_alpha);
  paras.Set<unsigned>("L_disk", params[std::string("merge_l_index")]);
  paras.Set<unsigned>("R_disk", params[std::string("range")]);
  paras.Set<float>("alpha_disk", ::merge_alpha);
  paras.Set<unsigned>("C", params[std::string("merge_maxc")]);
  paras.Set<unsigned>("beamwidth", params[std::string("beam_width")]);
  paras.Set<unsigned>("nodes_to_cache",
                      params[std::string("disk_search_node_cache_count")]);
  paras.Set<unsigned>("num_search_threads",
                      params[std::string("disk_search_nthreads")]);

  const std::string             working_folder = ::TMP_FOLDER;
  diskann::Metric               metric = diskann::Metric::L2;
  diskann::MergeInsert<T, TagT> merge_insert(
      paras, ndims, mem_prefix, base_prefix, merge_prefix, dist_cmp, metric,
      ::save_index_as_one_file, working_folder);
  for (size_t i = 0; i < n_iters; i++) {
    std::cout << "ITER : " << i << std::endl;
    run_iter<T>(merge_insert, mem_prefix, active_tags, inactive_tags,
                query_file, gs_file, data.get());
  }
  /*
  std::cout << "Done running all iterations, now merging any leftover points."
            << std::endl;
  std::future_status merge_status, insert_status, delete_status;
  do {
    merge_status = ::merge_future.wait_for(std::chrono::milliseconds(1));
    insert_status = ::insert_future.wait_for(std::chrono::milliseconds(1));
    delete_status = ::delete_future.wait_for(std::chrono::milliseconds(1));

    //    search_kernel<T>(merge_insert, active_tags, query_file, gs_file,
    //    false);
  } while ((merge_status != std::future_status::ready) ||
           (insert_status != std::future_status::ready) ||
           (delete_status != std::future_status::ready));
  merge_kernel(merge_insert);
  */
  search_kernel<T, TagT>(merge_insert, active_tags, query_file, gs_file, true);
}

int main(int argc, char **argv) {
  if (argc < 20) {
    std::cout << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <WORKING_FOLDER> <base_prefix> "
                 "<merge_prefix> <mem_prefix> <L_mem> <alpha_mem> <L_disk> "
                 "<alpha_disk> "
              << " <full_data_bin> <single_file[0/1]> <query_bin> <truthset>"
              << " <n_iters> <total_insert_count> <total_delete_count> <range> "
                 "<recall_k> "
                 "<search_L1> <search_L2> <search_L3> ...."
              << "\n WARNING: Other parameters set inside CPP source."
              << std::endl;
    exit(-1);
  } else {
    std::cout << "This driver file only works with uint32 type tags"
              << std::endl;
  }
  std::cout.setf(std::ios::unitbuf);

  int         arg_no = 1;
  std::string index_type = argv[arg_no++];
  TMP_FOLDER = argv[arg_no++];
  std::cout << "TMP_FOLDER : " << TMP_FOLDER << std::endl;
  std::string base_prefix(argv[arg_no++]);
  std::string merge_prefix(argv[arg_no++]);
  std::string mem_prefix(argv[arg_no++]);
  unsigned    L_mem = (unsigned) atoi(argv[arg_no++]);
  float       alpha_mem = (float) atof(argv[arg_no++]);
  unsigned    L_disk = (unsigned) atoi(argv[arg_no++]);
  float       alpha_disk = (float) atof(argv[arg_no++]);
  std::string data_bin(argv[arg_no++]);
  int         single_file = atoi(argv[arg_no++]);
  std::string query_bin(argv[arg_no++]);
  std::string gt_bin(argv[arg_no++]);
  int         n_iters = atoi(argv[arg_no++]);
  uint32_t    insert_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    delete_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    range = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    recall_k = (uint32_t) atoi(argv[arg_no++]);

  for (int ctr = arg_no; ctr < argc; ctr++) {
    _u32 curL = std::atoi(argv[ctr]);
    if (curL >= recall_k)
      ::Lvec.push_back(curL);
  }

  params[std::string("n_iters")] = n_iters;
  params[std::string("insert_count")] = insert_count;
  params[std::string("delete_count")] = delete_count;
  params[std::string("range")] = range;
  params[std::string("recall_k")] = recall_k;

  // hard-coded params
  params[std::string("disk_search_node_cache_count")] = 100;
  params[std::string("disk_search_nthreads")] = 16;
  params[std::string("beam_width")] = 4;
  params[std::string("mem_l_index")] = L_mem;
  mem_alpha = alpha_mem;
  merge_alpha = alpha_disk;
  params[std::string("mem_nthreads")] = 32;
  params[std::string("merge_maxc")] = (uint32_t)(range * 2.5);
  params[std::string("merge_l_index")] = L_disk;

  if (single_file == 1)
    ::save_index_as_one_file = true;
  else
    ::save_index_as_one_file = false;

  std::string active_tags_filename;
  if (single_file)
    active_tags_filename = base_prefix + "_disk.index";
  else
    active_tags_filename = base_prefix + "_disk.index.tags";

  if (index_type == std::string("float")) {
    diskann::DistanceL2 dist_cmp;
    run_all_iters<float>(base_prefix, merge_prefix, mem_prefix, data_bin,
                         active_tags_filename, query_bin, gt_bin, &dist_cmp);
  } else if (index_type == std::string("uint8")) {
    diskann::DistanceL2UInt8 dist_cmp;
    run_all_iters<uint8_t>(base_prefix, merge_prefix, mem_prefix, data_bin,
                           active_tags_filename, query_bin, gt_bin, &dist_cmp);
  } else if (index_type == std::string("int8")) {
    diskann::DistanceL2Int8 dist_cmp;
    run_all_iters<int8_t>(base_prefix, merge_prefix, mem_prefix, data_bin,
                          active_tags_filename, query_bin, gt_bin, &dist_cmp);
  } else {
    std::cout << "Unsupported type : " << index_type << "\n";
  }
  std::cout << "Exiting\n";
}
