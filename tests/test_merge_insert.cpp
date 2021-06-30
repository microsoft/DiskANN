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

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define NUM_INSERT_THREADS 30
#define NUM_DELETE_THREADS 1
#define NUM_SEARCH_THREADS 10

// random number generator
std::random_device dev;
std::mt19937       rng(dev());

tsl::robin_map<std::string, uint32_t> params;
float                                 mem_alpha, merge_alpha;
uint32_t              medoid_id = std::numeric_limits<uint32_t>::max();
std::atomic_bool      _insertions_done(false);
std::vector<uint32_t> Lvec;

template<typename T>
void seed_iter(tsl::robin_set<uint32_t> &active_set,
               tsl::robin_set<uint32_t> &inactive_set, T *all_points,
               const std::string &       inserted_points_file,
               const std::string &       inserted_tags_file,
               tsl::robin_set<uint64_t> &deleted_tags) {
  const uint32_t insert_count = params[std::string("insert_count")];
  const uint32_t delete_count = params[std::string("delete_count")];
  const uint32_t ndims = params[std::string("ndims")];
  diskann::cout << "ITER: start = " << active_set.size() << ", "
                << inactive_set.size() << "\n";

  // pick `delete_count` tags
  std::vector<uint32_t> active_vec(active_set.begin(), active_set.end());
  std::shuffle(active_vec.begin(), active_vec.end(), rng);
  std::vector<uint32_t> delete_vec(active_vec.begin(),
                                   active_vec.begin() + delete_count);
  for (auto iter : delete_vec)
    deleted_tags.insert(iter);
  active_set.clear();
  active_set.insert(active_vec.begin() + delete_count, active_vec.end());
  // pick `insert_count` tags
  std::vector<uint32_t> inactive_vec(inactive_set.begin(), inactive_set.end());
  std::shuffle(inactive_vec.begin(), inactive_vec.end(), rng);
  std::vector<uint32_t> insert_vec(inactive_vec.begin(),
                                   inactive_vec.begin() + insert_count);
  inactive_set.clear();
  diskann::cout << "ITER: INSERT - " << insert_vec.size() << " IDs in "
                << inserted_tags_file << "\n";
  inactive_set.insert(inactive_vec.begin() + insert_count, inactive_vec.end());
  std::sort(insert_vec.begin(), insert_vec.end());
  std::ofstream inserted_tags_writer(inserted_tags_file, std::ios::trunc);
  for (auto &id : insert_vec) {
    inserted_tags_writer << id << std::endl;
  }
  inserted_tags_writer.close();
  std::ofstream inserted_points_writer(inserted_points_file, std::ios::binary);
  T *new_pts = new T[(uint64_t) insert_vec.size() * (uint64_t) ndims];
  for (uint64_t idx = 0; idx < insert_vec.size(); idx++) {
    uint64_t actual_idx = insert_vec[idx];
    T *      src_ptr = all_points + actual_idx * (uint64_t) ndims;
    T *      dest_ptr = new_pts + idx * (uint64_t) ndims;
    std::memcpy(dest_ptr, src_ptr, ndims * sizeof(T));
  }
  uint32_t npts_u32 = insert_vec.size(), ndims_u32 = ndims;
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
  malloc_stats();
}

float compute_active_recall(const uint64_t *result_tags,
                            const uint32_t  result_count,
                            const uint32_t *gs_tags, const uint32_t gs_count,
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

template<typename T>
void search_disk_index(const std::string &             index_prefix_path,
                       const tsl::robin_set<uint32_t> &inactive_tags,
                       const std::string &             query_path,
                       const std::string &             gs_path) {
  std::string pq_prefix = index_prefix_path + "_pq";
  std::string disk_index_file = index_prefix_path + "_disk.index";
  std::string warmup_query_file = index_prefix_path + "_sample_data.bin";
  uint32_t    beamwidth = params[std::string("beam_width")];
  uint32_t    num_threads = params[std::string("disk_search_nthreads")];
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
  diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                         &gt_tags);
  if (gt_num != query_num) {
    diskann::cout
        << "Error. Mismatch in number of queries and ground truth data"
        << std::endl;
  }

  // load PQ Flash Index
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<T, uint64_t>> _pFlashIndex(
      new diskann::PQFlashIndex<T, uint64_t>(reader));
  int res = _pFlashIndex->load(num_threads, pq_prefix.c_str(),
                               disk_index_file.c_str(), true);
  if (res != 0) {
    std::cerr << "Failed to load index.\n";
    exit(-1);
  }

  // prep for search
  std::vector<uint32_t> query_result_ids;
  std::vector<uint64_t> query_result_tags;
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
        nullptr, query_result_tags.data() + (i * recall_at));
  }

  // compute mean recall, IOs
  float mean_recall = 0.0f;
  for (uint32_t i = 0; i < query_num; i++) {
    auto *result_tags = query_result_tags.data() + (i * recall_at);
    auto *gs_tags = gt_tags + (i * gt_dim);
    float query_recall = compute_active_recall(result_tags, recall_at, gs_tags,
                                               gt_dim, inactive_tags);
    mean_recall += query_recall;
  }
  mean_recall /= query_num;
  float mean_ios = (float) diskann::get_mean_stats(
      stats, query_num,
      [](const diskann::QueryStats &stats) { return stats.n_ios; });
  diskann::cout << "PQFlashIndex :: recall-" << recall_at << "@" << recall_at
                << ": " << mean_recall << ", mean IOs: " << mean_ios << "\n";
  diskann::aligned_free(query);
  delete[] stats;
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
}

template<typename T>
void search_kernel(diskann::MergeInsert<T> &       merge_insert,
                   const tsl::robin_set<uint32_t> &inactive_tags,
                   const std::string &query_path, const std::string &gs_path,
                   bool print_stats = false) {
  uint32_t    beamwidth = params[std::string("beam_width")];
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
  diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                         &gt_tags);
  if (gt_num != query_num) {
    diskann::cout
        << "Error. Mismatch in number of queries and ground truth data"
        << std::endl;
  }

  if (print_stats) {
    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
                  << std::setw(18) << "Mean Latency (ms)" << std::setw(15)
                  << "99.9 Latency" << std::setw(12) << recall_string
                  << std::setw(12) << "Mean disk IOs" << std::endl;

    diskann::cout
        << "==============================================================="
           "==============="
        << std::endl;
  }

  // prep for search
  std::vector<uint32_t> query_result_ids;
  std::vector<uint64_t> query_result_tags;
  std::vector<float>    query_result_dists;
  query_result_ids.resize(recall_at * query_num);
  query_result_dists.resize(recall_at * query_num);
  query_result_tags.resize(recall_at * query_num);
  diskann::QueryStats * stats = new diskann::QueryStats[query_num];
  std::vector<uint64_t> query_result_ids_64(recall_at * query_num);

  size_t end_count = 1;
  if (print_stats)
    end_count = ::Lvec.size();
  else
    end_count = 1;
  for (size_t test_id = 0; test_id < end_count; test_id++) {
    uint32_t L = Lvec[test_id];
    if (print_stats) {
      std::vector<double> latency_stats(query_num, 0);
      auto                s = std::chrono::high_resolution_clock::now();
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
      auto e = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = e - s;
      float                         qps = (float) (query_num / diff.count());

      // compute mean recall, IOs
      float mean_recall = 0.0f;
      for (uint32_t i = 0; i < query_num; i++) {
        auto *result_tags = query_result_tags.data() + (i * recall_at);
        auto *gs_tags = gt_tags + (i * gt_dim);
        float query_recall = compute_active_recall(
            result_tags, recall_at, gs_tags, gt_dim, inactive_tags);
        mean_recall += query_recall;
      }
      mean_recall /= query_num;
      float mean_ios = (float) diskann::get_mean_stats(
          stats, query_num,
          [](const diskann::QueryStats &stats) { return stats.n_ios; });
      std::sort(latency_stats.begin(), latency_stats.end());
      diskann::cout << std::setw(4) << L << std::setw(12) << qps
                    << std::setw(18)
                    << std::accumulate(latency_stats.begin(),
                                       latency_stats.end(), 0) /
                           (float) query_num
                    << std::setw(15)
                    << (float) latency_stats[(_u64)(0.999 * query_num)]
                    << std::setw(12) << mean_recall << std::setw(12) << mean_ios
                    << std::endl;
    } else {
      diskann::Timer timer;
#pragma parallel omp for num_threads(NUM_SEARCH_THREADS)
      for (_s64 i = 0; i < (int64_t) query_num; i++) {
        merge_insert.search_sync(query + (i * query_aligned_dim), recall_at, L,
                                 (query_result_tags.data() + (i * recall_at)),
                                 query_result_dists.data() + (i * recall_at),
                                 stats + i);
        //      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      diskann::cout << "Concurrent search over " << NUM_SEARCH_THREADS
                    << " threads in " << timer.elapsed() / 1000 << " ms"
                    << std::endl;
    }
  }
  diskann::aligned_free(query);
  delete[] stats;
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
}

template<typename T, typename TagT = uint64_t>
void insertion_kernel(diskann::MergeInsert<T> &merge_insert,
                      const std::string &      mem_pts_file,
                      const std::string &      mem_tags_file) {
  /*	if(::_insertions_done.load())
      {
          diskann::cout << "Insertions_done is true at the beginning of
     insertion
     kernel" << std::endl;
          exit(-1);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5000)); */
  T *    data_insert = nullptr;
  size_t npts, ndim, aligned_dim;
  diskann::load_aligned_bin<T>(mem_pts_file, data_insert, npts, ndim,
                               aligned_dim);
  std::vector<TagT> tags;
  tags.reserve(npts);
  tags.resize(npts);
  std::ifstream tag_file;
  tag_file = std::ifstream(mem_tags_file);
  if (!tag_file.is_open()) {
    std::cerr << "Tag file not found." << std::endl;
  }
  TagT   tag;
  size_t j = 0;
  while (tag_file >> tag) {
    tags[j] = tag;
    j++;
  }
  diskann::cout << "Tags loaded." << std::endl;
  tag_file.close();
  size_t         i;
  diskann::Timer timer;
#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (i = 0; i < npts; i++) {
    if (merge_insert.insert(data_insert + i * aligned_dim, tags[i]) != 0)
      diskann::cout << "Point " << i << "could not be inserted." << std::endl;
  }
  diskann::cout << "Mem index insertion time : " << timer.elapsed() / 1000
                << " ms" << std::endl;
  ::_insertions_done.store(true);
  delete[] data_insert;
}

template<typename T, typename TagT = uint64_t>
void deletion_kernel(diskann::MergeInsert<T> &merge_insert,
                     tsl::robin_set<TagT> &   del_tags) {
  //	std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  diskann::Timer timer;
  for (auto iter : del_tags) {
    merge_insert.lazy_delete(iter);
  }
  diskann::cout << "Deletion time : " << timer.elapsed() / 1000 << " ms"
                << std::endl;
}

template<typename T>
void merge_kernel(diskann::MergeInsert<T> &merge_insert) {
  while (true) {
    merge_insert.trigger_merge();
    if (::_insertions_done.load()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

/*template<typename T>
void merge_kernel(diskann::MergeInsert<T> &merge_insert) {
    merge_insert.final_merge();
}
*/
template<typename T>
void run_iter(diskann::MergeInsert<T> &merge_insert,
              const std::string &base_prefix, const std::string &merge_prefix,
              const std::string &       mem_prefix,
              tsl::robin_set<uint32_t> &active_set,
              tsl::robin_set<uint32_t> &inactive_set,
              const std::string &query_file, const std::string &gs_file,
              T *data, diskann::Distance<T> *dist_cmp) {
  uint64_t initial_count = active_set.size() + inactive_set.size();
  // files for mem-DiskANN
  std::string mem_pts_file = mem_prefix + ".data_orig";
  std::string mem_index_file = mem_prefix + ".index";
  std::string mem_tags_file = mem_prefix + ".tags_orig";

  diskann::cout << "ITER: Seeding iteration"
                << "\n";
  // seed the iteration
  tsl::robin_set<uint64_t> deleted_tags;
  seed_iter(active_set, inactive_set, data, mem_pts_file, mem_tags_file,
            deleted_tags);
  assert(active_set.size() + inactive_set.size() == initial_count);
  std::future<void> delete_future =
      std::async(std::launch::async, deletion_kernel<T>, std::ref(merge_insert),
                 std::ref(deleted_tags));
  std::future<void> insert_future;
  insert_future = std::async(std::launch::async, insertion_kernel<T>,
                             std::ref(merge_insert), std::ref(mem_pts_file),
                             std::ref(mem_tags_file));
  std::future<void> merge_future =
      std::async(std::launch::async, merge_kernel<T>, std::ref(merge_insert));
  /*  if(::_insertions_done.load())
    {
        ::_insertions_done.store(false);
       insert_future = std::async(
        std::launch::async, insertion_kernel<T>, std::ref(merge_insert),
        std::ref(mem_pts_file), std::ref(mem_tags_file));
    }
    */
  std::future_status insert_status, merge_status, delete_status;

  do {
    insert_status = insert_future.wait_for(std::chrono::milliseconds(1));
    merge_status = merge_future.wait_for(std::chrono::milliseconds(1));
    delete_status = delete_future.wait_for(std::chrono::milliseconds(1));

    if ((insert_status == std::future_status::timeout) ||
        (merge_status == std::future_status::timeout) ||
        (delete_status == std::future_status::timeout)) {
      //      search_kernel<T>(merge_insert, inactive_set, query_file, gs_file);
    } else {
      if ((insert_status == std::future_status::ready) &&
          (merge_status == std::future_status::ready) &&
          (delete_status == std::future_status::ready)) {
        merge_insert.final_merge();
      }
    }
  } while (insert_status != std::future_status::ready ||
           merge_status != std::future_status::ready);

  /*
  do {
    insert_status = insert_future.wait_for(std::chrono::milliseconds(1));
    merge_status = merge_future.wait_for(std::chrono::milliseconds(1));
    delete_status = delete_future.wait_for(std::chrono::milliseconds(1));

    if ((merge_status == std::future_status::timeout) ||
        (delete_status == std::future_status::timeout)) {
      search_kernel<T>(merge_insert, inactive_set, query_file, gs_file);
    } else {
      if ((merge_status == std::future_status::ready) &&
          (delete_status == std::future_status::ready) && (insert_status ==
std::future_status::ready)) {
//        merge_insert.final_merge();
    break;
      }
    }
  } while (delete_status != std::future_status::ready || insert_status !=
std::future_status::ready ||
           merge_status != std::future_status::ready);
*/

  //  diskann::cout << "Searching all indices" << std::endl;
  search_kernel<T>(merge_insert, inactive_set, query_file, gs_file, true);
}

template<typename T>
void run_all_iters(std::string &base_prefix, std::string &merge_prefix,
                   const std::string &mem_prefix, const std::string &data_file,
                   const std::string &active_tags_file,
                   const std::string &query_file, const std::string &gs_file,
                   diskann::Distance<T> *dist_cmp) {
  // load all data points
  std::unique_ptr<T[]> data;
  uint64_t             npts = 0, ndims = 0;
  diskann::load_bin<T>(data_file, data, npts, ndims);
  params[std::string("ndims")] = ndims;

  int n_iters = params["n_iters"];
  // load active tags
  std::ifstream            reader(active_tags_file);
  tsl::robin_set<uint32_t> active_tags;
  uint32_t                 tag;
  while (reader >> tag) {
    active_tags.insert(tag);
  }

  // read medoid ID from base_prefix
  std::ifstream disk_reader(base_prefix + "_disk.index", std::ios::binary);
  disk_reader.seekg(2 * sizeof(uint64_t), std::ios::beg);
  uint64_t medoid = std::numeric_limits<uint64_t>::max();
  disk_reader.read((char *) &medoid, sizeof(uint64_t));
  diskann::cout << "Detected medoid = " << medoid
                << " ==> excluding from insert/deletes.\n";
  ::medoid_id = medoid;

  // generate inactive tags
  tsl::robin_set<uint32_t> inactive_tags;
  for (uint32_t i = 0; i < npts; i++) {
    auto iter = active_tags.find(i);
    if (iter == active_tags.end()) {
      inactive_tags.insert(i);
    }
  }

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

  diskann::MergeInsert<T> merge_insert(paras, 200000, ndims, mem_prefix,
                                       base_prefix, merge_prefix, dist_cmp);
  for (size_t i = 0; i < n_iters; i++) {
    diskann::cout << "ITER : " << i << std::endl;
    run_iter<T>(merge_insert, base_prefix, merge_prefix, mem_prefix,
                active_tags, inactive_tags, query_file, gs_file, data.get(),
                dist_cmp);
  }
  //  merge_kernel(merge_insert);
}

int main(int argc, char **argv) {
  if (argc < 19) {
    diskann::cout
        << "Correct usage: " << argv[0]
        << " <type[int8/uint8/float]> <WORKING_FOLDER> <base_prefix> "
           "<merge_prefix> <mem_prefix> <L_mem> <alpha_mem> <L_disk> "
           "<alpha_disk> "
        << " <full_data_bin> <query_bin> <truthset>"
        << " <n_iters> <total_insert_count> <total_delete_count> <range> "
           "<recall_k> "
           "<search_L1> <search_L2> <search_L3> ...."
        << "\n WARNING: Other parameters set inside CPP source." << std::endl;
    exit(-1);
  }
  diskann::cout.setf(std::ios::unitbuf);

  int         arg_no = 1;
  std::string index_type = argv[arg_no++];
  TMP_FOLDER = argv[arg_no++];
  std::string base_prefix(argv[arg_no++]);
  std::string merge_prefix(argv[arg_no++]);
  std::string mem_prefix(argv[arg_no++]);
  unsigned    L_mem = (unsigned) atoi(argv[arg_no++]);
  float       alpha_mem = (float) atof(argv[arg_no++]);
  unsigned    L_disk = (unsigned) atoi(argv[arg_no++]);
  float       alpha_disk = (float) atof(argv[arg_no++]);
  std::string data_bin(argv[arg_no++]);
  std::string query_bin(argv[arg_no++]);
  std::string gt_bin(argv[arg_no++]);
  int         n_iters = atoi(argv[arg_no++]);
  uint32_t    insert_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    delete_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    range = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    recall_k = (uint32_t) atoi(argv[arg_no++]);

  for (int ctr = arg_no; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
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

  if (index_type == std::string("float")) {
    diskann::DistanceL2 dist_cmp;
    run_all_iters<float>(base_prefix, merge_prefix, mem_prefix, data_bin,
                         base_prefix + "_disk.index.tags", query_bin, gt_bin,
                         &dist_cmp);
  } else if (index_type == std::string("uint8")) {
    diskann::DistanceL2UInt8 dist_cmp;
    run_all_iters<uint8_t>(base_prefix, merge_prefix, mem_prefix, data_bin,
                           base_prefix + "_disk.index.tags", query_bin, gt_bin,
                           &dist_cmp);
  } else if (index_type == std::string("int8")) {
    diskann::DistanceL2Int8 dist_cmp;
    run_all_iters<int8_t>(base_prefix, merge_prefix, mem_prefix, data_bin,
                          base_prefix + "_disk.index.tags", query_bin, gt_bin,
                          &dist_cmp);
  } else {
    diskann::cout << "Unsupported type : " << index_type << "\n";
  }
  diskann::cout << "Exiting\n";
}
