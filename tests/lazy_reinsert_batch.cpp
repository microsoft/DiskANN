// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <index.h>
#include <iomanip>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <string.h>

#include "aux_utils.h"
#include "utils.h"
#include "tsl/robin_set.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

template<typename T, typename TagT>
void search_kernel(
    T* query, size_t query_num, size_t query_aligned_dim, const int recall_at,
    std::vector<_u64> Lvec, diskann::Index<T, TagT>& index,
    const std::string&       truthset_file,
    tsl::robin_set<unsigned> active_tags = tsl::robin_set<unsigned>()) {
  unsigned* gt_ids = NULL;
  unsigned* gt_tags = NULL;
  float*    gt_dists = NULL;
  size_t    gt_num, gt_dim;
  diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim,
                         &gt_tags);

  float*    query_result_dists = new float[recall_at * query_num];
  unsigned* query_result_ids = new unsigned[recall_at * query_num];
  TagT*     query_result_tags = new TagT[recall_at * query_num];
  memset(query_result_dists, 0, sizeof(float) * recall_at * query_num);
  memset(query_result_tags, 0, sizeof(TagT) * recall_at * query_num);
  memset(query_result_ids, 0, sizeof(unsigned) * recall_at * query_num);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  diskann::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
                << std::setw(18) << "Mean Latency (ms)" << std::setw(15)
                << "99.9 Latency" << std::setw(12) << recall_string
                << std::endl;

  diskann::cout
      << "==============================================================="
         "==============="
      << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    std::vector<double> latency_stats(query_num, 0);
    memset(query_result_dists, 0, sizeof(float) * recall_at * query_num);
    memset(query_result_tags, 0, sizeof(TagT) * recall_at * query_num);
    memset(query_result_ids, 0, sizeof(unsigned) * recall_at * query_num);
    _u64 L = Lvec[test_id];
    auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      index.search_with_tags(query + i * query_aligned_dim, recall_at, (_u32) L,
                             query_result_tags + i * recall_at,
                             query_result_dists + i * recall_at);
      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000;
      //      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;
    float                         qps = (float) (query_num / diff.count());

    float recall;
    if (active_tags.size() > 0) {
      recall = (float) diskann::calculate_recall(
          (_u32) query_num, gt_ids, gt_dists, (_u32) gt_dim, query_result_tags,
          (_u32) recall_at, (_u32) recall_at, active_tags);
    } else {
      recall = (float) diskann::calculate_recall(
          (_u32) query_num, gt_ids, gt_dists, (_u32) gt_dim, query_result_tags,
          (_u32) recall_at, (_u32) recall_at);
    }

    std::sort(latency_stats.begin(), latency_stats.end());
    diskann::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
                  << std::accumulate(latency_stats.begin(), latency_stats.end(),
                                     0) /
                         (float) query_num
                  << std::setw(15)
                  << (float) latency_stats[(_u64)(0.999 * query_num)]
                  << std::setw(12) << recall << std::endl;
  }
  delete[] query_result_dists;
  delete[] query_result_ids;
  delete[] query_result_tags;
}

template<typename T, typename TagT>
int build_incremental_index(const std::string& data_path,
                            const std::string& memory_index_file,
                            const unsigned L, const unsigned R,
                            const unsigned C, const unsigned num_rnds,
                            const float alpha, const std::string& save_path,
                            const unsigned num_cycles, int fraction,
                            const std::string& query_file,
                            const std::string& truthset_file,
                            const int recall_at, std::vector<_u64> Lvec) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<unsigned>("num_rnds", num_rnds);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  diskann::Index<T, TagT> index(diskann::Metric::L2, dim, num_points + 100, 1,
                                false, true, 0);

  auto tag_path = memory_index_file + ".tags";
  index.load(memory_index_file.c_str());
  diskann::cout << "Loaded index and tags and data" << std::endl;
  T*     query = NULL;
  size_t query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);
  diskann::cout << "Search on static index" << std::endl;
  search_kernel(query, query_num, query_aligned_dim, recall_at, Lvec, index,
                truthset_file);
  unsigned i = 0;
  while (i < num_cycles) {
    size_t                   delete_size = (num_points / 100) * fraction;
    tsl::robin_set<unsigned> delete_set;
    while (delete_set.size() < delete_size)
      delete_set.insert(rand() % num_points);
    std::vector<TagT> delete_vector(delete_set.begin(), delete_set.end());
    tsl::robin_set<unsigned> active_tags;
    for (size_t j = 0; j < num_points; j++) {
      if (delete_set.find(j) == delete_set.end()) {
        active_tags.insert(j);
      }
    }
    diskann::cout << "\nDeleting " << delete_vector.size() << " elements... ";

    {
      index.enable_delete();
      std::vector<TagT> failed_tags;
      if (index.lazy_delete(delete_set, failed_tags) < 0) {
        std::cerr << "Error in delete_points" << std::endl;
      }
      if (failed_tags.size() > 0) {
        std::cerr << "Failed to delete " << failed_tags.size() << " tags"
                  << std::endl;
      }
      /*
            std::string save_del_path =
                save_path + "_" + std::to_string(i) + ".delete";
            index.save(save_del_path.c_str());
            index.load(save_del_path.c_str());
            */
      diskann::Timer del_timer;
      diskann::cout
          << "Starting consolidation of deletes and compacting data.....";
      index.consolidate(paras);
      diskann::cout << "completed in " << del_timer.elapsed() / 1000000.0
                    << "sec." << std::endl;

      diskann::cout << "Search post deletion....." << std::endl;
      search_kernel(query, query_num, query_aligned_dim, recall_at, Lvec, index,
                    truthset_file, active_tags);
    }

    {
      diskann::Timer timer;
#pragma omp parallel for
      for (size_t i = 0; i < delete_vector.size(); i++) {
        unsigned p = delete_vector[i];
        index.insert_point(data_load + (size_t) p * (size_t) aligned_dim, paras,
                           p);
      }
      diskann::cout << "Re-incremental time: " << timer.elapsed() / 1000
                    << "ms\n";
      search_kernel(query, query_num, query_aligned_dim, recall_at, Lvec, index,
                    truthset_file);
    }
    i++;
  }

  delete[] data_load;

  return 0;
}

int main(int argc, char** argv) {
  if (argc < 16) {
    diskann::cout
        << "Correct usage: " << argv[0]
        << " <type>[int8/uint8/float] <data_file> <index_file> <L> <R> "
           "<C> <alpha> "
           "<num_rounds> "
        << "<save_graph_file> <#batches> "
           "<#batch_del_size> <query_file> <truthset_file> <recall@> "
           "<L1> <L2> ...."
        << std::endl;
    exit(-1);
  }

  int               arg_no = 4;
  unsigned          L = (unsigned) atoi(argv[arg_no++]);
  unsigned          R = (unsigned) atoi(argv[arg_no++]);
  unsigned          C = (unsigned) atoi(argv[arg_no++]);
  float             alpha = (float) std::atof(argv[arg_no++]);
  unsigned          num_rnds = (unsigned) std::atoi(argv[arg_no++]);
  std::string       save_path(argv[arg_no++]);
  unsigned          num_cycles = (unsigned) atoi(argv[arg_no++]);
  int               fraction = (int) atoi(argv[arg_no++]);
  std::string       query_file(argv[arg_no++]);
  std::string       truthset(argv[arg_no++]);
  int               recall_at = (int) std::atoi(argv[arg_no++]);
  std::vector<_u64> Lvec;

  for (int ctr = 15; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    diskann::cout
        << "No valid Lsearch found. Lsearch must be at least recall_at."
        << std::endl;
    return -1;
  }

  if (std::string(argv[1]) == std::string("int8"))
    build_incremental_index<int8_t, unsigned>(
        argv[2], argv[3], L, R, C, num_rnds, alpha, save_path, num_cycles,
        fraction, query_file, truthset, recall_at, Lvec);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_incremental_index<uint8_t, unsigned>(
        argv[2], argv[3], L, R, C, num_rnds, alpha, save_path, num_cycles,
        fraction, query_file, truthset, recall_at, Lvec);
  else if (std::string(argv[1]) == std::string("float"))
    build_incremental_index<float, unsigned>(
        argv[2], argv[3], L, R, C, num_rnds, alpha, save_path, num_cycles,
        fraction, query_file, truthset, recall_at, Lvec);
  else
    diskann::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
