// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <future>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

template<typename T, typename TagT>
void search_kernel(T* query, size_t query_num, size_t query_aligned_dim,
                   const int recall_at, std::vector<_u64> Lvec,
                   diskann::Index<T, TagT>& index) {
  uint32_t* results = new uint32_t[recall_at * query_num];
  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
#pragma omp parallel for num_threads(8)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      _u64 L = Lvec[test_id];
      index.search(query + i * query_aligned_dim, recall_at, L,
                   results + i * recall_at);
    }
  }
  delete[] results;
}

template<typename T, typename TagT>
void insert_kernel(T* data_load, diskann::Index<T, TagT>& index,
                   diskann::Parameters& parameters, size_t num_points,
                   size_t aligned_dim, unsigned start) {
  diskann::cout << "Insertion thread" << std::endl;
  diskann::Timer timer;
// do not parallelize without making _iterate_to_fixed_point() thread safe
#pragma omp parallel for num_threads(16)
  for (unsigned insertions = start; insertions < num_points; ++insertions) {
    index.insert_point(data_load + insertions * aligned_dim, parameters,
                       insertions);
  }

  unsigned time_secs = timer.elapsed() / 1000000;
  diskann::cout << "Insert time : " << time_secs << " s\n"
                << "Inserts/sec : " << num_points / time_secs << std::endl;
}
template<typename T>
int build_incremental_index(const std::string& data_path, const unsigned L,
                            const unsigned R, const unsigned C,
                            const unsigned num_rnds, const float alpha,
                            const std::string& save_path,
                            const std::string& query_file, const int recall_at,
                            std::vector<_u64> Lvec) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<unsigned>("num_rnds", num_rnds);
  paras.Set<bool>("saturate_graph", 0);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  unsigned num_prebuilt = 1;

  typedef int             TagT;
  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, 1, true, false);

  {
    std::vector<TagT> tags(1);
    std::iota(tags.begin(), tags.end(), 0);

    diskann::Timer timer;
    index.build(data_path.c_str(), num_points, paras, tags);
    diskann::cout << "Index build time: " << timer.elapsed() / 1000 << "ms\n";
  }

  diskann::cout << "Saved batch build" << std::endl;
  T*     query = NULL;
  size_t query_num, query_dim, query_aligned_dim;

  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);
  {
    std::future<void> future = std::async(
        std::launch::async, insert_kernel<T, TagT>, data_load, std::ref(index),
        std::ref(paras), num_points, aligned_dim, num_prebuilt);

    std::future_status status;

    unsigned total_queries = 0;
    do {
      status = future.wait_for(std::chrono::milliseconds(1));
      if (status == std::future_status::deferred) {
        diskann::cout << "deferred\n";
      } else if (status == std::future_status::timeout) {
        search_kernel(query, query_num, query_aligned_dim, recall_at, Lvec,
                      index);
        total_queries += query_num;
        diskann::cout << "total queries: " << total_queries << std::endl;
      } else if (status == std::future_status::ready) {
        diskann::cout << "Insertions complete!\n";
      }
    } while (status != std::future_status::ready);
  }
  auto save_path_inc = save_path + ".inc";
  index.save(save_path_inc.c_str());
  diskann::cout << "Saved file post insertions" << std::endl;

  delete[] data_load;

  return 0;
}

int main(int argc, char** argv) {
  if (argc < 12) {
    diskann::cout << "Correct usage: " << argv[0]
                  << " type[int8/uint8/float] data_file L R C alpha "
                     "num_rounds "
                  << "save_graph_file query_file recall@ L1 L2....."
                  << std::endl;
    exit(-1);
  }

  unsigned    L = (unsigned) atoi(argv[3]);
  unsigned    R = (unsigned) atoi(argv[4]);
  unsigned    C = (unsigned) atoi(argv[5]);
  float       alpha = (float) std::atof(argv[6]);
  unsigned    num_rnds = (unsigned) std::atoi(argv[7]);
  std::string save_path(argv[8]);
  std::string query_file(argv[9]);
  int         recall_at = (int) atoi(argv[10]);

  std::vector<_u64> Lvec;

  for (int ctr = 11; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= (_u64) recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    diskann::cout
        << "No valid Lsearch found. Lsearch must be at least recall_at."
        << std::endl;
    return -1;
  }

  if (std::string(argv[1]) == std::string("int8"))
    build_incremental_index<int8_t>(argv[2], L, R, C, num_rnds, alpha,
                                    save_path, query_file, recall_at, Lvec);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_incremental_index<uint8_t>(argv[2], L, R, C, num_rnds, alpha,
                                     save_path, query_file, recall_at, Lvec);
  else if (std::string(argv[1]) == std::string("float"))
    build_incremental_index<float>(argv[2], L, R, C, num_rnds, alpha, save_path,
                                   query_file, recall_at, Lvec);
  else
    diskann::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
