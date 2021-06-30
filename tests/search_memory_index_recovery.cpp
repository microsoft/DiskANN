// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <omp.h>
#include <set>
#include <memory>
#include <string.h>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "aux_utils.h"
#include "index.h"
#include "memory_mapper.h"
#include "utils.h"

template<typename T>
int search_memory_index(int argc, char** argv) {
  T*                query = nullptr;
  unsigned*         gt_ids = nullptr;
  unsigned*         gt_tags = nullptr;
  float*            gt_dists = nullptr;
  size_t            query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  std::vector<_u64> Lvec;

  int         arg_no = 2;
  std::string data_file(argv[arg_no++]);
  std::string memory_index_file(argv[arg_no++]);
  int         frozen_pt = std::atoi(argv[arg_no++]);
  int         tags = std::atoi(argv[arg_no++]);
  std::string query_bin(argv[arg_no++]);
  std::string truthset_bin(argv[arg_no++]);
  _u64        recall_at = std::atoi(argv[arg_no++]);
  std::string result_output_prefix(argv[arg_no++]);

  bool calc_recall_flag = false;

  for (int ctr = 10; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at."
              << std::endl;
    return -1;
  }

  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);

  if (file_exists(truthset_bin)) {
    diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                           &gt_tags);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data"
                << std::endl;
    }
    calc_recall_flag = true;
  }

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  _u64 data_num, data_dim;
  diskann::get_bin_metadata(data_file, data_num, data_dim);
  std::vector<std::shared_ptr<diskann::Index<T>>> indices;
  indices.reserve(6);
  for (int j = 0; j < 6; j++) {
    indices.emplace_back(std::make_shared<diskann::Index<T>>(
        diskann::L2, data_dim, data_num, frozen_pt));
    if (tags == 1)
      indices[j]->load(memory_index_file.c_str(), data_file.c_str(), true);
    else {
      indices[j]->load(memory_index_file.c_str(), data_file.c_str(), false);
    }
  }
  std::cout << "Index loaded" << std::endl;

  diskann::Parameters paras;
  std::string         recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18)
            << "Mean Latency (ms)" << std::setw(15) << "99.9 Latency"
            << std::setw(12) << recall_string << std::endl;
  std::cout << "==============================================================="
               "==============="
            << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());
  std::vector<std::vector<unsigned>> query_result_tags(Lvec.size());

  std::vector<std::vector<double>> latency_stats(6);
  for (int j = 0; j < 6; j++) {
    latency_stats[j].reserve(query_num);
    for (int k = 0; k < query_num; k++)
      latency_stats[j][k] = 0;
  }

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];
    //      recall_at = L;
    //    query_result_ids[test_id].resize(recall_at * query_num);
    //    query_result_tags[test_id].resize(recall_at * query_num);

    auto s = std::chrono::high_resolution_clock::now();
    //   #pragma omp parallel for schedule(dynamic, 1) num_threads(96)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
#pragma omp parallel for num_threads(6)
      for (int j = 0; j < 6; j++) {
        std::vector<uint32_t> result_ids(recall_at);
        auto                  qs = std::chrono::high_resolution_clock::now();
        indices[j]->search(query + i * query_aligned_dim, recall_at, (_u32) L,
                           result_ids.data());
        auto qe = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = qe - qs;
        latency_stats[j][i] = diff.count() * 1000;
      }
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    float qps = (float) (query_num / diff.count());

    float recall = 0;
    //    if (calc_recall_flag)

    /*      recall = (float) diskann::calculate_recall(
              (_u32) query_num, gt_ids, gt_dists, (_u32) gt_dim,
              query_result_ids[test_id].data(), (_u32) recall_at, (_u32)
       recall_at);
              */

    for (int j = 0; j < 6; j++)
      std::sort(latency_stats[j].begin(), latency_stats[j].end());
    double mean_latency = 0;
    for (uint64_t q = 0; q < query_num; q++) {
      double mean_latency_temp = 0;
      for (int j = 0; j < 6; j++) {
        mean_latency_temp += latency_stats[j][q];
      }
      mean_latency += (mean_latency_temp / 6);
    }
    mean_latency /= query_num;

    double latency_99 = 0;
    for (int j = 0; j < 6; j++) {
      latency_99 += latency_stats[j][(_u64)(0.99 * query_num)];
    }

    latency_99 /= 6;

    std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
              << (float) mean_latency << std::setw(15) << (float) latency_99
              << std::setw(12) << recall << std::endl;
  }

  /*  std::cout << "Done searching. Now saving results " << std::endl;
    _u64 test_id = 0;
    for (auto L : Lvec) {
      //      recall_at = L;
      std::string cur_result_path =
          result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
      diskann::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                              query_num, recall_at);
      test_id++;
    }
  */
  diskann::aligned_free(query);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 10) {
    std::cout
        << "Usage: " << argv[0]
        << "  [index_type<float/int8/uint8>]  [data_file.bin]  "
           "[memory_index_path]  [frozen_pt[0/1]] [tags[0/1]]"
           " [query_file.bin]  [truthset.bin (use \"null\" for none)] "
           " [K]  [result_output_prefix] "
           " [L1]  [L2] etc. See README for more information on parameters. "
        << std::endl;
    exit(-1);
  }
  if (std::string(argv[1]) == std::string("int8"))
    search_memory_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_memory_index<uint8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("float"))
    search_memory_index<float>(argc, argv);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
