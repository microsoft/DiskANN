// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "index.h"
#include "memory_mapper.h"
#include "utils.h"

namespace po = boost::program_options;

template<typename T>
int search_memory_index(diskann::Metric& metric, const std::string& index_path,
                        const std::string& result_path_prefix,
                        const std::string& query_file,
                        std::string& truthset_file, const unsigned num_threads,
                        const unsigned               recall_at,
                        const std::vector<unsigned>& Lvec) {
  // Load the query file
  T*        query = nullptr;
  unsigned* gt_ids = nullptr;
  float*    gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  // Check for ground truth
  bool calc_recall_flag = false;
  if (truthset_file != std::string("null") && file_exists(truthset_file)) {
    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data"
                << std::endl;
    }
    calc_recall_flag = true;
  } else {
    diskann::cout << " Truthset file " << truthset_file
                  << " not found. Not computing recall." << std::endl;
  }

  // Load the index
  diskann::Index<T, uint32_t> index(metric, query_dim, 0, true, true, false);
  index.load(index_path.c_str(), num_threads,
             *(std::max_element(Lvec.begin(), Lvec.end())));
  std::cout << "Index loaded" << std::endl;
  if (metric == diskann::FAST_L2)
    index.optimize_index_layout();

  diskann::Parameters paras;
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18)
            << "Avg dist cmps" << std::setw(20) << "Mean Latency (mus)"
            << std::setw(15) << "99.9 Latency";
  if (calc_recall_flag)
    std::cout << std::setw(12) << recall_string;
  std::cout << std::endl;
  std::cout << "==============================================================="
               "=================="
            << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());
  std::vector<float>                 latency_stats(query_num, 0);
  std::vector<unsigned>              cmp_stats(query_num, 0);

  // index.enable_delete();
  // index.disable_delete(paras, true);

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    uint32_t* query_result_tags = new uint32_t[recall_at * query_num];
    _u64      L = Lvec[test_id];
    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }
    query_result_ids[test_id].resize(recall_at * query_num);

    // std::vector<uint32_t> tags(index.get_num_points());
    // std::iota(tags.begin(), tags.end(), 0);

    std::vector<T*> res = std::vector<T*>();

    auto s = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      cmp_stats[i] = index.search_with_tags(
          query + i * query_aligned_dim, recall_at, L,
          query_result_tags + i * recall_at, nullptr, res);

      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000000;
    }
    for (int64_t i = 0; i < (int64_t) query_num * recall_at; i++) {
      query_result_ids[test_id][i] = *(query_result_tags + i);
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    float qps = (query_num / diff.count());

    float recall = 0;
    if (calc_recall_flag)
      recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                         query_result_ids[test_id].data(),
                                         recall_at, recall_at);

    std::sort(latency_stats.begin(), latency_stats.end());
    float mean_latency =
        std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) /
        query_num;

    float avg_cmps =
        (float) std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) /
        (float) query_num;

    std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
              << avg_cmps << std::setw(20) << (float) mean_latency
              << std::setw(15)
              << (float) latency_stats[(_u64)(0.999 * query_num)];
    if (calc_recall_flag)
      std::cout << std::setw(12) << recall;
    std::cout << std::endl;

    // delete[] query_result_tags;
  }

  std::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  for (auto L : Lvec) {
    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }
    std::string cur_result_path =
        result_path_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    // diskann::save_bin<_u32>(cur_result_path,
    // query_result_ids[test_id].data(), query_num, recall_at);
    test_id++;
  }
  diskann::aligned_free(query);

  return 0;
}

int main(int argc, char** argv) {
  std::string data_type, dist_fn, index_path_prefix, result_path, query_file,
      gt_file;
  unsigned              num_threads, K;
  std::vector<unsigned> Lvec;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips/fast_l2/cosine>");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&index_path_prefix)->required(),
                       "Path prefix to the index");
    desc.add_options()("result_path",
                       po::value<std::string>(&result_path)->required(),
                       "Path prefix for saving results of the queries");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
    desc.add_options()(
        "gt_file",
        po::value<std::string>(&gt_file)->default_value(std::string("null")),
        "ground truth file for the queryset");
    desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                       "Number of neighbors to be returned");
    desc.add_options()("search_list,L",
                       po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                       "List of L values of search");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if ((dist_fn == std::string("mips")) && (data_type == std::string("float"))) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else if ((dist_fn == std::string("fast_l2")) &&
             (data_type == std::string("float"))) {
    metric = diskann::Metric::FAST_L2;
  } else {
    std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                 "supported in general, and mips/fast_l2 only for floating "
                 "point data."
              << std::endl;
    return -1;
  }

  try {
    if (data_type == std::string("int8"))
      return search_memory_index<int8_t>(metric, index_path_prefix, result_path,
                                         query_file, gt_file, num_threads, K,
                                         Lvec);
    else if (data_type == std::string("uint8"))
      return search_memory_index<uint8_t>(metric, index_path_prefix,
                                          result_path, query_file, gt_file,
                                          num_threads, K, Lvec);
    else if (data_type == std::string("float"))
      return search_memory_index<float>(metric, index_path_prefix, result_path,
                                        query_file, gt_file, num_threads, K,
                                        Lvec);
    else {
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
      return -1;
    }
  } catch (std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index search failed." << std::endl;
    return -1;
  }
}