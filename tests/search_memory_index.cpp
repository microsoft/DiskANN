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
                        const std::string& truthset_file,
                        const unsigned num_threads, const unsigned recall_at,
                        const bool                   print_all_recalls,
                        const std::vector<unsigned>& Lvec, const bool dynamic,
                        const bool tags, const bool show_qps_per_thread) {
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
  using TagT = uint32_t;
  diskann::Index<T, TagT> index(metric, query_dim, 0, dynamic, tags);
  std::cout << "Index class instantiated" << std::endl;
  index.load(index_path.c_str(), num_threads,
             *(std::max_element(Lvec.begin(), Lvec.end())));
  std::cout << "Index loaded" << std::endl;
  if (metric == diskann::FAST_L2)
    index.optimize_index_layout();

  std::cout << "Using " << num_threads << " threads to search" << std::endl;
  diskann::Parameters paras;
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);
  const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
  unsigned          table_width = 0;
  if (tags) {
    std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title
              << std::setw(20) << "Mean Latency (mus)" << std::setw(15)
              << "99.9 Latency";
    table_width += 4 + 12 + 20 + 15;
  } else {
    std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title
              << std::setw(18) << "Avg dist cmps" << std::setw(20)
              << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
    table_width += 4 + 12 + 18 + 20 + 15;
  }
  unsigned       recalls_to_print = 0;
  const unsigned first_recall = print_all_recalls ? 1 : recall_at;
  if (calc_recall_flag) {
    for (unsigned curr_recall = first_recall; curr_recall <= recall_at;
         curr_recall++) {
      std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
    }
    recalls_to_print = recall_at + 1 - first_recall;
    table_width += recalls_to_print * 12;
  }
  std::cout << std::endl;
  std::cout << std::string(table_width, '=') << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());
  std::vector<float>                 latency_stats(query_num, 0);
  std::vector<unsigned>              cmp_stats;
  if (not tags) {
    cmp_stats = std::vector<unsigned>(query_num, 0);
  }

  std::vector<TagT> query_result_tags;
  if (tags) {
    query_result_tags.resize(recall_at * query_num);
  }

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];
    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }

    query_result_ids[test_id].resize(recall_at * query_num);
    std::vector<T*> res = std::vector<T*>();

    auto s = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      if (metric == diskann::FAST_L2) {
        index.search_with_optimized_layout(
            query + i * query_aligned_dim, recall_at, L,
            query_result_ids[test_id].data() + i * recall_at);
      } else if (tags) {
        index.search_with_tags(query + i * query_aligned_dim, recall_at, L,
                               query_result_tags.data() + i * recall_at,
                               nullptr, res);
        for (int64_t r = 0; r < (int64_t) recall_at; r++) {
          query_result_ids[test_id][recall_at * i + r] =
              query_result_tags[recall_at * i + r];
        }
      } else {
        cmp_stats[i] =
            index
                .search(query + i * query_aligned_dim, recall_at, L,
                        query_result_ids[test_id].data() + i * recall_at)
                .second;
      }
      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000000;
    }
    std::chrono::duration<double> diff =
        std::chrono::high_resolution_clock::now() - s;

    float displayed_qps = static_cast<float>(query_num) / diff.count();

    if (show_qps_per_thread)
      displayed_qps /= num_threads;

    std::vector<float> recalls;
    if (calc_recall_flag) {
      recalls.reserve(recalls_to_print);
      for (unsigned curr_recall = first_recall; curr_recall <= recall_at;
           curr_recall++) {
        recalls.push_back(diskann::calculate_recall(
            query_num, gt_ids, gt_dists, gt_dim,
            query_result_ids[test_id].data(), recall_at, curr_recall));
      }
    }

    std::sort(latency_stats.begin(), latency_stats.end());
    float mean_latency =
        std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) /
        static_cast<float>(query_num);

    float avg_cmps =
        (float) std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) /
        (float) query_num;

    if (tags) {
      std::cout << std::setw(4) << L << std::setw(12) << displayed_qps
                << std::setw(20) << (float) mean_latency << std::setw(15)
                << (float) latency_stats[(_u64) (0.999 * query_num)];
    } else {
      std::cout << std::setw(4) << L << std::setw(12) << displayed_qps
                << std::setw(18) << avg_cmps << std::setw(20)
                << (float) mean_latency << std::setw(15)
                << (float) latency_stats[(_u64) (0.999 * query_num)];
    }
    for (float recall : recalls) {
      std::cout << std::setw(12) << recall;
    }
    std::cout << std::endl;
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
    diskann::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                            query_num, recall_at);
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
  bool                  print_all_recalls, dynamic, tags, show_qps_per_thread;

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
    desc.add_options()("print_all_recalls", po::bool_switch(&print_all_recalls),
                       "Print recalls at all positions, from 1 up to specified "
                       "recall_at value");
    desc.add_options()("search_list,L",
                       po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                       "List of L values of search");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("dynamic",
                       po::value<bool>(&dynamic)->default_value(false),
                       "Whether the index is dynamic. Default false.");
    desc.add_options()("tags", po::value<bool>(&tags)->default_value(false),
                       "Whether to search with tags. Default false.");
    desc.add_options()("qps_per_thread", po::bool_switch(&show_qps_per_thread),
                       "Print overall QPS divided by the number of threads in "
                       "the output table");

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

  if (dynamic && not tags) {
    std::cerr
        << "Tags must be enabled while searching dynamically built indices"
        << std::endl;
    return -1;
  }

  try {
    if (data_type == std::string("int8")) {
      return search_memory_index<int8_t>(metric, index_path_prefix, result_path,
                                         query_file, gt_file, num_threads, K,
                                         print_all_recalls, Lvec, dynamic, tags,
                                         show_qps_per_thread);
    }

    else if (data_type == std::string("uint8")) {
      return search_memory_index<uint8_t>(
          metric, index_path_prefix, result_path, query_file, gt_file,
          num_threads, K, print_all_recalls, Lvec, dynamic, tags,
          show_qps_per_thread);
    } else if (data_type == std::string("float")) {
      return search_memory_index<float>(metric, index_path_prefix, result_path,
                                        query_file, gt_file, num_threads, K,
                                        print_all_recalls, Lvec, dynamic, tags,
                                        show_qps_per_thread);
    } else {
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
      return -1;
    }
  } catch (std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index search failed." << std::endl;
    return -1;
  }
}
