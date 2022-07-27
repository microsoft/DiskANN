// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <boost/program_options.hpp>
#include <future>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "aux_utils.h"
#include "index.h"
#include "utils.h"
#include "memory_mapper.h"

namespace po = boost::program_options;

template<typename T>
int search_memory_index(diskann::Metric& metric, const std::string& index_path,
                        const std::string& result_path_prefix,
                        const std::string& query_file,
                        std::string& truthset_file, const unsigned num_threads,
                        const unsigned               recall_at,
                        const std::vector<unsigned>& Lvec, const bool dynamic,
                        const bool tags, std::vector<std::vector<float>>& history) {
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

  diskann::Index<T, uint32_t> index(metric, query_dim, 1, dynamic, dynamic);
  index.load(index_path.c_str(), num_threads,
             *(std::max_element(Lvec.begin(), Lvec.end())));
  std::cout << "Index loaded" << std::endl;
  if (metric == diskann::FAST_L2)
    index.optimize_index_layout();

  diskann::Parameters paras;
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  if (tags) {
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(20) << "Mean Latency (mus)" << std::setw(15)
              << "99.9 Latency";
  } else {
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(18) << "Avg dist cmps" << std::setw(20)
              << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
  }
  if (calc_recall_flag)
    std::cout << std::setw(12) << recall_string;
  std::cout << std::endl;
  std::cout << "==============================================================="
               "=================="
            << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());
  std::vector<float>                 latency_stats(query_num, 0);
  std::vector<unsigned>              cmp_stats;
  if (not tags) {
    cmp_stats = std::vector<unsigned>(query_num, 0);
  }

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    uint32_t*       query_result_tags;
    std::vector<T*> res = std::vector<T*>();
    if (tags) {
      query_result_tags = new uint32_t[recall_at * query_num];
    }
    _u64 L = Lvec[test_id];
    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }
    query_result_ids[test_id].resize(recall_at * query_num);

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
                               query_result_tags + i * recall_at, nullptr, res);
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

    if (tags) {
      for (int64_t i = 0; i < (int64_t) query_num * recall_at; i++) {
        query_result_ids[test_id][i] = *(query_result_tags + i);
      }
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    float qps = (query_num / diff.count());

    float recall = 0;
    if (calc_recall_flag) {
      recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                         query_result_ids[test_id].data(),
                                         recall_at, recall_at);
      history[test_id].push_back(recall);
    }

    std::sort(latency_stats.begin(), latency_stats.end());
    float mean_latency =
        std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) /
        query_num;

    float avg_cmps =
        (float) std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) /
        (float) query_num;

    if (tags) {
      std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(20)
                << (float) mean_latency << std::setw(15)
                << (float) latency_stats[(_u64)(0.999 * query_num)];
    } else {
      std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
                << avg_cmps << std::setw(20) << (float) mean_latency
                << std::setw(15)
                << (float) latency_stats[(_u64)(0.999 * query_num)];
    }
    if (calc_recall_flag)
      std::cout << std::setw(12) << recall;
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

std::string get_save_filename(const std::string& save_path,
                              size_t             points_to_skip,
                              size_t             last_point_threshold) {
  std::string final_path = save_path;
  final_path += std::to_string(points_to_skip) + "-";
  final_path += std::to_string(last_point_threshold);
  return final_path;
}

// build index via insertion, then delete and reinsert every point
// in batches of 10% graph size
template<typename T>
void test_batch_deletes(const std::string& data_path, const unsigned L,
                        const unsigned R, const float alpha,
                        const unsigned            thread_count,
                        const std::string& save_path, const std::string& query_path, const int rounds,
                        std::string& gt_file, const std::string& query_file,
                        const std::string& res_path) {
  const unsigned C = 500;
  const bool     saturate_graph = false;

  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", saturate_graph);
  paras.Set<unsigned>("num_rnds", 1);
  paras.Set<unsigned>("num_threads", thread_count);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path, data_load, num_points, dim,
                               aligned_dim);

  T*     query_load = NULL;
  size_t num_query_points, query_dim, query_aligned_dim;

  diskann::load_aligned_bin<T>(query_path, query_load, num_query_points,
                               query_dim, query_aligned_dim);

  using TagT = uint32_t;
  unsigned   num_frozen = 1;
  const bool enable_tags = true;
  const bool support_eager_delete = false;
  const bool concurrent_consolidate = false;
  const bool queries_present = true;

  auto num_frozen_str = getenv("TTS_NUM_FROZEN");

  if (num_frozen_str != nullptr) {
    num_frozen = std::atoi(num_frozen_str);
    std::cout << "Overriding num_frozen to" << num_frozen << std::endl;
  }

  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, true, paras,
                                paras, enable_tags, support_eager_delete,
                                concurrent_consolidate, queries_present,
                                num_query_points);

  size_t num_initial_points = 1;

  std::vector<TagT> tags(num_initial_points);  
  std::iota(tags.begin(), tags.end(), 0);

  index.build(data_load, num_initial_points, paras, tags, query_load, num_query_points);

  std::cout << "Inserting every point into the index" << std::endl;

  diskann::Timer index_timer;

#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
  for (int64_t j = num_initial_points; j < (int64_t) num_points; j++) {
    index.insert_point(&data_load[j * aligned_dim], static_cast<TagT>(j));
  }

  double seconds = index_timer.elapsed() / 1000000.0;

  std::cout << "Inserted points in " << seconds << " seconds" << std::endl;

  index.marked_graph_stats();

  index.save(save_path.c_str());
  std::cout << std::endl;
  std::cout << std::endl;

  std::vector<std::vector<float>> history(5);

  std::vector<unsigned> Lvec;
  Lvec.push_back(10);
  Lvec.push_back(20);
  Lvec.push_back(50);
  Lvec.push_back(100);
  Lvec.push_back(200);

  diskann::Metric metric;
  metric = diskann::Metric::L2;
  search_memory_index<T>(metric, save_path, res_path, query_file,
                             gt_file, thread_count, 10, Lvec, true, true,
                             history);

  // CYCLING START

  std::cout << "Beginning cycling " << std::endl; 

  int parts = 20;
  int points_in_part;

  std::vector<double> delete_times;
  std::vector<double> insert_times;

  for (int i = 0; i < rounds; i++) {
    std::cout << "ROUND " << i + 1 << std::endl;
    std::cout << std::endl;

    std::vector<int64_t> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    int points_seen = 0;
    for (int j = 0; j < parts/2; j++) {
      if (j == parts - 1)
        points_in_part = num_points - points_seen;
      else
        points_in_part = num_points / parts;

      // DELETIONS
      std::cout << "Deleting " << points_in_part
                << " points from the index..." << std::endl;
      index.enable_delete();
      tsl::robin_set<TagT> deletes;
      for (int k = points_seen; k < points_seen + points_in_part; k++) {
        deletes.insert(static_cast<TagT>(indices[k]));
      }
      std::vector<TagT> failed_deletes;
      index.lazy_delete(deletes, failed_deletes);
      omp_set_num_threads(thread_count);
      diskann::Timer delete_timer;
      index.consolidate_deletes(paras);
      double elapsedSeconds = delete_timer.elapsed() / 1000000.0;

      std::cout << "Deleted " << points_in_part << " points in "
                << elapsedSeconds << " seconds" << std::endl;

      delete_times.push_back(elapsedSeconds);


      // RE-INSERTIONS
      std::cout << "Re-inserting the same " << points_in_part
                << " points from the index..." << std::endl;
      diskann::Timer insert_timer;
#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
      for (int64_t k = points_seen;
           k < (int64_t) points_seen + points_in_part; k++) {
        index.insert_point(&data_load[indices[k] * aligned_dim],
                                static_cast<TagT>(indices[k]));
      }
      elapsedSeconds = insert_timer.elapsed() / 1000000.0;

      std::cout << "Inserted " << points_in_part << " points in "
                << elapsedSeconds << " seconds" << std::endl;
      std::cout << std::endl;

      insert_times.push_back(elapsedSeconds);
      index.marked_graph_stats();

      points_seen += points_in_part;
      index.save(save_path.c_str());

      
      search_memory_index<T>(metric, save_path, res_path, query_file,
                             gt_file, thread_count, 10, Lvec, true, true,
                             history);
    }
  }

  std::cout << "Recall Lists: " << std::endl;
  for (int i=0; i<5; i++){
    std::cout << "Recall at L = " << Lvec[i] << std::endl; 
    std::vector<float> recall_list = history[i];
    for(float rec : recall_list) std::cout << rec << std::endl;
    std::cout << std::endl; 
  }

  double avg_delete = ((double) std::accumulate(delete_times.begin(),
                                                delete_times.end(), 0.0)) /
                      ((double) delete_times.size());
  double avg_insert = ((double) std::accumulate(insert_times.begin(),
                                                insert_times.end(), 0.0)) /
                      ((double) insert_times.size());
  std::cout << "Average time for deletions " << avg_delete << " seconds"
            << std::endl;
  std::cout << "Average time for insertions " << avg_insert << " seconds"
            << std::endl;
  std::cout << std::endl;

}

int main(int argc, char** argv) {
  std::string data_type, data_path, save_path, sample_query_path, gt_file, query_file, res_path;
  unsigned    num_threads, R, L;
  float       alpha;
  int                 rounds;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("data_path",
                       po::value<std::string>(&data_path)->required(),
                       "Input data file in bin format");
    desc.add_options()("save_path",
                       po::value<std::string>(&save_path)->required(),
                       "Path prefix for saving index file components");
    desc.add_options()("sample_query_path",
                       po::value<std::string>(&sample_query_path)->required(),
                       "Sample query path");
    desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                       "Ground truth file");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file");
    desc.add_options()("res_path",
                       po::value<std::string>(&res_path)->required(),
                       "Res path");
    desc.add_options()("max_degree,R",
                       po::value<uint32_t>(&R)->default_value(64),
                       "Maximum graph degree");
    desc.add_options()(
        "Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
        "Build complexity, higher value results in better graphs");
    desc.add_options()(
        "alpha", po::value<float>(&alpha)->default_value(1.2f),
        "alpha controls density and diameter of graph, set 1 for sparse graph, "
        "1.2 or 1.4 for denser graphs with lower diameter");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("rounds", po::value<int>(&rounds)->default_value(2),
                       "Number of rounds");

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

  try {
    if (data_type == std::string("int8"))
      test_batch_deletes<int8_t>(data_path, L, R, alpha, num_threads,
                                 save_path, sample_query_path, rounds, gt_file, query_file,
                                 res_path);
    else if (data_type == std::string("uint8"))
      test_batch_deletes<uint8_t>(data_path, L, R, alpha, num_threads,
                                  save_path, sample_query_path, rounds, gt_file, query_file,
                                  res_path);
    else if (data_type == std::string("float"))
      test_batch_deletes<float>(data_path, L, R, alpha, num_threads,
                                save_path, sample_query_path, rounds, gt_file, query_file,
                                res_path);
    else
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "Caught unknown exception" << std::endl;
    exit(-1);
  }

  return 0;
}