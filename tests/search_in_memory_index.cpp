#include <index_nsg.h>
#include <omp.h>
#include <string.h>
#include <cstring>
#include <iomanip>
#include "util.h"
#ifndef __NSG_WINDOWS__
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "MemoryMapper.h"

float calc_recall(unsigned num_queries, unsigned* gold_std, unsigned dim_gs,
                  unsigned* our_results, unsigned dim_or, unsigned recall_at) {
  bool*    this_point = new bool[recall_at];
  unsigned total_recall = 0;

  for (size_t i = 0; i < num_queries; i++) {
    for (unsigned j = 0; j < recall_at; j++)
      this_point[j] = false;
    for (size_t j1 = 0; j1 < recall_at; j1++)
      for (size_t j2 = 0; j2 < dim_or; j2++)
        if (gold_std[i * (size_t) dim_gs + j1] ==
            our_results[i * (size_t) dim_or + j2]) {
          if (this_point[j1] == false)
            total_recall++;
          this_point[j1] = true;
        }
  }
  return ((float) total_recall) / ((float) num_queries) *
         (100.0 / ((float) recall_at));
}

template<typename T>
int aux_main(int argc, char** argv) {
  if ((argc != 7)) {
    std::cout << argv[0] << " data_type [int8/uint8/float] data_bin_file "
                            "query_bin_file groundtruth_bin nsg_path "
                            "recall@"
              << std::endl;
    exit(-1);
  }

  int      bfs_init = 0;
  unsigned beam_width = 4;

  T*        query_load = NULL;
  unsigned* gt_load = NULL;
  size_t    query_num, query_dim;
  size_t    gt_num, gt_dim;

  //  NSG::load_bin<T>(argv[2], data_load, points_num, dim);
  NSG::load_bin<T>(argv[3], query_load, query_num, query_dim);
  NSG::load_bin<unsigned>(argv[4], gt_load, gt_num, gt_dim);
  std::string rand_nsg_path(argv[5]);
  unsigned    recall_at = atoi(argv[6]);
  std::string recall_string = std::string("Recall@") + std::string(argv[6]);

  if (gt_num != query_num) {
    std::cout << "Ground truth does not match number of queries. " << std::endl;
    exit(-1);
  }

  if (recall_at > gt_dim) {
    std::cout << "Ground truth has only " << gt_dim
              << " elements. Calculating recall at " << gt_dim << std::endl;
    recall_at = gt_num;
  }

  std::vector<unsigned> Lvec;
  unsigned              curL = 8;
  while (curL < 2048) {
    Lvec.push_back(curL);
    if (curL < 16)
      curL += 1;
    else if (curL < 32)
      curL += 2;
    else if (curL < 64)
      curL += 4;
    else if (curL < 128)
      curL += 8;
    else if (curL < 256)
      curL += 16;
    else if (curL < 512)
      curL += 32;
    else if (curL < 1024)
      curL += 64;
    else
      curL += 128;
  }

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  //  data_load = NSG::data_align(data_load, points_num, dim);
  query_load = NSG::data_align(query_load, query_num, query_dim);
  std::cout << "query data loaded and aligned" << std::endl;

  NSG::IndexNSG<T> index(NSG::L2, argv[2]);
  index.load(rand_nsg_path.c_str());  // to load NSG
  std::cout << "Index loaded" << std::endl;

  std::vector<unsigned> start_points;
  if (bfs_init) {
    index.populate_start_points_bfs(start_points);
    std::cout << "Initialized starting points based on BFS" << std::endl;
  }

  NSG::Parameters paras;
  std::cout << std::setw(8) << "Ls" << std::setw(16) << recall_string
            << std::setw(16) << "Latency" << std::setw(16) << "Cmps"
            << std::setw(16) << "Hops" << std::endl;
  std::cout << "==============================================================="
               "============="
               "======="
            << std::endl;
  unsigned  K = recall_at;
  unsigned* res = new unsigned[(size_t) query_num * K];
  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    unsigned L = Lvec[test_id];
    if (L < recall_at)
      continue;

    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);

    long long total_hops = 0;
    long long total_cmps = 0;

    auto s = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < query_num; i++) {
      auto ret =
          index.beam_search(query_load + i * query_dim, K, paras,
                            res + ((size_t) i) * K, beam_width, start_points);
#pragma omp atomic
      total_hops += ret.first;
#pragma omp atomic
      total_cmps += ret.second;
    }
    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;
    float                         latency =
        omp_get_max_threads() * (diff.count() / query_num) * (1000000);

    float avg_hops = (float) total_hops / (float) query_num;
    float avg_cmps = (float) total_cmps / (float) query_num;
    float recall = calc_recall(query_num, gt_load, gt_dim, res, K, recall_at);

    std::cout << std::setw(8) << L << std::setw(16) << recall << std::setw(16)
              << latency << std::setw(16) << avg_cmps << std::setw(16)
              << avg_hops << std::endl;
    if (recall > 99.5) {
      break;
    }
  }

  return 0;
}

int main(int argc, char** argv) {
  if ((argc != 7)) {
    std::cout << argv[0] << " data_type [int8/uint8/float] data_bin_file "
                            "query_bin_file groundtruth_bin nsg_path "
                            "recall@"
              << std::endl;
    exit(-1);
  }
  if (std::string(argv[1]) == std::string("int8"))
    aux_main<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    aux_main<uint8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("float"))
    aux_main<float>(argc, argv);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
