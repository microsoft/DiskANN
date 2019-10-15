#include <index_nsg.h>
#include <omp.h>
#include <string.h>
#include <cstring>
#include <iomanip>
#include <set>
#include "utils.h"
#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

/*float calc_recall(unsigned num_queries, unsigned* gold_std, unsigned dim_gs,
                  unsigned* our_results, unsigned dim_or, unsigned recall_at) {
  bool*    this_point = new bool[recall_at];
  unsigned total_recall = 0;

  for (size_t i = 0; i < num_queries; i++) {
    std::set<unsigned> unique_results;
    for (uint32_t j = 0; j < recall_at; j++) {
      unique_results.insert(our_results[i * dim_or + j]);
    }
    if (unique_results.size() < recall_at) {
      std::cout << "Point " << i << " has duplicates. " << std::endl;
      for (auto p : unique_results)
        std::cout << p << " ";
      std::cout << std::endl;
    }
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
}*/

template<typename T>
int search_memory_index(int argc, char** argv) {
  T*                query = nullptr;
  size_t            query_num, query_dim, query_aligned_dim;
  std::vector<_u64> Lvec;

  /*  unsigned* gt_load = NULL;
    size_t    gt_num, gt_dim, gt_aligned_dim;*/

  std::string data_file(argv[2]);
  std::string memory_index_file(argv[3]);
  std::string query_bin(argv[4]);
  _u64        recall_at = std::atoi(argv[5]);
  _u32        beam_width = std::atoi(argv[6]);
  std::string result_output_prefix(argv[7]);
  //  std::string gt(argv[9]);

  for (int ctr = 8; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at"
              << std::endl;
    return -1;
  }

  std::cout << "Search parameters: beamwidth: " << beam_width << std::endl;

  NSG::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                           query_aligned_dim);
  //  NSG::load_aligned_bin<unsigned>(gt, gt_load, gt_num, gt_dim,
  //  gt_aligned_dim);

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  NSG::IndexNSG<T> index(NSG::L2, data_file.c_str());
  index.load(memory_index_file.c_str());  // to load NSG
  std::cout << "Index loaded" << std::endl;

  std::vector<unsigned> start_points;
  index.populate_start_points_ep(start_points);

  NSG::Parameters paras;
  std::cout << std::setw(8) << "Ls" << std::setw(16) << "Latency" << std::endl;
  std::cout << "==============================" << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];
    query_result_ids[test_id].resize(recall_at * query_num);

    //  long long total_hops = 0;
    // long long total_cmps = 0;

    auto s = std::chrono::high_resolution_clock::now();
    //#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto ret =
          index.beam_search(query + i * query_aligned_dim, recall_at, L,
                            query_result_ids[test_id].data() + i * recall_at,
                            beam_width, start_points);

      /* #pragma omp atomic
            total_hops += ret.first;
      #pragma omp atomic
            total_cmps += ret.second;*/
    }
    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;

    float latency = (diff.count() / query_num) * (1000000);
    //   float avg_hops = (float) total_hops / (float) query_num;
    // float avg_cmps = (float) total_cmps / (float) query_num;
    /*  float recall =
    calc_recall(query_num, gt_load, gt_dim, query_result_ids[test_id].data(),
                recall_at, recall_at);

     std::cout << std::setw(8) << L << std::setw(16) << recall <<
     std::setw(16)
               << latency << std::setw(16) << avg_cmps << std::setw(16)
               << avg_hops << std::endl;
     if (recall > 99.5)
       break;*/

    std::cout << std::setw(8) << L << std::setw(16) << latency << std::endl;
  }

  std::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  for (auto L : Lvec) {
    std::string cur_result_path =
        result_output_prefix + std::to_string(L) + "_idx_uint32.bin";
    NSG::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                        query_num, recall_at);
    test_id++;
  }

  NSG::aligned_free(query);
  return 0;
}

int main(int argc, char** argv) {
  if (argc <= 8) {
    std::cout << "Usage: " << argv[0]
              << " <index_type[float/int8/uint8]>  <full_data_bin>  "
                 "<memory_index_path>  "
                 "<query_bin> "
                 "<recall@> <beam_width> <result_output_prefix> <gt_file> <L1> "
                 "<L2> ... "
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
