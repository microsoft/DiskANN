#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "utils.h"

float calc_recall_set(unsigned num_queries, unsigned* gold_std, unsigned dim_gs,
                      unsigned* our_results, unsigned dim_or,
                      unsigned recall_at, unsigned subset_size) {
  std::cout << "dim_gs: " << dim_gs << ", dim_or: " << dim_or
            << ", recall_at: " << recall_at << " num_queries = " << num_queries
            << "\n";
  unsigned           total_recall = 0;
  std::set<unsigned> gt, res;

  for (size_t i = 0; i < num_queries; i++) {
    gt.clear();
    res.clear();
    unsigned* gt_vec = gold_std + dim_gs * i;
    unsigned* res_vec = our_results + dim_or * i;
    gt.insert(gt_vec, gt_vec + recall_at);
    res.insert(res_vec, res_vec + subset_size);
    unsigned cur_recall = 0;
    for (auto& v : gt) {
      if (res.find(v) != res.end()) {
        cur_recall++;
      }
    }
    // std::cout << " idx: " << i << ", interesection: " << cur_recall << "\n";
    total_recall += cur_recall;
  }
  return ((float) total_recall) / ((float) num_queries) *
         (100.0 / ((float) recall_at));
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cout << argv[0]
              << " ground_truth_bin rand-nsg_result_bin  r1 r2 (to calculate "
                 "recall r1@r2). By default, r2 will be set to r1."
              << std::endl;
    exit(-1);
  }
  unsigned* gold_std = NULL;
  unsigned* our_results = NULL;
  size_t    points_num, points_num_gs, points_num_or;
  size_t    dim_gs;
  size_t    dim_or;
  //  load_data(argv[1], gold_std, points_num, dim_gs);
  //  load_data(argv[2], our_results, points_num, dim_or);
  NSG::load_Tvecs<unsigned>(argv[1], gold_std, points_num_gs, dim_gs);
  NSG::load_Tvecs<unsigned>(argv[2], our_results, points_num_or, dim_or);

  if (points_num_gs != points_num_or) {
    std::cout
        << "Error. Number of queries mismatch in ground truth and our results"
        << std::endl;
    return -1;
  }
  points_num = points_num_gs;

  uint32_t recall_at = std::atoi(argv[3]);
  uint32_t subset_size = dim_or;
  if (argc == 5)
    subset_size = std::atoi(argv[4]);

  if ((dim_or < recall_at) || (recall_at > dim_gs)) {
    std::cout << "ground truth has size " << dim_gs << "; our set has "
              << dim_or << " points. Asking for recall " << recall_at
              << std::endl;
    return -1;
  }
  std::cout << "calculating recall " << recall_at << "@" << subset_size
            << std::endl;
  float recall_val = calc_recall_set(points_num, gold_std, dim_gs, our_results,
                                     dim_or, recall_at, subset_size);

  //  double avg_recall = (recall*1.0)/(points_num*1.0);
  std::cout << "avg. recall " << recall_at << " at " << subset_size << " is "
            << recall_val << "\n";
}
