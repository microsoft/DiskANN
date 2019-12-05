#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "utils.h"

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cout << argv[0] << " <ground_truth_bin> <our_results_bin>  <r1> "
                            "<optional: r2 to calculate "
                            "recall r1@r2. By default, r2 will be set to r1>"
              << std::endl;
    exit(-1);
  }
  unsigned* gold_std = NULL;
  unsigned* our_results = NULL;
  size_t    points_num, points_num_gs, points_num_or;
  size_t    dim_gs;
  size_t    dim_or;
  diskann::load_bin<unsigned>(argv[1], gold_std, points_num_gs, dim_gs);
  diskann::load_bin<unsigned>(argv[2], our_results, points_num_or, dim_or);

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
  std::cout << "Calculating recall " << recall_at << "@" << subset_size
            << std::endl;
  float recall_val =
      diskann::calc_recall_set(points_num, gold_std, dim_gs, our_results,
                               dim_or, recall_at, subset_size);

  //  double avg_recall = (recall*1.0)/(points_num*1.0);
  std::cout << "Avg. recall " << recall_at << " at " << subset_size << " is "
            << recall_val << "\n";
}
