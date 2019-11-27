//#include <distances.h>
//#include <indexing.h>
#include <index.h>
#include <math_utils.h>
#include "partition_and_pq.h"

// DEPRECATED: NEED TO REPROGRAM

int main(int argc, char** argv) {
  auto s = std::chrono::high_resolution_clock::now();

  if (argc != 8) {
    std::cout << argv[0]
              << " format: data type <int8/uint8/float> base_set train_set "
                 "num_clusters "
                 "max_reps prefix_for_working_directory k_base "
              << std::endl;
    exit(-1);
  }
  size_t num_clusters = std::atoi(argv[4]);
  size_t max_reps = std::atoi(argv[5]);
  size_t k_base = std::atoi(argv[7]);
  if (std::string(argv[1]) == std::string("float"))
    partition<float>(argv[2], argv[3], num_clusters, max_reps, argv[6], k_base);
  else if (std::string(argv[1]) == std::string("int8"))
    partition<int8_t>(argv[2], argv[3], num_clusters, max_reps, argv[6],
                      k_base);
  else if (std::string(argv[1]) == std::string("uint8"))
    partition<uint8_t>(argv[2], argv[3], num_clusters, max_reps, argv[6],
                       k_base);
  else
    std::cout << "unsupported data format. use float/int8/uint8" << std::endl;
}
