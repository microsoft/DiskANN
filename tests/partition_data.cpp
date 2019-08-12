//#include <distances.h>
//#include <indexing.h>
#include <index_nsg.h>
#include <math_utils.h>
#include "partition_and_pq.h"

// DEPRECATED: NEED TO REPROGRAM

int main(int argc, char** argv) {
  auto s = std::chrono::high_resolution_clock::now();

  if (argc != 7) {
    std::cout << "Usage:\n"
              << argv[0] << "  datatype<int8/uint8/float>  <data_path>"
                            "  <prefix_path>  <sampling_rate>  "
                            "  <num_partitions>  <k_index>"
              << std::endl;
    exit(-1);
  }

  const std::string data_path(argv[2]);
  const std::string prefix_path(argv[3]);
  const float       sampling_rate = atof(argv[4]);
  const size_t      num_partitions = (size_t) std::atoi(argv[5]);
  const size_t      max_reps = 15;
  const size_t      k_index = (size_t) std::atoi(argv[6]);

  if (std::string(argv[1]) == std::string("float"))
    partition<float>(data_path, sampling_rate, num_partitions, max_reps,
                     prefix_path, k_index);
  else if (std::string(argv[1]) == std::string("int8"))
    partition<int8_t>(data_path, sampling_rate, num_partitions, max_reps,
                      prefix_path, k_index);
  else if (std::string(argv[1]) == std::string("uint8"))
    partition<uint8_t>(data_path, sampling_rate, num_partitions, max_reps,
                       prefix_path, k_index);
  else
    std::cout << "unsupported data format. use float/int8/uint8" << std::endl;
}
