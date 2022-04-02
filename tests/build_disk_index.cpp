// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "omp.h"

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

template<typename T>
bool build_index(const char* dataFilePath, const char* indexFilePath,
                 const char* indexBuildParameters, diskann::Metric metric) {
  return diskann::build_disk_index<T>(dataFilePath, indexFilePath,
                                      indexBuildParameters, metric);
}

int main(int argc, char** argv) {
  if (argc != 11) {
    std::cout << "Usage: " << argv[0]
              << "   data_type<float/int8/uint8>   dist_fn<l2/mips>   "
                 "data_file.bin   index_prefix_path  "
                 "R(graph degree)   L(build complexity)   "
                 "B(search memory allocation in GB)   "
                 "M(build memory allocation in GB)   "
                 "T(#threads)   PQ_disk_bytes"
              << std::endl;
    return -1;
  }

  diskann::Metric metric = diskann::Metric::L2;

  if (std::string(argv[2]) == std::string("mips"))
    metric = diskann::Metric::INNER_PRODUCT;

  std::string params = std::string(argv[5]) + " " + std::string(argv[6]) + " " +
                       std::string(argv[7]) + " " + std::string(argv[8]) + " " +
                       std::string(argv[9]) + " " + std::string(argv[10]);
  try {
    if (std::string(argv[1]) == std::string("float"))
      return build_index<float>(argv[3], argv[4], params.c_str(), metric);
    else if (std::string(argv[1]) == std::string("int8"))
      return build_index<int8_t>(argv[3], argv[4], params.c_str(), metric);
    else if (std::string(argv[1]) == std::string("uint8"))
      return build_index<uint8_t>(argv[3], argv[4], params.c_str(), metric);
    else {
      diskann::cerr << "Error. Unsupported data type" << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index build failed." << std::endl;
    return -1;
  }
}
