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
              << "  [data_type<float/int8/uint8>]  [dist_fn: l2/mips] "
                 "[data_file.bin]  "
                 "[index_prefix_path]  "
                 "[R]  [L]  [B]  [M]  [T] [PQ_disk_bytes (for very large "
                 "dimensionality, use 0 for full vectors)]. See README for "
                 "more information on "
                 "parameters."
              << std::endl;
  } else {
    diskann::Metric metric = diskann::Metric::L2;

    if (std::string(argv[2]) == std::string("mips"))
      metric = diskann::Metric::INNER_PRODUCT;

    std::string params = std::string(argv[5]) + " " + std::string(argv[6]) +
                         " " + std::string(argv[7]) + " " +
                         std::string(argv[8]) + " " + std::string(argv[9]) +
                         " " + std::string(argv[10]);
    if (std::string(argv[1]) == std::string("float"))
      build_index<float>(argv[3], argv[4], params.c_str(), metric);
    else if (std::string(argv[1]) == std::string("int8"))
      build_index<int8_t>(argv[3], argv[4], params.c_str(), metric);
    else if (std::string(argv[1]) == std::string("uint8"))
      build_index<uint8_t>(argv[3], argv[4], params.c_str(), metric);
    else
      std::cout << "Error. wrong file type" << std::endl;
  }
}
