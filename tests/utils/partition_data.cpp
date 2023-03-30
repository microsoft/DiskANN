// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <math_utils.h>
#include "cached_io.h"
#include "partition.h"

// DEPRECATED: NEED TO REPROGRAM

int main(int argc, char** argv) {
  if (argc != 7 && argc != 8) {
    std::cout << "Usage:\n"
              << argv[0]
              << "  datatype<int8/uint8/float>  <data_path>"
                 "  <prefix_path>  <sampling_rate>  "
                 "  <num_partitions>  <k_index>"
                 "  [optionally: --write_hmetis_file] "
              << std::endl;
    exit(-1);
  }

  const std::string data_path(argv[2]);
  const std::string prefix_path(argv[3]);
  const float       sampling_rate = atof(argv[4]);
  const size_t      num_partitions = (size_t) std::atoi(argv[5]);
  const size_t      max_reps = 15;
  const size_t      k_index = (size_t) std::atoi(argv[6]);
  bool              write_hmetis_file = false;
  if (argc == 8) {
    write_hmetis_file = true;
    if (std::string(argv[7]) != std::string("--write_hmetis_file")) {
      std::cout << "Last parameter, if present, must be --write_hmetis_file"
                << std::endl;
      exit(-1);
    }
    if (k_index != 1) {
      std::cout << "k_index must be 1 when writing hmetis file" << std::endl;
	  exit(-1);
    }
  }

  if (std::string(argv[1]) == std::string("float"))
    partition<float>(data_path, sampling_rate, num_partitions, max_reps,
                     prefix_path, k_index, write_hmetis_file);
  else if (std::string(argv[1]) == std::string("int8"))
    partition<int8_t>(data_path, sampling_rate, num_partitions, max_reps,
                      prefix_path, k_index, write_hmetis_file);
  else if (std::string(argv[1]) == std::string("uint8"))
    partition<uint8_t>(data_path, sampling_rate, num_partitions, max_reps,
                       prefix_path, k_index, write_hmetis_file);
  else
    std::cout << "unsupported data format. use float/int8/uint8" << std::endl;
}
