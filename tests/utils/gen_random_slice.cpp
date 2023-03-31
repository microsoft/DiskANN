// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include "partition.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    std::cout << argv[0]
              << " data_type [float/int8/uint8] base_bin_file "
                 "sample_output_prefix sampling_probability"
                 " [optionally: --gen_complement]"
              << std::endl;
    exit(-1);
  }

  if (argc == 6 && std::string(argv[5]) != std::string("--gen_complement")) {
	std::cout << "Last parameter, if present, must be --gen_complement"
			  << std::endl;
	exit(-1);
  }

  std::string base_file(argv[2]);
  std::string output_prefix(argv[3]);
  float       sampling_rate = (float) (std::atof(argv[4]));
  bool        gen_complement = false;
  if (argc == 6)
    gen_complement = true;

  if (std::string(argv[1]) == std::string("float")) {
    gen_random_slice<float>(base_file, output_prefix, sampling_rate,
                            gen_complement);
  } else if (std::string(argv[1]) == std::string("int8")) {
    gen_random_slice<int8_t>(base_file, output_prefix, sampling_rate,
                            gen_complement);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    gen_random_slice<uint8_t>(base_file, output_prefix, sampling_rate,
                             gen_complement);
  } else
    std::cout << "Unsupported type. Use float/int8/uint8." << std::endl;
  return 0;
}
