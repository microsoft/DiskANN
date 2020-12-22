// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "aux_utils.h"
#include "cached_io.h"
#include "utils.h"

template<typename T>
int create_disk_layout(int argc, char **argv) {
  if (argc != 5) {
    std::cout << argv[0] << " data_type <float/int8/uint8> data_bin "
                            "vamana_index_file output_diskann_index_file"
              << std::endl;
    exit(-1);
  }
  std::string base_file(argv[2]);
  std::string vamana_file(argv[3]);
  std::string output_file(argv[4]);
  diskann::create_disk_layout<T>(base_file, vamana_file, output_file);
  return 0;
}

int main(int argc, char **argv) {
  int ret_val = -1;
  if (std::string(argv[1]) == std::string("float"))
    ret_val = create_disk_layout<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    ret_val = create_disk_layout<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    ret_val = create_disk_layout<uint8_t>(argc, argv);
  else {
    std::cout << "unsupported type. use int8/uint8/float " << std::endl;
    ret_val = -2;
  }
  return ret_val;
}
