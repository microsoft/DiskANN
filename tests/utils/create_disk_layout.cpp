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
int create_disk_layout(char **argv) {
  std::string vamana_file(argv[2]);
  std::string base_file(argv[3]);
  std::string tags_file(argv[4]);
  std::string pq_pivots_file(argv[5]);
  std::string pq_vectors_file(argv[6]);
  std::string output_file(argv[7]);
  bool        single_index_flag = false;
  if (base_file == "null")
    single_index_flag = true;
  diskann::create_disk_layout<T, uint32_t>(vamana_file, base_file, "",
                                           pq_pivots_file, pq_vectors_file,
                                           single_index_flag, output_file);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 8) {
    diskann::cout << argv[0]
                  << " data_type <float/int8/uint8> vamana_index_file "
                     " data_file tags_bin pq_pivots_file pq_vectors_file "
                     "output_diskann_file"
                  << std::endl;
    exit(-1);
  }
  int ret_val = -1;
  if (std::string(argv[1]) == std::string("float"))
    ret_val = create_disk_layout<float>(argv);
  else if (std::string(argv[1]) == std::string("int8"))
    ret_val = create_disk_layout<int8_t>(argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    ret_val = create_disk_layout<uint8_t>(argv);
  else {
    diskann::cout << "unsupported type. use int8/uint8/float " << std::endl;
    ret_val = -2;
  }
  return ret_val;
}
