#include <utils.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <pq_flash_index_nsg.h>
#include <index_nsg.h>
#include "cached_io.h"

template<typename T>
int create_disk_layout(int argc, char **argv) {
  diskann::PQFlashIndex<T> _pFlashIndex;
  std::string              base_file(argv[2]);
  std::string              rand_nsg_file(argv[3]);
  std::string              output_file(argv[4]);
  _pFlashIndex.create_disk_layout(base_file, rand_nsg_file, output_file);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout
        << argv[0]
        << " data_type <float/int8/uint8> data_bin rand-nsg_path output_file"
        << std::endl;
    exit(-1);
  }
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
