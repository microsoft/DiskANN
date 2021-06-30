// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <future>
#include <Neighbor_Tag.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <sync_index.h>
#include <time.h>
#include <timer.h>
#include <cstring>
#include <iomanip>

#include "aux_utils.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    diskann::cout << "Correct usage : " << argv[0]
                  << " <input_file_name> <save_file_prefix>" << std::endl;
    exit(-1);
  }

  std::string input(argv[1]);
  std::string output(argv[2]);

  float* data = nullptr;
  size_t in_num, in_dim, in_aligned_dim;
  diskann::load_aligned_bin<float>(input.c_str(), data, in_num, in_dim,
                                   in_aligned_dim);
  diskann::save_bin<float>(output + ".bin", data, in_num, in_aligned_dim);

  return 0;
}
