// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <omp.h>
#include <string.h>
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"

template<typename T>
int build_in_memory_index(const std::string& data_path, const unsigned R,
                          const unsigned L, const float alpha,
                          const std::string& save_path,
                          const unsigned     num_threads) {
  diskann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>(
      "C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", 0);
  paras.Set<unsigned>("num_threads", num_threads);

  diskann::Index<T> index(diskann::FAST_L2, data_path.c_str());
  auto              s = std::chrono::high_resolution_clock::now();
  index.build(paras);
  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << "Usage: " << argv[0]
              << "  [data_type<int8/uint8/float>]  [data_file.bin]  "
                 "[output_index_file]  "
              << "[R]  [L]  [alpha]"
              << "  [num_threads_to_use]. See README for more information on "
                 "parameters."
              << std::endl;
    exit(-1);
  }

  const std::string data_path(argv[2]);
  const std::string save_path(argv[3]);
  const unsigned    R = (unsigned) atoi(argv[4]);
  const unsigned    L = (unsigned) atoi(argv[5]);
  const float       alpha = (float) atof(argv[6]);
  const unsigned    num_threads = (unsigned) atoi(argv[7]);

  if (std::string(argv[1]) == std::string("int8"))
    build_in_memory_index<int8_t>(data_path, R, L, alpha, save_path,
                                  num_threads);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_in_memory_index<uint8_t>(data_path, R, L, alpha, save_path,
                                   num_threads);
  else if (std::string(argv[1]) == std::string("float"))
    build_in_memory_index<float>(data_path, R, L, alpha, save_path,
                                 num_threads);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
