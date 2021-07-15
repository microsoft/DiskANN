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
int build_in_memory_index(const std::string&     data_path,
                          const diskann::Metric& metric, const unsigned R,
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

  diskann::Index<T> index(metric, data_path.c_str());
  auto              s = std::chrono::high_resolution_clock::now();
  index.build(paras);
  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cout << "Usage: " << argv[0]
              << "  [data_type<int8/uint8/float>] [l2/mips] [data_file.bin]  "
                 "[output_index_file]  "
              << "[R]  [L]  [alpha]"
              << "  [num_threads_to_use]. See README for more information on "
                 "parameters."
              << std::endl;
    exit(-1);
  }

  _u32 ctr = 2;

  diskann::Metric metric;
  if (std::string(argv[ctr]) == std::string("mips"))
    metric = diskann::Metric::INNER_PRODUCT;
  else if (std::string(argv[ctr]) == std::string("l2"))
    metric = diskann::Metric::L2;
  else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product support."
              << std::endl;
    return -1;
  }
  ctr++;

  const std::string data_path(argv[ctr++]);
  const std::string save_path(argv[ctr++]);
  const unsigned    R = (unsigned) atoi(argv[ctr++]);
  const unsigned    L = (unsigned) atoi(argv[ctr++]);
  const float       alpha = (float) atof(argv[ctr++]);
  const unsigned    num_threads = (unsigned) atoi(argv[ctr++]);

  if (std::string(argv[1]) == std::string("int8"))
    build_in_memory_index<int8_t>(data_path, metric, R, L, alpha, save_path,
                                  num_threads);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_in_memory_index<uint8_t>(data_path, metric, R, L, alpha, save_path,
                                   num_threads);
  else if (std::string(argv[1]) == std::string("float"))
    build_in_memory_index<float>(data_path, metric, R, L, alpha, save_path,
                                 num_threads);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
