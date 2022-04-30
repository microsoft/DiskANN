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
#include "ann_exception.h"

template<typename T, typename TagT = uint32_t>
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

  _u64 data_num, data_dim;
  diskann::get_bin_metadata(data_path, data_num, data_dim);

  diskann::Index<T, TagT> index(metric, data_dim, data_num, false, 
                                false);
  auto                    s = std::chrono::high_resolution_clock::now();
  index.build(data_path.c_str(), data_num, paras);

  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cout << "Usage: " << argv[0] << "  "
              << "data_type<int8/uint8/float>  dist_fn<l2/mips> "
              << "data_file.bin   output_index_file  "
              << "R(graph degree)   L(build complexity)  "
              << "alpha(graph diameter control)   T(num_threads)" << std::endl;
    return -1;
  }

  _u32 ctr = 2;

  diskann::Metric metric;
  if (std::string(argv[ctr]) == std::string("mips")) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (std::string(argv[ctr]) == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (std::string(argv[ctr]) == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product/Cosine are supported."
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

  try {
    if (std::string(argv[1]) == std::string("int8"))
      return build_in_memory_index<int8_t>(data_path, metric, R, L, alpha,
                                           save_path, num_threads);
    else if (std::string(argv[1]) == std::string("uint8"))
      return build_in_memory_index<uint8_t>(data_path, metric, R, L, alpha,
                                            save_path, num_threads);
    else if (std::string(argv[1]) == std::string("float"))
      return build_in_memory_index<float>(data_path, metric, R, L, alpha,
                                          save_path, num_threads);
    else {
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index build failed." << std::endl;
    return -1;
  }
}
