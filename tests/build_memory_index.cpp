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
int build_in_memory_index(const std::string& data_path, _u32 dist_fn, const unsigned R,
                          const unsigned L, const float alpha,
                          const std::string& save_path,
                          const unsigned     num_threads, const std::string& learn_path) {
  diskann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>(
      "C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", 0);
  paras.Set<unsigned>("num_threads", num_threads);

  diskann::Metric metric;
  if (dist_fn == 0)
    metric = diskann::L2;
  else if (dist_fn == 1)
      metric = diskann::INNER_PRODUCT;
  else {
    std::cout<<"Error. Unsupported distance type. Exitting" << std::endl;
    return -1;
  }

  _u64 data_pts, data_dim, learn_pts = 0;
  diskann::get_bin_metadata(data_path, data_pts, data_dim);
  if (learn_path != "") {
  diskann::get_bin_metadata(learn_path, learn_pts, data_dim);
  }


  diskann::Index<T> index(metric, data_path.c_str(), data_pts + learn_pts);
  if (learn_pts > 0) {
  index.setup_learn_data(learn_path);
  }
  auto              s = std::chrono::high_resolution_clock::now();
  index.build(paras);
  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 9 && argc != 10) {
    std::cout << "Usage: " << argv[0]
              << "  [data_type<int8/uint8/float>] [dist_fn 0 for L2, 1 for inner product] [data_file.bin]  "
                 "[output_index_file]  "
              << "[R]  [L]  [alpha]"
              << "  [num_threads_to_use] [optional: learn_path]. See README for more information on "
                 "parameters."
              << std::endl;
    exit(-1);
  }

  _u32 ctr = 2;

  _u32 dist_fn = (_u32) atoi(argv[ctr++]);
  const std::string data_path(argv[ctr++]);
  const std::string save_path(argv[ctr++]);
  const unsigned    R = (unsigned) atoi(argv[ctr++]);
  const unsigned    L = (unsigned) atoi(argv[ctr++]);
  const float       alpha = (float) atof(argv[ctr++]);
  const unsigned    num_threads = (unsigned) atoi(argv[ctr++]);
  std::string learn_path = "";
  if (argc == 10) {
    learn_path = std::string(argv[ctr++]);
  }

  if (std::string(argv[1]) == std::string("int8"))
    build_in_memory_index<int8_t>(data_path, dist_fn, R, L, alpha, save_path,
                                  num_threads, learn_path);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_in_memory_index<uint8_t>(data_path, dist_fn, R, L, alpha, save_path,
                                   num_threads, learn_path);
  else if (std::string(argv[1]) == std::string("float"))
    build_in_memory_index<float>(data_path, dist_fn, R, L, alpha, save_path,
                                 num_threads, learn_path);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
