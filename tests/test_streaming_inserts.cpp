// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
// #include <sync_index.h>
#include <future>
#include <time.h>
#include <timer.h>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"



template<typename T>
int build_incremental_index(const std::string& data_path, const unsigned L,
                            const unsigned R, const unsigned C,
                            const unsigned num_rnds, const float alpha,
                            const std::string& save_path,
                            const unsigned     num_frozen,
                            const unsigned threads) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", false);
  paras.Set<unsigned>("num_rnds", num_rnds);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  typedef int TagT;


  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, true, true,
                                 false);
  {
    std::vector<TagT> tags((int64_t) num_points*.5);
    std::iota(tags.begin(), tags.end(), 0);


    diskann::Timer timer;
    index.build(data_path.c_str(), (int64_t) num_points*.5, paras, tags);
    std::cout << "Index build time: " << timer.elapsed() / 1000 << "ms\n";

  }


  int64_t inserts_start = (int64_t) (.5*num_points);
  int64_t inserts_end = (int64_t) (num_points);

  {
    diskann::Timer timer;

    auto inserts = std::async(std::launch::async, [&] (){
      #pragma omp        parallel for num_threads(threads)
        for (int64_t j = inserts_start; j < inserts_end; ++j) {
          index.insert_point(data_load + j * aligned_dim, paras, j);
        }
    });

    inserts.wait(); 


  
    std::cout << "Time Elapsed" << timer.elapsed() / 1000 << "ms\n";

  }
      index.save(save_path.c_str());


  delete[] data_load;

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 11) {
    std::cout << "Correct usage: " << argv[0]
              << " type[int8/uint8/float] data_file L R C alpha "
                 "num_rounds "
              << "save_graph_file  #frozen_points #threads" << std::endl;
    exit(-1);
  }

  unsigned    L = (unsigned) atoi(argv[3]);
  unsigned    R = (unsigned) atoi(argv[4]);
  unsigned    C = (unsigned) atoi(argv[5]);
  float       alpha = (float) std::atof(argv[6]);
  unsigned    num_rnds = (unsigned) std::atoi(argv[7]);
  std::string save_path(argv[8]);
  // unsigned    num_incr = (unsigned) atoi(argv[9]);
  unsigned    num_frozen = (unsigned) atoi(argv[9]);
  unsigned    num_threads = (unsigned) atoi(argv[10]);

  if (std::string(argv[1]) == std::string("int8"))
    build_incremental_index<int8_t>(argv[2], L, R, C, num_rnds, alpha,
                                    save_path, num_frozen, num_threads);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_incremental_index<uint8_t>(argv[2], L, R, C, num_rnds, alpha,
                                     save_path, num_frozen, num_threads);
  else if (std::string(argv[1]) == std::string("float"))
    build_incremental_index<float>(argv[2], L, R, C, num_rnds, alpha, save_path,
                                   num_frozen, num_threads);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
