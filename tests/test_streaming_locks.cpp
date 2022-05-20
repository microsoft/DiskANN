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
                            const unsigned      threads) {
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
  int64_t deletes_offset = (uint64_t) (.5*num_points);
  int64_t deletes_size = (int64_t) (.25*num_points);
  {
    //tags need to start at .5*data_size, increase to .75*data_size, then 0-.25*data_size
    std::vector<TagT> tags((int64_t) num_points*.5);
    std::iota(tags.begin(), tags.begin()+deletes_size, deletes_offset);
    std::iota(tags.begin()+deletes_size, tags.end(), 0); 

    diskann::Timer timer;
    index.build(data_path.c_str(), (int64_t) num_points*.5, paras, tags);
    std::cout << "Index build time: " << timer.elapsed() / 1000 << "ms\n";
  }

    

  std::vector<unsigned> delete_vector(deletes_size);
  std::iota(delete_vector.begin(), delete_vector.end(), deletes_offset);


  int sub_threads = (threads + 1) / 2;

  int64_t inserts_size = (int64_t) (.25*num_points);
  int64_t inserts_start = (int64_t) (.5*num_points);
  int64_t inserts_end = (int64_t) (.75*num_points);
  std::vector<TagT> insert_tags(inserts_size);
  std::iota(insert_tags.begin(), insert_tags.end(), inserts_size);

  {
    diskann::Timer timer;
    index.enable_delete();

    auto inserts = std::async(std::launch::async, [&] (){
      #pragma omp        parallel for num_threads(sub_threads)
        for (int64_t j = inserts_start; j < inserts_end; ++j) {
          index.insert_point(data_load + j * aligned_dim, paras, insert_tags[j-inserts_start]);
        }
    });



    auto deletes = std::async(std::launch::async, [&] (){
      #pragma omp parallel for num_threads(sub_threads)
        for (size_t i = 0; i < delete_vector.size(); i++) {
          unsigned p = delete_vector[i];
          if (index.lazy_delete(p) != 0)
            std::cerr << "Delete tag " << p << " not found" << std::endl;
        }
      if (index.disable_delete_2(paras, true) != 0) {
        std::cerr << "Disable delete failed" << std::endl;
      }
    });

    inserts.wait(); 
    deletes.wait();

  
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
