// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
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
                            const unsigned R, const float alpha,
                            const std::string& save_path,
                            const unsigned     num_incr) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", 750);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", false);
  paras.Set<unsigned>("num_rnds", 2);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  typedef uint32_t TagT;

  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, true,
                                false,  // single file index
                                true);
  {
    std::vector<TagT> tags(num_points - num_incr);
    std::iota(tags.begin(), tags.end(), 0);

    diskann::Timer timer;
    index.build(data_path.c_str(), num_points - num_incr, paras, tags);
    diskann::cout << "Index build time: " << timer.elapsed() / 1000 << "ms\n";
    index.save(save_path.c_str());
  }

  {
    diskann::Timer timer;
#pragma omp parallel for
    for (_s64 i = num_points - num_incr; i < (_s64) num_points; ++i) {
      index.insert_point(data_load + i * aligned_dim, paras, (TagT) i);
    }
    diskann::cout << "Incremental time: " << timer.elapsed() / 1000 << "ms\n";
    auto save_path_inc = save_path + ".inc";
    index.save(save_path_inc.c_str());
  }

  tsl::robin_set<unsigned> delete_list;
  while (delete_list.size() < num_incr)
    delete_list.insert((uint32_t)(rand() % num_points));
  diskann::cout << "Deleting " << delete_list.size() << " elements"
                << std::endl;
  std::vector<unsigned> delete_vector;

  for (auto p : delete_list) {
    delete_vector.emplace_back(p);
  }
  diskann::cout << "Size of delete_vector : " << delete_vector.size()
                << std::endl;
  {
    index.enable_delete();
    for (size_t i = 0; i < delete_vector.size(); i++) {
      unsigned p = delete_vector[i];
      if (index.lazy_delete(p) != 0)
        std::cerr << "Delete tag " << p << " not found" << std::endl;
    }
  }

  auto save_path_del = save_path + ".delete";
  index.save(save_path_del.c_str());
  index.load(save_path_del.c_str());
  index.consolidate(paras);

  {
    index.reposition_frozen_point_to_end();
    std::vector<_u32> reinsert_vec;
    for (auto p : delete_list)
      reinsert_vec.emplace_back(p);

    diskann::Timer timer;
#pragma omp parallel for
    for (_s64 p = 0; p < (_s64) reinsert_vec.size(); p++) {
      index.insert_point(
          data_load + (size_t)(reinsert_vec[p]) * (size_t) aligned_dim, paras,
          (reinsert_vec[p]));
    }
    diskann::cout << "Re-incremental time: " << timer.elapsed() / 1000
                  << "ms\n";
  }

  auto save_path_reinc = save_path + ".reinc";
  index.save(save_path_reinc.c_str());

  delete[] data_load;

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 8) {
    diskann::cout << "Correct usage: " << argv[0]
                  << " type[int8/uint8/float] data_file L R alpha "
                  << "save_graph_file #incr_points " << std::endl;
    exit(-1);
  }

  unsigned    L = (unsigned) atoi(argv[3]);
  unsigned    R = (unsigned) atoi(argv[4]);
  float       alpha = (float) std::atof(argv[5]);
  std::string save_path(argv[6]);
  unsigned    num_incr = (unsigned) atoi(argv[7]);

  if (std::string(argv[1]) == std::string("int8"))
    build_incremental_index<int8_t>(argv[2], L, R, alpha, save_path, num_incr);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_incremental_index<uint8_t>(argv[2], L, R, alpha, save_path, num_incr);
  else if (std::string(argv[1]) == std::string("float"))
    build_incremental_index<float>(argv[2], L, R, alpha, save_path, num_incr);
  else
    diskann::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
