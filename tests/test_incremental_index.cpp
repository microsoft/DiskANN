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

int main(int argc, char** argv) {
  if (argc != 10) {
    std::cout << "Correct usage: " << argv[0]
              << " data_file L R C alpha num_rounds "
              << "save_graph_file #incr_points #frozen_points" << std::endl;
    exit(-1);
  }

  float* data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<float>(argv[1], data_load, num_points, dim,
                                   aligned_dim);

  unsigned    L = (unsigned) atoi(argv[2]);
  unsigned    R = (unsigned) atoi(argv[3]);
  unsigned    C = (unsigned) atoi(argv[4]);
  float       alpha = (float) std::atof(argv[5]);
  unsigned    num_rnds = (unsigned) std::atoi(argv[6]);
  std::string save_path(argv[7]);
  unsigned    num_incr = (unsigned) atoi(argv[8]);
  unsigned    num_frozen = (unsigned) atoi(argv[9]);

  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", false);
  paras.Set<unsigned>("num_rnds", num_rnds);

  typedef int                 TagT;
  diskann::Index<float, TagT> index(diskann::L2, argv[1], num_points,
                                    num_points - num_incr, num_frozen, true,
                                    true, true);
  {
    std::vector<TagT> tags(num_points - num_incr);
    std::iota(tags.begin(), tags.end(), 0);

    if (argc > 10) {
      std::string frozen_points_file(argv[10]);
      index.generate_random_frozen_points(frozen_points_file.c_str());
    } else
      index.generate_random_frozen_points();

    diskann::Timer timer;
    index.build(paras, tags);
    std::cout << "Index build time: " << timer.elapsed() / 1000 << "ms\n";
  }

  std::vector<diskann::Neighbor>       pool, tmp;
  tsl::robin_set<unsigned>             visited;
  std::vector<diskann::SimpleNeighbor> cut_graph;
  index.readjust_data(num_frozen);

  {
    diskann::Timer timer;
    for (size_t i = num_points - num_incr; i < num_points; ++i) {
      index.insert_point(data_load + i * aligned_dim, paras, pool, tmp, visited,
                         cut_graph, i);
    }
    std::cout << "Incremental time: " << timer.elapsed() / 1000 << "ms\n";
    auto save_path_inc = save_path + ".inc";
    index.save(save_path_inc.c_str());
  }

  tsl::robin_set<unsigned> delete_list;
  while (delete_list.size() < num_incr)
    delete_list.insert(rand() % num_points);
  std::cout << "Deleting " << delete_list.size() << " elements" << std::endl;

  {
    diskann::Timer timer;
    index.enable_delete();
    for (auto p : delete_list)

      if (index.eager_delete(p, paras) != 0)
        //    if (index.delete_point(p) != 0)
        std::cerr << "Delete tag " << p << " not found" << std::endl;

    if (index.disable_delete(paras, true) != 0) {
      std::cerr << "Disable delete failed" << std::endl;
      return -1;
    }
    std::cout << "Delete time: " << timer.elapsed() / 1000 << "ms\n";
  }

  auto save_path_del = save_path + ".del";
  index.save(save_path_del.c_str());

  index.readjust_data(num_frozen);
  {
    diskann::Timer timer;
    for (auto p : delete_list) {
      index.insert_point(data_load + (size_t) p * (size_t) aligned_dim, paras,
                         pool, tmp, visited, cut_graph, p);
    }
    std::cout << "Re-incremental time: " << timer.elapsed() / 1000 << "ms\n";
  }

  auto save_path_reinc = save_path + ".reinc";
  index.save(save_path_reinc.c_str());

  delete[] data_load;

  return 0;
}
