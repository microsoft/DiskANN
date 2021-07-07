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
                            const unsigned R, const unsigned C,
                            const unsigned num_rnds, const float alpha,
                            const std::string& save_path,
                            const unsigned num_incr, const unsigned num_frozen,
                            const int del_mode) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<unsigned>("num_rnds", num_rnds);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  typedef int TagT;

  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, num_frozen, true,
                                true);
  diskann::cout << "num_points = " << num_points;
  {
    std::vector<TagT> tags(num_points - num_incr);
    std::iota(tags.begin(), tags.end(), 0);

    diskann::Timer timer;
    index.build(data_path.c_str(), num_points - num_incr, paras, tags);
    diskann::cout << "Index build time: " << timer.elapsed() / 1000 << "ms\n";
  }

  unsigned insert_size = (num_points / 100) * 2;
  unsigned rounds = 100 / 2;

  tsl::robin_set<unsigned> insert_list;
  tsl::robin_set<unsigned> delete_list;
  tsl::robin_set<unsigned> used_tags;
  tsl::robin_set<unsigned> inserted_tags;
  tsl::robin_set<unsigned> deleted_tags;
  unsigned                 num_curr_pts = num_points - num_incr + num_frozen;

  for (unsigned i = 0; i < rounds; i++) {
    diskann::cout << i << std::endl << std::endl;
    /*___________________________Insertion____________________*/

    while (insert_list.size() < insert_size) {
      insert_list.insert(rand() % num_points);
    }
    unsigned tag_p;
    size_t   res = used_tags.size();
    while (used_tags.size() < res + insert_size) {
      tag_p = rand() % num_points + 1;
      size_t temp = used_tags.size();
      used_tags.insert(tag_p);
      if (used_tags.size() > temp)
        inserted_tags.insert(tag_p);
    }
    diskann::cout << "Inserting " << insert_size << " points" << std::endl;
    tsl::robin_set<unsigned>::iterator it = inserted_tags.begin();
    diskann::Timer                     insert_timer;
    for (auto p : insert_list) {
      index.insert_point(data_load + p * aligned_dim, paras, *it);
      it++;
    }
    diskann::cout << "Insertion time " << insert_timer.elapsed() / 1000
                  << "ms\n";

    auto save_path_inc =
        save_path + ".inc" + std::to_string(i);  // 10 -denotes that 10% of base
                                                 // points are being inserted at
                                                 // a time
    index.save(save_path_inc.c_str());

    num_curr_pts += insert_size;
    diskann::cout << "Number of points in the index post insertion "
                  << num_curr_pts << std::endl;
    insert_list.clear();

    /*_________________________Deleting points___________________________*/

    unsigned delete_size = (unsigned) (num_curr_pts / 10);
    while (deleted_tags.size() < delete_size) {
      auto                               r = rand() % used_tags.size();
      tsl::robin_set<unsigned>::iterator iter = used_tags.begin();
      for (unsigned j = 0; j < r; j++)
        iter++;
      size_t res = delete_list.size();
      delete_list.insert(*iter);
      if (delete_list.size() > res)
        deleted_tags.insert(*iter);
    }
    diskann::cout << "Deleting " << delete_size << " points from the index"
                  << std::endl;

    index.enable_delete();
    for (auto p : deleted_tags) {
      if (del_mode == 0) {
        if (index.eager_delete(p, paras) != 0) {
          std::cerr << "Delete tag " << p << " not found" << std::endl;
        } else
          used_tags.erase(p);
      } else {
        if (index.lazy_delete(p) != 0)
          std::cerr << "Delete tag" << p << "not found" << std::endl;
        else
          used_tags.erase(p);
      }
    }
    if (index.disable_delete(paras, true) != 0) {
      std::cerr << "Disable delete failed" << std::endl;
      return -1;
    }
    diskann::cout << "Delete time in this phase " << del_timer.elapsed() / 1000
                  << "ms\n";
    deleted_tags.clear();
    inserted_tags.clear();

    auto save_path_del = save_path + ".del" + std::to_string(i);
    index.save(save_path_del.c_str());

    num_curr_pts -= delete_size;
    diskann::cout << "Number of points in the graph currently = "
                  << num_curr_pts << std::endl;
  }

  {
    diskann::Timer timer;
    for (size_t i = num_points - num_incr; i < num_points; ++i) {
      index.insert_point(data_load + i * aligned_dim, paras, i);
    }
    diskann::cout << "Incremental time: " << timer.elapsed() / 1000 << "ms\n";
    auto save_path_inc = save_path + ".inc";
    index.save(save_path_inc.c_str());
  }

  delete[] data_load;

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 12) {
    diskann::cout << "Correct usage: " << argv[0]
                  << " type[int8/uint8/float] data_file L R C alpha "
                     "num_rounds "
                  << "save_graph_file #incr_points #frozen_points "
                     "delete_mode[0-eager/1-lazy]"
                  << std::endl;
    exit(-1);
  }

  unsigned    L = (unsigned) atoi(argv[3]);
  unsigned    R = (unsigned) atoi(argv[4]);
  unsigned    C = (unsigned) atoi(argv[5]);
  float       alpha = (float) std::atof(argv[6]);
  unsigned    num_rnds = (unsigned) std::atoi(argv[7]);
  std::string save_path(argv[8]);
  unsigned    num_incr = (unsigned) atoi(argv[9]);
  unsigned    num_frozen = (unsigned) atoi(argv[10]);
  int         del_mode = (int) atoi(argv[11]);

  if (std::string(argv[1]) == std::string("int8"))
    build_incremental_index<int8_t>(argv[2], L, R, C, num_rnds, alpha,
                                    save_path, num_incr, num_frozen, del_mode);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_incremental_index<uint8_t>(argv[2], L, R, C, num_rnds, alpha,
                                     save_path, num_incr, num_frozen, del_mode);
  else if (std::string(argv[1]) == std::string("float"))
    build_incremental_index<float>(argv[2], L, R, C, num_rnds, alpha, save_path,
                                   num_incr, num_frozen, del_mode);
  else
    diskann::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
