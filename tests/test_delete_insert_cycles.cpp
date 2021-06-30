// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>

#include "utils.h"
#include "tsl/robin_set.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

template<typename T>
int build_incremental_index(const std::string& data_path,
                            const std::string& memory_index_file,
                            const unsigned L, const unsigned R,
                            const unsigned C, const unsigned num_rnds,
                            const float alpha, const std::string& save_path,
                            const unsigned num_cycles, int fraction, int tags,
                            int delete_mode) {
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

  bool eager;
  if (delete_mode == 0) {
    eager = false;
  } else {
    eager = true;
  }

  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, 1, true, eager);

  if (tags == 0)
    index.load(memory_index_file.c_str(), data_path.c_str());
  else {
    auto tag_path = memory_index_file + ".tags";
    index.load(memory_index_file.c_str(), data_path.c_str(), true,
               tag_path.c_str());
  }

  unsigned i = 0;
  while (i < num_cycles) {
    size_t              delete_size = (num_points / 100) * fraction;
    tsl::robin_set<int> delete_set;
    while (delete_set.size() < delete_size)
      delete_set.insert(rand() % num_points);
    std::vector<TagT> delete_vector(delete_set.begin(), delete_set.end());
    diskann::cout << "\nDeleting " << delete_vector.size() << " elements... ";

    {
      index.enable_delete();
      if (delete_mode == 0) {  // Lazy delete
        diskann::Timer    del_timer;
        std::vector<TagT> failed_tags;
        if (index.lazy_delete(delete_set, failed_tags) < 0)
          std::cerr << "Error in delete_points" << std::endl;
        if (failed_tags.size() > 0)
          std::cerr << "Failed to delete " << failed_tags.size() << " tags"
                    << std::endl;
        diskann::cout << "completed in " << del_timer.elapsed() / 1000000.0
                      << "sec.\n"
                      << "Starting consolidation... " << std::flush;

        diskann::Timer timer;
        if (index.disable_delete(paras, true) != 0) {
          std::cerr << "Disable delete failed" << std::endl;
          return -1;
        }
        diskann::cout << "completed in " << timer.elapsed() / 1000000.0
                      << "sec.\n"
                      << std::endl;

      } else {  // Eager delete
        diskann::Timer timer;
#pragma omp parallel for
        for (auto iter = delete_vector.begin(); iter < delete_vector.end();
             ++iter) {
          if (index.eager_delete(*iter, paras, delete_mode) != 0)
            std::cerr << "Delete tag " << *iter << " not found" << std::endl;
        }
        diskann::cout << "Eager delete time: " << timer.elapsed() / 1000000.0
                      << "sec\n";
        if (index.disable_delete(paras, false) != 0) {
          std::cerr << "Disable delete failed" << std::endl;
          return -1;
        }
      }
    }

    auto save_path_del = save_path + ".del" + std::to_string(i);
    index.save(save_path_del.c_str());

    {
      diskann::Timer timer;
#pragma omp parallel for
      for (size_t i = 0; i < delete_vector.size(); i++) {
        unsigned p = delete_vector[i];
        index.insert_point(data_load + (size_t) p * (size_t) aligned_dim, paras,
                           p);
      }
      diskann::cout << "Re-incremental time: " << timer.elapsed() / 1000
                    << "ms\n";
    }

    auto save_path_reinc = save_path + ".reinc" + std::to_string(i);
    index.save(save_path_reinc.c_str());
    i++;
  }

  delete[] data_load;

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 14) {
    diskann::cout
        << "Correct usage: " << argv[0]
        << " type[int8/uint8/float] data_file index_file L R C alpha "
           "num_rounds "
        << "save_graph_file #cycles "
           "#percent_points_deleted_in_each_cycle tags_available?[0/1] "
           "delete_mode[0-lazy/1-eager-all/2-eager-selected] "
        << std::endl;
    exit(-1);
  }

  unsigned    L = (unsigned) atoi(argv[4]);
  unsigned    R = (unsigned) atoi(argv[5]);
  unsigned    C = (unsigned) atoi(argv[6]);
  float       alpha = (float) std::atof(argv[7]);
  unsigned    num_rnds = (unsigned) std::atoi(argv[8]);
  std::string save_path(argv[9]);
  unsigned    num_cycles = (unsigned) atoi(argv[10]);
  int         fraction = (int) atoi(argv[11]);
  int         tags = (int) atoi(argv[12]);
  int         delete_mode = (int) atoi(argv[13]);

  if (std::string(argv[1]) == std::string("int8"))
    build_incremental_index<int8_t>(argv[2], argv[3], L, R, C, num_rnds, alpha,
                                    save_path, num_cycles, fraction, tags,
                                    delete_mode);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_incremental_index<uint8_t>(argv[2], argv[3], L, R, C, num_rnds, alpha,
                                     save_path, num_cycles, fraction, tags,
                                     delete_mode);
  else if (std::string(argv[1]) == std::string("float"))
    build_incremental_index<float>(argv[2], argv[3], L, R, C, num_rnds, alpha,
                                   save_path, num_cycles, fraction, tags,
                                   delete_mode);
  else
    diskann::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
