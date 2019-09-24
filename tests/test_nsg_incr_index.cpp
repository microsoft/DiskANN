#include <index_nsg.h>
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
  if (argc != 8) {
    std::cout << "Correct usage: " << argv[0]
              << " data_file L R C alpha num_rounds "
              << "save_graph_file  " << std::endl;
    exit(-1);
  }

  float* data_load = NULL;
  size_t num_points, dim, aligned_dim;

  NSG::load_aligned_bin<float>(argv[1], data_load, num_points, dim,
                               aligned_dim);

  unsigned    L = (unsigned) atoi(argv[2]);
  unsigned    R = (unsigned) atoi(argv[3]);
  unsigned    C = (unsigned) atoi(argv[4]);
  float       alpha = (float) std::atof(argv[5]);
  unsigned    num_rnds = (unsigned) std::atoi(argv[6]);
  std::string save_path(argv[7]);

  NSG::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<unsigned>("num_rnds", num_rnds);

  num_points = 20000;
  unsigned num_incr = 1000;

  typedef int TagT;

  NSG::IndexNSG<float, TagT> index(NSG::L2, argv[1], num_points,
                                   num_points - num_incr, true);
  {
    std::vector<TagT> tags(num_points - num_incr);
    std::iota(tags.begin(), tags.end(), 0);

    NSG::Timer timer;
    index.build(paras, tags);
    std::cout << "Index build time: " << timer.elapsed() / 1000 << "ms\n";
  }

  std::vector<NSG::Neighbor>       pool, tmp;
  tsl::robin_set<unsigned>         visited;
  std::vector<NSG::SimpleNeighbor> cut_graph;

  {
    NSG::Timer timer;
    for (size_t i = num_points - num_incr; i < num_points; ++i) {
      index.insert_point(data_load + i * dim, paras, pool, tmp, visited,
                         cut_graph, i);
      std::cout << i << std::endl;
    }
    std::cout << "Incremental time: " << timer.elapsed() / 1000 << "ms\n";
    index.save(save_path.c_str());
  }

  tsl::robin_set<unsigned> delete_list;
  while (delete_list.size() < num_incr)
    delete_list.insert((rand() * rand() * rand()) % num_points);
  std::cout << "Deleting " << delete_list.size() << " elements" << std::endl;

  {
    NSG::Timer timer;
    index.enable_delete();
    for (auto p : delete_list)
      if (index.delete_point(p) != 0)
        std::cerr << "Delete tag " << p << " not found" << std::endl;

    if (index.disable_delete(paras, true) != 0) {
      std::cerr << "Disable delete failed" << std::endl;
      return -1;
    }
    std::cout << "Delete time: " << timer.elapsed() / 1000 << "ms\n";
  }

  {
    NSG::Timer timer;
    for (auto p : delete_list)
      index.insert_point(data_load + (size_t) p * (size_t) dim, paras, pool,
                         tmp, visited, cut_graph, p);
    std::cout << "Re-incremental time: " << timer.elapsed() / 1000 << "ms\n";
    auto save_path_reinc = save_path + ".reinc";
    index.save(save_path_reinc.c_str());
  }

  delete[] data_load;

  return 0;
}
