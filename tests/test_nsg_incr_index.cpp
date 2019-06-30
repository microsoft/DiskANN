//
// Created by 付聪 on 2017/6/21.
//

#include <index_nsg.h>

#include <omp.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "util.h"

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << "Correct usage: " << argv[0]
              << " data_file L R C alpha num_rounds "
              << "save_graph_file  " << std::endl;
    exit(-1);
  }

  float* data_load = NULL;
  size_t points_num, dim;

  //  load_file_into_data<float>(argv[1], data_load, points_num, dim);
  NSG::load_Tvecs<float>(argv[1], data_load, points_num, dim);
  data_load = NSG::data_align(data_load, points_num, dim);
  std::cout << "Data loaded and aligned" << std::endl;

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

  unsigned num_incr = 100000;

  NSG::IndexNSG<float> index(dim, points_num - num_incr, NSG::L2, nullptr,
                             points_num);
  {
    auto s = std::chrono::high_resolution_clock::now();
    index.BuildRandomHierarchical(data_load, paras);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Indexing time: " << diff.count() << "\n";
  }
  {
    std::vector<NSG::Neighbor>       pool, tmp;
    tsl::robin_set<unsigned>         visited;
    std::vector<NSG::SimpleNeighbor> cut_graph;

    auto s = std::chrono::high_resolution_clock::now();
    for (unsigned i = points_num - num_incr; i < points_num; ++i)
      index.insert_point(data_load + (size_t) i * (size_t) dim, paras, pool,
                         tmp, visited, cut_graph);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Incremental time: " << diff.count() << "\n";
  }

  index.Save(save_path.c_str());
  // index.Save_Inner_Vertices(argv[5]);

  return 0;
}
