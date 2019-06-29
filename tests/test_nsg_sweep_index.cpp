//
// Created by 付聪 on 2017/6/21.
//

#include <index_nsg.h>
#include <omp.h>
#include <string.h>
#include "util.h"

#ifndef __NSG_WINDOWS__
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "MemoryMapper.h"

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << "Correct usage\n"
              << argv[0] << " data_file L R C "
              << "save_graph_file " << std::endl;
    exit(-1);
  }

  float* data_load = NULL;
  size_t points_num, dim;

  load_file_into_data<float>(argv[1], data_load, points_num, dim);
  data_load = NSG::data_align(data_load, points_num, dim);
  std::cout << "Data loaded and aligned" << std::endl;

  //  unsigned    nn_graph_deg = (unsigned) atoi(argv[3]);
  unsigned    L = (unsigned) atoi(argv[2]);
  unsigned    R = (unsigned) atoi(argv[3]);
  unsigned    C = (unsigned) atoi(argv[4]);
  std::string save_path(argv[5]);
  float       alpha = 1.2f;  //(float) std::atof(argv[6]);
  //  float       p_val = 0.05f;  //(float) std::atof(argv[7]);
  //  unsigned    num_hier = (float) std::atof(argv[8]);
  //  unsigned num_syncs = 150;  //(float) std::atof(argv[8]);
  unsigned num_rnds = 2;
  //  unsigned    innerL = (unsigned) atoi(argv[11]);
  //  unsigned innerR = R;  //(unsigned) atoi(argv[10]);
  //  unsigned    innerC = (unsigned) atoi(argv[13]);

  //  if (nn_graph_deg > R) {
  //    std::cerr << "Error: nn_graph_degree must be <= R" << std::endl;
  //    exit(-1);
  //  }

  NSG::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<unsigned>("num_rnds", num_rnds);
  paras.Set<std::string>("save_path", save_path);
  //  paras.Set<float>("p_val", p_val);
  //  paras.Set<unsigned>("innerL", innerL);
  //  paras.Set<unsigned>("innerC", innerC);
  //  paras.Set<unsigned>("num_syncs", num_syncs);

  NSG::IndexNSG<float> index(dim, points_num, NSG::L2, nullptr);
  auto                 s = std::chrono::high_resolution_clock::now();
  index.BuildRandomHierarchical(data_load, paras);
  auto                          e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.Save(save_path.c_str());

  return 0;
}
