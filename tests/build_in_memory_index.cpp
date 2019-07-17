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

template<typename T>
int aux_main(int argc, char** argv) {
  if (argc != 9) {
    std::cout
        << "Correct usage: " << argv[0]
        << " data_type [float/int8/uint8] data_file L R C num_rounds alpha"
        << "save_graph_file " << std::endl;
    exit(-1);
  }
  T*     data_load = NULL;
  size_t points_num, dim;

  //  NSG::load_bin<T>(argv[2], data_load, points_num, dim);
  //  data_load = NSG::data_align<T>(data_load, points_num, dim);
  //  std::cout << "Data loaded and aligned" << std::endl;

  unsigned    L = (unsigned) atoi(argv[3]);
  unsigned    R = (unsigned) atoi(argv[4]);
  unsigned    C = (unsigned) atoi(argv[5]);
  unsigned    num_rnds = (unsigned) atoi(argv[6]);
  float       alpha = (float) atof(argv[7]);
  std::string save_path(argv[8]);

  NSG::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<unsigned>("num_rnds", num_rnds);
  paras.Set<float>("alpha", alpha);

  NSG::IndexNSG<T> index(NSG::L2, argv[2]);
  auto             s = std::chrono::high_resolution_clock::now();
  index.build(paras);
  auto                          e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cout
        << "Correct usage: " << argv[0]
        << " data_type[int8/uint8/float] data_bin_file L R C num_rounds alpha "
        << "save_graph_file " << std::endl;
    exit(-1);
  }
  if (std::string(argv[1]) == std::string("int8"))
    aux_main<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    aux_main<uint8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("float"))
    aux_main<float>(argc, argv);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
