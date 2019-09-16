//
// Created by 付聪 on 2017/6/21.
//

#include <index_nsg.h>
#include <omp.h>
#include <string.h>
#include "utils.h"

#ifndef __NSG_WINDOWS__
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"

template<typename T>
int build_in_memory_index(const std::string& data_path, const unsigned L,
                          const unsigned R, const unsigned C,
                          const unsigned num_rnds, const float alpha,
                          const std::string& save_path) {
  NSG::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<unsigned>("num_rnds", num_rnds);
  paras.Set<float>("alpha", alpha);

  NSG::IndexNSG<T> index(NSG::L2, data_path);
  auto             s = std::chrono::high_resolution_clock::now();
  index.build(paras);
  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cout << "Usage:\n"
              << argv[0] << "  data_type<int8/uint8/float>  <data_file.bin>"
              << "  L  R  C  #rounds  alpha"
              << "  <output_graph_prefix>" << std::endl;
    exit(-1);
  }

  const std::string data_path(argv[2]);
  const unsigned    L = (unsigned) atoi(argv[3]);
  const unsigned    R = (unsigned) atoi(argv[4]);
  const unsigned    C = (unsigned) atoi(argv[5]);
  const unsigned    num_rnds = (unsigned) atoi(argv[6]);
  const float       alpha = (float) atof(argv[7]);
  const std::string save_path =
      std::string(argv[8]) + std::string("_unopt.rnsg");

  if (std::string(argv[1]) == std::string("int8"))
    build_in_memory_index<int8_t>(data_path, L, R, C, num_rnds, alpha,
                                  save_path);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_in_memory_index<uint8_t>(data_path, L, R, C, num_rnds, alpha,
                                   save_path);
  else if (std::string(argv[1]) == std::string("float"))
    build_in_memory_index<float>(data_path, L, R, C, num_rnds, alpha,
                                 save_path);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
