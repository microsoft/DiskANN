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
int aux_main(int argc, char** argv) {
  std::string load_path(argv[3]);
  unsigned    new_degree = (unsigned) atoi(argv[4]);
  float       alpha = (float) atof(argv[5]);
  std::string save_path(argv[6]);

  NSG::Parameters paras;
  //  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", 1000);
  //  paras.Set<unsigned>("num_rnds", num_rnds);
  paras.Set<float>("alpha", alpha);

  NSG::IndexNSG<T> index(NSG::L2, argv[2]);
  auto             s = std::chrono::high_resolution_clock::now();
  index.load(load_path.c_str());
  index.truncate_degree(paras, new_degree);
  auto                          e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "Re-indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout
        << "Correct usage: " << argv[0]
        << " data_type [float/int8/uint8] data_file old_index new_degree alpha"
        << "new_index " << std::endl;
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
