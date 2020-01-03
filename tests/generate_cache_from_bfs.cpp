//#include <distances.h>
//#include <indexing.h>
#include <index.h>
#include <math_utils.h>
#include <omp.h>
#include <pq_flash_index.h>
#include <string.h>
#include <time.h>
#include <atomic>
#include <cstring>
#include <iomanip>
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

template<typename T>
int generate_cache_list(int argc, char** argv) {
  // load warmup bin
  std::string pq_centroids_file(argv[2]);
  std::string compressed_data_file(argv[3]);
  std::string disk_index_file(argv[4]);
  std::string medoids_file(argv[5]);
  std::string centroid_data_file(argv[6]);
  _u64        num_cache_nodes = std::atoi(argv[7]);
  std::string cache_list_bin(argv[8]);

  _u32                     num_threads = 32;
  diskann::PQFlashIndex<T> _pFlashIndex;

  int res =
      _pFlashIndex.load(num_threads, pq_centroids_file.c_str(),
                        compressed_data_file.c_str(), disk_index_file.c_str());
  if (res != 0) {
    return res;
  }

  _pFlashIndex.load_entry_points(medoids_file, centroid_data_file);

  std::vector<uint32_t> node_list;
  _pFlashIndex.cache_bfs_levels(num_cache_nodes, node_list);

  std::cout << "Saving cache list to file " << cache_list_bin.c_str()
            << std::endl;
  diskann::save_bin<uint32_t>(cache_list_bin.c_str(), node_list.data(),
                              node_list.size(), 1);
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cout << "Usage: " << argv[0]
              << " <index_type[float/int8/uint8]>  <pq_centroids_bin> "
                 "<compressed_data_bin> <disk_index_path>  "
                 "<medoids_bin (use \"null\" if none)> <centroid_float_bin> "
                 "(use \"null\" if no medoids file) "
                 " <num_nodes_to_cache> "
                 "<output_cache_list_bin> "
              << std::endl;
    exit(-1);
  }
  if (std::string(argv[1]) == std::string("float"))
    generate_cache_list<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    generate_cache_list<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    generate_cache_list<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8"
              << std::endl;
}
