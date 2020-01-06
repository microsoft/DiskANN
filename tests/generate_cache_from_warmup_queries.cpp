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
  T*     warmup = nullptr;
  size_t warmup_num, ndims, warmup_aligned_dim;

  std::string pq_centroids_file(argv[2]);
  std::string compressed_data_file(argv[3]);
  std::string disk_index_file(argv[4]);
  std::string medoids_file(argv[5]);
  std::string centroid_data_file(argv[6]);
  std::string warmup_bin(argv[7]);
  _u64        Lsearch = std::atoi(argv[8]);
  _u32        beam_width = std::atoi(argv[9]);
  _u64        num_cache_nodes = std::atoi(argv[10]);
  std::string cache_list_bin(argv[11]);

  _u32 num_threads = 32;
  _u32 cache_nlevels = 3;

  std::cout << "Search parameters: #threads: " << num_threads
            << ", beamwidth: " << beam_width << std::endl;

  diskann::load_aligned_bin<T>(warmup_bin, warmup, warmup_num, ndims,
                               warmup_aligned_dim);

  diskann::PQFlashIndex<T> _pFlashIndex;

  _pFlashIndex.set_cache_create_flag();
  int res =
      _pFlashIndex.load(num_threads, pq_centroids_file.c_str(),
                        compressed_data_file.c_str(), disk_index_file.c_str());
  if (res != 0) {
    return res;
  }

  _pFlashIndex.load_entry_points(medoids_file, centroid_data_file);
  _pFlashIndex.cache_medoid_nhoods();

  std::cout << "Caching BFS levels " << cache_nlevels << " around medoid(s)."
            << std::endl;
  _pFlashIndex.cache_bfs_levels(cache_nlevels);

  omp_set_num_threads(num_threads);
  unsigned recall_at = 1;
  _u64*    warmup_res = new _u64[recall_at * warmup_num];
  float*   warmup_dists = new float[recall_at * warmup_num];

#pragma omp parallel for schedule(dynamic, 1)
  for (_s64 i = 0; i < (int32_t) warmup_num; i++) {
    _pFlashIndex.cached_beam_search(warmup + (i * warmup_aligned_dim),
                                    recall_at, Lsearch,
                                    warmup_res + (i * recall_at),
                                    warmup_dists + (i * recall_at), beam_width);
  }

  diskann::aligned_free(warmup);
  delete[] warmup_res;
  delete[] warmup_dists;

  std::cout << "Saving cache list to file " << cache_list_bin.c_str()
            << std::endl;
  _pFlashIndex.save_cached_nodes(num_cache_nodes, cache_list_bin);

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 12) {
    std::cout << "Usage: " << argv[0]
              << " <index_type[float/int8/uint8]>  <pq_centroids_bin> "
                 "<compressed_data_bin> <disk_index_path>  "
                 "<medoids_bin (use \"null\" if none)> <centroid_float_bin> "
                 "(use \"null\" if no medoids file) "
                 "<warmup_bin>  "
                 "<Lsearch> <beam_width> <num_nodes_to_cache> "
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
