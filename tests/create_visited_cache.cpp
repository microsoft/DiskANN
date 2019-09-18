//#include <distances.h>
//#include <indexing.h>
#include <index_nsg.h>
#include <math_utils.h>
#include <omp.h>
#include <pq_flash_index_nsg.h>
#include <string.h>
#include <time.h>
#include <atomic>
#include <cstring>
#include <iomanip>
#include "partition_and_pq.h"
#include "timer.h"

#ifndef __NSG_WINDOWS__
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

template<typename T>
bool load_index(const char* indexFilePath, const char* warmupParameters,
                NSG::PQFlashNSG<T>*& _pFlashIndex) {
  std::stringstream parser;
  parser << std::string(warmupParameters);
  std::string              cur_param;
  std::vector<std::string> param_list;
  while (parser >> cur_param)
    param_list.push_back(cur_param);

  if (param_list.size() != 3) {
    std::cerr << "Correct usage of parameters is \n"
                 "BeamWidth[1] cache_nlevels[2] nthreads[3]"
              << std::endl;
    return 1;
  }

  const std::string index_prefix_path(indexFilePath);

  // convert strs into params
  std::string data_bin = index_prefix_path + "_compressed.bin";
  std::string pq_tables_bin = index_prefix_path + "_pq_pivots.bin";
  std::string medoids_bin = index_prefix_path + "_medoids.bin";

  // determine nchunks
  std::string params_path = index_prefix_path + "_params.bin";
  std::string node_visit_bin = index_prefix_path + "_visit_ctr.bin";

  _u32 dim32, num_points32, num_chunks32, num_centers32, chunk_size32;

  std::ifstream pq_meta_reader(pq_tables_bin, std::ios::binary);
  pq_meta_reader.read((char*) &num_centers32, sizeof(uint32_t));
  pq_meta_reader.read((char*) &dim32, sizeof(uint32_t));
  pq_meta_reader.close();

  std::ifstream compressed_meta_reader(data_bin, std::ios::binary);
  compressed_meta_reader.read((char*) &num_points32, sizeof(uint32_t));
  compressed_meta_reader.read((char*) &num_chunks32, sizeof(uint32_t));
  compressed_meta_reader.close();

  // infer chunk_size
  chunk_size32 = DIV_ROUND_UP(dim32, num_chunks32);

  _u64 m_dimension = (_u64) dim32;
  _u64 n_chunks = (_u64) num_chunks32;
  _u64 chunk_size = chunk_size32;

  std::string nsg_disk_opt = index_prefix_path + "_diskopt.rnsg";

  _u64        beam_width = (_u64) std::atoi(param_list[0].c_str());
  _u64        cache_nlevels = (_u64) std::atoi(param_list[1].c_str());
  _u64        nthreads = (_u64) std::atoi(param_list[2].c_str());
  std::string stars(40, '*');
  std::cout << stars << "\nPQ -- n_chunks: " << n_chunks
            << ", chunk_size: " << chunk_size << ", data_dim: " << m_dimension
            << "\n";
  std::cout << "Search meta-params -- beam_width: " << beam_width
            << ", cache_nlevels: " << cache_nlevels
            << ", nthreads: " << nthreads << "\n"
            << stars << "\n";

  // create object

  // set create_visit_cache to true when creating _pFlashIndex
  _pFlashIndex = new NSG::PQFlashNSG<T>(true);

  _pFlashIndex->load(data_bin.c_str(), nsg_disk_opt.c_str(),
                     pq_tables_bin.c_str(), chunk_size, n_chunks, m_dimension,
                     nthreads, medoids_bin.c_str());

  // cache bfs levels
  _pFlashIndex->cache_bfs_levels(cache_nlevels);

  return 0;
}

// Search several vectors, return their neighbors' distance and ids.
// Both distances & ids are returned arraies of neighborCount elements,
// And need to be allocated by invoker, which capacity should be greater
// than [warmup_num * neighborCount].
template<typename T>
void search_index(NSG::PQFlashNSG<T>* _pFlashIndex, const char* vector,
                  uint64_t warmup_num, uint64_t neighborCount, float* distances,
                  uint64_t* ids, _u64 L) {
  //  _u64     L = 6 * neighborCount;
  //  _u64     L = 12;
  const T*  warmup_load = (const T*) vector;
#pragma omp parallel for schedule(dynamic, 1) num_threads(16)
  for (_s64 i = 0; i < warmup_num; i++)
    _pFlashIndex->cached_beam_search(
        warmup_load + (i * _pFlashIndex->aligned_dim), neighborCount, L,
        ids + (i * neighborCount), distances + (i * neighborCount), 6);
}

template<typename T>
int create_visited_cache(int argc, char** argv) {
  NSG::PQFlashNSG<T>* _pFlashIndex;

  // load warmup bin
  T*     warmup = nullptr;
  size_t warmup_num, ndims, warmup_aligned_dim;
  _u64   curL = 0;
  _u64   num_cache_nodes = 0;

  NSG::load_aligned_bin<T>(argv[3], warmup, warmup_num, ndims,
                           warmup_aligned_dim);
  curL = atoi(argv[4]);
  num_cache_nodes = atoi(argv[5]);

  ndims = warmup_aligned_dim;

  // for warmup search
  {
    // load the index
    bool res = load_index(argv[2], "8 2 16", _pFlashIndex);
    omp_set_num_threads(16);

    // ERROR CHECK
    if (res == 1) {
      std::cout << "Error detected loading the index" << std::endl;
      exit(-1);
    }

    int    recall_at = 5;
    _u64*  warmup_res = new _u64[recall_at * warmup_num];
    _u32*  warmup_res32 = new _u32[warmup_num * recall_at];
    float* warmup_dists = new float[recall_at * warmup_num];

    // execute queries
    search_index(_pFlashIndex, (const char*) warmup, warmup_num, recall_at,
                 warmup_dists, warmup_res, curL);

    std::cout << "Saving visit ctr file" << std::endl;
    std::string node_cache_path = std::string(argv[2]) + "_visit_ctr.bin";
    _pFlashIndex->save_cached_nodes(num_cache_nodes, node_cache_path);

    NSG::aligned_free(warmup);
    delete[] warmup_res;
    delete[] warmup_res32;
    delete[] warmup_dists;
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << "Usage: " << argv[0]
              << " <index_type[float/int8/uint8]>  <index_prefix_path>  "
                 "<warmup_bin> LS num_cached_nodes"
              << std::endl;
    exit(-1);
  }
  if (std::string(argv[1]) == std::string("float"))
    create_visited_cache<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    create_visited_cache<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    create_visited_cache<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8"
              << std::endl;
}
