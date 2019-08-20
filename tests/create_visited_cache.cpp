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
bool load_index(const char* indexFilePath, const char* queryParameters,
                NSG::PQFlashNSG<T>*& _pFlashIndex) {
  std::stringstream parser;
  parser << std::string(queryParameters);
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
  std::string data_bin = index_prefix_path + "_compressed_uint32.bin";
  std::string pq_tables_bin = index_prefix_path + "_pq_pivots.bin";

  // determine nchunks
  std::string params_path = index_prefix_path + "_params.bin";
  std::string node_visit_bin = index_prefix_path + "_visit_ctr.bin";

  uint32_t* params;
  size_t    nargs, one;
  NSG::load_bin<uint32_t>(params_path.c_str(), params, nargs, one);

  // infer chunk_size
  _u64 m_dimension = (_u64) params[3];
  _u64 n_chunks = (_u64) params[4];
  _u64 chunk_size = (_u64)(m_dimension / n_chunks);
  // corrected n_chnks in case it is dimension is not divisible by original
  // num_chunks
  n_chunks = DIV_ROUND_UP(m_dimension, chunk_size);

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
                     nthreads);

  // cache bfs levels
  _pFlashIndex->cache_bfs_levels(cache_nlevels);

  return 0;
}

// Search several vectors, return their neighbors' distance and ids.
// Both distances & ids are returned arraies of neighborCount elements,
// And need to be allocated by invoker, which capacity should be greater
// than [query_num * neighborCount].
template<typename T>
std::tuple<float, float, float> search_index(
    NSG::PQFlashNSG<T>* _pFlashIndex, const char* vector, uint64_t query_num,
    uint64_t neighborCount, float* distances, uint64_t* ids, _u64 L) {
  //  _u64     L = 6 * neighborCount;
  //  _u64     L = 12;
  const T*         query_load = (const T*) vector;
  NSG::QueryStats* stats = new NSG::QueryStats[query_num];
  NSG::Timer       timer;
#pragma omp        parallel for schedule(dynamic, 1) num_threads(16)
  for (_u64 i = 0; i < query_num; i++)
    _pFlashIndex->cached_beam_search(
        query_load + (i * _pFlashIndex->aligned_dim), neighborCount, L,
        ids + (i * neighborCount), distances + (i * neighborCount), 6,
        stats + i);

  float mean_latency = NSG::get_percentile_stats(
      stats, query_num, 0.5,
      [](const NSG::QueryStats& stats) { return stats.total_us; });

  float latency_99 = NSG::get_percentile_stats(
      stats, query_num, 0.99,
      [](const NSG::QueryStats& stats) { return stats.total_us; });

  float mean_io = NSG::get_percentile_stats(
      stats, query_num, 0.5,
      [](const NSG::QueryStats& stats) { return stats.n_ios; });

  delete[] stats;
  return std::make_tuple(mean_latency, latency_99, mean_io);
}

template<typename T>
int aux_main(int argc, char** argv) {
  NSG::PQFlashNSG<T>* _pFlashIndex;

  // load query bin
  T*     query = nullptr;
  size_t query_num, ndims;
  _u64   curL = 0;
  _u64   num_cache_nodes = 0;

  NSG::load_bin<T>(argv[3], query, query_num, ndims);
  curL = atoi(argv[4]);
  num_cache_nodes = atoi(argv[5]);

  query = NSG::data_align<T>(query, query_num, ndims);
  ndims = ROUND_UP(ndims, 8);

  // for query search
  {
    // load the index
    bool res = load_index(argv[2], "8 3 16", _pFlashIndex);
    omp_set_num_threads(16);

    // ERROR CHECK
    if (res == 1) {
      std::cout << "Error detected loading the index" << std::endl;
      exit(-1);
    }

    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);

    std::cout << std::setw(8) << "Ls" << std::setw(16) << "Avg Latency"
              << std::setw(16) << "99 Latency" << std::setw(16)
              << "Avg Disk I/Os" << std::endl;
    std::cout << "============================================================="
                 "============"
                 "======="
              << std::endl;

    int    recall_at = 5;
    _u64*  query_res = new _u64[recall_at * query_num];
    _u32*  query_res32 = new _u32[query_num * recall_at];
    float* query_dists = new float[recall_at * query_num];

    // execute queries
    std::tuple<float, float, float> q_stats;
    q_stats = search_index(_pFlashIndex, (const char*) query, query_num,
                           recall_at, query_dists, query_res, curL);

    std::cout << "Saving visit ctr file" << std::endl;
    std::string node_cache_path = std::string(argv[2]) + "_visit_ctr.bin";
    _pFlashIndex->save_cached_nodes(num_cache_nodes, node_cache_path);

    NSG::aligned_free(query);
    delete[] query_res;
    delete[] query_res32;
    delete[] query_dists;
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << "Usage: " << argv[0]
              << " <index_type[float/int8/uint8]>  <index_prefix_path>  "
                 "<query_bin> LS num_cached_nodes"
              << std::endl;
    exit(-1);
  }
  if (std::string(argv[1]) == std::string("float"))
    aux_main<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    aux_main<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    aux_main<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8"
              << std::endl;
}
