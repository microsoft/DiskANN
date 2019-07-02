//#include <distances.h>
//#include <indexing.h>
#include <efanna2e/index_nsg.h>
#include <efanna2e/pq_flash_index_nsg.h>
#include <efanna2e/util.h>
#include <math_utils.h>
#include <partitionAndPQ.h>
#include "utils.h"

// #define TRAINING_SET_SIZE 2000000
#define TRAINING_SET_SIZE 2000000

template<typename T>
bool LoadIndex(const char* indexFilePath, const char* queryParameters,
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
  uint32_t*   params;
  size_t      nargs, one;
  NSG::load_bin<uint32_t>(params_path.c_str(), params, nargs, one);

  // infer chunk_size
  _u64 m_dimension = (_u64) params[3];
  _u64 n_chunks = (_u64) params[4];
  _u64 chunk_size = (_u64)(m_dimension / n_chunks);

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

  _pFlashIndex = new NSG::PQFlashNSG<T>();
//  _pFlashIndex->reset(new NSG::PQFlashNSG<T>());

  // load index
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
// than [queryCount * neighborCount].
template<typename T>
void SearchIndex(NSG::PQFlashNSG<T>* _pFlashIndex, const char* vector,
                 uint64_t queryCount, uint64_t neighborCount,
                 float* distances, uint64_t* ids) {
  _u64     L = 12;
  const T* query_load = (const T*) vector;
  // #pragma omp parallel for schedule(dynamic, 1)
  for (_s64 i = 0; i < queryCount; i++) {
    _pFlashIndex->cached_beam_search(
        query_load + (i * _pFlashIndex->data_dim), neighborCount, L,
        ids + (i * neighborCount), distances + (i * neighborCount), 5);
  }
}

template<typename T>
int aux_main(int argc, char** argv) {
  // argv[1]: data file
  // argv[2]: output_file_pattern

  //  ANNIndex::IANNIndex* intf = new NSG::NSGInterface<T>(0, ANNIndex::DT_L2);
  NSG::PQFlashNSG<T>* _pFlashIndex;
  // for query search
  {
    // load the index
    bool res = LoadIndex(argv[1], "4 4 16", _pFlashIndex);
    // ERROR CHECK
    if (res == 1) {
      std::cout << "Error detected loading the index" << std::endl;
      exit(-1);
    }

    // load query fvecs
    T*       query = nullptr;
    size_t nqueries, ndims;
//    NSG::aligned_load_Tvecs<T>(argv[3], query, nqueries, ndims);
    NSG::load_bin<T> (argv[3], query, nqueries, ndims);
    query = NSG::data_align(query, nqueries, ndims);
    ndims = ROUND_UP(ndims, 8);

    // query params/output
    _u64   k = 5, L = 30;
    _u64*  query_res = new _u64[k * nqueries];
    float* query_dists = new float[k * nqueries];

    // execute queries
    SearchIndex(_pFlashIndex, (const char*) query, nqueries, k,
                      query_dists, query_res);

    // compute recall
    write_Tvecs_unsigned(argv[4], query_res, nqueries, k);

    NSG::aligned_free(query);
    delete[] query_res;
    delete[] query_dists;
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cout << "Usage: " << argv[0]
              << " <index_file_prefix> [bin] <index_type> [float/int8/uint8] "
                 "<query_Tvecs> <query_res>"
              << std::endl;
    exit(-1);
  }
  if (std::string(argv[2]) == std::string("float"))
    aux_main<float>(argc, argv);
  else if (std::string(argv[2]) == std::string("int8"))
    aux_main<int8_t>(argc, argv);
  else if (std::string(argv[2]) == std::string("uint8"))
    aux_main<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8"
              << std::endl;
}
