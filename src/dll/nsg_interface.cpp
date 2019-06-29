#include <chrono>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dll/nsg_interface.h"
#include "index_nsg.h"
#include "util.h"
#include "partitionAndPQ.h"

namespace NSG {

#define TRAINING_SET_SIZE 2000000
  template<typename T>
  __cdecl NSGInterface<T>::NSGInterface(unsigned __int32       dimension,
                                        ANNIndex::DistanceType distanceType)
      : ANNIndex::IANNIndex(dimension, distanceType), _pNsgIndex(nullptr) {
    if (distanceType == ANNIndex::DT_L2) {
      _compareMetric = NSG::Metric::L2;
    } else if (distanceType == ANNIndex::DT_InnerProduct) {
      _compareMetric = NSG::Metric::INNER_PRODUCT;
    } else {
      throw std::exception("Only DT_L2 and DT_InnerProduct are supported.");
    }

  }  // namespace NSG

  template<typename T>
  NSGInterface<T>::~NSGInterface<T>() {
  }

  template<typename T>
  // In implementation, the file path can be a file or folder.
  bool NSGInterface<T>::BuildIndex(const char* dataFilePath,
                                   const char* indexFilePath,
                                   const char* indexBuildParameters) {
    std::stringstream parser;
    parser << std::string(indexBuildParameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() != 4) {
      std::cout
          << "Correct usage of parameters is L (indexing search list size) "
             "R (max degree) C (visited list maximum size) B (approximate "
             "compressed number of bytes per datapoint to store in "
             "memory) "
          << std::endl;
      return 1;
    }

    std::string index_prefix_path(indexFilePath);
    std::string index_params_path = index_prefix_path + "_params.bin";
    std::string train_file_path = index_prefix_path + "_training_set_float.bin";
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path =
        index_prefix_path + "_compressed_uint32.bin";
    std::string randnsg_path = index_prefix_path + "_unopt.rnsg";
    std::string diskopt_path = index_prefix_path + "_diskopt.rnsg";

    unsigned L = (unsigned) atoi(param_list[0].c_str());
    unsigned R = (unsigned) atoi(param_list[1].c_str());
    unsigned C = (unsigned) atoi(param_list[2].c_str());
    size_t   num_pq_chunks = (size_t) atoi(param_list[3].c_str());

    T* data_load = NULL;

    size_t points_num, dim;

    NSG::load_bin<T>(dataFilePath, data_load, points_num, dim);
    data_load = NSG::data_align(data_load, points_num, dim);
    std::cout << "Data loaded and aligned" << std::endl;

    uint32_t* params_array = new uint32_t[5];
    params_array[0] = (uint32_t) L;
    params_array[1] = (uint32_t) R;
    params_array[2] = (uint32_t) C;
    params_array[3] = (uint32_t) dim;
    params_array[4] = (uint32_t) num_pq_chunks;
    NSG::save_bin<uint32_t>(index_params_path.c_str(), params_array, 5, 1);
    std::cout << "Saving params to " << index_params_path << "\n";

    auto s = std::chrono::high_resolution_clock::now();

    if (points_num > 2 * TRAINING_SET_SIZE) {
      gen_random_slice(data_load, points_num, dim, train_file_path.c_str(),
                       (size_t) TRAINING_SET_SIZE);
    } else {
      float* float_data = new float[points_num * dim];
      for (size_t i = 0; i < points_num; i++) {
        for (size_t j = 0; j < dim; j++) {
          float_data[i * dim + j] = data_load[i * dim + j];
        }
      }

      NSG::save_bin<float>(train_file_path.c_str(), float_data, points_num,
                           dim);
      delete[] float_data;
    }

    //  unsigned    nn_graph_deg = (unsigned) atoi(argv[3]);

    generate_pq_pivots<T>(train_file_path, 256, num_pq_chunks, 15,
                          pq_pivots_path);
    generate_pq_data_from_pivots<T>(data_load, points_num, dim, 256,
                                    num_pq_chunks, pq_pivots_path,
                                    pq_compressed_vectors_path);

    NSG::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<float>("alpha", 1.2f);
    paras.Set<unsigned>("num_rnds", 2);
    paras.Set<std::string>("save_path", randnsg_path);

    _pNsgIndex = std::unique_ptr<NSG::IndexNSG<T>>(
        new NSG::IndexNSG<T>(dim, points_num, _compareMetric, nullptr));
    _pNsgIndex->BuildRandomHierarchical(data_load, paras);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "Indexing time: " << diff.count() << "\n";

    _pNsgIndex->Save(randnsg_path.c_str());

    _pNsgIndex->save_disk_opt_graph(diskopt_path.c_str());

    return 0;
  }

  template<typename T>
  // Load index form file.
  bool NSGInterface<T>::LoadIndex(const char* indexFilePath,
                                  const char* queryParameters) {
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
    load_bin<uint32_t>(params_path.c_str(), params, nargs, one);

    // infer chunk_size
    this->m_dimension = (_u64) params[3];
    this->n_chunks = (_u64) params[4];
    this->chunk_size = (_u64)(this->m_dimension/ n_chunks);

    std::string nsg_disk_opt = index_prefix_path + "_diskopt.rnsg";

    this->beam_width = (_u64) std::atoi(param_list[0].c_str());
    _u64        cache_nlevels = (_u64) std::atoi(param_list[1].c_str());
    _u64        nthreads = (_u64) std::atoi(param_list[2].c_str());
    std::string stars(40, '*');
    std::cout << stars << "\nPQ -- n_chunks: " << n_chunks
              << ", chunk_size: " << chunk_size << ", data_dim: " << this->m_dimension
              << "\n";
    std::cout << "Search meta-params -- beam_width: " << beam_width
              << ", cache_nlevels: " << cache_nlevels
              << ", nthreads: " << nthreads << "\n"
              << stars << "\n";

    // create object
    _pFlashIndex.reset(new PQFlashNSG<T>());

    // load index
    _pFlashIndex->load(data_bin.c_str(), nsg_disk_opt.c_str(),
                       pq_tables_bin.c_str(), chunk_size, n_chunks, this->m_dimension,
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
  void NSGInterface<T>::SearchIndex(const char*       vector,
                                    unsigned __int64  queryCount,
                                    unsigned __int64  neighborCount,
                                    float*            distances,
                                    unsigned __int64* ids) const {
    _u64      L = 6 * neighborCount;
    const T*  query_load = (const T*) vector;
// #pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < queryCount; i++) {
      _pFlashIndex->cached_beam_search(
          query_load + (i * m_dimension), neighborCount, L,
          ids + (i * neighborCount), distances + (i * neighborCount),
          this->beam_width);
    }
  }

  template class NSGInterface<int8_t>;
  template class NSGInterface<float>;
  template class NSGInterface<uint8_t>;

}  // namespace NSG
