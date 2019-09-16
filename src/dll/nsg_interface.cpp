#include <chrono>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dll/nsg_interface.h"
#include "index_nsg.h"
#include "partition_and_pq.h"
#include "utils.h"

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

    if (param_list.size() != 5) {
      std::cout
          << "Correct usage of parameters is L (indexing search list size) "
             "R (max degree) C (visited list maximum size) B (approximate "
             "compressed number of bytes per datapoint to store in "
             "memory) Training-Set-Sampling-Rate-For-PQ-Generation"
          << std::endl;
      return 1;
    }

    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path =
        index_prefix_path + "_compressed.bin";
    std::string randnsg_path = index_prefix_path + "_unopt.rnsg";
    std::string diskopt_path = index_prefix_path + "_diskopt.rnsg";

    unsigned L = (unsigned) atoi(param_list[0].c_str());
    unsigned R = (unsigned) atoi(param_list[1].c_str());
    unsigned C = (unsigned) atoi(param_list[2].c_str());
    size_t   num_pq_chunks = (size_t) atoi(param_list[3].c_str());
    float    training_set_sampling_rate = atof(param_list[4].c_str());

    auto s = std::chrono::high_resolution_clock::now();

    float* train_data;
    size_t train_size, train_dim;

    // generates random sample and sets it to train_data and updates train_size
    gen_random_slice<T>(dataFilePath, training_set_sampling_rate, train_data,
                        train_size, train_dim);

    std::cout << "Training loaded of size " << train_size << std::endl;

    //  unsigned    nn_graph_deg = (unsigned) atoi(argv[3]);

    generate_pq_pivots(train_data, train_size, train_dim, 256, num_pq_chunks,
                       15, pq_pivots_path);
    generate_pq_data_from_pivots<T>(dataFilePath, 256, num_pq_chunks,
                                    pq_pivots_path, pq_compressed_vectors_path);

    delete[] train_data;

    NSG::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<float>("alpha", 1.2f);
    paras.Set<unsigned>("num_rnds", 2);
    paras.Set<std::string>("save_path", randnsg_path);

    _pNsgIndex = std::unique_ptr<NSG::IndexNSG<T>>(
        new NSG::IndexNSG<T>(_compareMetric, dataFilePath));

    if (file_exists(randnsg_path.c_str())) {
      _pNsgIndex->load(randnsg_path.c_str());
    } else {
      _pNsgIndex->build(paras);
      _pNsgIndex->save(randnsg_path.c_str());
    }

    _pNsgIndex->save_disk_opt_graph(diskopt_path.c_str());

    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "Indexing time: " << diff.count() << "\n";

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

    if (param_list.size() != 4) {
      std::cerr << "Correct usage of parameters is \n"
                   "Lsearch[1] BeamWidth[2] cache_nlevels[3] nthreads[4]"
                << std::endl;
      return 1;
    }

    const std::string index_prefix_path(indexFilePath);

    // convert strs into params
    std::string data_bin = index_prefix_path + "_compressed.bin";
    std::string pq_tables_bin = index_prefix_path + "_pq_pivots.bin";
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

    _u64*  node_visit_list = NULL;
    size_t one = 0, num_cache_nodes = 0;

    std::string nsg_disk_opt = index_prefix_path + "_diskopt.rnsg";
    std::string medoids_file = index_prefix_path + "_medoids.bin";

    this->Lsearch = (_u64) std::atoi(param_list[0].c_str());
    this->beam_width = (_u64) std::atoi(param_list[1].c_str());
    int         cache_nlevels = (_u64) std::atoi(param_list[2].c_str());
    _u64        nthreads = (_u64) std::atoi(param_list[3].c_str());

    if (cache_nlevels == -1 && file_exists(node_visit_bin))
      NSG::load_bin<_u64>(node_visit_bin.c_str(), node_visit_list,
                          num_cache_nodes, one);
    else if (!file_exists(node_visit_bin))
      cache_nlevels = 2;

    std::string stars(40, '*');
    std::cout << stars << "\nPQ -- n_chunks: " << this->n_chunks
              << ", chunk_size: " << this->chunk_size
              << ", data_dim: " << this->m_dimension << "\n";
    std::cout << "Search meta-params -- beam_width: " << this->beam_width
              << ", cache_nlevels: " << cache_nlevels
              << ", nthreads: " << nthreads << "\n"
              << stars << "\n";

    // create object
    _pFlashIndex.reset(new PQFlashNSG<T>());

    // load index
    _pFlashIndex->load(data_bin.c_str(), nsg_disk_opt.c_str(),
                       pq_tables_bin.c_str(), this->chunk_size, this->n_chunks,
                       this->m_dimension, nthreads, medoids_file.c_str());

    // cache bfs levels
    if (cache_nlevels > 0)
      _pFlashIndex->cache_bfs_levels(cache_nlevels);
    else
      _pFlashIndex->cache_visited_nodes(node_visit_list, num_cache_nodes);

    //    free(params);  // Gopal. Caller has to free the 'params' variable.
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
    //    _u64      L = 6 * neighborCount;

    _u64 aligned_data_dim = ROUND_UP(this->m_dimension, 8);

    const T* query_load = (const T*) vector;
#pragma omp  parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < queryCount; i++) {
      _pFlashIndex->cached_beam_search(
          query_load + (i * aligned_data_dim), neighborCount, this->Lsearch,
          ids + (i * neighborCount), distances + (i * neighborCount),
          this->beam_width);
    }
  }

  template class NSGInterface<int8_t>;
  template class NSGInterface<float>;
  template class NSGInterface<uint8_t>;

}  // namespace NSG
