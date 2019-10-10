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
      return false;
    }

    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path =
        index_prefix_path + "_compressed.bin";
    std::string randnsg_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";

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

    _pNsgIndex->build(paras);
    _pNsgIndex->save(randnsg_path.c_str());
    _pFlashIndex.reset(new PQFlashNSG<T>());
    _pFlashIndex->create_disk_layout(std::string(dataFilePath), randnsg_path,
                                     disk_index_path);

    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;

    std::cout << "Indexing time: " << diff.count() << "\n";

    return true;
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
      return false;
    }

    const std::string index_prefix_path(indexFilePath);

    // convert strs into params
    std::string data_bin = index_prefix_path + "_compressed.bin";
    std::string pq_tables_bin = index_prefix_path + "_pq_pivots.bin";
    std::string disk_index_file = index_prefix_path + "_disk.index";
    std::string medoids_file = index_prefix_path + "_medoids.bin";

    size_t data_dim, num_pq_centers;
    NSG::get_bin_metadata(pq_tables_bin, num_pq_centers, data_dim);
    this->m_dimension = (_u32) data_dim;
    this->aligned_dimension = ROUND_UP(this->m_dimension, 8);

    this->Lsearch = (_u64) std::atoi(param_list[0].c_str());
    this->beam_width = (_u64) std::atoi(param_list[1].c_str());
    int  cache_nlevels = (_u64) std::atoi(param_list[2].c_str());
    _u64 nthreads = (_u64) std::atoi(param_list[3].c_str());

    // create object
    _pFlashIndex.reset(new PQFlashNSG<T>());

    // load index
    _pFlashIndex->load(nthreads, pq_tables_bin.c_str(), data_bin.c_str(),
                       disk_index_file.c_str(), medoids_file.c_str());

    // cache bfs levels
    _pFlashIndex->cache_bfs_levels(cache_nlevels);
    //    free(params);  // Gopal. Caller has to free the 'params' variable.
    return true;
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
    const T* query = (const T*) vector;
    //#pragma omp  parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < queryCount; i++) {
      _pFlashIndex->cached_beam_search(
          query + (i * this->aligned_dimension), neighborCount, this->Lsearch,
          ids + (i * neighborCount), distances + (i * neighborCount),
          this->beam_width);
      //      std::cout << i << std::endl;
    }
  }

  extern "C" __declspec(dllexport) ANNIndex::IANNIndex* CreateObjectFloat(
      unsigned __int32 dimension, ANNIndex::DistanceType distanceType) {
    return new NSG::NSGInterface<float>(dimension, distanceType);
  }

  extern "C" __declspec(dllexport) void ReleaseObjectFloat(
      ANNIndex::IANNIndex* object) {
    NSG::NSGInterface<float>* subclass =
        dynamic_cast<NSG::NSGInterface<float>*>(object);
    if (subclass != nullptr) {
      delete subclass;
    }
  }
  
  template class NSGInterface<int8_t>;
  template class NSGInterface<float>;
  template class NSGInterface<uint8_t>;

}  // namespace NSG


