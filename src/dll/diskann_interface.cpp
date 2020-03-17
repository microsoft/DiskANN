#include <chrono>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "omp.h"
#include "mkl.h"

#include "aux_utils.h"
#include "dll/diskann_interface.h"
#include "index.h"
#include "partition_and_pq.h"
#include "utils.h"

namespace diskann {

  template<typename T>
  __cdecl DiskANNInterface<T>::DiskANNInterface(
    unsigned __int32 dimension, ANNIndex::DistanceType distanceType)
    : ANNIndex::IANNIndex(dimension, distanceType), _pNsgIndex(nullptr) {
    if (distanceType == ANNIndex::DT_L2) {
      _compareMetric = diskann::Metric::L2;
    } else if (distanceType == ANNIndex::DT_InnerProduct) {
      _compareMetric = diskann::Metric::INNER_PRODUCT;
    } else {
      throw std::exception("Only DT_L2 and DT_InnerProduct are supported.");
    }

  }  // namespace diskann

  template<typename T>
  DiskANNInterface<T>::~DiskANNInterface<T>() {
  }

  template<typename T>
  // In implementation, the file path can be a file or folder.
  bool DiskANNInterface<T>::BuildIndex(const char* dataFilePath,
                                       const char* indexFilePath,
                                       const char* indexBuildParameters) {
    return diskann::build_disk_index<T>(dataFilePath, indexFilePath,
                                        indexBuildParameters, _compareMetric);
  }

  template<typename T>
  // Load index form file.
  bool DiskANNInterface<T>::LoadIndex(const char* indexFilePath,
                                      const char* queryParameters) {
    std::cout << "Loading index " << indexFilePath << " for search"
              << std::endl;
    std::stringstream parser;
    parser << std::string(queryParameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    std::cout << "Pushed " << param_list.size() << " parameters" << std::endl;

    if (param_list.size() != 2) {
      std::cerr << "Correct usage of parameters is \n"
                   "Lsearch[1] nthreads[2]"
                << std::endl;
      return false;
    }

    const std::string index_prefix_path(indexFilePath);

    // convert strs into params
    std::string data_bin = index_prefix_path + "_compressed.bin";
    std::string pq_tables_bin = index_prefix_path + "_pq_pivots.bin";
    std::string disk_index_file = index_prefix_path + "_disk.index";
    std::string medoids_file = index_prefix_path + "_medoids.bin";
    std::string cache_list_file = index_prefix_path + "_cache_list.bin";
    std::string sample_data_file = index_prefix_path + "_sample_data.bin";

    size_t data_dim, num_pq_centers;
    diskann::get_bin_metadata(pq_tables_bin, num_pq_centers, data_dim);
    this->m_dimension = (_u32) data_dim;
    this->m_aligned_dimension = ROUND_UP(this->m_dimension, 8);

    this->Lsearch = (_u64) std::atoi(param_list[0].c_str());
    uint64_t num_cache_nodes = NUM_NODES_TO_CACHE;
    auto     nthreads = (_u32) std::atoi(param_list[1].c_str());

    try {
      _pFlashIndex.reset(new PQFlashIndex<T>());
      _pFlashIndex->load(nthreads, pq_tables_bin.c_str(), data_bin.c_str(),
                         disk_index_file.c_str());
      _pFlashIndex->load_entry_points(medoids_file, "");
      _pFlashIndex->cache_medoid_nhoods();
      std::vector<uint32_t> node_list;
      // cache bfs levels
      _pFlashIndex->cache_bfs_levels(num_cache_nodes, node_list);
      _pFlashIndex->load_cache_list(node_list);
      node_list.clear();
      node_list.shrink_to_fit();

      uint64_t tuning_sample_num = 0, tuning_sample_dim = 0,
               tuning_sample_aligned_dim = 0;
      T* tuning_sample =
        diskann::load_warmup<T>(sample_data_file, tuning_sample_num,
                                this->m_dimension, this->m_aligned_dimension);
      this->beam_width = diskann::optimize_beamwidth<T>(
        _pFlashIndex, tuning_sample, tuning_sample_num,
        tuning_sample_aligned_dim, (uint32_t) this->Lsearch);
      std::cout << "Loaded DiskANN index with L: " << this->Lsearch
                << " (calculated) beam width: " << this->beam_width
                << " nthreads: " << nthreads << std::endl;

      return true;
    } catch (const diskann::ANNException& ex) {
      std::cerr << ex.message();
      return false;
    }
  }

  // Search several vectors, return their neighbors' distance and ids.
  // Both distances & ids are returned arraies of neighborCount elements,
  // And need to be allocated by invoker, which capacity should be greater
  // than [queryCount * neighborCount].
  template<typename T>
  void DiskANNInterface<T>::SearchIndex(const char*       vector,
                                        unsigned __int64  queryCount,
                                        unsigned __int64  neighborCount,
                                        float*            distances,
                                        unsigned __int64* ids) const {
    try {
      const T* query = (const T*) vector;
      for (_u64 i = 0; i < queryCount; i++) {
        _pFlashIndex->cached_beam_search(
          query + (i * this->m_dimension), neighborCount, this->Lsearch,
          ids + (i * neighborCount), distances + (i * neighborCount),
          this->beam_width);
      }
    } catch (const diskann::ANNException& ex) {
      std::cerr << ex.message();
    }
  }

  extern "C" __declspec(dllexport) ANNIndex::IANNIndex* CreateObjectFloat(
    unsigned __int32 dimension, ANNIndex::DistanceType distanceType) {
    return new diskann::DiskANNInterface<float>(dimension, distanceType);
  }

  extern "C" __declspec(dllexport) void ReleaseObjectFloat(
    ANNIndex::IANNIndex* object) {
    diskann::DiskANNInterface<float>* subclass =
      dynamic_cast<diskann::DiskANNInterface<float>*>(object);
    if (subclass != nullptr) {
      delete subclass;
    }
  }

  extern "C" __declspec(dllexport) ANNIndex::IANNIndex* CreateObjectByte(
    unsigned __int32 dimension, ANNIndex::DistanceType distanceType) {
    return new diskann::DiskANNInterface<int8_t>(dimension, distanceType);
  }

  extern "C" __declspec(dllexport) void ReleaseObjectByte(
    ANNIndex::IANNIndex* object) {
    diskann::DiskANNInterface<int8_t>* subclass =
      dynamic_cast<diskann::DiskANNInterface<int8_t>*>(object);
    if (subclass != nullptr) {
      delete subclass;
    }
  }

  template class DiskANNInterface<int8_t>;
  template class DiskANNInterface<float>;
  template class DiskANNInterface<uint8_t>;

}  // namespace diskann
