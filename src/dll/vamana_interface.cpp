#include <chrono>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dll/vamana_interface.h"
#include "index.h"
#include "utils.h"

namespace diskann {

#define TRAINING_SET_SIZE 3000000
  template<typename T>
  __cdecl VamanaInterface<T>::VamanaInterface(
      unsigned __int32 dimension, ANNIndex::DistanceType distanceType)
      : ANNIndex::IANNIndex(dimension, distanceType), _pIndex(nullptr) {
    if (distanceType == ANNIndex::DT_L2) {
      _compareMetric = diskann::Metric::L2;
    } else if (distanceType == ANNIndex::DT_InnerProduct) {
      _compareMetric = diskann::Metric::INNER_PRODUCT;
    } else {
      throw std::exception("Only DT_L2 and DT_InnerProduct are supported.");
    }

  }  // namespace diskann

  template<typename T>
  VamanaInterface<T>::~VamanaInterface<T>() {
  }

  template<typename T>
  // In implementation, the file path can be a file or folder.
  bool VamanaInterface<T>::BuildIndex(const char* dataFilePath,
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
             "R (max degree) C (visited list maximum size) T (Max Threads To "
             "Use) D (data_dimension)"
          << std::endl;
      return false;
    }

    std::string index_prefix_path(indexFilePath);
    std::string mem_index_file = index_prefix_path + "_mem.index";

    unsigned L = (unsigned) atoi(param_list[0].c_str());
    unsigned R = (unsigned) atoi(param_list[1].c_str());
    unsigned C = (unsigned) atoi(param_list[2].c_str());
    unsigned num_threads = (unsigned) atoi(param_list[3].c_str());
    unsigned data_dim = (unsigned) atoi(param_list[4].c_str());
    auto     s = std::chrono::high_resolution_clock::now();

    this->m_dimension = data_dim;
    this->aligned_dimension = ROUND_UP(data_dim, 8);

    diskann::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<float>("alpha", 3.0);
    paras.Set<unsigned>("num_rnds", 2);
    paras.Set<unsigned>("num_threads", num_threads);

    _pIndex = std::unique_ptr<diskann::Index<T>>(
        new diskann::Index<T>(_compareMetric, dataFilePath));

    _pIndex->build(paras);
    _pIndex->save(mem_index_file.c_str());

    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;

    std::cout << "Indexing time: " << diff.count() << "\n";

    return true;
  }

  template<typename T>
  // Load index form file.
  bool VamanaInterface<T>::LoadIndex(const char* indexFilePath,
                                     const char* queryParameters) {
    std::stringstream parser;
    parser << std::string(queryParameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() != 3) {
      std::cerr << "Correct usage of parameters is \n"
                   "Lsearch[1] Data_File_Path[2] nthreads[3]"
                << std::endl;
      return false;
    }

    const std::string index_prefix_path(indexFilePath);

    // convert strs into params
    this->Lsearch = (_u64) std::atoi(param_list[0].c_str());
    std::string data_bin = param_list[1];
    _u64        nthreads = (_u64) std::atoi(param_list[2].c_str());

    std::string mem_index_file = index_prefix_path + "_mem.index";

    //    size_t data_dim, num_pq_centers;
    //    diskann::get_bin_metadata(pq_tables_bin, num_pq_centers, data_dim);
    //    this->m_dimension = (_u32) data_dim;
    //    this->aligned_dimension = ROUND_UP(this->m_dimension, 8);

    // create object
    _pIndex.reset(new Index<T>(_compareMetric, data_bin.c_str()));

    // load index
    _pIndex->load(mem_index_file.c_str());

    return true;
  }

  // Search several vectors, return their neighbors' distance and ids.
  // Both distances & ids are returned arraies of neighborCount elements,
  // And need to be allocated by invoker, which capacity should be greater
  // than [queryCount * neighborCount].
  // QUERIES HAVE TO BE ALIGNED TO DIMENSION EQUAL TO MULTIPLE OF 8, PADDED WITH
  // ZEROS IF NEEDED
  template<typename T>
  void VamanaInterface<T>::SearchIndex(const char*       vector,
                                       unsigned __int64  queryCount,
                                       unsigned __int64  neighborCount,
                                       float*            distances,
                                       unsigned __int64* ids) const {
    const T*              query = (const T*) vector;
    std::vector<unsigned> start_points(0);
    //#pragma omp  parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < queryCount; i++) {
      _pNsgindex->beam_search(query + i * this->aligned_dim, neighborCount,
                              this->Lsearch, 1, start_points,
                              ids + (i * neighborCount),
                              distances + (i * neighborCount));
    }
  }

  extern "C" __declspec(dllexport) ANNIndex::IANNIndex* CreateObjectFloat(
      unsigned __int32 dimension, ANNIndex::DistanceType distanceType) {
    return new diskann::VamanaInterface<float>(dimension, distanceType);
  }

  extern "C" __declspec(dllexport) void ReleaseObjectFloat(
      ANNIndex::IANNIndex* object) {
    diskann::VamanaInterface<float>* subclass =
        dynamic_cast<diskann::VamanaInterface<float>*>(object);
    if (subclass != nullptr) {
      delete subclass;
    }
  }

  template class VamanaInterface<int8_t>;
  template class VamanaInterface<float>;
  template class VamanaInterface<uint8_t>;

}  // namespace diskann
