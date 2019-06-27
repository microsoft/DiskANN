#include <chrono>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dll/nsg_interface.h"
#include "efanna2e/index_nsg.h"
#include "efanna2e/util.h"
#include "partitionAndPQ.h"
#include "utils.h"

namespace NSG {

#define TRAINING_SET_SIZE 2000000
 template <typename T>
  __cdecl NSGInterface<T>::NSGInterface(unsigned __int32       dimension,
                             ANNIndex::DistanceType distanceType)
      : ANNIndex::IANNIndex(dimension, distanceType),
        _pNsgIndex(nullptr) 
  {
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

  template <typename T>
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

    T* data_load = NULL;
    size_t points_num, dim;

    load_file_into_data<T>(dataFilePath, data_load, points_num, dim);
    data_load = NSG::data_align(data_load, points_num, dim);
    std::cout << "Data loaded and aligned" << std::endl;
    std::string train_file;

    auto        s = std::chrono::high_resolution_clock::now();
    std::string working_file_prefix(indexFilePath);

    if (points_num > 2 * TRAINING_SET_SIZE) {
      train_file = working_file_prefix + "_train.fvecs";
      if (!file_exists(train_file)) {
        gen_random_slice(data_load, points_num, dim, train_file.c_str(),
                         (size_t) TRAINING_SET_SIZE);
      } else
        std::cout << "Train file exists. Using it" << std::endl;
    } else {
      train_file = std::string(dataFilePath);
    }

    //  unsigned    nn_graph_deg = (unsigned) atoi(argv[3]);
    unsigned L = (unsigned) atoi(param_list[0].c_str());
    unsigned R = (unsigned) atoi(param_list[1].c_str());
    unsigned C = (unsigned) atoi(param_list[2].c_str());
    size_t   num_chunks = (size_t) atoi(param_list[3].c_str());

    generate_pq_pivots(train_file.c_str(), 256, num_chunks, 15,
                       working_file_prefix.c_str());

    generate_pq_data_from_pivots(train_file.c_str(), 256, num_chunks,
                                 working_file_prefix.c_str());

    NSG::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<float>("alpha", 1.2f);
    paras.Set<unsigned>("num_rnds", 2);
    paras.Set<std::string>("save_path",
                           std::string(working_file_prefix + "_degree-" +
                                       std::to_string(R) + ".rnsg"));

    _pNsgIndex = std::unique_ptr<NSG::IndexNSG<T> >(new NSG::IndexNSG<T>(dim, points_num, _compareMetric, nullptr));
    _pNsgIndex->BuildRandomHierarchical(data_load, paras);
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "Indexing time: " << diff.count() << "\n";
    _pNsgIndex->Save(std::string(working_file_prefix + "_degree-" +
                           std::to_string(R) + ".rnsg")
                   .c_str());
    return 0;
  }

  template <typename T>
  // Load index form file.
  bool NSGInterface<T>::LoadIndex(const char* indexFilePath,
                               const char* queryParameters) 
  {
    return 0;
  }

  // Search several vectors, return their neighbors' distance and ids.
  // Both distances & ids are returned arraies of neighborCount elements,
  // And need to be allocated by invoker, which capicity should be greater
  // than queryCount * neighborCount.
  template<typename T>
  void NSGInterface<T>::SearchIndex(const char*       vector,
                                 unsigned __int64  queryCount,
                                 unsigned __int64  neighborCount,
                                 float*            distances,
                                 unsigned __int64* ids) const 
  {
  }

  template class NSGInterface<int8_t>;
  template class NSGInterface<float>;
  template class NSGInterface<uint8_t>;


}  // namespace NSG