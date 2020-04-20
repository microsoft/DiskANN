#include <chrono>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "omp.h"
#include "mkl.h"

#include "logger.h"
#include "aux_utils.h"
#include "dll/diskann_interface.h"
#include "dll/bing_aligned_file_reader.h"
#include "index.h"
#include "partition_and_pq.h"
#include "utils.h"

namespace diskann {

  template<typename T>
  DiskANNInterface<T>::DiskANNInterface(
      unsigned __int32 dimension, ANNIndex::DistanceType distanceType,
      std::shared_ptr<ANNIndex::IDiskPriorityIO> diskIO)
      : ANNIndex::IANNIndex(dimension, distanceType, diskIO),
        _pNsgIndex(nullptr) {
    if (distanceType == ANNIndex::DT_L2) {
      _compareMetric = diskann::Metric::L2;
    } else {
      throw std::exception("Only DT_L2 is supported.");
    }

#ifdef EXEC_ENV_OLS
    if (!diskIO) {
      throw std::exception(
          "Disk IO cannot be null when invoked from ANN search.");
    }
    _pDiskIO = diskIO;
#endif
  }

  template<typename T>
  DiskANNInterface<T>::~DiskANNInterface<T>() {
  }

  template<typename T>
  // In implementation, the file path can be a file or folder.
  bool DiskANNInterface<T>::BuildIndex(const char* dataFilePath,
                                       const char* indexFilePath,
                                       const char* indexBuildParameters) {
    if (diskann::build_disk_index<T>(dataFilePath, indexFilePath,
                                     indexBuildParameters, _compareMetric)) {
      return writeSharedStoreIniFile(indexFilePath);
    } else {
      return false;
    }
  }

  template<typename T>
  // Load index form file.
  bool DiskANNInterface<T>::LoadIndex(const char* indexFilePath,
                                      const char* queryParameters) {
    diskann::cout << "Loading index " << indexFilePath << " for search"
              << std::endl;
    std::stringstream parser;
    parser << std::string(queryParameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    diskann::cout << "Pushed " << param_list.size() << " parameters"
                  << std::endl;

    if (param_list.size() != 3) {
      std::cerr << "Correct usage of parameters is \n"
                   "Lsearch[1] nthreads[2] beamwidth[3]"
                << std::endl;
      return false;
    }

    const std::string index_prefix_path(indexFilePath);

    // convert strs into params
    std::string pq_prefix = index_prefix_path + "_pq";
    std::string pq_tables_bin = pq_prefix + "_pivots.bin";
    std::string disk_index_file = index_prefix_path + "_disk.index";
    std::string sample_data_file = index_prefix_path + "_sample_data.bin";

    size_t data_dim, num_pq_centers;
    diskann::get_bin_metadata(pq_tables_bin, num_pq_centers, data_dim);
    this->m_dimension = (_u32) data_dim;
    this->m_aligned_dimension = ROUND_UP(this->m_dimension, 8);

    this->Lsearch = (_u64) std::atoi(param_list[0].c_str());
    uint64_t num_cache_nodes = NUM_NODES_TO_CACHE;
    auto     nthreads = (_u32) std::atoi(param_list[1].c_str());
    auto     beamwidth = (_u32) std::atoi(param_list[2].c_str());

    try {
      std::shared_ptr<AlignedFileReader> reader = nullptr;

#ifdef _WINDOWS
#ifndef USE_BING_INFRA
      reader.reset(new WindowsAlignedFileReader());
#else
      _pDiskIO->Initialize(disk_index_file.c_str());
      reader.reset(new BingAlignedFileReader(_pDiskIO));
#endif
#else
      reader.reset(new LinuxAlignedFileReader());
#endif

      _pFlashIndex.reset(new PQFlashIndex<T>(reader));
      int res = _pFlashIndex->load(nthreads, pq_prefix.c_str(),
                                   disk_index_file.c_str());
      if (res != 0) {
        return false;
      }

      std::vector<uint32_t> node_list;
      // cache bfs levels
      _pFlashIndex->cache_bfs_levels(num_cache_nodes, node_list);
      _pFlashIndex->load_cache_list(node_list);
      node_list.clear();
      node_list.shrink_to_fit();

      uint64_t tuning_sample_num = 0;

      diskann::cout << "Warming up index... " << std::flush;
      T* tuning_sample =
          diskann::load_warmup<T>(sample_data_file, tuning_sample_num,
                                  this->m_dimension, this->m_aligned_dimension);
      std::vector<uint64_t> tuning_sample_result_ids_64(tuning_sample_num, 0);
      std::vector<float>    tuning_sample_result_dists(tuning_sample_num, 0);
#pragma omp                 parallel for schedule(dynamic, 1)
      for (_s64 i = 0; i < (int64_t) tuning_sample_num; i++) {
        _pFlashIndex->cached_beam_search(
            tuning_sample + (i * this->m_aligned_dimension), 1, WARMUP_L,
            tuning_sample_result_ids_64.data() + (i * 1),
            tuning_sample_result_dists.data() + (i * 1), 4);
      }
      diskann::cout << "..done" << std::endl;

      if (beamwidth == 0) {
        this->beam_width = diskann::optimize_beamwidth<T>(
            _pFlashIndex, tuning_sample, tuning_sample_num,
            this->m_aligned_dimension, (uint32_t) this->Lsearch);
        diskann::cout << "Loaded DiskANN index with L: " << this->Lsearch
                  << " (calculated) beam width: " << this->beam_width
                  << " nthreads: " << nthreads << std::endl;

      } else {
        this->beam_width = beamwidth;
        diskann::cout << "Loaded DiskANN index with L: " << this->Lsearch
                  << " (specified) beam width: " << this->beam_width
                  << " nthreads: " << nthreads << std::endl;
      }
      return true;
    } catch (const diskann::ANNException& ex) {
      diskann::cerr << ex.message() << std::endl;
      return false;
    }
  }

  // Load index from memory blob.
  template<typename T>
  DISKANN_DLLEXPORT bool DiskANNInterface<T>::LoadIndex(
      const std::vector<ANNIndex::FileBlob>& files,
      const char*                            queryParameters) {
    throw diskann::ANNException("Not implemented", -1);
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

  //Private methods:
  template <typename T>
  bool DiskANNInterface<T>::writeSharedStoreIniFile(const char* indexPathPrefix) {
    //Load template from the current folder. 
    std::ifstream in(".\\SharedFileStoreTemplate.ini");
    if (in.is_open()) {
      std::string contents((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());

      size_t searchIndex = 0;
      size_t replaceIndex = std::string::npos;
      while ( (replaceIndex = contents.find(INDEX_PATH_PREFIX_PLACEHOLDER, searchIndex)) != std::string::npos) {
        contents.replace(replaceIndex, PATH_PREFIX_PLACEHOLDER_LEN,
                         indexPathPrefix);
        searchIndex += strlen(indexPathPrefix);
      }
      std::ofstream out(std::string(indexPathPrefix) + "_SharedStore.ini");
      out << contents;
      out.close();
      in.close();
      return true;
    } else {
      std::cerr << "Could not find template file: SharedFileStoreTemplate.ini "
                   "in the current directory. Please contact the DiskANN team."
                << std::endl;
      return false;
    }
  }

  extern "C" __declspec(dllexport) ANNIndex::IANNIndex* CreateObjectFloat(

      unsigned __int32 dimension, ANNIndex::DistanceType distanceType,
      std::shared_ptr<ANNIndex::IDiskPriorityIO> ptr) {
    return new diskann::DiskANNInterface<float>(dimension, distanceType, ptr);
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
      unsigned __int32 dimension, ANNIndex::DistanceType distanceType,
      std::shared_ptr<ANNIndex::IDiskPriorityIO> ptr) {
    return new diskann::DiskANNInterface<int8_t>(dimension, distanceType, ptr);
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
