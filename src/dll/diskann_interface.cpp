#include <chrono>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "omp.h"
#include "mkl.h"

#include "diskann_interface.h"
#include "logger.h"
#include "aux_utils.h"
#include "bing_aligned_file_reader.h"
#include "index.h"
#include "partition_and_pq.h"
#include "utils.h"

#ifdef EXEC_ENV_OLS
#include "ANNLogging.h"
#include "util.h"
#endif

namespace diskann {
  // File local utility functions

  bool parseParameters(const std::string&        queryParameters,
                       std::vector<std::string>& param_list);
  std::string readContentsOfFile(const char* fileName);
  // File local utility functions end.

  // File local constants START
  static const std::string SHARED_STORE_INI_TEMPLATE_FILE =
      ".\\SharedFileStoreTemplate.ini";
  static const std::string INI_TEMPLATE_PATTERN_TO_REPLACE = "{0}";
  // File local constants END

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
#endif
    _pDiskIO = diskIO;
    diskann::cout << "Created index of type: " << typeid(T).name() << std::endl;
  }

  template<typename T>
  DiskANNInterface<T>::~DiskANNInterface<T>() {
  }

  template<typename T>
  // In implementation, the file path can be a file or folder.
  bool DiskANNInterface<T>::BuildIndex(const char* dataFilePath,
                                       const char* indexFilePath,
                                       const char* indexBuildParameters) {
    try {
      bool success = diskann::build_disk_index<T>(
          dataFilePath, indexFilePath, indexBuildParameters, _compareMetric);
      // No need to generate this ini file, index build pipeline will cover it.
      // success = success && writeSharedStoreIniFile(indexFilePath);
      return success;
    } catch (const diskann::ANNException& ex) {
      diskann::cerr << ex.message() << std::endl;
      return false;
    } catch (const std::exception& ex) {
      diskann::cerr << ex.what() << std::endl;
      return false;
    }
  }

  template<typename T>
  bool DiskANNInterface<T>::LoadIndex(
      const std::vector<ANNIndex::FileBlob>& files,
      const char*                            queryParameters) {
    diskann::cout << "Loading index from file blobs: " << std::endl;

#ifdef EXEC_ENV_OLS
    addBlobsToMemoryMappedFiles(files);
    std::string index_prefix_path = _mmFiles.filesPrefix();
#endif

    std::vector<std::string> param_list;
    if (false == parseParameters(queryParameters, param_list)) {
      return false;
    }

    std::string pq_prefix = index_prefix_path + "_pq";
    std::string pq_tables_bin = pq_prefix + "_pivots.bin";
    std::string disk_index_file = index_prefix_path + "_disk.index";
    std::string sample_data_file = index_prefix_path + "_sample_data.bin";

    size_t data_dim, num_pq_centers;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(_mmFiles, pq_tables_bin, num_pq_centers, data_dim);
#else
    get_bin_metadata(pq_tables_bin, num_pq_centers, data_dim);
#endif

    this->m_dimension = (_u32) data_dim;
    this->m_aligned_dimension = ROUND_UP(this->m_dimension, 8);
    this->Lsearch = (_u64) std::atoi(param_list[0].c_str());
    uint64_t num_cache_nodes = NUM_NODES_TO_CACHE;
    auto     nthreads = (_u32) std::atoi(param_list[1].c_str());
    auto     beamwidth = (_u32) std::atoi(param_list[2].c_str());

    try {
      _pDiskIO->Initialize(disk_index_file.c_str());
      _pReader.reset(new diskann::BingAlignedFileReader(_pDiskIO));
      _pFlashIndex.reset(new PQFlashIndex<T>(_pReader));

      int res;
#ifdef EXEC_ENV_OLS
      res = _pFlashIndex->load(_mmFiles, nthreads, pq_prefix.c_str(),
                               disk_index_file.c_str());
#else
      res = _pFlashIndex->load(nthreads, pq_prefix.c_str(),
                               disk_index_file.c_str());
#endif
      if (res != 0) {
        diskann::cerr << "Failed to load PQFlashIndex. PQFile: " << pq_prefix
                      << " Disk index file: " << disk_index_file << std::endl;
        return false;
      }

      std::vector<uint32_t> node_list;
      // cache bfs levels
      _pFlashIndex->cache_bfs_levels(num_cache_nodes, node_list);
      _pFlashIndex->load_cache_list(node_list);
      node_list.clear();
      node_list.shrink_to_fit();

      uint32_t tuningSampleNum = 0;
      T* tuningSample = loadTuningSample(sample_data_file, tuningSampleNum);
      warmupIndex(tuningSample, tuningSampleNum, nthreads);
      optimizeBeamwidth(tuningSample, tuningSampleNum, beamwidth, nthreads);
      aligned_free(tuningSample);
      return true;

    } catch (diskann::ANNException& ex) {
      diskann::cerr << ex.message() << std::endl;
      return false;
    }
  }

  template<typename T>
  // Load index form file.
  bool DiskANNInterface<T>::LoadIndex(const char* indexFilePath,
                                      const char* queryParameters) {
#ifdef EXEC_ENV_OLS
    diskann::cerr
        << "Cannot call LoadIndex with file prefix while running in OLS"
        << std::endl;
    return false;
#else
    diskann::cout << "Loading index from files with prefix: " << indexFilePath
                  << " for search" << std::endl;

    std::vector<std::string> param_list;
    if (false == parseParameters(queryParameters, param_list)) {
      return false;
    }

    std::string index_prefix_path(indexFilePath);
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
      _pDiskIO->Initialize(disk_index_file.c_str());
      _pReader.reset(new diskann::BingAlignedFileReader(_pDiskIO));
      _pFlashIndex.reset(new PQFlashIndex<T>(_pReader));
      int res = _pFlashIndex->load(nthreads, pq_prefix.c_str(),
                                   disk_index_file.c_str());
      if (res != 0) {
        diskann::cerr << "Failed to load PQFlashIndex. PQFile: " << pq_prefix
                      << " Disk index file: " << disk_index_file << std::endl;
        return false;
      }

      std::vector<uint32_t> node_list;
      // cache bfs levels
      _pFlashIndex->cache_bfs_levels(num_cache_nodes, node_list);
      _pFlashIndex->load_cache_list(node_list);
      node_list.clear();
      node_list.shrink_to_fit();

      uint32_t tuningSampleNum = 0;
      T* tuningSample = loadTuningSample(sample_data_file, tuningSampleNum);
      warmupIndex(tuningSample, tuningSampleNum);
      optimizeBeamwidth(tuningSample, tuningSampleNum, beamwidth, nthreads);
      aligned_free(tuningSample);

      return true;
    } catch (const diskann::ANNException& ex) {
      diskann::cerr << ex.message() << std::endl;
      return false;
    }
#endif
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
      diskann::cerr << ex.message() << std::endl;
    }
  }

  // PRIVATE FUNCTIONS START
  template<typename T>
  void DiskANNInterface<T>::addBlobsToMemoryMappedFiles(
      const std::vector<ANNIndex::FileBlob>& files) {
    for (auto file : files) {
      FileContent fc(const_cast<void*>(file.data), file.size);
      _mmFiles.addFile(file.path, fc);
      diskann::cout << "Loaded file: " << file.path
                    << " with size: " << file.size << std::endl;
    }
  }

  template<typename T>
  T* DiskANNInterface<T>::loadTuningSample(const std::string& sample_data_file,
                                           uint32_t& tuning_sample_num) {
    uint64_t sample_num = 0;
#ifdef EXEC_ENV_OLS

    T* tuning_sample =
        diskann::load_warmup<T>(_mmFiles, sample_data_file, sample_num,
                                this->m_dimension, this->m_aligned_dimension);
#else
    T* tuning_sample =
        diskann::load_warmup<T>(sample_data_file, sample_num, this->m_dimension,
                                this->m_aligned_dimension);
#endif
    tuning_sample_num = (uint32_t)(sample_num);
    return tuning_sample;
  }

  template<typename T>
  void DiskANNInterface<T>::warmupIndex(T*       tuning_sample,
                                        uint32_t tuning_sample_num,
                                        uint32_t nthreads) {
    diskann::cout << "Warming up index... " << std::flush;

    std::vector<uint64_t> tuning_sample_result_ids_64(tuning_sample_num, 0);
    std::vector<float>    tuning_sample_result_dists(tuning_sample_num, 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (_s64 i = 0; i < (int64_t) tuning_sample_num; i++) {
      _pFlashIndex->cached_beam_search(
          tuning_sample + (i * this->m_aligned_dimension), 1, WARMUP_L,
          tuning_sample_result_ids_64.data() + (i * 1),
          tuning_sample_result_dists.data() + (i * 1), 4);
    }
    diskann::cout << "..done" << std::endl;
  }

  template<typename T>
  void DiskANNInterface<T>::optimizeBeamwidth(T*       tuning_sample,
                                              uint32_t tuning_sample_num,
                                              uint32_t beamwidth,
                                              uint32_t nthreads) {
    if (beamwidth == 0) {
      this->beam_width = diskann::optimize_beamwidth<T>(
          _pFlashIndex, tuning_sample, tuning_sample_num,
          this->m_aligned_dimension, (uint32_t) this->Lsearch, nthreads);
      diskann::cout << "Loaded DiskANN index with L: " << this->Lsearch
                    << " (calculated) beam width: " << this->beam_width
                    << " nthreads: " << nthreads << std::endl;

    } else {
      this->beam_width = beamwidth;
      diskann::cout << "Loaded DiskANN index with L: " << this->Lsearch
                    << " (specified) beam width: " << this->beam_width
                    << " nthreads: " << nthreads << std::endl;
    }
  }

  // Writing the INI File that tells OLS what to load in memory and what
  // to keep on disk.

  template<typename T>
  bool DiskANNInterface<T>::writeSharedStoreIniFile(
      const char* indexPathPrefix) {
    diskann::cout << "Writing INI file " << std::flush;

    std::string iniTemplate =
        readContentsOfFile(SHARED_STORE_INI_TEMPLATE_FILE.c_str());
    if (iniTemplate.empty()) {
      return false;
    } else {
      size_t      index = 0;
      std::string indexPathPrefixStr(indexPathPrefix);
      while ((index = iniTemplate.find(INI_TEMPLATE_PATTERN_TO_REPLACE,
                                       index)) != std::string::npos) {
        iniTemplate.replace(index, INI_TEMPLATE_PATTERN_TO_REPLACE.length(),
                            indexPathPrefix);
        index = index + indexPathPrefixStr.length();
      }

      std::ofstream iniFile(indexPathPrefixStr + "_ssf.ini");
      iniFile << iniTemplate << std::endl;

      diskann::cout << "done." << std::endl;
      return true;
    }
  }
  // PRIVATE FUNCTIONS END

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

  // File local utility functions START
  // Support two patterns of parameters:
  // 1) "10 20 0" as original one.
  // 2) "CandidateListSize=10,SearchThreadCount=20,BeamWidth=0" to help partner
  // understanding.
  bool parseParameters(const std::string&        queryParameters,
                       std::vector<std::string>& param_list) {
#ifdef EXEC_ENV_OLS
    if (queryParameters.find('=') != std::string::npos) {
      std::map<std::string, std::string> parameterMap;
      similarity::SplitParameterValues(queryParameters.c_str(), parameterMap);

      int parsedParameterCount = 0;
      int value = 0;
      if (similarity::ParseIntParameterValue(
              parameterMap, ParameterCandidateListSize, value)) {
        param_list.emplace_back(
            parameterMap.find(ParameterCandidateListSize)->second);
        parsedParameterCount++;
      } else {
        param_list.emplace_back("20");
      }

      if (similarity::ParseIntParameterValue(
              parameterMap, ParameterSearchThreadCount, value)) {
        param_list.emplace_back(
            parameterMap.find(ParameterSearchThreadCount)->second);
        parsedParameterCount++;
      } else {
        param_list.emplace_back("40");
      }

      if (similarity::ParseIntParameterValue(parameterMap, ParameterBeamWidth,
                                             value)) {
        param_list.emplace_back(parameterMap.find(ParameterBeamWidth)->second);
        parsedParameterCount++;
      } else {
        param_list.emplace_back("0");
      }

      if (parsedParameterCount < parameterMap.size()) {
        diskann::cerr << "Input parameters : " << queryParameters
                      << " contain unrecognized parameter, only "
                      << ParameterCandidateListSize << ", "
                      << ParameterSearchThreadCount << ", "
                      << ParameterBeamWidth << "allowed(case sensitive)!"
                      << std::endl;
        param_list.clear();
        return false;
      }
    } else {
#endif
      std::stringstream parser;
      parser << std::string(queryParameters);
      std::string cur_param;

      while (parser >> cur_param)
        param_list.push_back(cur_param);

      if (param_list.size() != 3) {
        diskann::cerr << "Correct usage of parameters is \n"
                         "Lsearch[1] nthreads[2] beamwidth[3]"
                      << std::endl;
        return false;
      }
#ifdef EXEC_ENV_OLS
    }
#endif

    diskann::cout << "Search parameter : ";
    for (const auto& parameter : param_list) {
      diskann::cout << parameter << " ";
    }
    diskann::cout << std::endl;

    return true;
  }
  std::string readContentsOfFile(const char* fileName) {
    std::ifstream configFile(fileName, std::ios::ate);
    size_t        fileSize = configFile.tellg();
    configFile.seekg(0);

    if (configFile.is_open()) {
      std::string contents;
      char*       buffer = new char[fileSize];
      memset(buffer, 0, fileSize);

      configFile.read(buffer, fileSize);
      contents.assign(buffer, fileSize);
      delete[] buffer;

      return contents;
    } else {
      diskann::cout << "Could not open config file template file: " << fileName
                    << std::endl;
      return std::string();
    }
  }

  // File local utility functions END

  template class DiskANNInterface<int8_t>;
  template class DiskANNInterface<float>;
  template class DiskANNInterface<uint8_t>;

}  // namespace diskann
