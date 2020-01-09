#include <chrono>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dll/diskann_interface.h"
#include "index.h"
#include "partition_and_pq.h"
#include "utils.h"

namespace diskann {

#define TRAINING_SET_SIZE 3000000
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
    std::stringstream parser;
    parser << std::string(indexBuildParameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() != 6) {
      std::cout
          << "Correct usage of parameters is L (indexing search list size) "
             "R (max degree) C (visited list maximum size) B (approximate "
             "compressed number of bytes per datapoint to store in "
             "memory) S (Sampling Rate For PQ Generation) T (Max Threads To "
             "Use)"
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
    unsigned num_threads = (unsigned) atoi(param_list[5].c_str());
    auto     s = std::chrono::high_resolution_clock::now();

    float* train_data;
    size_t train_size, train_dim;

    // generates random sample and sets it to train_data and updates train_size
    gen_random_slice<T>(dataFilePath, training_set_sampling_rate, train_data,
                        train_size, train_dim);

    std::string warmup_file_prefix = index_prefix_path + "_warmup";
    gen_random_slice<T>(dataFilePath, warmup_file_prefix, 0.01);

    std::cout << "Training loaded of size " << train_size << std::endl;

    generate_pq_pivots(train_data, train_size, train_dim, 256, num_pq_chunks,
                       20, pq_pivots_path);
    generate_pq_data_from_pivots<T>(dataFilePath, 256, num_pq_chunks,
                                    pq_pivots_path, pq_compressed_vectors_path);

    delete[] train_data;

    diskann::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<float>("alpha", 4.0);
    paras.Set<unsigned>("num_rnds", 2);
    paras.Set<unsigned>("num_threads", num_threads);
    paras.Set<std::string>("save_path", randnsg_path);

    _pNsgIndex = std::unique_ptr<diskann::Index<T>>(
        new diskann::Index<T>(_compareMetric, dataFilePath));

    _pNsgIndex->build(paras);
    _pNsgIndex->save(randnsg_path.c_str());
    _pFlashIndex.reset(new PQFlashIndex<T>());
    _pFlashIndex->create_disk_layout(std::string(dataFilePath), randnsg_path,
                                     disk_index_path);

    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;

    std::cout << "Indexing time: " << diff.count() << "\n";

    return true;
  }

  template<typename T>
  // Load index form file.
  bool DiskANNInterface<T>::LoadIndex(const char* indexFilePath,
                                      const char* queryParameters) {
    std::stringstream parser;
    parser << std::string(queryParameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() != 4) {
      std::cerr << "Correct usage of parameters is \n"
                   "Lsearch[1] BeamWidth[2] num_cache_nodes[3] nthreads[4]"
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
    std::string centroid_data_file = index_prefix_path + "_centroids_float.bin";
    std::string cache_warmup_file = index_prefix_path + "_warmup_data.bin";

    size_t data_dim, num_pq_centers;
    diskann::get_bin_metadata(pq_tables_bin, num_pq_centers, data_dim);
    this->m_dimension = (_u32) data_dim;
    this->aligned_dimension = ROUND_UP(this->m_dimension, 8);

    this->Lsearch = (_u64) std::atoi(param_list[0].c_str());
    this->beam_width = (_u64) std::atoi(param_list[1].c_str());
    uint64_t num_cache_nodes = (_u64) std::atoi(param_list[2].c_str());
    _u64     nthreads = (_u64) std::atoi(param_list[3].c_str());

    // create object
    _pFlashIndex.reset(new PQFlashIndex<T>());

    // load index
    _pFlashIndex->load(nthreads, pq_tables_bin.c_str(), data_bin.c_str(),
                       disk_index_file.c_str());
    _pFlashIndex->load_entry_points(medoids_file, centroid_data_file);
    _pFlashIndex->cache_medoid_nhoods();
    std::vector<uint32_t> node_list;
    // cache bfs levels
    _pFlashIndex->cache_bfs_levels(num_cache_nodes, node_list);
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();
    //    free(params);  // Gopal. Caller has to free the 'params' variable.
    //

    uint64_t warmup_L = this->Lsearch;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T*       warmup = nullptr;
    if (file_exists(cache_warmup_file)) {
      diskann::load_aligned_bin<T>(cache_warmup_file, warmup, warmup_num,
                                   warmup_dim, warmup_aligned_dim);
      if (warmup_dim != this->m_dimension) {
        std::cout << "Error in warmup file. Dimension mismatch." << std::endl;
        exit(-1);
      }
    } else {
      warmup_num = 100000;
      warmup_dim = this->m_dimension;
      warmup_aligned_dim = this->aligned_dimension;
      std::cout << "Generating random warmup file with dim " << warmup_dim
                << " and aligned dim " << warmup_aligned_dim << std::flush;
      diskann::alloc_aligned(((void**) &warmup),
                             warmup_num * warmup_aligned_dim * sizeof(T),
                             8 * sizeof(T));
      std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
      std::random_device              rd;
      std::mt19937                    gen(rd());
      std::uniform_int_distribution<> dis(-128, 127);
      for (uint32_t i = 0; i < warmup_num; i++) {
        for (uint32_t d = 0; d < warmup_dim; d++) {
          warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
        }
      }
      std::cout << "..done" << std::endl;
    }
    std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
    std::vector<float>    warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) warmup_num; i++) {
      _pFlashIndex->cached_beam_search(
          warmup + (i * warmup_aligned_dim), 1, warmup_L,
          warmup_result_ids_64.data() + (i * 1),
          warmup_result_dists.data() + (i * 1), beam_width);
    }
    std::cout << "Done warming up threads and cache." << std::endl;
    if (warmup != nullptr)
      diskann::aligned_free(warmup);
    std::cout << "DiskANN Index Loaded." << std::endl;
    return true;
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
    //    _u64      L = 6 * neighborCount;
    std::cout << this->aligned_dimension << " is aligned dimension "
              << std::endl;
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

  template class DiskANNInterface<int8_t>;
  template class DiskANNInterface<float>;
  template class DiskANNInterface<uint8_t>;

}  // namespace diskann
