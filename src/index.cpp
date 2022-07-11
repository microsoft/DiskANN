// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <type_traits>
#include <omp.h>
#include <atomic>
#include <random>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include <unordered_map>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "common_includes.h"
#include "logger.h"
#include "exceptions.h"
#include "aligned_file_reader.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "parameters.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "windows_customizations.h"
#include "ann_exception.h"
#include "boost/dynamic_bitset.hpp"

#ifdef _WINDOWS
#include <xmmintrin.h>
#endif
#include "index.h"

#define MAX_POINTS_FOR_USING_BITSET 10000000

namespace diskann {

  template<typename T>
  inline T diskann_max(T left, T right) {
    return left > right ? left : right;
  }
  // QueryScratch functions
  template<typename T>
  InMemQueryScratch<T>::InMemQueryScratch() {
    search_l = indexing_l = r = 0;
    // pointers are initialized in the header itself.
  }
  template<typename T>
  void InMemQueryScratch<T>::setup(uint32_t search_l, uint32_t indexing_l,
                                   uint32_t r, size_t dim) {
    if (search_l == 0 || indexing_l == 0 || r == 0 || dim == 0) {
      std::stringstream ss;
      ss << "In InMemQueryScratch, one of search_l = " << search_l
         << ", indexing_l = " << indexing_l << ", dim = " << dim
         << " or r = " << r << " is zero." << std::endl;
      throw diskann::ANNException(ss.str(), -1);
    }
    indices = new uint32_t[search_l];     // only used by search
    interim_dists = new float[search_l];  // only used by search
    memset(indices, 0, sizeof(uint32_t) * search_l);
    memset(interim_dists, 0, sizeof(float) * search_l);
    this->search_l = search_l;
    this->indexing_l = indexing_l;
    this->r = r;

    auto   aligned_dim = ROUND_UP(dim, 8);
    size_t allocSize = aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, aligned_dim * sizeof(T));

    auto l_to_use = std::max(search_l, indexing_l);

    _des = new std::vector<unsigned>();
    _des->reserve(2 * r);
    _pool = new std::vector<Neighbor>();
    _pool->reserve(l_to_use * 10);
    _visited = new tsl::robin_set<unsigned>();
    _visited->reserve(l_to_use * 2);
    _best_l_nodes = new std::vector<Neighbor>();
    _best_l_nodes->resize(l_to_use + 1);
    _inserted_into_pool_rs = new tsl::robin_set<unsigned>();
    _inserted_into_pool_rs->reserve(l_to_use * 20);
    _inserted_into_pool_bs = new boost::dynamic_bitset<>();
  }

  template<typename T>
  void InMemQueryScratch<T>::clear() {
    memset(indices, 0, sizeof(uint32_t) * search_l);
    memset(interim_dists, 0, sizeof(float) * search_l);
    _pool->clear();
    _visited->clear();
    _des->clear();
    _inserted_into_pool_rs->clear();
    _inserted_into_pool_bs->reset();
  }

  template<typename T>
  void InMemQueryScratch<T>::resize_for_query(uint32_t new_search_l) {
    if (search_l < new_search_l) {
      if (indices != nullptr) {
        delete[] indices;
      }
      indices = new uint32_t[new_search_l];

      if (interim_dists != nullptr) {
        delete[] interim_dists;
      }
      interim_dists = new float[new_search_l];
      search_l = new_search_l;
    }
  }

  template<typename T>
  void InMemQueryScratch<T>::destroy() {
    if (indices != nullptr) {
      delete[] indices;
      indices = nullptr;
    }
    if (interim_dists != nullptr) {
      delete[] interim_dists;
      interim_dists = nullptr;
    }
    if (_pool != nullptr) {
      delete _pool;
      _pool = nullptr;
    }
    if (_visited != nullptr) {
      delete _visited;
      _visited = nullptr;
    }
    if (_des != nullptr) {
      delete _des;
      _des = nullptr;
    }
    if (_best_l_nodes != nullptr) {
      delete _best_l_nodes;
      _best_l_nodes = nullptr;
    }
    if (aligned_query != nullptr) {
      aligned_free(aligned_query);
      aligned_query = nullptr;
    }

    if (_inserted_into_pool_rs != nullptr) {
      delete _inserted_into_pool_rs;
      _inserted_into_pool_rs = nullptr;
    }
    if (_inserted_into_pool_bs != nullptr) {
      delete _inserted_into_pool_bs;
      _inserted_into_pool_bs = nullptr;
    }

    search_l = indexing_l = r = 0;
  }

  // Class to avoid the hassle of pushing and popping the query scratch.
  template<typename T>
  class ScratchStoreManager {
   public:
    diskann::InMemQueryScratch<T>          _scratch;
    ConcurrentQueue<InMemQueryScratch<T>> &_query_scratch;
    ScratchStoreManager(ConcurrentQueue<InMemQueryScratch<T>> &query_scratch)
        : _query_scratch(query_scratch) {
      _scratch = query_scratch.pop();
      while (_scratch.indices == nullptr) {
        query_scratch.wait_for_push_notify();
        _scratch = query_scratch.pop();
      }
    }
    InMemQueryScratch<T> scratch_space() {
      return _scratch;
    }

    ~ScratchStoreManager() {
      _scratch.clear();
      _query_scratch.push(_scratch);
      _query_scratch.push_notify_all();
    }

   private:
    ScratchStoreManager(const ScratchStoreManager<T> &);
    ScratchStoreManager<T> &operator=(const ScratchStoreManager<T> &);
  };

  // Initialize an index with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T, typename TagT>
  Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points,
                        const bool dynamic_index, const Parameters &indexParams,
                        const Parameters &searchParams, const bool enable_tags,
                        const bool   support_eager_delete,
                        const bool   concurrent_consolidate,
                        const bool   queries_present,
                        const size_t max_query_points)
      : Index(m, dim, max_points, dynamic_index, enable_tags,
              support_eager_delete, concurrent_consolidate, queries_present,
              max_query_points) {
    _indexingQueueSize = indexParams.Get<uint32_t>("L");
    _indexingRange = indexParams.Get<uint32_t>("R");
    _indexingMaxC = indexParams.Get<uint32_t>("C");
    _indexingAlpha = indexParams.Get<float>("alpha");

    uint32_t num_threads_srch = searchParams.Get<uint32_t>("num_threads");
    uint32_t num_threads_indx = indexParams.Get<uint32_t>("num_threads");
    uint32_t num_threads = std::max(num_threads_srch, num_threads_indx);
    uint32_t search_l = searchParams.Get<uint32_t>("L");

    initialize_query_scratch(num_threads, search_l, _indexingQueueSize,
                             _indexingRange, dim);
  }

  template<typename T, typename TagT>
  Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points,
                        const bool dynamic_index, const bool enable_tags,
                        const bool   support_eager_delete,
                        const bool   concurrent_consolidate,
                        const bool   queries_present,
                        const size_t max_query_points)
      : _dist_metric(m), _dim(dim), _max_points(max_points),
        _dynamic_index(dynamic_index), _enable_tags(enable_tags),
        _support_eager_delete(support_eager_delete),
        _conc_consolidate(concurrent_consolidate),
        _queries_present(queries_present), _max_query_points(max_query_points) {
    if (dynamic_index && !enable_tags) {
      throw diskann::ANNException(
          "ERROR: Eager Deletes must have Dynamic Indexing enabled.", -1,
          __FUNCSIG__, __FILE__, __LINE__);
      diskann::cerr
          << "WARNING: Dynamic Indices must have tags enabled. Auto-enabling."
          << std::endl;
      _enable_tags = true;
    }
    if (support_eager_delete && !dynamic_index) {
      diskann::cerr << "ERROR: Eager Deletes must have Dynamic Indexing "
                       "enabled. Exitting."
                    << std::endl;
      throw diskann::ANNException(
          "ERROR: Eager deletes are possible only if dynamic indexing is "
          "enabled. Exitting.",
          -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    _aligned_dim = ROUND_UP(_dim, 8);

    if (dynamic_index || _queries_present) {
      _num_frozen_pts = 1;
    }
    // Sanity check. While logically it is correct, max_points = 0 causes
    // downstream problems.
    if (_max_points == 0) {
      _max_points = 1;
    }

    alloc_aligned(((void **) &_data),
                  (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(T),
                  8 * sizeof(T));
    std::memset(_data, 0,
                (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(T));

    _start = (unsigned) _max_points;

    _final_graph.resize(_max_points + _num_frozen_pts);

    if (_support_eager_delete) {
      _in_graph.reserve(_max_points + _num_frozen_pts);
      _in_graph.resize(_max_points + _num_frozen_pts);
    }

    if (m == diskann::Metric::COSINE && std::is_floating_point<T>::value) {
      // This is safe because T is float inside the if block.
      this->_distance = (Distance<T> *) new AVXNormalizedCosineDistanceFloat();
      this->_normalize_vecs = true;
      std::cout << "Normalizing vectors and using L2 for cosine "
                   "AVXNormalizedCosineDistanceFloat()."
                << std::endl;
    } else {
      this->_distance = get_distance_function<T>(m);
    }

    _locks = std::vector<std::mutex>(_max_points + _num_frozen_pts);

    if (_support_eager_delete)
      _locks_in = std::vector<std::mutex>(_max_points + _num_frozen_pts);

    if (_queries_present) {
      alloc_aligned(((void **) &_query_data),
                    (_max_query_points) *_aligned_dim * sizeof(T),
                    8 * sizeof(T));
      std::memset(_data, 0, (_max_query_points) *_aligned_dim * sizeof(T));
      _query_graph.reserve(_max_query_points);
      _query_graph.resize(_max_query_points);
      _query_nn.reserve(_max_query_points);
      _query_nn.resize(_max_query_points);
      _query_locks = std::vector<std::mutex>(_max_query_points);
      _query_nn_locks = std::vector<std::mutex>(_max_query_points);
    }
  }

  template<typename T, typename TagT>
  Index<T, TagT>::~Index() {
    // Ensure that no other activity is happening before dtor()
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tul(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> tdl(_delete_lock);

    for (auto &lock : _locks) {
      LockGuard lg(lock);
    }
    for (auto &lock : _locks_in) {
      LockGuard lg(lock);
    }

    if (this->_distance != nullptr) {
      delete this->_distance;
      this->_distance = nullptr;
    }
    if (this->_data != nullptr) {
      aligned_free(this->_data);
      this->_data = nullptr;
    }

    while (!_query_scratch.empty()) {
      auto val = _query_scratch.pop();
      while (val.indices == nullptr) {
        _query_scratch.wait_for_push_notify();
        val = _query_scratch.pop();
      }
      val.destroy();
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::initialize_query_scratch(uint32_t num_threads,
                                                uint32_t search_l,
                                                uint32_t indexing_l, uint32_t r,
                                                size_t dim) {
    for (uint32_t i = 0; i < num_threads; i++) {
      InMemQueryScratch<T> scratch;
      scratch.setup(search_l, indexing_l, r, dim);
      _query_scratch.push(scratch);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::clear_index() {
    memset(_data, 0,
           _aligned_dim * (_max_points + _num_frozen_pts) * sizeof(T));
    _nd = 0;
    for (size_t i = 0; i < _final_graph.size(); i++)
      _final_graph[i].clear();

    _tag_to_location.clear();
    _location_to_tag.clear();

    // What about empty slots, delete_set, etc?
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_tags(std::string tags_file) {
    if (!_enable_tags) {
      diskann::cout << "Not saving tags as they are not enabled." << std::endl;
      return 0;
    }
    size_t tag_bytes_written;
    TagT * tag_data = new TagT[_nd + _num_frozen_pts];
    for (_u32 i = 0; i < _nd; i++) {
      if (_location_to_tag.find(i) != _location_to_tag.end()) {
        tag_data[i] = _location_to_tag[i];
      } else {
        // catering to future when tagT can be any type.
        std::memset((char *) &tag_data[i], 0, sizeof(TagT));
      }
    }
    if (_num_frozen_pts > 0) {
      std::memset((char *) &tag_data[_start], 0, sizeof(TagT));
    }
    try {
      tag_bytes_written =
          save_bin<TagT>(tags_file, tag_data, _nd + _num_frozen_pts, 1);
    } catch (std::system_error &e) {
      throw FileException(tags_file, e, __FUNCSIG__, __FILE__, __LINE__);
    }
    delete[] tag_data;
    return tag_bytes_written;
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_data(std::string data_file, const int version) {
    if (version == 0)
      return save_data_in_base_dimensions(
          data_file, _data, _nd + _num_frozen_pts, _dim, _aligned_dim);
    else
      return save_data_in_base_dimensions(
          data_file, _query_data, _max_query_points, _dim, _aligned_dim);
  }

  // save the graph index on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_graph_internal(
      std::string graph_file, std::vector<std::vector<unsigned>> &final_graph,
      size_t &nd, size_t num_frozen_pts, unsigned &max_observed_degree,
      unsigned start) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    _u64 file_offset = 0;  // we will use this if we want
    out.seekp(file_offset, out.beg);
    _u64 index_size = 24;
    _u32 max_degree = 0;
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_observed_degree, sizeof(unsigned));
    unsigned ep_u32 = start;
    out.write((char *) &ep_u32, sizeof(unsigned));
    out.write((char *) &num_frozen_pts, sizeof(_u64));
    for (unsigned i = 0; i < nd + num_frozen_pts; i++) {
      unsigned GK = (unsigned) final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) final_graph[i].data(), GK * sizeof(unsigned));
      max_degree = final_graph[i].size() > max_degree
                       ? (_u32) final_graph[i].size()
                       : max_degree;
      index_size += (_u64)(sizeof(unsigned) * (GK + 1));
    }
    out.seekp(file_offset, out.beg);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_degree, sizeof(_u32));
    out.close();
    return index_size;  // number of bytes written
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_graph(std::string graph_file, const int version) {
    if (version == 0)
      return save_graph_internal(graph_file, _final_graph, _nd, _num_frozen_pts,
                                 _max_observed_degree, _start);
    else 
      return save_graph_internal(graph_file, _query_graph, _max_query_points, 0,
                                 _max_observed_qdegree, _query_start);
    // else
    //   return save_graph_internal(graph_file, _query_nn, _max_query_points, 0,
    //                              _indexingRange, 0);
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_delete_list(const std::string &filename) {
    if (_delete_set.size() == 0) {
      return 0;
    }
    std::unique_ptr<_u32[]> delete_list =
        std::make_unique<_u32[]>(_delete_set.size());
    _u32 i = 0;
    for (auto &del : _delete_set) {
      delete_list[i++] = del;
    }
    return save_bin<_u32>(filename, delete_list.get(), _delete_set.size(), 1);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save(const char *filename) {
    // first check if no thread is inserting
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_lock<std::shared_timed_mutex> lock(_update_lock);
    _num_points_lock.lock();

    compact_frozen_point();
    if (!_save_as_one_file) {
      std::string graph_file = std::string(filename);
      std::string tags_file = std::string(filename) + ".tags";
      std::string data_file = std::string(filename) + ".data";
      std::string delete_list_file = std::string(filename) + ".del";
      std::string query_data_file = std::string(filename) + ".qdata";
      std::string query_graph_file = std::string(filename) + ".qgraph";
      // std::string query_nn_file = std::string(filename) + ".qnn";

      // Because the save_* functions use append mode, ensure that
      // the files are deleted before save. Ideally, we should check
      // the error code for delete_file, but will ignore now because
      // delete should succeed if save will succeed.
      delete_file(graph_file);
      save_graph(graph_file);
      delete_file(data_file);
      save_data(data_file);
      delete_file(tags_file);
      save_tags(tags_file);
      delete_file(delete_list_file);
      save_delete_list(delete_list_file);
      if (_queries_present) {
        delete_file(query_data_file);
        save_data(query_data_file);
        delete_file(query_graph_file);
        save_graph(query_graph_file, 1);
        // delete_file(query_nn_file);
        // save_graph(query_nn_file, 2);
      }

    } else {
      diskann::cout << "Save index in a single file currently not supported. "
                       "Not saving the index."
                    << std::endl;
    }

    reposition_frozen_point_to_end();

    _num_points_lock.unlock();
    auto stop = std::chrono::high_resolution_clock::now();
    auto timespan =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    diskann::cout << "Time taken for save: " << timespan.count() << "s."
                  << std::endl;
  }

#ifdef EXEC_ENV_OLS
  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_tags(AlignedFileReader &reader) {
#else
  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_tags(const std::string tag_filename) {
    if (_enable_tags && !file_exists(tag_filename)) {
      diskann::cerr << "Tag file provided does not exist!" << std::endl;
      throw diskann::ANNException("Tag file provided does not exist!", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }
#endif
    if (!_enable_tags) {
      diskann::cout << "Tags not loaded as tags not enabled." << std::endl;
      return 0;
    }

    size_t file_dim, file_num_points;
    TagT * tag_data;
#ifdef EXEC_ENV_OLS
    load_bin<TagT>(reader, tag_data, file_num_points, file_dim);
#else
    load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points,
                   file_dim);
#endif

    if (file_dim != 1) {
      std::stringstream stream;
      stream << "ERROR: Found " << file_dim << " dimensions for tags,"
             << "but tag file must have 1 dimension." << std::endl;
      diskann::cerr << stream.str() << std::endl;
      delete[] tag_data;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    size_t num_data_points =
        _num_frozen_pts > 0 ? file_num_points - 1 : file_num_points;
    for (_u32 i = 0; i < (_u32) num_data_points; i++) {
      TagT tag = *(tag_data + i);
      if (_delete_set.find(i) == _delete_set.end()) {
        _location_to_tag[i] = tag;
        _tag_to_location[tag] = (_u32) i;
      }
    }
    diskann::cout << "Tags loaded." << std::endl;
    delete[] tag_data;
    return file_num_points;
  }

  template<typename T, typename TagT>
#ifdef EXEC_ENV_OLS
  size_t Index<T, TagT>::load_data(AlignedFileReader &reader) {
#else
  size_t Index<T, TagT>::load_data(std::string filename) {
#endif
    size_t file_dim, file_num_points;
#ifdef EXEC_ENV_OLS
    diskann::get_bin_metadata(reader, file_num_points, file_dim);
#else
    if (!file_exists(filename)) {
      std::stringstream stream;
      stream << "ERROR: data file " << filename << " does not exist."
             << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    diskann::get_bin_metadata(filename, file_num_points, file_dim);
#endif

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != _dim) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << _dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (file_num_points > _max_points) {
      //_num_points_lock is already locked in load()
      std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      resize(file_num_points);
    }

#ifdef EXEC_ENV_OLS
    copy_aligned_data_from_file<T>(reader, _data, file_num_points, file_dim,
                                   _aligned_dim);
#else
    copy_aligned_data_from_file<T>(filename.c_str(), _data, file_num_points,
                                   file_dim, _aligned_dim);
#endif
    return file_num_points;
  }

#ifdef EXEC_ENV_OLS
  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_delete_set(AlignedFileReader &reader) {
#else
  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_delete_set(const std::string &filename) {
#endif
    std::unique_ptr<_u32[]> delete_list;
    _u64                    npts, ndim;

#ifdef EXEC_ENV_OLS
    diskann::load_bin<_u32>(reader, delete_list, npts, ndim);
#else
    diskann::load_bin<_u32>(filename, delete_list, npts, ndim);
#endif
    assert(ndim == 1);
    for (uint32_t i = 0; i < npts; i++) {
      _delete_set.insert(delete_list[i]);
    }
    return npts;
  }

  // load the index from file and update the max_degree, start (navigating
  // node id), and _final_graph (adjacency list)
  template<typename T, typename TagT>
#ifdef EXEC_ENV_OLS
  void Index<T, TagT>::load(AlignedFileReader &reader, uint32_t num_threads,
                            uint32_t search_l) {
#else
  void Index<T, TagT>::load(const char *filename, uint32_t num_threads,
                            uint32_t search_l) {
#endif
    _num_points_lock.lock();

    size_t tags_file_num_pts = 0, flags_file_num_pts = 0, graph_num_pts = 0,
           data_file_num_pts = 0;

    if (!_save_as_one_file) {
      // For DLVS Store, we will not support saving the index in multiple files.
#ifndef EXEC_ENV_OLS
      std::string data_file = std::string(filename) + ".data";
      std::string tags_file = std::string(filename) + ".tags";
      std::string delete_set_file = std::string(filename) + ".del";
      std::string graph_file = std::string(filename);
      data_file_num_pts = load_data(data_file);
      if (file_exists(delete_set_file)) {
        load_delete_set(delete_set_file);
      }
      if (_enable_tags) {
        tags_file_num_pts = load_tags(tags_file);
      }
      graph_num_pts = load_graph(graph_file, data_file_num_pts);
#endif

    } else {
      diskann::cout << "Single index file saving/loading support not yet "
                       "enabled. Not loading the index."
                    << std::endl;
      return;
    }

    if (data_file_num_pts != graph_num_pts ||
        (data_file_num_pts != tags_file_num_pts && _enable_tags)) {
      std::stringstream stream;
      stream << "ERROR: When loading index, loaded " << data_file_num_pts
             << " points from datafile, " << graph_num_pts
             << " from graph, and " << tags_file_num_pts
             << " tags, with num_frozen_pts being set to " << _num_frozen_pts
             << " in constructor." << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (_queries_present && flags_file_num_pts != tags_file_num_pts) {
      std::stringstream stream;
      stream << "ERROR: When loading index, loaded " << tags_file_num_pts
             << " points from tag file, " << graph_num_pts
             << " from graph, and " << flags_file_num_pts
             << " flags, with num_frozen_pts being set to " << _num_frozen_pts
             << " in constructor." << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    _nd = data_file_num_pts - _num_frozen_pts;
    _empty_slots.clear();
    for (auto i = _nd; i < _max_points; i++) {
      _empty_slots.insert((uint32_t) i);
    }

    _lazy_done = _delete_set.size() != 0;

    reposition_frozen_point_to_end();
    diskann::cout << "Num frozen points:" << _num_frozen_pts << " _nd: " << _nd
                  << " _start: " << _start
                  << " size(_location_to_tag): " << _location_to_tag.size()
                  << " size(_tag_to_location):" << _tag_to_location.size()
                  << " Max points: " << _max_points << std::endl;

    _search_queue_size = search_l;
    // For incremental index, _query_scratch is initialized in the constructor.
    // For the bulk index, the params required to initialize _query_scratch
    // are known only at load time, hence this check and the call to
    // initialize_q_s().
    if (_query_scratch.size() == 0) {
      initialize_query_scratch(num_threads, search_l, search_l,
                               (uint32_t) _max_range_of_loaded_graph, _dim);
    }

    _has_built = true;

    _num_points_lock.unlock();
  }

#ifdef EXEC_ENV_OLS
  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_graph(AlignedFileReader &reader,
                                    size_t             expected_num_points) {
#else

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_graph(std::string filename,
                                    size_t      expected_num_points) {
#endif
    size_t expected_file_size;
    _u64   file_frozen_pts;

#ifdef EXEC_ENV_OLS
    int header_size = 2 * sizeof(_u64) + 2 * sizeof(unsigned);
    std::unique_ptr<char[]> header = std::make_unique<char[]>(header_size);
    read_array(reader, header.get(), header_size);

    expected_file_size = *((_u64 *) header.get());
    _max_observed_degree = *((_u32 *) (header.get() + sizeof(_u64)));
    _start = *((_u32 *) (header.get() + sizeof(_u64) + sizeof(unsigned)));
    file_frozen_pts = *((_u64 *) (header.get() + sizeof(_u64) +
                                  sizeof(unsigned) + sizeof(unsigned)));
#else

    _u64 file_offset = 0;  // will need this for single file format support
    std::ifstream in(filename, std::ios::binary);
    in.seekg(file_offset, in.beg);
    in.read((char *) &expected_file_size, sizeof(_u64));
    in.read((char *) &_max_observed_degree, sizeof(unsigned));
    in.read((char *) &_start, sizeof(unsigned));
    in.read((char *) &file_frozen_pts, sizeof(_u64));
    _u64 vamana_metadata_size =
        sizeof(_u64) + sizeof(_u32) + sizeof(_u32) + sizeof(_u64);

#endif
    diskann::cout << "From graph header, expected_file_size: "
                  << expected_file_size
                  << ", _max_observed_degree: " << _max_observed_degree
                  << ", _start: " << _start
                  << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    if (file_frozen_pts != _num_frozen_pts) {
      std::stringstream stream;
      if (file_frozen_pts == 1) {
        stream << "ERROR: When loading index, detected dynamic index, but "
                  "constructor asks for static index. Exitting."
               << std::endl;
      } else {
        stream << "ERROR: When loading index, detected static index, but "
                  "constructor asks for dynamic index. Exitting."
               << std::endl;
      }
      diskann::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

#ifdef EXEC_ENV_OLS
    diskann::cout << "Loading vamana graph from reader..." << std::flush;
#else
    diskann::cout << "Loading vamana graph " << filename << "..." << std::flush;
#endif

    // Sanity check. If user provides more points than max_points
    // resize the _final_graph to the larger size.
    if (_max_points < expected_num_points) {
      diskann::cout << "Number of points in data: " << expected_num_points
                    << " is greater than max_points: " << _max_points
                    << " Setting max points to: " << expected_num_points
                    << std::endl;
      _final_graph.resize(expected_num_points + _num_frozen_pts);
      _max_points = expected_num_points;
    }
#ifdef EXEC_ENV_OLS
    _u32 nodes_read = 0;
    _u64 cc = 0;
    _u64 graph_offset = file_offset + header_size;
    while (nodes_read < expected_num_points) {
      _u32 k;
      read_value(reader, k, graph_offset);
      graph_offset += sizeof(_u32);
      std::vector<_u32> tmp(k);
      tmp.reserve(k);
      read_array(reader, tmp.data(), k, graph_offset);
      graph_offset += k * sizeof(_u32);
      cc += k;
      _final_graph[nodes_read].swap(tmp);
      nodes_read++;
      if (nodes_read % 1000000 == 0) {
        diskann::cout << "." << std::flush;
      }
      if (k > _max_range_of_loaded_graph) {
        _max_range_of_loaded_graph = k;
      }
    }
#else

    size_t   bytes_read = vamana_metadata_size;
    size_t   cc = 0;
    unsigned nodes_read = 0;
    while (bytes_read != expected_file_size) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (k == 0) {
        diskann::cerr << "ERROR: Point found with no out-neighbors, point#"
                      << nodes_read << std::endl;
      }

      cc += k;
      ++nodes_read;
      std::vector<unsigned> tmp(k);
      tmp.reserve(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));
      _final_graph[nodes_read - 1].swap(tmp);
      bytes_read += sizeof(uint32_t) * ((_u64) k + 1);
      if (nodes_read % 10000000 == 0)
        diskann::cout << "." << std::flush;
      if (k > _max_range_of_loaded_graph) {
        _max_range_of_loaded_graph = k;
      }
    }
#endif

    diskann::cout << "done. Index has " << nodes_read << " nodes and " << cc
                  << " out-edges, _start is set to " << _start << std::endl;
    return nodes_read;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::get_vector_by_tag(TagT &tag, T *vec) {
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      diskann::cout << "Tag " << tag << " does not exist" << std::endl;
      return -1;
    }

    size_t location = _tag_to_location[tag];
    memcpy((void *) vec, (void *) (_data + location * _aligned_dim),
           (size_t) _dim * sizeof(T));
    return 0;
  }

  /**************************************************************
   *      Support for Static Index Building and Searching
   **************************************************************/

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T, typename TagT>
  unsigned Index<T, TagT>::calculate_entry_point_internal(size_t &nd, T *data) {
    // allocate and init centroid
    float *center = new float[_aligned_dim]();
    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] = 0;

    for (size_t i = 0; i < nd; i++)
      for (size_t j = 0; j < _aligned_dim; j++)
        center[j] += (float) data[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] /= (float) nd;

    // compute all to one distance
    float *distances = new float[nd]();
#pragma omp parallel for schedule(static, 65536)
    for (_s64 i = 0; i < (_s64) nd; i++) {
      // extract point and distance reference
      float &  dist = distances[i];
      const T *cur_vec = data + (i * (size_t) _aligned_dim);
      dist = 0;
      float diff = 0;
      for (size_t j = 0; j < _aligned_dim; j++) {
        diff =
            (center[j] - (float) cur_vec[j]) * (center[j] - (float) cur_vec[j]);
        dist += diff;
      }
    }
    // find imin
    unsigned min_idx = 0;
    float    min_dist = distances[0];
    for (unsigned i = 1; i < nd; i++) {
      if (distances[i] < min_dist) {
        min_idx = i;
        min_dist = distances[i];
      }
    }

    delete[] distances;
    delete[] center;
    return min_idx;
  }

  template<typename T, typename TagT>
  unsigned Index<T, TagT>::calculate_entry_point(const int version) {
    if (version == 0)
      return calculate_entry_point_internal(_nd, _data);
    else
      return calculate_entry_point_internal(_max_query_points, _query_data);
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point_internal(
      const T *node_coords, const unsigned Lsize,
      const std::vector<unsigned> &init_ids,
      std::vector<Neighbor> &      expanded_nodes_info,
      tsl::robin_set<unsigned> &   expanded_nodes_ids,
      std::vector<Neighbor> &best_L_nodes, std::vector<unsigned> &des,
      tsl::robin_set<unsigned> &inserted_into_pool_rs,
      boost::dynamic_bitset<> &inserted_into_pool_bs, T *data,
      size_t &max_points, size_t &num_frozen_pts,
      std::vector<std::vector<unsigned>> &final_graph, const unsigned &start,
      bool ret_frozen, bool search_invocation, const int version) {
    for (unsigned i = 0; i < Lsize + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }
    if (!search_invocation) {
      expanded_nodes_ids.clear();
      expanded_nodes_info.clear();
      des.clear();
    }

    unsigned l = 0;
    Neighbor nn;

    bool fast_iterate =
        (max_points + num_frozen_pts) <= MAX_POINTS_FOR_USING_BITSET;

    if (fast_iterate) {
      auto total_num_points = max_points + num_frozen_pts;
      if (inserted_into_pool_bs.size() < total_num_points) {
        // hopefully using 2X will reduce the number of allocations.
        auto resize_size = 2 * total_num_points > MAX_POINTS_FOR_USING_BITSET
                               ? MAX_POINTS_FOR_USING_BITSET
                               : 2 * total_num_points;
        inserted_into_pool_bs.resize(resize_size);
      }
    }

    for (auto id : init_ids) {
      if (id >= max_points + num_frozen_pts) {
        diskann::cerr << "Wrong id : " << id << std::endl;
        throw diskann::ANNException(
            std::string("Wrong id") + std::to_string(id), -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }
      nn = Neighbor(id,
                    _distance->compare(data + _aligned_dim * (size_t) id,
                                       node_coords, (unsigned) _aligned_dim),
                    true);
      if (fast_iterate) {
        if (inserted_into_pool_bs[id] == 0) {
          inserted_into_pool_bs[id] = 1;
          best_L_nodes[l++] = nn;
        }
      } else {
        if (inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end()) {
          inserted_into_pool_rs.insert(id);
          best_L_nodes[l++] = nn;
        }
      }
      if (l == Lsize)
        break;
    }

    // sort best_L_nodes based on distance of each point to node_coords
    std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
    unsigned k = 0;
    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (k < l) {
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;
        if (!(best_L_nodes[k].id == start && num_frozen_pts > 0 &&
              !ret_frozen)) {
          if (!search_invocation) {
            expanded_nodes_info.emplace_back(best_L_nodes[k]);
            expanded_nodes_ids.insert(n);
          }
        }
        des.clear();
        if (_dynamic_index && version == 0) {
          LockGuard guard(_locks[n]);
          for (unsigned m = 0; m < final_graph[n].size(); m++) {
            if (final_graph[n][m] >= max_points + num_frozen_pts) {
              throw diskann::ANNException(std::string("Wrong id found") +
                                              std::to_string(final_graph[n][m]),
                                          -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            des.emplace_back(final_graph[n][m]);
          }
        } else {
          for (unsigned m = 0; m < final_graph[n].size(); m++) {
            if (final_graph[n][m] >= max_points + num_frozen_pts) {
              throw diskann::ANNException(std::string("Wrong id found") +
                                              std::to_string(final_graph[n][m]),
                                          -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            des.emplace_back(final_graph[n][m]);
          }
        }

        for (unsigned m = 0; m < des.size(); ++m) {
          unsigned id = des[m];
          bool     id_is_missing = fast_iterate ? inserted_into_pool_bs[id] == 0
                                            : inserted_into_pool_rs.find(id) ==
                                                  inserted_into_pool_rs.end();
          if (id_is_missing) {
            if (fast_iterate) {
              inserted_into_pool_bs[id] = 1;
            } else {
              inserted_into_pool_rs.insert(id);
            }
            if ((m + 1) < des.size()) {
              auto nextn = des[m + 1];
              diskann::prefetch_vector(
                  (const char *) data + _aligned_dim * (size_t) nextn,
                  sizeof(T) * _aligned_dim);
            }

            cmps++;
            float dist = _distance->compare(node_coords,
                                            data + _aligned_dim * (size_t) id,
                                            (unsigned) _aligned_dim);

            if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
              continue;

            Neighbor nn(id, dist, true);
            unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < Lsize)
              ++l;
            if (r < nk)
              nk = r;
          }
        }

        if (nk <= k)
          k = nk;
        else
          ++k;
      } else
        k++;
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point(
      const T *node_coords, const unsigned Lsize,
      const std::vector<unsigned> &init_ids,
      std::vector<Neighbor> &      expanded_nodes_info,
      tsl::robin_set<unsigned> &   expanded_nodes_ids,
      std::vector<Neighbor> &best_L_nodes, std::vector<unsigned> &des,
      tsl::robin_set<unsigned> &inserted_into_pool_rs,
      boost::dynamic_bitset<> &inserted_into_pool_bs, bool ret_frozen,
      bool search_invocation, const int version) {
    if (version == 0)
      return iterate_to_fixed_point_internal(
          node_coords, Lsize, init_ids, expanded_nodes_info, expanded_nodes_ids,
          best_L_nodes, des, inserted_into_pool_rs, inserted_into_pool_bs,
          _data, _max_points, _num_frozen_pts, _final_graph, _start, ret_frozen,
          search_invocation, version);
    else {
      size_t num_frozen_points = 0;
      return iterate_to_fixed_point_internal(
          node_coords, Lsize, init_ids, expanded_nodes_info, expanded_nodes_ids,
          best_L_nodes, des, inserted_into_pool_rs, inserted_into_pool_bs,
          _query_data, _max_query_points, num_frozen_points, _query_graph,
          _query_start, ret_frozen, search_invocation, version);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_expanded_nodes(
      const size_t node_id, const unsigned Lindex,
      std::vector<unsigned>     init_ids,
      std::vector<Neighbor> &   expanded_nodes_info,
      tsl::robin_set<unsigned> &expanded_nodes_ids, std::vector<unsigned> &des,
      std::vector<Neighbor> &   best_L_nodes,
      tsl::robin_set<unsigned> &inserted_into_pool_rs,
      boost::dynamic_bitset<> & inserted_into_pool_bs) {
    const T *node_coords = _data + _aligned_dim * node_id;

    if (init_ids.size() == 0)
      init_ids.emplace_back(_start);

    iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info,
                           expanded_nodes_ids, best_L_nodes, des,
                           inserted_into_pool_rs, inserted_into_pool_bs);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_expanded_nodes_internal(
      const size_t node_id, const unsigned Lindex,
      std::vector<unsigned>     init_ids,
      std::vector<Neighbor> &   expanded_nodes_info,
      tsl::robin_set<unsigned> &expanded_nodes_ids, T *data,
      const int version) {
    const T *node_coords = data + _aligned_dim * node_id;

    if (init_ids.size() == 0)
      init_ids.emplace_back(_start);

    std::vector<unsigned> des;
    std::vector<Neighbor> best_L_nodes;
    best_L_nodes.resize(Lindex + 1);
    tsl::robin_set<unsigned> inserted_into_pool_rs;
    boost::dynamic_bitset<>  inserted_into_pool_bs;

    iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info,
                           expanded_nodes_ids, best_L_nodes, des,
                           inserted_into_pool_rs, inserted_into_pool_bs, true,
                           false, version);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_expanded_nodes(
      const size_t node_id, const unsigned Lindex,
      std::vector<unsigned>     init_ids,
      std::vector<Neighbor> &   expanded_nodes_info,
      tsl::robin_set<unsigned> &expanded_nodes_ids, const int version) {
    if (version == 0)
      get_expanded_nodes_internal(node_id, Lindex, init_ids,
                                  expanded_nodes_info, expanded_nodes_ids,
                                  _data, version);
    else {
      get_expanded_nodes_internal(node_id, Lindex, init_ids,
                                  expanded_nodes_info, expanded_nodes_ids,
                                  _query_data, version);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const float alpha, const unsigned degree,
                                    const unsigned         maxc,
                                    std::vector<Neighbor> &result, const int version) {
    auto               pool_size = (_u32) pool.size();
    std::vector<float> occlude_factor(pool_size, 0);
    occlude_list(pool, alpha, degree, maxc, result, occlude_factor, version);
  }

    template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const float alpha, const unsigned degree,
                                    const unsigned         maxc,
                                    std::vector<Neighbor> &result,
                                    std::vector<float> &   occlude_factor, const int version){
    if(version == 0) occlude_list_internal(pool, alpha, degree, maxc, result, occlude_factor, _data);
    else occlude_list_internal(pool, alpha, degree, maxc, result, occlude_factor, _query_data);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list_internal(std::vector<Neighbor> &pool,
                                    const float alpha, const unsigned degree,
                                    const unsigned         maxc,
                                    std::vector<Neighbor> &result,
                                    std::vector<float> &   occlude_factor, T* data) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
      unsigned start = 0;
      float eps = cur_alpha + 0.01f;  // used for MIPS, where we store a value
                                      // of eps in cur_alpha to
      // denote pruned out entries which we can skip in later rounds.
      while (result.size() < degree && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        for (unsigned t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          float djk = _distance->compare(
              data + _aligned_dim * (size_t) pool[t].id,
              data + _aligned_dim * (size_t) p.id, (unsigned) _aligned_dim);
          if (_dist_metric == diskann::Metric::L2 ||
              _dist_metric == diskann::Metric::COSINE) {
            occlude_factor[t] =
                std::max(occlude_factor[t], pool[t].distance / djk);
          } else if (_dist_metric == diskann::Metric::INNER_PRODUCT) {
            // Improvization for flipping max and min dist for MIPS
            float x = -pool[t].distance;
            float y = -djk;
            if (y > cur_alpha * x) {
              occlude_factor[t] = std::max(occlude_factor[t], eps);
            }
          }
        }
        start++;
      }
      cur_alpha *= 1.2;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_neighbors(const unsigned         location,
                                       std::vector<Neighbor> &pool,
                                       std::vector<unsigned> &pruned_list, const int version) {
    prune_neighbors(location, pool, _indexingRange, _indexingMaxC,
                    _indexingAlpha, pruned_list, version);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_neighbors(const unsigned         location,
                                       std::vector<Neighbor> &pool,
                                       const _u32             range,
                                       const _u32  max_candidate_size,
                                       const float alpha,
                                       std::vector<unsigned> &pruned_list, const int version) {
    if (pool.size() == 0) {
      std::stringstream ss;
      ss << "Thread id:" << std::this_thread::get_id()
         << " Pool address: " << &pool << std::endl;
      std::cout << ss.str();
      throw diskann::ANNException("Pool passed to prune_neighbors is empty",
                                  -1);
    }

    _max_observed_degree = (std::max)(_max_observed_degree, range);

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list(pool, alpha, range, max_candidate_size, result,
                 occlude_factor, version);

    // Add all the nodes in result into a variable called cut_graph
    // So this contains all the neighbors of id location
    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      if (iter.id != location)
        pruned_list.emplace_back(iter.id);
    }

    if (_saturate_graph && alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if ((std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) ==
             pruned_list.end()) &&
            pool[i].id != location)
          pruned_list.emplace_back(pool[i].id);
      }
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::batch_inter_insert_internal(
      unsigned n, const std::vector<unsigned> &pruned_list, const _u32 range,
      std::vector<unsigned> &need_to_sync, size_t &max_points,
      size_t &num_frozen_pts, std::vector<std::mutex> &locks,
      std::vector<std::vector<unsigned>> &final_graph) {
    // assert(!src_pool.empty());

    for (auto des : pruned_list) {
      if (des == n)
        continue;
      // des.id is the id of the neighbors of n
      assert(des >= 0 && des < max_points + num_frozen_pts);
      if (des > max_points)
        diskann::cout << "error. " << des << " exceeds max_pts" << std::endl;
      // des_pool contains the neighbors of the neighbors of n

      {
        LockGuard guard(locks[des]);
        if (std::find(final_graph[des].begin(), final_graph[des].end(), n) ==
            final_graph[des].end()) {
          final_graph[des].push_back(n);
          if (final_graph[des].size() > (unsigned) (range * GRAPH_SLACK_FACTOR))
            need_to_sync[des] = 1;
        }
      }  // des lock is released by this point
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::batch_inter_insert(
      unsigned n, const std::vector<unsigned> &pruned_list,
      std::vector<unsigned> &need_to_sync, const int version) {
    if (version == 0)
      batch_inter_insert_internal(n, pruned_list, _indexingRange, need_to_sync,
                                  _max_points, _num_frozen_pts, _locks,
                                  _final_graph);
    else {
      size_t num_frozen_points = 0;
      batch_inter_insert_internal(n, pruned_list, _indexingRange, need_to_sync,
                                  _max_query_points, num_frozen_points,
                                  _query_locks, _query_graph);
    }
  }

  template<typename T, typename TagT>
  std::pair<int, std::vector<double>> Index<T, TagT>::inter_insert(unsigned               n,
                                    std::vector<unsigned> &pruned_list,
                                    const _u32 range, bool update_in_graph) {
    std::vector<double> stitch_times(3, 0.0);

    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    int num_pruned = 0;

    for (auto des : src_pool) {
      // des.id is the id of the neighbors of n
      assert(des >= 0 && des < _max_points + _num_frozen_pts);
      // des_pool contains the neighbors of the neighbors of n
      auto &                des_pool = _final_graph[des];
      std::vector<unsigned> copy_of_neighbors;
      bool                  prune_needed = false;
      {
        LockGuard guard(_locks[des]);
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < (_u64)(GRAPH_SLACK_FACTOR * range)) {
            des_pool.emplace_back(n);
            if (update_in_graph) {
              LockGuard guard(_locks_in[n]);
              _in_graph[n].emplace_back(des);
            }
            prune_needed = false;
          } else {
            copy_of_neighbors = des_pool;
            prune_needed = true;
            num_pruned++;
          }
        }
      }  // des lock is released by this point

      if (prune_needed) {
        copy_of_neighbors.push_back(n);
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);

        size_t reserveSize =
            (size_t)(std::ceil(1.05 * GRAPH_SLACK_FACTOR * range));
        dummy_visited.reserve(reserveSize);
        dummy_pool.reserve(reserveSize);

        for (auto cur_nbr : copy_of_neighbors) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != des) {
            float dist =
                _distance->compare(_data + _aligned_dim * (size_t) des,
                                   _data + _aligned_dim * (size_t) cur_nbr,
                                   (unsigned) _aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        std::vector<unsigned> new_out_neighbors;
        prune_neighbors(des, dummy_pool, new_out_neighbors);
        {
          LockGuard guard(_locks[des]);
          // updating in_graph of out-neighbors of des
          if (update_in_graph) {
            for (auto out_nbr : _final_graph[des]) {
              {
                LockGuard guard(_locks_in[out_nbr]);
                for (unsigned i = 0; i < _in_graph[out_nbr].size(); i++) {
                  if (_in_graph[out_nbr][i] == des) {
                    _in_graph[out_nbr].erase(_in_graph[out_nbr].begin() + i);
                    break;
                  }
                }
              }
            }
          }

          _final_graph[des].clear();
          for (auto new_nbr : new_out_neighbors) {
            _final_graph[des].emplace_back(new_nbr);
            if (update_in_graph) {
              LockGuard guard(_locks_in[new_nbr]);
              _in_graph[new_nbr].emplace_back(des);
            }
          }
        }
        if(_queries_present){
          auto times = insert_and_stitch(des);
          stitch_times[0] += times[0];
          stitch_times[1] += times[1];
          stitch_times[2] += times[2];
        }
      }
    }
    return std::make_pair(num_pruned, stitch_times);
  }

  template<typename T, typename TagT>
  std::pair<int, std::vector<double>> Index<T, TagT>::inter_insert(unsigned               n,
                                    std::vector<unsigned> &pruned_list,
                                    bool                   update_in_graph) {
    return inter_insert(n, pruned_list, _indexingRange, update_in_graph);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::link_internal(
      Parameters &parameters, std::vector<std::vector<unsigned>> &final_graph,
      size_t &nd, unsigned &start, T *data, size_t &num_frozen_pts,
      size_t &max_points, const int version) {
    unsigned num_threads = parameters.Get<unsigned>("num_threads");
    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    uint32_t num_syncs =
        (unsigned) DIV_ROUND_UP(nd + num_frozen_pts, (64 * 64));
    if (num_syncs < 40)
      num_syncs = 40;
    diskann::cout << "Number of syncs: " << num_syncs << std::endl;

    // should only be able to saturate graph if RobustStitch will not be used
    if (!_queries_present)
      _saturate_graph = parameters.Get<bool>("saturate_graph");

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    _indexingQueueSize = parameters.Get<unsigned>("L");  // Search list size
    _indexingRange = parameters.Get<unsigned>("R");
    _indexingMaxC = parameters.Get<unsigned>("C");
    const float last_round_alpha = parameters.Get<float>("alpha");
    unsigned    L = _indexingQueueSize;

    std::vector<unsigned> Lvec;
    Lvec.push_back(L);
    Lvec.push_back(L);
    const unsigned NUM_RNDS = 2;
    _indexingAlpha = 1.0f;

    /* visit_order is a vector that is initialized to the entire graph */
    std::vector<unsigned>          visit_order;
    std::vector<diskann::Neighbor> pool, tmp;
    tsl::robin_set<unsigned>       visited;
    visit_order.reserve(nd + num_frozen_pts);
    for (unsigned i = 0; i < (unsigned) nd; i++) {
      visit_order.emplace_back(i);
    }

    if (num_frozen_pts > 0 && version == 0)
      visit_order.emplace_back((unsigned) max_points);

    // std::cout << "at start point" << std::endl;
    // if there are frozen points, the first such one is set to be the _start
    if (_num_frozen_pts > 0 && version == 0)
      start = (unsigned) max_points;
    else
      start = calculate_entry_point(version);

    std::cout << "Start id " << start << std::endl;

    if (_support_eager_delete && version == 0) {
      _in_graph.reserve(max_points + num_frozen_pts);
      _in_graph.resize(max_points + num_frozen_pts);
    }

    for (uint64_t p = 0; p < nd; p++) {
      final_graph[p].reserve(
          (size_t)(std::ceil(_indexingRange * GRAPH_SLACK_FACTOR * 1.05)));
    }

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // creating a initial list to begin the search process. it has _start and
    // random other nodes
    std::set<unsigned> unique_start_points;
    unique_start_points.insert(start);

    std::vector<unsigned> init_ids;
    for (auto pt : unique_start_points)
      init_ids.emplace_back(pt);

    diskann::Timer link_timer;
    for (uint32_t rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
      L = Lvec[rnd_no];

      if (rnd_no == NUM_RNDS - 1) {
        if (last_round_alpha > 1)
          _indexingAlpha = last_round_alpha;
      }

      double   sync_time = 0, total_sync_time = 0;
      double   inter_time = 0, total_inter_time = 0;
      size_t   inter_count = 0, total_inter_count = 0;
      unsigned progress_counter = 0;

      size_t round_size = DIV_ROUND_UP(nd, num_syncs);  // size of each batch
      std::vector<unsigned> need_to_sync(max_points + num_frozen_pts, 0);

      std::vector<std::vector<unsigned>> pruned_list_vector(round_size);

      for (uint32_t sync_num = 0; sync_num < num_syncs; sync_num++) {
        size_t start_id = sync_num * round_size;
        size_t end_id =
            (std::min)(nd + num_frozen_pts, (sync_num + 1) * round_size);

        auto s = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff;

#pragma omp parallel for schedule(dynamic)
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          auto                     node = visit_order[node_ctr];
          size_t                   node_offset = node_ctr - start_id;
          tsl::robin_set<unsigned> visited;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          // get nearest neighbors of n in tmp. pool contains all the
          // points that were checked along with their distance from
          // n. visited contains all the points visited, just the ids
          std::vector<Neighbor> pool;
          pool.reserve(L * 2);
          visited.reserve(L * 2);
          get_expanded_nodes(node, L, init_ids, pool, visited, version);
          // check the neighbors of the query that are not part of
          // visited, check their distance to the query, and add it to
          if (!final_graph[node].empty())
            for (auto id : final_graph[node]) {
              if (visited.find(id) == visited.end() && id != node) {
                float dist = _distance->compare(
                    data + _aligned_dim * (size_t) node,
                    data + _aligned_dim * (size_t) id, (unsigned) _aligned_dim);
                pool.emplace_back(Neighbor(id, dist, true));
                visited.insert(id);
              }
            }
          prune_neighbors(node, pool, pruned_list, version);
        }
        diff = std::chrono::high_resolution_clock::now() - s;
        sync_time += diff.count();

// prune_neighbors will check pool, and remove some of the points and
// create a cut_graph, which contains neighbors for point n
#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          _u64                   node = visit_order[node_ctr];
          size_t                 node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          final_graph[node].clear();
          for (auto id : pruned_list) {
            final_graph[node].emplace_back(id);
          }
        }
        s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = start_id; node_ctr < (_s64) end_id; ++node_ctr) {
          auto                   node = visit_order[node_ctr];
          _u64                   node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          batch_inter_insert(node, pruned_list, need_to_sync, version);
          pruned_list.clear();
          pruned_list.shrink_to_fit();
        }

#pragma omp parallel for schedule(dynamic, 65536)
        for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size());
             node_ctr++) {
          auto node = visit_order[node_ctr];
          if (need_to_sync[node] != 0) {
            // std::cout << "sync step" << std::endl;
            need_to_sync[node] = 0;
            inter_count++;
            tsl::robin_set<unsigned> dummy_visited(0);
            std::vector<Neighbor>    dummy_pool(0);
            std::vector<unsigned>    new_out_neighbors;

            for (auto cur_nbr : final_graph[node]) {
              if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                  cur_nbr != node) {
                float dist =
                    _distance->compare(data + _aligned_dim * (size_t) node,
                                       data + _aligned_dim * (size_t) cur_nbr,
                                       (unsigned) _aligned_dim);
                dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
                dummy_visited.insert(cur_nbr);
              }
            }
            prune_neighbors(node, dummy_pool, new_out_neighbors, version);

            final_graph[node].clear();
            for (auto id : new_out_neighbors)
              final_graph[node].emplace_back(id);
          }
        }

        diff = std::chrono::high_resolution_clock::now() - s;
        inter_time += diff.count();

        if ((sync_num * 100) / num_syncs > progress_counter) {
          diskann::cout.precision(4);
          diskann::cout << "Completed  (round: " << rnd_no
                        << ", sync: " << sync_num << "/" << num_syncs
                        << " with L " << L << ")"
                        << " sync_time: " << sync_time << "s"
                        << "; inter_time: " << inter_time << "s" << std::endl;

          total_sync_time += sync_time;
          total_inter_time += inter_time;
          total_inter_count += inter_count;
          sync_time = 0;
          inter_time = 0;
          inter_count = 0;
          progress_counter += 5;
        }
      }

// Gopal. Splittng nsg_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#ifdef USE_TCMALLOC
      MallocExtension::instance()->ReleaseFreeMemory();
#endif
      if (nd > 0) {
        diskann::cout << "Completed Pass " << rnd_no << " of data using L=" << L
                      << " and alpha=" << parameters.Get<float>("alpha")
                      << ". Stats: ";
        diskann::cout << "search+prune_time=" << total_sync_time
                      << "s, inter_time=" << total_inter_time
                      << "s, inter_count=" << total_inter_count << std::endl;
      }
    }
    if (nd > 0) {
      diskann::cout << "Link time: "
                    << ((double) link_timer.elapsed() / (double) 1000000) << "s"
                    << std::endl;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::link(Parameters &parameters, const int version) {
    if (version == 0)
      link_internal(parameters, _final_graph, _nd, _start, _data,
                    _num_frozen_pts, _max_points, version);
    else {
      size_t num_frozen_points = 0;
      link_internal(parameters, _query_graph, _max_query_points, _query_start,
                    _query_data, num_frozen_points, _max_query_points, version);
    }
  }

  // find the nearest neighbors in _final_graph of every point in _query_graph
  // and add them to the _query_nn field
  template<typename T, typename TagT>
  std::vector<double> Index<T, TagT>::insert_and_stitch(unsigned location) {

    std::vector<double> times(3);

    auto                  L = _indexingQueueSize/2;
    std::vector<unsigned> init_ids;
    init_ids.emplace_back(_query_start);
    tsl::robin_set<unsigned> visited;
    std::vector<Neighbor>    pool;
    pool.reserve(L * 2);
    visited.reserve(L * 2);
    const T *node_coords = _data + _aligned_dim * location;

    std::vector<unsigned> des;
    std::vector<Neighbor> best_L_nodes;
    best_L_nodes.resize(L + 1);
    tsl::robin_set<unsigned> inserted_into_pool_rs;
    boost::dynamic_bitset<>  inserted_into_pool_bs;

    int version = 1; //flag to search in query graph

    diskann::Timer search_timer;

    iterate_to_fixed_point(node_coords, L, init_ids, pool, visited,
                           best_L_nodes, des, inserted_into_pool_rs,
                           inserted_into_pool_bs, true, false, version);

    double search_time = search_timer.elapsed() / 1000000.0;
    times[0] = search_time;
    // find the notes in _query_nn that should be updated to include location

    diskann::Timer update_timer;

    std::vector<unsigned> to_stitch;
    for(auto nbor : best_L_nodes){
      unsigned nbh = nbor.id; 
      { 
        assert(nbh <= _max_query_points);
        LockGuard guard(_query_nn_locks[nbh]);
        float dist =
                    _distance->compare(_query_data + _aligned_dim * (size_t) nbh,
                                       _data + _aligned_dim * (size_t) location,
                                       (unsigned) _aligned_dim);
        if((_query_nn[nbh][_query_nn[nbh].size()-1]).distance > dist){
          size_t k = _query_nn[nbh].size();
          int result = InsertIntoPool(_query_nn[nbh].data(), k, Neighbor(location, dist, true));
          if(result != (int) k+1){
            to_stitch.push_back(nbh);
          }
        }
      }
    } 

    double update_time = update_timer.elapsed() / 1000000.0;
    times[1] = update_time;

    // robustStitch location based on the nodes in to_stitch

    diskann::Timer stitch_timer; 

    size_t total_capacity = _indexingRange - _final_graph[location].size();
    size_t capacity;
    if(to_stitch.size() == 0) capacity = 0;
    else capacity = std::floor(total_capacity/ (double) to_stitch.size());
    for(auto query_ctr : to_stitch){
      int c = 0; //number of nodes added to _final_graph[nbh]
      int j = 0; //iter count over _query_nn[query_ctr]
      { 
        LockGuard guard(_locks[location]);
        while (c < (int) capacity &&
               j < (int) _query_nn[query_ctr].size() &&
               _final_graph[location].size() < _indexingRange) {
          auto candidate = _query_nn[query_ctr][j].id;
          assert(candidate < _max_points + _num_frozen_pts);
          if (candidate != location &&
              (std::find(_final_graph[location].begin(), _final_graph[location].end(),
                         candidate) == _final_graph[location].end())) {  
            _final_graph[location].push_back(candidate);
            c++;
          }
          j++;
        }
      } 
    }
    double stitch_time = stitch_timer.elapsed() / 1000000.0;
    times[2] = stitch_time;
    return times; 
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::erase_query_nn(tsl::robin_set<unsigned> delete_set){
#pragma omp parallel for schedule(dynamic)
    for (_s64 node_ctr = (_s64) 0; node_ctr < (_s64) _max_query_points;
         ++node_ctr) {
      std::vector<Neighbor> new_pool;
      {
        LockGuard guard(_query_nn_locks[node_ctr]);
        for(auto nbor : _query_nn[node_ctr]){
          if(delete_set.find(nbor.id) == delete_set.end()){
            new_pool.push_back(nbor);
          }
        }
        if(new_pool.size() != _indexingRange){
          while(new_pool.size() != _indexingRange){
            new_pool.push_back(Neighbor(0, std::numeric_limits<float>::max(), false));
          }
          _query_nn[node_ctr].clear();
          for(size_t i=0; i<_indexingRange; i++){
            _query_nn[node_ctr][i] = new_pool[i];
          }
        }
      }
    }
  }


  // find the nearest neighbors in _final_graph of every point in _query_graph
  // and add them to the _query_nn field
  template<typename T, typename TagT>
  void Index<T, TagT>::populate_query_nn() {
    std::cout << "Finding nearest neighbors of query nodes... " << std::endl; 

    diskann::Timer query_timer;

#pragma omp parallel for schedule(dynamic)
      for (_s64 node_ctr = (_s64) 0; node_ctr < (_s64) _max_query_points;
           ++node_ctr) {
        _query_nn[node_ctr].reserve(_indexingRange*2+1);
      }

    auto                  L = _indexingQueueSize*2;
    std::vector<unsigned> init_ids;
    init_ids.emplace_back(_start);
#pragma omp parallel for schedule(dynamic)
    for (_s64 node_ctr = (_s64) 0; node_ctr < (_s64) _max_query_points;
         ++node_ctr) {
      tsl::robin_set<unsigned> visited;
      std::vector<Neighbor>    pool;
      pool.reserve(L * 2);
      visited.reserve(L * 2);
      const T *node_coords = _query_data + _aligned_dim * node_ctr;

      std::vector<unsigned> des;
      std::vector<Neighbor> best_L_nodes;
      best_L_nodes.resize(L + 1);
      tsl::robin_set<unsigned> inserted_into_pool_rs;
      boost::dynamic_bitset<>  inserted_into_pool_bs;

      iterate_to_fixed_point(node_coords, L, init_ids, pool, visited,
                             best_L_nodes, des, inserted_into_pool_rs,
                             inserted_into_pool_bs);

      std::sort(pool.begin(), pool.end());
      size_t k = pool.size();
      if(k < _indexingRange*2){
        pool.resize(_indexingRange*2);
        for(size_t j=k; j<_indexingRange*2; j++){
          pool[j].distance = std::numeric_limits<float>::max();
        }
      }
      _query_nn[node_ctr] = pool;
    }

    double seconds = query_timer.elapsed() / 1000000.0;
    std::cout << "Time for finding query nearest neighbors: "
              << seconds << "s" << std::endl;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::robust_stitch(tsl::robin_set<unsigned> pruned_nodes) {
    std::cout << "Beginning RobustStitch routine... " << std::endl;
    std::unordered_map<unsigned, int> query_indegree;
    std::unordered_map<unsigned, int> per_node_capacity;

    for(unsigned node : pruned_nodes){
      query_indegree[node] = 0;
      per_node_capacity[node] = 0;
    }

    diskann::Timer stitch_timer;
    //calculate the in-degrees of every node in pruned_nodes
    for (_s64 query_ctr = 0; query_ctr < (_s64) _max_query_points;
         query_ctr++) { 
      //need a lock on _query_nn[query_ctr] here unless we totally ditch concurrent updates to _query_nn
      for (auto nbh : _query_nn[query_ctr]) {
        if(pruned_nodes.find(nbh.id) != pruned_nodes.end()) query_indegree[nbh.id]++;
      }
    }

//calculate the capacity of every node in pruned_nodes
#pragma omp parallel for schedule(dynamic, 65536)
    for (_s64 node = 0; node < (_s64) _nd;
         node++) {
      if(pruned_nodes.find(node) != pruned_nodes.end()){
        int total_capacity;
      {
        LockGuard guard(_locks[node]);
        total_capacity = _indexingRange - (int) (_final_graph[node].size());
      }
      if (query_indegree[node] != 0) {
        int per_query_capacity =
            std::floor(total_capacity / ((double) query_indegree[node]));
        per_node_capacity[node] = per_query_capacity;
      }
      }
    }

    std::cout << "Stitching nodes now..." << std::endl;


//stitch every node in pruned_nodes with nonzero capacity
#pragma omp parallel for schedule(dynamic, 65536)
    for (_s64 query_ctr = 0; query_ctr < (_s64) _max_query_points;
         query_ctr++) {
      for (auto nbor : _query_nn[query_ctr]) {
        unsigned nbh = nbor.id;
        if (per_node_capacity[nbh] > 0 && (pruned_nodes.find(nbh) != pruned_nodes.end())){
          int c = 0; //number of nodes added to _final_graph[nbh]
          int j = 0; //iter count over _query_nn[query_ctr]
          {
            LockGuard guard(_locks[nbh]);
            while (c < per_node_capacity[nbh] &&
                   j < (int) _query_nn[query_ctr].size() &&
                   _final_graph[nbh].size() < _indexingRange) {
              auto candidate = _query_nn[query_ctr][j].id;
              if (candidate != nbh &&
                  (std::find(_final_graph[nbh].begin(), _final_graph[nbh].end(),
                             candidate) == _final_graph[nbh].end())) {
                _final_graph[nbh].push_back(candidate);
                c++;
              }
              j++;
            }
          }
        }
      }
    }

    double seconds = stitch_timer.elapsed() / 1000000.0;
    std::cout << "RobustStitch time: "
              << seconds << "s" << std::endl;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::robust_stitch() {
    std::cout << "Beginning RobustStitch routine... " << std::endl;
    std::vector<int> changed(_nd);

    std::vector<int> query_indegree(_nd, 0);
    std::vector<int> per_node_capacity(_nd, 0);

    diskann::Timer stitch_timer;

    for (_s64 query_ctr = 0; query_ctr < (_s64) _max_query_points;
         query_ctr++) {
      for (auto nbh : _query_nn[query_ctr]) {
        query_indegree[nbh.id]++;
      }
    }


#pragma omp parallel for schedule(dynamic, 65536)
    for (_s64 node_ctr = 0; node_ctr < (_s64) _nd; node_ctr++) {
      auto node = node_ctr;

      int total_capacity = _indexingRange - (int) (_final_graph[node].size());
      if (query_indegree[node] != 0) {
        int per_query_capacity =
            std::floor(total_capacity / ((double) query_indegree[node]));
        per_node_capacity[node] = per_query_capacity;
      }
    }

    std::cout << "Stitching nodes now..." << std::endl;

    for (_s64 query_ctr = 0; query_ctr < (_s64) _max_query_points;
         query_ctr++) {
      for (auto nbor : _query_nn[query_ctr]) {
        unsigned nbh = nbor.id;
        if (per_node_capacity[nbh] > 0){
          changed[nbh] = 1;
          int c = 0; //number of nodes added to _final_graph[nbh]
          int j = 0; //iter count over _query_nn[query_ctr]
          while (c < per_node_capacity[nbh] &&
                 j < (int) _query_nn[query_ctr].size() &&
                 _final_graph[nbh].size() < _indexingRange) {
            auto candidate = _query_nn[query_ctr][j].id;
            if (candidate != nbh &&
                (std::find(_final_graph[nbh].begin(), _final_graph[nbh].end(),
                           candidate) == _final_graph[nbh].end())) {
              _final_graph[nbh].push_back(candidate);
              c++;
            }
            j++;
          }
        }
      }
    }

    double seconds = stitch_timer.elapsed() / 1000000.0;
    std::cout << "RobustStitch time: "
              << seconds << "s" << std::endl;
    std::cout << "Number of nodes stitched: "
              << std::accumulate(changed.begin(), changed.end(), 0)
              << std::endl;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_all_nbrs(const Parameters &parameters) {
    const unsigned range = parameters.Get<unsigned>("R");

    diskann::Timer timer;
#pragma omp parallel for
    for (_s64 node = 0; node < (_s64)(_max_points + _num_frozen_pts); node++) {
      if ((size_t) node < _nd || (size_t) node == _max_points) {
        if (_final_graph[node].size() > range) {
          tsl::robin_set<unsigned> dummy_visited(0);
          std::vector<Neighbor>    dummy_pool(0);
          std::vector<unsigned>    new_out_neighbors;

          for (auto cur_nbr : _final_graph[node]) {
            if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                cur_nbr != node) {
              float dist =
                  _distance->compare(_data + _aligned_dim * (size_t) node,
                                     _data + _aligned_dim * (size_t) cur_nbr,
                                     (unsigned) _aligned_dim);
              dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
              dummy_visited.insert(cur_nbr);
            }
          }
          prune_neighbors((_u32) node, dummy_pool, new_out_neighbors);

          _final_graph[node].clear();
          for (auto id : new_out_neighbors)
            _final_graph[node].emplace_back(id);
        }
      }
    }

    diskann::cout << "Prune time : " << timer.elapsed() / 1000 << "ms"
                  << std::endl;
    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < (_nd + _num_frozen_pts); i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      diskann::cout << "Index built with degree: max:" << max << "  avg:"
                    << (float) total / (float) (_nd + _num_frozen_pts)
                    << "  min:" << min << "  count(deg<2):" << cnt << std::endl;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build_with_data_populated(
      Parameters &parameters, const std::vector<TagT> &tags) {
    diskann::cout << "Starting index build with " << _nd << " points... "
                  << std::endl;

    if (_nd < 1)
      throw ANNException("Error: Trying to build an index with 0 points", -1,
                         __FUNCSIG__, __FILE__, __LINE__);

    if (_enable_tags && tags.size() != _nd) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << _nd << " points from file,"
             << "but tags vector is of size " << tags.size() << "."
             << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (_enable_tags) {
      for (size_t i = 0; i < tags.size(); ++i) {
        _tag_to_location[tags[i]] = (unsigned) i;
        _location_to_tag[(unsigned) i] = tags[i];
      }
    }

    generate_frozen_point();
    link(parameters);

    if (_support_eager_delete) {
      update_in_graph();  // copying values to in_graph
    }

    if(!_queries_present){
      size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
      for (size_t i = 0; i < _nd; i++) {
        auto &pool = _final_graph[i];
        max = (std::max)(max, pool.size());
        min = (std::min)(min, pool.size());
        total += pool.size();
        if (pool.size() < 2)
          cnt++;
      }
      diskann::cout << "Index built with degree: max:" << max
                    << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                    << "  min:" << min << "  count(deg<2):" << cnt << std::endl;

      _max_observed_degree = (std::max)((unsigned) max, _max_observed_degree);
      _has_built = true;
    }
    

    if (_queries_present) {
      std::cout << "Beginning query index build" << std::endl;
      int version = 1;
      link(parameters, version);

      size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
      for (size_t i = 0; i < _max_query_points; i++) {
        auto &pool = _query_graph[i];
        max = (std::max)(max, pool.size());
        min = (std::min)(min, pool.size());
        total += pool.size();
        if (pool.size() < 2)
          cnt++;
      }
      diskann::cout << "Query index built with degree: max:" << max
                    << "  avg:" << (float) total / (float) (_max_query_points)
                    << "  min:" << min << "  count(deg<2):" << cnt << std::endl;

      _max_observed_qdegree = (std::max)((unsigned) max, _max_observed_qdegree);


      // tsl::robin_set<unsigned> dummy_set; 
      // bool from_empty = true;
      populate_query_nn();
      robust_stitch();

      max = 0; min = SIZE_MAX; total = 0; cnt = 0;
      for (size_t i = 0; i < _nd; i++) {
        auto &pool = _final_graph[i];
        max = (std::max)(max, pool.size());
        min = (std::min)(min, pool.size());
        total += pool.size();
        if (pool.size() < 2)
          cnt++;
      }
      diskann::cout << "Index built with degree: max:" << max
                    << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                    << "  min:" << min << "  count(deg<2):" << cnt << std::endl;

      _max_observed_degree = (std::max)((unsigned) max, _max_observed_degree);
      _has_built = true;
    }

  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const T *data, const size_t num_points_to_load,
                             Parameters &             parameters,
                             const std::vector<TagT> &tags, const T *query_data,
                             const size_t num_query_points_to_load) {
    _nd = num_points_to_load;

    memcpy((char *) _data, (char *) data,
           _aligned_dim * num_points_to_load * sizeof(T));
    if (query_data != nullptr) {
      if (_queries_present) {
        // memcpy((char *) (_data + _aligned_dim * num_points_to_load),
        //        (char *) query_data,
        //        _aligned_dim * num_query_points_to_load * sizeof(T));
        memcpy((char *) _query_data, (char *) query_data,
               _aligned_dim * num_query_points_to_load * sizeof(T));
      } else {
        std::stringstream stream;
        stream << "ERROR: query file provided but flags not enabled"
               << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
    }

    if (_normalize_vecs) {
      for (uint64_t i = 0; i < num_points_to_load; i++) {
        normalize(_data + _aligned_dim * i, _aligned_dim);
      }
    }

    build_with_data_populated(parameters, tags);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char *             filename,
                             const size_t             num_points_to_load,
                             Parameters &             parameters,
                             const std::vector<TagT> &tags,
                             const char *             query_filename,
                             const size_t num_query_points_to_load) {
    if (!file_exists(filename)) {
      diskann::cerr << "Data file " << filename
                    << " does not exist!!! Exiting...." << std::endl;
      std::stringstream stream;
      stream << "Data file " << filename << " does not exist." << std::endl;
      diskann::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      throw diskann::ANNException("Can not build with an empty file", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::get_bin_metadata(filename, file_num_points, file_dim);
    if (file_num_points > _max_points) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << num_points_to_load
             << " points and file has " << file_num_points << " points, but "
             << "index can support only " << _max_points
             << " points as specified in constructor." << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (num_points_to_load > file_num_points) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << num_points_to_load
             << " points and file has only " << file_num_points << " points."
             << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (file_dim != _dim) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << _dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    size_t qfile_num_points, qfile_dim;
    qfile_num_points = 0;

    if (query_filename != nullptr && _queries_present) {
      if (!file_exists(query_filename)) {
        diskann::cerr << "Query data file " << filename
                      << " does not exist!!! Exiting...." << std::endl;
        std::stringstream stream;
        stream << "Query data file " << query_filename << " does not exist."
               << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }

      diskann::get_bin_metadata(query_filename, qfile_num_points, qfile_dim);
      if (qfile_num_points > _max_query_points) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << qfile_num_points
               << " points and files have " << qfile_num_points
               << " points, but "
               << "index can support only " << _max_query_points
               << " points as specified in constructor." << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }

      if (num_query_points_to_load > qfile_num_points) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_query_points_to_load
               << " query points and file has only " << qfile_num_points
               << " query points." << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }

      if (qfile_dim != _dim) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but query file has " << qfile_dim << " dimension."
               << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
    } else if (query_filename != nullptr && !_queries_present) {
      std::stringstream stream;
      stream << "ERROR: query file provided but flags not enabled" << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    copy_aligned_data_from_file<T>(filename, _data, file_num_points, file_dim,
                                   _aligned_dim);

    if (_queries_present)
      copy_aligned_data_from_file<T>(query_filename, _query_data,
                                     qfile_num_points, file_dim, _aligned_dim);

    if (_normalize_vecs) {
      for (uint64_t i = 0; i < file_num_points; i++) {
        normalize(_data + _aligned_dim * i, _aligned_dim);
      }
    }

    diskann::cout << "Using only first " << num_points_to_load
                  << " from file.. " << std::endl;

    _nd = num_points_to_load;
    build_with_data_populated(parameters, tags);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char * filename,
                             const size_t num_points_to_load,
                             Parameters &parameters, const char *tag_filename,
                             const char * query_filename,
                             const size_t num_query_points_to_load) {
    std::vector<TagT> tags;

    if (_enable_tags) {
      if (tag_filename == nullptr) {
        throw ANNException("Tag filename is null, while _enable_tags is set",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
      } else {
        if (file_exists(tag_filename)) {
          diskann::cout << "Loading tags from " << tag_filename
                        << " for vamana index build" << std::endl;
          TagT * tag_data = nullptr;
          size_t npts, ndim;
          diskann::load_bin(tag_filename, tag_data, npts, ndim);
          if (npts < num_points_to_load) {
            std::stringstream sstream;
            sstream << "Loaded " << npts
                    << " tags, insufficient to populate tags for "
                    << num_points_to_load << "  points to load";
            throw diskann::ANNException(sstream.str(), -1, __FUNCSIG__,
                                        __FILE__, __LINE__);
          }
          for (size_t i = 0; i < num_points_to_load; i++) {
            tags.push_back(tag_data[i]);
          }
          delete[] tag_data;
        } else {
          throw diskann::ANNException(
              std::string("Tag file") + tag_filename + " does not exist", -1,
              __FUNCSIG__, __FILE__, __LINE__);
        }
      }
    }
    build(filename, num_points_to_load, parameters, tags, query_filename,
          num_query_points_to_load);
  }

  template<typename T, typename TagT>
  template<typename IdType>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *      query,
                                                       const size_t   K,
                                                       const unsigned L,
                                                       IdType *       indices,
                                                       float *distances) {
    ScratchStoreManager<T> manager(_query_scratch);
    auto                   scratch = manager.scratch_space();

    return search_impl(query, K, L, indices, distances, scratch);
  }

  template<typename T, typename TagT>
  template<typename IdType>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search_impl(
      const T *query, const size_t K, const unsigned L, IdType *indices,
      float *distances, InMemQueryScratch<T> &scratch) {
    std::vector<Neighbor> &   expanded_nodes_info = scratch.pool();
    tsl::robin_set<unsigned> &expanded_nodes_ids = scratch.visited();
    std::vector<unsigned> &   des = scratch.des();
    std::vector<Neighbor>     best_L_nodes = scratch.best_l_nodes();
    tsl::robin_set<unsigned> &inserted_into_pool_rs =
        scratch.inserted_into_pool_rs();
    boost::dynamic_bitset<> &inserted_into_pool_bs =
        scratch.inserted_into_pool_bs();

    std::vector<unsigned> init_ids;

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_start);
    }
    T *aligned_query = scratch.aligned_query;
    memcpy(aligned_query, query, _dim * sizeof(T));

    if (_normalize_vecs) {
      normalize((float *) aligned_query, _dim);
    }

    auto retval = iterate_to_fixed_point(
        aligned_query, L, init_ids, expanded_nodes_info, expanded_nodes_ids,
        best_L_nodes, des, inserted_into_pool_rs, inserted_into_pool_bs, true,
        true);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      if (it.id < _max_points) {
        indices[pos] =
            (IdType) it.id;  // safe because our indices are always uint32_t and
                             // IDType will be uint32_t or uint64_t
        if (distances != nullptr) {
#ifdef EXEC_ENV_OLS
          distances[pos] = it.distance;  // DLVS expects negative distances
#else
          distances[pos] = _dist_metric == diskann::Metric::INNER_PRODUCT
                               ? -1 * it.distance
                               : it.distance;
#endif
        }
        pos++;
      }
      if (pos == K)
        break;
    }
    return retval;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const uint64_t K,
                                          const unsigned L, TagT *tags,
                                          float *           distances,
                                          std::vector<T *> &res_vectors) {
    ScratchStoreManager<T> manager(_query_scratch);
    auto                   scratch = manager.scratch_space();

    if (L > scratch.search_l) {
      scratch.resize_for_query(L);
      diskann::cout << "Expanding query scratch_space. Was created with Lsize: "
                    << scratch.search_l << " but search L is: " << L
                    << std::endl;
    }
    _u32 * indices = scratch.indices;
    float *dist_interim = scratch.interim_dists;
    search_impl(query, L, L, indices, dist_interim, scratch);

    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t                                    pos = 0;

    for (int i = 0; i < (int) L; ++i)
      if (_location_to_tag.find(indices[i]) != _location_to_tag.end()) {
        tags[pos] = _location_to_tag[indices[i]];

        if (res_vectors.size() > 0) {
          memcpy(res_vectors[pos], _data + ((size_t) indices[i]) * _aligned_dim,
                 _dim * sizeof(T));
        }

        if (distances != nullptr) {
#ifdef EXEC_ENV_OLS
          distances[pos] = dist_interim[i];  // DLVS expects negative distances
#else
          distances[pos] = _dist_metric == INNER_PRODUCT ? -1 * dist_interim[i]
                                                         : dist_interim[i];
#endif
        }
        pos++;
        // If res_vectors.size() < k, clip at the value.
        if (pos == K || pos == res_vectors.size())
          break;
      }

    return pos;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::get_num_points() {
    return _nd;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::get_max_points() {
    return _max_points;
  }

  /*************************************************
   *      Support for Incremental Update
   *************************************************/

  template<typename T, typename TagT>
  int Index<T, TagT>::generate_frozen_point() {
    if (_num_frozen_pts == 0)
      return 0;

    if (_nd == 0) {
      memset(_data + (_max_points) *_aligned_dim, 0, _aligned_dim * sizeof(T));
      return 1;
    }
    size_t res = calculate_entry_point();
    memcpy(_data + _max_points * _aligned_dim, _data + res * _aligned_dim,
           _aligned_dim * sizeof(T));
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::enable_delete() {
    assert(_enable_tags);

    if (!_enable_tags) {
      diskann::cerr << "Tags must be instantiated for deletions" << std::endl;
      return -2;
    }

    std::unique_lock<std::shared_timed_mutex> update_lock(_update_lock);
    if (_data_compacted) {
      for (unsigned slot = (unsigned) _nd; slot < _max_points; ++slot) {
        _empty_slots.insert(slot);
      }
    }

    _lazy_done = false;
    _eager_done = false;

    if (_support_eager_delete) {
      _in_graph.resize(_max_points + _num_frozen_pts);
      _in_graph.reserve(_max_points + _num_frozen_pts);
      update_in_graph();
    }
    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::release_location() {
    LockGuard guard(_num_points_lock);
    _nd--;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::eager_delete(const TagT tag, const Parameters &parameters,
                                   int delete_mode) {
    if (_lazy_done && (!_data_compacted)) {
      diskann::cout << "Lazy delete requests issued but data not consolidated, "
                       "cannot proceed with eager deletes."
                    << std::endl;
      return -1;
    }

    unsigned id;  // since we will return if tag is not found, ok to leave it
                  // uninitialized.
    {
      std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        diskann::cerr << "Delete tag " << tag << " not found" << std::endl;
        return -1;
      }
      id = _tag_to_location[tag];
    }

    {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);
      _location_to_tag.erase(_tag_to_location[tag]);
      _tag_to_location.erase(tag);
    }

    {
      // id will be valid because if not, it'll return in the {} above.
      std::unique_lock<std::shared_timed_mutex> lock(_delete_lock);
      _delete_set.insert(id);
      _empty_slots.insert(id);
    }

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    // delete point from out-neighbors' in-neighbor list
    {
      LockGuard guard(_locks[id]);
      for (size_t i = 0; i < _final_graph[id].size(); i++) {
        unsigned j = _final_graph[id][i];
        {
          LockGuard guard(_locks_in[j]);
          for (unsigned k = 0; k < _in_graph[j].size(); k++) {
            if (_in_graph[j][k] == id) {
              _in_graph[j].erase(_in_graph[j].begin() + k);
              break;
            }
          }
        }
      }
    }

    tsl::robin_set<unsigned> in_nbr;
    {
      LockGuard guard(_locks_in[id]);
      for (unsigned i = 0; i < _in_graph[id].size(); i++)
        in_nbr.insert(_in_graph[id][i]);
    }
    assert(_in_graph[id].size() == in_nbr.size());

    std::vector<Neighbor>    pool, tmp;
    tsl::robin_set<unsigned> visited;
    std::vector<unsigned>    intersection;
    unsigned                 Lindex = parameters.Get<unsigned>("L");
    std::vector<unsigned>    init_ids;

    if (delete_mode == 2) {
      // constructing list of in-neighbors to be processed
      get_expanded_nodes(id, Lindex, init_ids, pool, visited);

      for (auto node : visited) {
        if (in_nbr.find(node) != in_nbr.end()) {
          intersection.push_back(node);
        }
      }
    }

    // deleting deleted point from all in-neighbors' out-neighbor list
    for (auto it : in_nbr) {
      LockGuard guard(_locks[it]);
      _final_graph[it].erase(
          std::remove(_final_graph[it].begin(), _final_graph[it].end(), id),
          _final_graph[it].end());
    }

    tsl::robin_set<unsigned> candidate_set;
    std::vector<Neighbor>    expanded_nghrs;
    std::vector<Neighbor>    result;

    for (size_t i = 0; i < intersection.size(); i++) {
      auto ngh = intersection[i];

      candidate_set.clear();
      expanded_nghrs.clear();
      result.clear();

      {
        std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
        if (_delete_set.find(ngh) != _delete_set.end())
          continue;
      }

      {
        LockGuard guard(_locks[ngh]);

        // constructing candidate set from out-neighbors of ngh and id
        {  // should a shared reader lock on delete_lock be held here at the
           // beginning of the two for loops or should it be held and release
           // for ech iteration of the for loops? Which is faster?

          std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
          for (auto j : _final_graph[id]) {
            if ((j != id) && (j != ngh) &&
                (_delete_set.find(j) == _delete_set.end()))
              candidate_set.insert(j);
          }

          for (auto j : _final_graph[ngh]) {
            if ((j != id) && (j != ngh) &&
                (_delete_set.find(j) == _delete_set.end()))
              candidate_set.insert(j);
          }
        }

        for (auto j : candidate_set)
          expanded_nghrs.push_back(
              Neighbor(j,
                       _distance->compare(_data + _aligned_dim * (size_t) ngh,
                                          _data + _aligned_dim * (size_t) j,
                                          (unsigned) _aligned_dim),
                       true));
        std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
        occlude_list(expanded_nghrs, alpha, range, maxc, result);

        // deleting ngh from its old out-neighbors' in-neighbor list
        for (auto iter : _final_graph[ngh]) {
          {
            LockGuard guard(_locks_in[iter]);
            for (unsigned k = 0; k < _in_graph[iter].size(); k++) {
              if (_in_graph[iter][k] == ngh) {
                _in_graph[iter].erase(_in_graph[iter].begin() + k);
                break;
              }
            }
          }
        }

        _final_graph[ngh].clear();

        // updating out-neighbors and in-neighbors of ngh
        {
          std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
          for (size_t i = 0; i < result.size(); i++) {
            auto j = result[i];
            if (_delete_set.find(j.id) == _delete_set.end()) {
              _final_graph[ngh].push_back(j.id);
              {
                LockGuard guard(_locks_in[j.id]);
                if (std::find(_in_graph[j.id].begin(), _in_graph[j.id].end(),
                              ngh) == _in_graph[j.id].end()) {
                  _in_graph[j.id].emplace_back(ngh);
                }
              }
            }
          }
        }
      }
    }

    _final_graph[id].clear();
    _in_graph[id].clear();

    release_location();

    _eager_done = true;
    _data_compacted = false;
    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::update_in_graph() {
    for (unsigned i = 0; i < _in_graph.size(); i++)
      _in_graph[i].clear();

    for (size_t i = 0; i < _final_graph.size();
         i++)  // copying to in-neighbor graph
      for (size_t j = 0; j < _final_graph[i].size(); j++)
        _in_graph[_final_graph[i][j]].emplace_back((_u32) i);
  }

  // bfs, not very sophisticated, but only needs to handle a few levels
  template<typename T, typename TagT>
  void Index<T, TagT>::bfs_up_to_level(const int           bfs_levels,
                                       std::set<unsigned> &level_set) {
    std::vector<std::set<unsigned>> level_sets(bfs_levels + 1);
    std::set<unsigned>              level_set_0;
    level_set_0.insert(_start);
    level_sets[0] = level_set_0;
    level_set.insert(_start);
    for (int i = 0; i < bfs_levels; i++) {
      std::set<unsigned> next_level_set;
      for (const unsigned j : level_sets[i]) {
        for (const unsigned ngh : _final_graph[j]) {
          next_level_set.insert(ngh);
        }
      }
      for (const unsigned j : next_level_set) {
        if (level_set.find(j) != level_set.end()) {
          next_level_set.erase(j);
        }
        level_set.insert(j);
      }
      level_sets[i + 1] = next_level_set;
    }
  }

  template<typename T, typename TagT>
  inline std::pair<bool, bool> Index<T, TagT>::process_delete(
      const tsl::robin_set<unsigned> &old_delete_set, size_t i,
      const unsigned &range, const unsigned &maxc, const float &alpha,
      const int delete_policy, std::set<unsigned> &level_set) {
    bool policy_none = false;
    bool policy_all = false;
    bool policy_closest = false;
    bool policy_random = false;
    bool policy_bfs = false;

    int num_closest = 5;
    int num_random = 5;

    if (delete_policy == 0) {
      policy_none = true;
    } else if (delete_policy == 1) {
      policy_all = true;
    } else if (delete_policy == 2) {
      policy_closest = true;
    } else if (delete_policy == 3) {
      policy_random = true;
    } else if (delete_policy == 4) {
      policy_bfs = true;
    } else {
      policy_all = true;
    }

    tsl::robin_set<unsigned> candidate_set;
    std::vector<Neighbor>    expanded_nghrs;
    std::vector<Neighbor>    result;

    bool modify = false;
    bool prune = false;

    if (policy_none) {
      for (auto ngh : _final_graph[(_u32) i]) {
        if (old_delete_set.find(ngh) != old_delete_set.end()) {
          modify = true;
        } else {
          candidate_set.insert(ngh);
        }
      }
    } else if (policy_all) {
      for (auto ngh : _final_graph[(_u32) i]) {
        if (old_delete_set.find(ngh) != old_delete_set.end()) {
          modify = true;

          // Add outgoing links from
          for (auto j : _final_graph[ngh])
            if (old_delete_set.find(j) == old_delete_set.end())
              candidate_set.insert(j);
        } else {
          candidate_set.insert(ngh);
        }
      }
    } else if (policy_closest) {
      for (auto ngh : _final_graph[(_u32) i]) {
        if (old_delete_set.find(ngh) != old_delete_set.end()) {
          modify = true;
          std::vector<Neighbor> intermediate_nbh;
          for (auto j : _final_graph[ngh]) {
            if (old_delete_set.find(j) == old_delete_set.end()) {
              intermediate_nbh.push_back(
                  Neighbor(j,
                           _distance->compare(_data + _aligned_dim * ngh,
                                              _data + _aligned_dim * (size_t) j,
                                              (unsigned) _aligned_dim),
                           true));
              std::sort(intermediate_nbh.begin(), intermediate_nbh.end());
              int k = std::min(num_closest, (int) intermediate_nbh.size());
              for (int m = 0; m < k; m++) {
                candidate_set.insert(intermediate_nbh[m].id);
              }
            }
          }
        } else {
          candidate_set.insert(ngh);
        }
      }
    } else if (policy_random) {
      for (auto ngh : _final_graph[(_u32) i]) {
        if (old_delete_set.find(ngh) != old_delete_set.end()) {
          modify = true;
          static thread_local std::mt19937 rng;
          std::vector<int>                 intermediate_candidates;
          // Add outgoing links from
          for (auto j : _final_graph[ngh])
            if (old_delete_set.find(j) == old_delete_set.end())
              intermediate_candidates.push_back(j);
          size_t k =
              std::min(intermediate_candidates.size(), (size_t) num_random);
          std::set<int> intermediate_set;
          while (intermediate_set.size() < k) {
            std::uniform_int_distribution<int> distribution(0, (int) _nd);
            int                                index =
                distribution(rng) % ((int) intermediate_candidates.size());
            intermediate_set.insert(intermediate_candidates[index]);
          }
          for (const int elt : intermediate_set) {
            candidate_set.insert(elt);
          }
        } else {
          candidate_set.insert(ngh);
        }
      }
    } else if (policy_bfs) {
      for (auto ngh : _final_graph[(_u32) i]) {
        if (old_delete_set.find(ngh) != old_delete_set.end()) {
          modify = true;

          if (level_set.find(ngh) != level_set.end()) {
            // Add outgoing links from
            for (auto j : _final_graph[ngh]) {
              if (old_delete_set.find(j) == _delete_set.end()) {
                candidate_set.insert(j);
              }
            }
          } else {
            std::vector<Neighbor> intermediate_nbh;
            for (auto j : _final_graph[ngh]) {
              if (old_delete_set.find(j) == old_delete_set.end()) {
                intermediate_nbh.push_back(Neighbor(
                    j,
                    _distance->compare(_data + _aligned_dim * ngh,
                                       _data + _aligned_dim * (size_t) j,
                                       (unsigned) _aligned_dim),
                    true));
                std::sort(intermediate_nbh.begin(), intermediate_nbh.end());
                int k = std::min(num_closest, (int) intermediate_nbh.size());
                for (int j = 0; j < k; j++) {
                  candidate_set.insert(intermediate_nbh[j].id);
                }
              }
            }
          }
        } else {
          candidate_set.insert(ngh);
        }
      }
    }

    if (modify) {
      if (candidate_set.size() <= (size_t) 1.5 * range) {
        _final_graph[(_u32) i].clear();
        for (auto j : candidate_set) {
          if (j != (_u32) i && (old_delete_set.find(j) == old_delete_set.end()))
            _final_graph[(_u32) i].push_back(j);
        }
      } else {
        prune = true;
        for (auto j : candidate_set) {
          expanded_nghrs.push_back(
              Neighbor(j,
                       _distance->compare(_data + _aligned_dim * i,
                                          _data + _aligned_dim * (size_t) j,
                                          (unsigned) _aligned_dim),
                       true));
        }
        std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
        occlude_list(expanded_nghrs, alpha, range, maxc, result);

        _final_graph[(_u32) i].clear();
        for (auto j : result) {
          if (j.id != (_u32) i &&
              (old_delete_set.find(j.id) == old_delete_set.end()))
            _final_graph[(_u32) i].push_back(j.id);
        }
        if(_queries_present) insert_and_stitch((_u32) i);
      }
    }
    return std::make_pair(modify, prune);
  }

  // Do not call consolidate_deletes() if you have not locked _num_points_lock.
  // Returns number of live points left after consolidation
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes(const Parameters &params,
                                             int               delete_policy) {
    if (!_enable_tags)
      throw diskann::ANNException("Point tag array not instantiated", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);

    if (_eager_done)
      throw ANNException("Can not consolidates eager deletes.", -1, __FUNCSIG__,
                         __FILE__, __LINE__);

    if (_tag_to_location.size() + _delete_set.size() != _nd) {
      throw diskann::ANNException(
          "Tags to points array wrong sized, _tag_to_location.size():  " +
              std::to_string(_tag_to_location.size()) +
              "  _delete_set.size():  " + std::to_string(_delete_set.size()) +
              "  _nd:  " + std::to_string(_nd),
          -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_location_to_tag.size() + _delete_set.size() != _nd) {
      throw diskann::ANNException(
          "Points to tags array wrong sized, _location_to_tag.size(): " +
              std::to_string(_location_to_tag.size()) +
              "  _delete_set.size():  " + std::to_string(_delete_set.size()) +
              "  _nd:  " + std::to_string(_nd),
          -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::cout << "Consolidating deletes... " << std::endl;

    std::unique_lock<std::shared_timed_mutex> update_lock(_update_lock,
                                                          std::defer_lock);
    if (!_conc_consolidate)
      update_lock.lock();

    bool expected_consolidate_value = false;
    if (_consolidate_active.compare_exchange_strong(expected_consolidate_value,
                                                    true) == false)
      throw ANNException(
          "Attenpting to run consolidation while another consolidation "
          "instance is active",
          -1, __FUNCSIG__, __FILE__, __LINE__);

    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    tsl::robin_set<unsigned> old_delete_set;
    {
      std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
      _delete_set.swap(old_delete_set);
    }

    const unsigned range = params.Get<unsigned>("R");
    const unsigned maxc = params.Get<unsigned>("C");
    const float    alpha = params.Get<float>("alpha");
    const unsigned num_threads = params.Get<unsigned>("num_threads") == 0
                                     ? omp_get_num_threads()
                                     : params.Get<unsigned>("num_threads");

    _u64     total_pts = _max_points + _num_frozen_pts;
    unsigned block_size = 1 << 10;
    _s64     total_blocks = DIV_ROUND_UP(total_pts, block_size);

    std::vector<int>   modified(total_pts, 0);
    std::vector<int>   pruned(total_pts, 0);
    std::set<unsigned> level_set;
    const int          bfs_levels = 3;

    if (delete_policy == 4) {
      if (_conc_consolidate == false) {
        bfs_up_to_level(bfs_levels, level_set);
        std::cout << level_set.size() << std::endl;
      } else {
        std::cout << "ERROR: cannot use delete policy 4 when concurrent "
                     "consolidate is enabled. Using default policy."
                  << std::endl;
        delete_policy = 1;
      }
    }

    if (delete_policy > 4) {
      std::cout
          << "ERROR: invalid delete policy specified. Using default policy."
          << std::endl;
      delete_policy = 1;
    }

    if(_queries_present) erase_query_nn(old_delete_set);

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (_s64 block = 0; block < total_blocks; ++block) {
      for (_s64 i = block * block_size;
           i < (_s64)((block + 1) * block_size) &&
           i < (_s64)(_max_points + _num_frozen_pts);
           i++) {
        if ((old_delete_set.find((_u32) i) == old_delete_set.end()) &&
            (_empty_slots.find((_u32) i) == _empty_slots.end())) {
          if (_conc_consolidate) {
            std::unique_lock<std::mutex> adj_list_lock(_locks[i]);
            auto stats = process_delete(old_delete_set, i, range, maxc, alpha,
                                        delete_policy, level_set);
            if (stats.first)
              modified[(_u32) i] = 1;
            if (stats.second)
              pruned[(_u32) i] = 1;
          } else {
            auto stats = process_delete(old_delete_set, i, range, maxc, alpha,
                                        delete_policy, level_set);
            if (stats.first)
              modified[(_u32) i] = 1;
            if (stats.second)
              pruned[(_u32) i] = 1;
          }
        }
      }
    }
    if (_support_eager_delete)
      update_in_graph();

    if (_conc_consolidate) {
      for (auto iter : old_delete_set)
        _empty_slots.insert(iter);
    } else {
      for (auto iter : old_delete_set)
        _empty_slots.insert(iter);
    }

    size_t ret_nd;
    {
      LockGuard guard(_num_points_lock);
      _nd -= old_delete_set.size();
      ret_nd = _nd;
      old_delete_set.clear();
    }

    //call stitch routine here
    //including swapping prune sets

    expected_consolidate_value = true;
    if (_consolidate_active.compare_exchange_strong(expected_consolidate_value,
                                                    false) == false)
      throw ANNException("Failed to change consolidation active to false", -1,
                         __FUNCSIG__, __FILE__, __LINE__);

    if (!_conc_consolidate)
      update_lock.unlock();



    compact_data();

    auto stop = std::chrono::high_resolution_clock::now();
    diskann::cout << "Time taken for consolidate_deletes() "
                  << std::chrono::duration_cast<std::chrono::duration<double>>(
                         stop - start)
                         .count()
                  << "s." << std::endl;
    diskann::cout << "Number of nodes modified: "
                  << std::accumulate(modified.begin(), modified.end(), 0)
                  << std::endl;
    diskann::cout << "Number of nodes pruned: "
                  << std::accumulate(pruned.begin(), pruned.end(), 0)
                  << std::endl;

    return ret_nd;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::consolidate(Parameters &parameters) {
    consolidate_deletes(parameters);
    compact_data();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_frozen_point() {
    if (_nd < _max_points) {
      if (_num_frozen_pts > 0) {
        // set new _start to be frozen point
        _start = (_u32) _nd;
        if (!_final_graph[_max_points].empty()) {
          for (unsigned i = 0; i < _nd; i++)
            for (unsigned j = 0; j < _final_graph[i].size(); j++)
              if (_final_graph[i][j] == _max_points)
                _final_graph[i][j] = (_u32) _nd;

          _final_graph[_nd].clear();
          for (unsigned k = 0; k < _final_graph[_max_points].size(); k++)
            _final_graph[_nd].emplace_back(_final_graph[_max_points][k]);

          _final_graph[_max_points].clear();
          if (_support_eager_delete)
            update_in_graph();

          memcpy((void *) (_data + _aligned_dim * _nd),
                 _data + (size_t) _aligned_dim * _max_points, sizeof(T) * _dim);
          memset((_data + (size_t) _aligned_dim * _max_points), 0,
                 sizeof(T) * _aligned_dim);
        }
      }
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data_for_search() {
    compact_data();
    compact_frozen_point();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data() {
    if (!_dynamic_index)
      throw ANNException("Can not compact a non-dynamic index", -1, __FUNCSIG__,
                         __FILE__, __LINE__);

    if (!_lazy_done && !_eager_done)
      return;

    if (_data_compacted) {
      diskann::cerr
          << "Warning! Calling compact_data() when _data_compacted is true!"
          << std::endl;
      return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto fnstart = start;

    std::vector<unsigned> new_location = std::vector<unsigned>(
        _max_points + _num_frozen_pts, (_u32) _max_points);

    _u32 new_counter = 0;

    for (_u32 old_counter = 0; old_counter < _max_points + _num_frozen_pts;
         old_counter++) {
      if (_location_to_tag.find(old_counter) != _location_to_tag.end()) {
        new_location[old_counter] = new_counter;
        new_counter++;
      }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    // If start node is removed, replace it.
    if (_delete_set.find(_start) != _delete_set.end()) {
      diskann::cerr << "Replacing start node which has been deleted... "
                    << std::flush;
      auto old_ep = _start;
      // First active neighbor of old start node is new start node
      for (auto iter : _final_graph[_start])
        if (_delete_set.find(iter) != _delete_set.end()) {
          _start = iter;
          break;
        }
      if (_start == old_ep) {
        throw diskann::ANNException(
            "ERROR: Did not find a replacement for start node.", -1,
            __FUNCSIG__, __FILE__, __LINE__);
      } else {
        assert(_delete_set.find(_start) == _delete_set.end());
      }
    }

    start = std::chrono::high_resolution_clock::now();
    double copy_time = 0;
    for (unsigned old = 0; old <= _max_points; ++old) {
      if ((new_location[old] < _max_points) ||
          (old == _max_points)) {  // If point continues to exist

        // Renumber nodes to compact the order
        for (size_t i = 0; i < _final_graph[old].size(); ++i) {
          if (new_location[_final_graph[old][i]] > _final_graph[old][i]) {
            std::stringstream sstream;
            sstream << "Error in compact_data(). Found point: " << old
                    << " whose " << i << "th neighbor has new location "
                    << new_location[_final_graph[old][i]]
                    << " that is greater than its old location: "
                    << _final_graph[old][i];
            if (_delete_set.find(_final_graph[old][i]) != _delete_set.end()) {
              sstream << std::endl
                      << " Point: " << old << " index: " << i
                      << " neighbor: " << _final_graph[old][i]
                      << " found in delete set of size: " << _delete_set.size()
                      << std::endl;
            } else {
              sstream << " Point: " << old
                      << " neighbor: " << _final_graph[old][i]
                      << " NOT found in delete set of size: "
                      << _delete_set.size() << std::endl;
            }
            throw diskann::ANNException(sstream.str(), -1, __FUNCSIG__,
                                        __FILE__, __LINE__);
          }
          _final_graph[old][i] = new_location[_final_graph[old][i]];
        }

        if (_support_eager_delete)
          for (size_t i = 0; i < _in_graph[old].size(); ++i) {
            if (new_location[_in_graph[old][i]] <= _in_graph[old][i])
              _in_graph[old][i] = new_location[_in_graph[old][i]];
          }

        // Move the data and adj list to the correct position
        auto c_start = std::chrono::high_resolution_clock::now();
        if (new_location[old] != old) {
          assert(new_location[old] < old);
          _final_graph[new_location[old]].swap(_final_graph[old]);
          if (_support_eager_delete)
            _in_graph[new_location[old]].swap(_in_graph[old]);
          memcpy((void *) (_data + _aligned_dim * (size_t) new_location[old]),
                 (void *) (_data + _aligned_dim * (size_t) old),
                 _aligned_dim * sizeof(T));
        }
        auto c_stop = std::chrono::high_resolution_clock::now();
        copy_time += std::chrono::duration_cast<std::chrono::duration<double>>(
                         c_stop - c_start)
                         .count();

      } else {
        _final_graph[old].clear();
      }
    }
    stop = std::chrono::high_resolution_clock::now();

    start = std::chrono::high_resolution_clock::now();
    _tag_to_location.clear();
    for (auto iter : _location_to_tag) {
      _tag_to_location[iter.second] = new_location[iter.first];
    }
    _location_to_tag.clear();
    for (auto iter : _tag_to_location) {
      _location_to_tag[iter.second] = iter.first;
    }

    for (_u64 old = _nd; old < _max_points; ++old) {
      _final_graph[old].clear();
    }
    _delete_set.clear();
    _empty_slots.clear();
    for (auto i = _nd; i < _max_points; i++) {
      _empty_slots.insert((uint32_t) i);
    }

    _lazy_done = false;
    _eager_done = false;
    _data_compacted = true;
    stop = std::chrono::high_resolution_clock::now();
    diskann::cout << "Time taken for compact_data(): "
                  << std::chrono::duration_cast<std::chrono::duration<double>>(
                         stop - fnstart)
                         .count()
                  << "s." << std::endl;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::reserve_location() {
    LockGuard guard(_num_points_lock);
    if (_nd >= _max_points) {
      return -1;
    }
    unsigned location;
    if (_data_compacted) {
      location = (unsigned) _nd;
      _empty_slots.erase(location);
    } else {
      // no need of delete_lock here, _num_points_lock will ensure no other
      // thread executes this block of code
      assert(_empty_slots.size() != 0);
      assert(_empty_slots.size() + _nd == _max_points);

      auto iter = _empty_slots.begin();
      location = *iter;
      _empty_slots.erase(iter);
      _delete_set.erase(location);
    }

    ++_nd;
    return location;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::reposition_point(unsigned old_location,
                                        unsigned new_location) {
    for (unsigned i = 0; i < _nd; i++)
      for (unsigned j = 0; j < _final_graph[i].size(); j++)
        if (_final_graph[i][j] == old_location)
          _final_graph[i][j] = (unsigned) new_location;

    _final_graph[new_location].clear();
    for (unsigned k = 0; k < _final_graph[_nd].size(); k++)
      _final_graph[new_location].emplace_back(_final_graph[old_location][k]);

    _final_graph[old_location].clear();

    if (_support_eager_delete) {
      update_in_graph();
    }
    memcpy((void *) (_data + (size_t) _aligned_dim * new_location),
           _data + (size_t) _aligned_dim * old_location,
           sizeof(T) * _aligned_dim);
    memset((_data + (size_t) _aligned_dim * old_location), 0,
           sizeof(T) * _aligned_dim);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::reposition_frozen_point_to_end() {
    if (_num_frozen_pts == 0)
      return;

    if (_nd == _max_points) {
      diskann::cout
          << "Not repositioning frozen point as it is already at the end."
          << std::endl;
      return;
    }
    reposition_point((_u32) _nd, (_u32) _max_points);
    _start = (_u32) _max_points;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::resize(size_t new_max_points) {
    auto start = std::chrono::high_resolution_clock::now();
    assert(_empty_slots.size() ==
           0);  // should not resize if there are empty slots.
#ifndef _WINDOWS
    T *new_data;
    alloc_aligned((void **) &new_data,
                  (new_max_points + 1) * _aligned_dim * sizeof(T),
                  8 * sizeof(T));
    memcpy(new_data, _data, (_max_points + 1) * _aligned_dim * sizeof(T));
    aligned_free(_data);
    _data = new_data;
#else
    realloc_aligned((void **) &_data,
                    (new_max_points + 1) * _aligned_dim * sizeof(T),
                    8 * sizeof(T));
#endif
    _final_graph.resize(new_max_points + 1);
    _locks = std::vector<std::mutex>(new_max_points + 1);
    if (_support_eager_delete) {
      _in_graph.resize(new_max_points + 1);
      _locks_in = std::vector<std::mutex>(new_max_points + 1);
    }

    reposition_point((_u32) _max_points, (_u32) new_max_points);
    _max_points = new_max_points;
    _start = (_u32) new_max_points;

    for (auto i = _nd; i < _max_points; i++) {
      _empty_slots.insert((uint32_t) i);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    diskann::cout << "Resizing took: "
                  << std::chrono::duration<double>(stop - start).count() << "s"
                  << std::endl;
  }

  template<typename T, typename TagT>
  std::pair<int, std::vector<double>> Index<T, TagT>::insert_point(const T *point, const TagT tag) {
    assert(_has_built);

    std::vector<double> stitch_times(3, 0.0);

    std::shared_lock<std::shared_timed_mutex> update_lock(_update_lock);

    unsigned range = _indexingRange;
    unsigned Lindex = _indexingQueueSize;

    // Avoiding the use of temporary containers.
    ScratchStoreManager<T> manager(_query_scratch);
    auto                   scratch = manager.scratch_space();

    std::vector<Neighbor> &   pool = scratch.pool();
    tsl::robin_set<unsigned> &visited = scratch.visited();

    {
      std::shared_lock<std::shared_timed_mutex> tsl(_tag_lock);
      if (_enable_tags &&
          (_tag_to_location.find(tag) != _tag_to_location.end())) {
        abort();
      }
    }

    auto location = reserve_location();
    if (location == -1) {
      update_lock.unlock();
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      if (_nd >= _max_points) {
        auto new_max_points = (size_t)(_max_points * INDEX_GROWTH_FACTOR);
        resize(new_max_points);
      }
      growth_lock.unlock();
      update_lock.lock();

      location = reserve_location();
      if (location == -1) {
        throw diskann::ANNException(
            "Cannot reserve location even after expanding graph. Terminating.",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }
    }

    int num_pruned;

    {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);

      _tag_to_location[tag] = location;
      _location_to_tag[location] = tag;
    }

    auto offset_data = _data + (size_t) _aligned_dim * location;
    memset((void *) offset_data, 0, sizeof(T) * _aligned_dim);
    memcpy((void *) offset_data, point, sizeof(T) * _dim);

    if (_normalize_vecs) {
      normalize((float *) offset_data, _dim);
    }

    pool.clear();
    visited.clear();
    std::vector<unsigned> pruned_list;
    // params.Get<unsigned>("L");

    std::vector<unsigned> init_ids;
    get_expanded_nodes(location, Lindex, init_ids, pool, visited, scratch.des(),
                       scratch.best_l_nodes(), scratch.inserted_into_pool_rs(),
                       scratch.inserted_into_pool_bs());

    for (unsigned i = 0; i < pool.size(); i++) {
      if (pool[i].id == (unsigned) location) {
        pool.erase(pool.begin() + i);
        visited.erase((unsigned) location);
        i--;
      } else if (_delete_set.find(pool[i].id) != _delete_set.end()) {
        pool.erase(pool.begin() + i);
        visited.erase((unsigned) pool[i].id);
        i--;
      }
    }

    prune_neighbors(location, pool, pruned_list);
    assert(_final_graph.size() == _max_points + _num_frozen_pts);

    if (_support_eager_delete) {
      for (unsigned i = 0; i < _final_graph[location].size(); i++) {
        {
          LockGuard guard(_locks_in[_final_graph[location][i]]);
          _in_graph[_final_graph[location][i]].erase(
              std::remove(_in_graph[_final_graph[location][i]].begin(),
                          _in_graph[_final_graph[location][i]].end(), location),
              _in_graph[_final_graph[location][i]].end());
        }
      }
    }

    _final_graph[location].clear();
    _final_graph[location].shrink_to_fit();
    _final_graph[location].reserve((_u64)(range * GRAPH_SLACK_FACTOR * 1.05));

    assert(!pruned_list.empty());
    {
      LockGuard guard(_locks[location]);
      for (auto link : pruned_list) {
        _final_graph[location].emplace_back(link);
        if (_support_eager_delete) {
          LockGuard guard(_locks_in[link]);
          _in_graph[link].emplace_back(location);
        }
      }
    }

    std::vector<double> times;
    assert(_final_graph[location].size() <= range);
    if (_support_eager_delete) {
      auto pair = inter_insert(location, pruned_list, 1);
      num_pruned = pair.first;
      times = pair.second;
    } else {
      auto pair = inter_insert(location, pruned_list, 0);
      num_pruned = pair.first;
      times = pair.second;
    }
    stitch_times[0] += times[0];
    stitch_times[1] += times[1];
    stitch_times[2] += times[2];
    if(_queries_present){
      auto Times = insert_and_stitch(location);
      stitch_times[0] += Times[0];
      stitch_times[1] += Times[1];
      stitch_times[2] += Times[2];
    } 
    return std::make_pair(num_pruned, times);
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::cleanup(){
    // tsl::robin_set<unsigned> dummy_set;
    populate_query_nn();
    robust_stitch();
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const TagT &tag) {
    if ((_eager_done) && (!_data_compacted)) {
      diskann::cerr << "Eager delete requests were issued but data was not "
                       "compacted, cannot proceed with lazy_deletes"
                    << std::endl;
      return -2;
    }
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    _lazy_done = true;

    {
      std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
      std::unique_lock<std::shared_timed_mutex> l(_delete_lock);

      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        diskann::cerr << "Delete tag not found" << std::endl;
        return -1;
      }
      assert(_tag_to_location[tag] < _max_points);

      _location_to_tag.erase(_tag_to_location[tag]);
      _tag_to_location.erase(tag);
      _delete_set.insert(_tag_to_location[tag]);
    }

    _data_compacted = false;
    return 0;
  }

  // TODO: Check if this function needs a shared_lock on _tag_lock.
  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const tsl::robin_set<TagT> &tags,
                                  std::vector<TagT> &         failed_tags) {
    if (failed_tags.size() > 0) {
      diskann::cerr << "failed_tags should be passed as an empty list"
                    << std::endl;
      return -3;
    }
    if ((_eager_done) && (!_data_compacted)) {
      diskann::cout << "Eager delete requests were issued but data was not "
                       "compacted, cannot proceed with lazy_deletes"
                    << std::endl;
      return -2;
    }
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    _lazy_done = true;

    for (auto tag : tags) {
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        failed_tags.push_back(tag);
      } else {
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
      }
    }

    _data_compacted = false;
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::extract_data(
      T *ret_data, std::unordered_map<TagT, unsigned> &tag_to_location) {
    if (!_data_compacted) {
      diskann::cerr
          << "Error! Data not compacted. Cannot give access to private data."
          << std::endl;
      return -1;
    }
    std::memset(ret_data, 0, (size_t) _aligned_dim * _nd * sizeof(T));
    std::memcpy(ret_data, _data, (size_t)(_aligned_dim) *_nd * sizeof(T));
    tag_to_location = _tag_to_location;
    return 0;
  }

  template<typename T, typename TagT>
  bool Index<T, TagT>::is_index_saved() {
    return _is_saved;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_active_tags(tsl::robin_set<TagT> &active_tags) {
    active_tags.clear();
    std::shared_lock<std::shared_timed_mutex> tul(_tag_lock);
    for (auto iter : _tag_to_location) {
      active_tags.insert(iter.first);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::print_status() const {
    diskann::cout << "------------------- Index object: " << (uint64_t) this
                  << " -------------------" << std::endl;
    diskann::cout << "Number of points: " << _nd << std::endl;
    diskann::cout << "Graph size: " << _final_graph.size() << std::endl;
    diskann::cout << "Location to tag size: " << _location_to_tag.size()
                  << std::endl;
    diskann::cout << "Tag to location size: " << _tag_to_location.size()
                  << std::endl;
    diskann::cout << "Number of empty slots: " << _empty_slots.size()
                  << std::endl;
    diskann::cout << std::boolalpha
                  << "Data compacted: " << this->_data_compacted
                  << " Lazy done: " << this->_lazy_done
                  << " Eager done: " << this->_eager_done << std::endl;
    diskann::cout << "---------------------------------------------------------"
                     "------------"
                  << std::endl;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::optimize_index_layout() {  // use after build or load
    _data_len = (_aligned_dim + 1) * sizeof(float);
    _neighbor_len = (_max_observed_degree + 1) * sizeof(unsigned);
    _node_size = _data_len + _neighbor_len;
    _opt_graph = (char *) malloc(_node_size * _nd);
    DistanceFastL2<T> *dist_fast = (DistanceFastL2<T> *) _distance;
    for (unsigned i = 0; i < _nd; i++) {
      char *cur_node_offset = _opt_graph + i * _node_size;
      float cur_norm = dist_fast->norm(_data + i * _aligned_dim, _aligned_dim);
      std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
      std::memcpy(cur_node_offset + sizeof(float), _data + i * _aligned_dim,
                  _data_len - sizeof(float));

      cur_node_offset += _data_len;
      unsigned k = _final_graph[i].size();
      std::memcpy(cur_node_offset, &k, sizeof(unsigned));
      std::memcpy(cur_node_offset + sizeof(unsigned), _final_graph[i].data(),
                  k * sizeof(unsigned));
      std::vector<unsigned>().swap(_final_graph[i]);
    }
    _final_graph.clear();
    _final_graph.shrink_to_fit();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::search_with_optimized_layout(const T *query, size_t K,
                                                    size_t    L,
                                                    unsigned *indices) {
    DistanceFastL2<T> *dist_fast = (DistanceFastL2<T> *) _distance;

    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    // std::mt19937 rng(rand());
    // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

    boost::dynamic_bitset<> flags{_nd, 0};
    unsigned                tmp_l = 0;
    unsigned *              neighbors =
        (unsigned *) (_opt_graph + _node_size * _start + _data_len);
    unsigned MaxM_ep = *neighbors;
    neighbors++;

    for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
      init_ids[tmp_l] = neighbors[tmp_l];
      flags[init_ids[tmp_l]] = true;
    }

    while (tmp_l < L) {
      unsigned id = rand() % _nd;
      if (flags[id])
        continue;
      flags[id] = true;
      init_ids[tmp_l] = id;
      tmp_l++;
    }

    for (unsigned i = 0; i < init_ids.size(); i++) {
      unsigned id = init_ids[i];
      if (id >= _nd)
        continue;
      _mm_prefetch(_opt_graph + _node_size * id, _MM_HINT_T0);
    }
    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
      unsigned id = init_ids[i];
      if (id >= _nd)
        continue;
      T *   x = (T *) (_opt_graph + _node_size * id);
      float norm_x = *x;
      x++;
      float dist =
          dist_fast->compare(x, query, norm_x, (unsigned) _aligned_dim);
      retset[i] = Neighbor(id, dist, true);
      flags[id] = true;
      L++;
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int) L) {
      int nk = L;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        _mm_prefetch(_opt_graph + _node_size * n + _data_len, _MM_HINT_T0);
        unsigned *neighbors =
            (unsigned *) (_opt_graph + _node_size * n + _data_len);
        unsigned MaxM = *neighbors;
        neighbors++;
        for (unsigned m = 0; m < MaxM; ++m)
          _mm_prefetch(_opt_graph + _node_size * neighbors[m], _MM_HINT_T0);
        for (unsigned m = 0; m < MaxM; ++m) {
          unsigned id = neighbors[m];
          if (flags[id])
            continue;
          flags[id] = 1;
          T *   data = (T *) (_opt_graph + _node_size * id);
          float norm = *data;
          data++;
          float dist =
              dist_fast->compare(query, data, norm, (unsigned) _aligned_dim);
          if (dist >= retset[L - 1].distance)
            continue;
          Neighbor nn(id, dist, true);
          int      r = InsertIntoPool(retset.data(), L, nn);

          // if(L+1 < retset.size()) ++L;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    for (size_t i = 0; i < K; i++) {
      indices[i] = retset[i].id;
    }
  }

  /*  Internals of the library */
  template<typename T, typename TagT>
  const float Index<T, TagT>::INDEX_GROWTH_FACTOR = 1.5f;

  // EXPORTS
  template DISKANN_DLLEXPORT class Index<float, int32_t>;
  template DISKANN_DLLEXPORT class Index<int8_t, int32_t>;
  template DISKANN_DLLEXPORT class Index<uint8_t, int32_t>;
  template DISKANN_DLLEXPORT class Index<float, uint32_t>;
  template DISKANN_DLLEXPORT class Index<int8_t, uint32_t>;
  template DISKANN_DLLEXPORT class Index<uint8_t, uint32_t>;
  template DISKANN_DLLEXPORT class Index<float, int64_t>;
  template DISKANN_DLLEXPORT class Index<int8_t, int64_t>;
  template DISKANN_DLLEXPORT class Index<uint8_t, int64_t>;
  template DISKANN_DLLEXPORT class Index<float, uint64_t>;
  template DISKANN_DLLEXPORT class Index<int8_t, uint64_t>;
  template DISKANN_DLLEXPORT class Index<uint8_t, uint64_t>;

  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<float, uint64_t>::search<uint64_t>(const float *query, const size_t K,
                                           const unsigned L, uint64_t *indices,
                                           float *distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<float, uint64_t>::search<uint32_t>(const float *query, const size_t K,
                                           const unsigned L, uint32_t *indices,
                                           float *distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<uint8_t, uint64_t>::search<uint64_t>(const uint8_t *query,
                                             const size_t K, const unsigned L,
                                             uint64_t *indices,
                                             float *   distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<uint8_t, uint64_t>::search<uint32_t>(const uint8_t *query,
                                             const size_t K, const unsigned L,
                                             uint32_t *indices,
                                             float *   distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<int8_t, uint64_t>::search<uint64_t>(const int8_t *query, const size_t K,
                                            const unsigned L, uint64_t *indices,
                                            float *distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<int8_t, uint64_t>::search<uint32_t>(const int8_t *query, const size_t K,
                                            const unsigned L, uint32_t *indices,
                                            float *distances);
  // TagT==uint32_t
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<float, uint32_t>::search<uint64_t>(const float *query, const size_t K,
                                           const unsigned L, uint64_t *indices,
                                           float *distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<float, uint32_t>::search<uint32_t>(const float *query, const size_t K,
                                           const unsigned L, uint32_t *indices,
                                           float *distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<uint8_t, uint32_t>::search<uint64_t>(const uint8_t *query,
                                             const size_t K, const unsigned L,
                                             uint64_t *indices,
                                             float *   distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<uint8_t, uint32_t>::search<uint32_t>(const uint8_t *query,
                                             const size_t K, const unsigned L,
                                             uint32_t *indices,
                                             float *   distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<int8_t, uint32_t>::search<uint64_t>(const int8_t *query, const size_t K,
                                            const unsigned L, uint64_t *indices,
                                            float *distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
                             Index<int8_t, uint32_t>::search<uint32_t>(const int8_t *query, const size_t K,
                                            const unsigned L, uint32_t *indices,
                                            float *distances);

}  // namespace diskann