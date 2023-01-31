// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <type_traits>
#include <omp.h>

#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include "boost/dynamic_bitset.hpp"

#include "memory_mapper.h"
#include "timer.h"
#include "windows_customizations.h"
#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#ifdef _WINDOWS
#include <xmmintrin.h>
#endif
#include "index.h"

#define MAX_POINTS_FOR_USING_BITSET 10000000

namespace diskann {
  // Initialize an index with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T, typename TagT>
  Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points,
                        const bool dynamic_index, const Parameters &indexParams,
                        const Parameters &searchParams, const bool enable_tags,
                        const bool concurrent_consolidate,
                        const bool pq_dist_build, const size_t num_pq_chunks,
                        const bool use_opq)
      : Index(m, dim, max_points, dynamic_index, enable_tags,
              concurrent_consolidate) {
    _indexingQueueSize = indexParams.Get<uint32_t>("L");
    _indexingRange = indexParams.Get<uint32_t>("R");
    _indexingMaxC = indexParams.Get<uint32_t>("C");
    _indexingAlpha = indexParams.Get<float>("alpha");

    uint32_t num_threads_srch = searchParams.Get<uint32_t>("num_threads");
    uint32_t num_threads_indx = indexParams.Get<uint32_t>("num_threads");
    uint32_t num_scratch_spaces = num_threads_srch + num_threads_indx;
    uint32_t search_l = searchParams.Get<uint32_t>("L");

    initialize_query_scratch(num_scratch_spaces, search_l, _indexingQueueSize,
                             _indexingRange, _indexingMaxC, dim);
  }

  template<typename T, typename TagT>
  Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points,
                        const bool dynamic_index, const bool enable_tags,
                        const bool concurrent_consolidate,
                        const bool pq_dist_build, const size_t num_pq_chunks,
                        const bool use_opq)
      : _dist_metric(m), _dim(dim), _max_points(max_points),
        _dynamic_index(dynamic_index), _enable_tags(enable_tags),
        _indexingMaxC(DEFAULT_MAXC), _query_scratch(nullptr),
        _conc_consolidate(concurrent_consolidate),
        _delete_set(new tsl::robin_set<unsigned>), _pq_dist(pq_dist_build),
        _use_opq(use_opq), _num_pq_chunks(num_pq_chunks) {
    if (dynamic_index && !enable_tags) {
      throw ANNException("ERROR: Dynamic Indexing must have tags enabled.", -1,
                         __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_pq_dist) {
      if (dynamic_index)
        throw ANNException(
            "ERROR: Dynamic Indexing not supported with PQ distance based "
            "index construction",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      if (m == diskann::Metric::INNER_PRODUCT)
        throw ANNException(
            "ERROR: Inner product metrics not yet supported with PQ distance "
            "base index",
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // data stored to _nd * aligned_dim matrix with necessary zero-padding
    _aligned_dim = ROUND_UP(_dim, 8);

    if (dynamic_index) {
      _num_frozen_pts = 1;
    }
    // Sanity check. While logically it is correct, max_points = 0 causes
    // downstream problems.
    if (_max_points == 0) {
      _max_points = 1;
    }
    const size_t total_internal_points = _max_points + _num_frozen_pts;

    if (_pq_dist) {
      if (_num_pq_chunks > _dim)
        throw diskann::ANNException("ERROR: num_pq_chunks > dim", -1,
                                    __FUNCSIG__, __FILE__, __LINE__);
      alloc_aligned(((void **) &_pq_data),
                    total_internal_points * _num_pq_chunks * sizeof(char),
                    8 * sizeof(char));
      std::memset(_pq_data, 0,
                  total_internal_points * _num_pq_chunks * sizeof(char));
    }
    alloc_aligned(((void **) &_data),
                  total_internal_points * _aligned_dim * sizeof(T),
                  8 * sizeof(T));
    std::memset(_data, 0, total_internal_points * _aligned_dim * sizeof(T));

    _start = (unsigned) _max_points;

    _final_graph.resize(total_internal_points);

    if (m == diskann::Metric::COSINE && std::is_floating_point<T>::value) {
      // This is safe because T is float inside the if block.
      this->_distance = (Distance<T> *) new AVXNormalizedCosineDistanceFloat();
      this->_normalize_vecs = true;
      diskann::cout << "Normalizing vectors and using L2 for cosine "
                       "AVXNormalizedCosineDistanceFloat()."
                    << std::endl;
    } else {
      this->_distance = get_distance_function<T>(m);
    }

    _locks = std::vector<non_recursive_mutex>(total_internal_points);

    if (enable_tags) {
      _location_to_tag.reserve(total_internal_points);
      _tag_to_location.reserve(total_internal_points);
    }
  }

  template<typename T, typename TagT>
  Index<T, TagT>::~Index() {
    // Ensure that no other activity is happening before dtor()
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    for (auto &lock : _locks) {
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
    if (_opt_graph != nullptr) {
      delete[] _opt_graph;
    }

    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    manager.destroy();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::initialize_query_scratch(uint32_t num_threads,
                                                uint32_t search_l,
                                                uint32_t indexing_l, uint32_t r,
                                                uint32_t maxc, size_t dim) {
    for (uint32_t i = 0; i < num_threads; i++) {
      auto scratch = new InMemQueryScratch<T>(search_l, indexing_l, r, maxc,
                                              dim, _pq_dist);
      _query_scratch.push(scratch);
    }
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_tags(std::string tags_file) {
    if (!_enable_tags) {
      diskann::cout << "Not saving tags as they are not enabled." << std::endl;
      return 0;
    }
    size_t tag_bytes_written;
    TagT  *tag_data = new TagT[_nd + _num_frozen_pts];
    for (_u32 i = 0; i < _nd; i++) {
      TagT tag;
      if (_location_to_tag.try_get(i, tag)) {
        tag_data[i] = tag;
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
  _u64 Index<T, TagT>::save_data(std::string data_file) {
    return save_data_in_base_dimensions(data_file, _data, _nd + _num_frozen_pts,
                                        _dim, _aligned_dim);
  }

  // save the graph index on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_graph(std::string graph_file) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    _u64 file_offset = 0;  // we will use this if we want
    out.seekp(file_offset, out.beg);
    _u64 index_size = 24;
    _u32 max_degree = 0;
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &_max_observed_degree, sizeof(unsigned));
    unsigned ep_u32 = _start;
    out.write((char *) &ep_u32, sizeof(unsigned));
    out.write((char *) &_num_frozen_pts, sizeof(_u64));
    for (unsigned i = 0; i < _nd + _num_frozen_pts; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _final_graph[i].data(), GK * sizeof(unsigned));
      max_degree = _final_graph[i].size() > max_degree
                       ? (_u32) _final_graph[i].size()
                       : max_degree;
      index_size += (_u64) (sizeof(unsigned) * (GK + 1));
    }
    out.seekp(file_offset, out.beg);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_degree, sizeof(_u32));
    out.close();
    return index_size;  // number of bytes written
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_delete_list(const std::string &filename) {
    if (_delete_set->size() == 0) {
      return 0;
    }
    std::unique_ptr<_u32[]> delete_list =
        std::make_unique<_u32[]>(_delete_set->size());
    _u32 i = 0;
    for (auto &del : *_delete_set) {
      delete_list[i++] = del;
    }
    return save_bin<_u32>(filename, delete_list.get(), _delete_set->size(), 1);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save(const char *filename, bool compact_before_save) {
    diskann::Timer timer;

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    if (compact_before_save) {
      compact_data();
      compact_frozen_point();
    } else {
      if (!_data_compacted) {
        throw ANNException(
            "Index save for non-compacted index is not yet implemented", -1,
            __FUNCSIG__, __FILE__, __LINE__);
      }
    }

    if (!_save_as_one_file) {
      std::string graph_file = std::string(filename);
      std::string tags_file = std::string(filename) + ".tags";
      std::string data_file = std::string(filename) + ".data";
      std::string delete_list_file = std::string(filename) + ".del";

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
    } else {
      diskann::cout << "Save index in a single file currently not supported. "
                       "Not saving the index."
                    << std::endl;
    }

    reposition_frozen_point_to_end();

    diskann::cout << "Time taken for save: " << timer.elapsed() / 1000000.0
                  << "s." << std::endl;
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
    TagT  *tag_data;
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
    _location_to_tag.reserve(num_data_points);
    _tag_to_location.reserve(num_data_points);
    for (_u32 i = 0; i < (_u32) num_data_points; i++) {
      TagT tag = *(tag_data + i);
      if (_delete_set->find(i) == _delete_set->end()) {
        _location_to_tag.set(i, tag);
        _tag_to_location[tag] = i;
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
      // update and tag lock acquired in load() before calling load_data
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
      _delete_set->insert(delete_list[i]);
    }
    return npts;
  }

  // load the index from file and update the max_degree, cur (navigating
  // node loc), and _final_graph (adjacency list)
  template<typename T, typename TagT>
#ifdef EXEC_ENV_OLS
  void Index<T, TagT>::load(AlignedFileReader &reader, uint32_t num_threads,
                            uint32_t search_l) {
#else
  void Index<T, TagT>::load(const char *filename, uint32_t num_threads,
                            uint32_t search_l) {
#endif
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    _has_built = true;

    size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0;

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

    _nd = data_file_num_pts - _num_frozen_pts;
    _empty_slots.clear();
    _empty_slots.reserve(_max_points);
    for (auto i = _nd; i < _max_points; i++) {
      _empty_slots.insert((uint32_t) i);
    }

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
                               (uint32_t) _max_range_of_loaded_graph,
                               _indexingMaxC, _dim);
    }
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
    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(filename, std::ios::binary);
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

    // If user provides more points than max_points
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
    _u64 graph_offset = header_size;
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

  template<typename T, typename TagT>
  unsigned Index<T, TagT>::calculate_entry_point() {
    //  TODO: need to compute medoid with PQ data too, for now sample at random
    if (_pq_dist) {
      size_t r = (size_t) rand() * (size_t) RAND_MAX + (size_t) rand();
      return (unsigned) (r % (size_t) _nd);
    }

    // allocate and init centroid
    float *center = new float[_aligned_dim]();
    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] = 0;

    for (size_t i = 0; i < _nd; i++)
      for (size_t j = 0; j < _aligned_dim; j++)
        center[j] += (float) _data[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] /= (float) _nd;

    // compute all to one distance
    float *distances = new float[_nd]();
#pragma omp parallel for schedule(static, 65536)
    for (_s64 i = 0; i < (_s64) _nd; i++) {
      // extract point and distance reference
      float   &dist = distances[i];
      const T *cur_vec = _data + (i * (size_t) _aligned_dim);
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
    for (unsigned i = 1; i < _nd; i++) {
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
  std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point(
      const T *query, const unsigned Lsize,
      const std::vector<unsigned> &init_ids, InMemQueryScratch<T> *scratch,
      bool ret_frozen, bool search_invocation) {
    std::vector<Neighbor> &expanded_nodes = scratch->pool();
    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    best_L_nodes.reserve(Lsize);
    tsl::robin_set<unsigned> &inserted_into_pool_rs =
        scratch->inserted_into_pool_rs();
    boost::dynamic_bitset<> &inserted_into_pool_bs =
        scratch->inserted_into_pool_bs();
    std::vector<unsigned> &id_scratch = scratch->id_scratch();
    std::vector<float> &dist_scratch = scratch->dist_scratch();
    assert(id_scratch.size() == 0);
    T *aligned_query = scratch->aligned_query();
    memcpy(aligned_query, query, _dim * sizeof(T));
    if (_normalize_vecs) {
      normalize((float *) aligned_query, _dim);
    }

    float *query_float;
    float *query_rotated;
    float *pq_dists;
    _u8   *pq_coord_scratch;
    // Intialize PQ related scratch to use PQ based distances
    if (_pq_dist) {
      // Get scratch spaces
      PQScratch<T> *pq_query_scratch = scratch->pq_scratch();
      query_float = pq_query_scratch->aligned_query_float;
      query_rotated = pq_query_scratch->rotated_query;
      pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;

      // Copy query vector to float and then to "rotated" query
      for (size_t d = 0; d < _dim; d++) {
        query_float[d] = (float) aligned_query[d];
      }
      pq_query_scratch->set(_dim, aligned_query);

      // center the query and rotate if we have a rotation matrix
      _pq_table.preprocess_query(query_rotated);
      _pq_table.populate_chunk_distances(query_rotated, pq_dists);

      pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;
    }

    if (expanded_nodes.size() > 0 || id_scratch.size() > 0) {
      throw ANNException("ERROR: Clear scratch space before passing.", -1,
                         __FUNCSIG__, __FILE__, __LINE__);
    }

    // Decide whether to use bitset or robin set to mark visited nodes
    auto total_num_points = _max_points + _num_frozen_pts;
    bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;

    if (fast_iterate) {
      if (inserted_into_pool_bs.size() < total_num_points) {
        // hopefully using 2X will reduce the number of allocations.
        auto resize_size = 2 * total_num_points > MAX_POINTS_FOR_USING_BITSET
                               ? MAX_POINTS_FOR_USING_BITSET
                               : 2 * total_num_points;
        inserted_into_pool_bs.resize(resize_size);
      }
    }

    // Lambda to determine if a node has been visited
    auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs,
                           &inserted_into_pool_rs](const unsigned id) {
      return fast_iterate ? inserted_into_pool_bs[id] == 0
                          : inserted_into_pool_rs.find(id) ==
                                inserted_into_pool_rs.end();
    };

    // Lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](
                             const std::vector<unsigned> &ids,
                             std::vector<float>          &dists_out) {
      diskann::aggregate_coords(ids, this->_pq_data, this->_num_pq_chunks,
                                pq_coord_scratch);
      diskann::pq_dist_lookup(pq_coord_scratch, ids.size(),
                              this->_num_pq_chunks, pq_dists, dists_out);
    };

    // Initialize the candidate pool with starting points
    for (auto id : init_ids) {
      if (id >= _max_points + _num_frozen_pts) {
        diskann::cerr << "Out of range loc found as an edge : " << id
                      << std::endl;
        throw diskann::ANNException(
            std::string("Wrong loc") + std::to_string(id), -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }

      if (is_not_visited(id)) {
        if (fast_iterate) {
          inserted_into_pool_bs[id] = 1;
        } else {
          inserted_into_pool_rs.insert(id);
        }

        float distance;
        if (_pq_dist)
          pq_dist_lookup(pq_coord_scratch, 1, this->_num_pq_chunks, pq_dists,
                         &distance);
        else
          distance = _distance->compare(_data + _aligned_dim * (size_t) id,
                                        aligned_query, (unsigned) _aligned_dim);
        Neighbor nn = Neighbor(id, distance);
        best_L_nodes.insert(nn);
      }
    }

    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (best_L_nodes.has_unexpanded_node()) {
      auto nbr = best_L_nodes.closest_unexpanded();   
      auto n = nbr.id;
      // Add node to expanded nodes to create pool for prune later
      if (!search_invocation &&
          (n != _start || _num_frozen_pts == 0 || ret_frozen)) {
        expanded_nodes.emplace_back(nbr);
      }
      // Find which of the nodes in des have not been visited before
      id_scratch.clear();
      dist_scratch.clear();
      {
        if (_dynamic_index)
          _locks[n].lock();
        for (auto id : _final_graph[n]) {
          assert(id < _max_points + _num_frozen_pts);
          if (is_not_visited(id)) {
            id_scratch.push_back(id);
          }
        }

        if (_dynamic_index)
          _locks[n].unlock();
      }

      // Mark nodes visited
      for (auto id : id_scratch) {
        if (fast_iterate) {
          inserted_into_pool_bs[id] = 1;
        } else {
          inserted_into_pool_rs.insert(id);
        }
      }

      // Compute distances to unvisited nodes in the expansion
      if (_pq_dist) {
        assert(dist_scratch.capacity() >= id_scratch.size());
        compute_dists(id_scratch, dist_scratch);
      } else {
        assert(dist_scratch.size() == 0);
        for (size_t m = 0; m < id_scratch.size(); ++m) {
          unsigned id = id_scratch[m];

          if (m + 1 < id_scratch.size()) {
            auto nextn = id_scratch[m + 1];
            diskann::prefetch_vector(
                (const char *) _data + _aligned_dim * (size_t) nextn,
                sizeof(T) * _aligned_dim);
          }

          dist_scratch.push_back( _distance->compare(
              aligned_query, _data + _aligned_dim * (size_t) id,
              (unsigned) _aligned_dim));
        }
      }
      cmps += id_scratch.size();

      // Insert <id, dist> pairs into the pool of candidates
      for (size_t m = 0; m < id_scratch.size(); ++m) {
        best_L_nodes.insert(Neighbor(id_scratch[m], dist_scratch[m]));
      }
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::search_for_point_and_prune(
      int location, _u32 Lindex, std::vector<unsigned> &pruned_list,
      InMemQueryScratch<T> *scratch) {
    std::vector<unsigned> init_ids;
    init_ids.emplace_back(_start);

    iterate_to_fixed_point(_data + _aligned_dim * location, Lindex, init_ids,
                           scratch, true, false);

    auto &pool = scratch->pool();

    for (unsigned i = 0; i < pool.size(); i++) {
      if (pool[i].id == (unsigned) location) {
        pool.erase(pool.begin() + i);
        i--;
      }
    }

    if (pruned_list.size() > 0) {
      throw diskann::ANNException("ERROR: non-empty pruned_list passed", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }

    prune_neighbors(location, pool, pruned_list, scratch);

    assert(!pruned_list.empty());
    assert(_final_graph.size() == _max_points + _num_frozen_pts);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(
      const unsigned location, std::vector<Neighbor> &pool, const float alpha,
      const unsigned degree, const unsigned maxc, std::vector<unsigned> &result,
      InMemQueryScratch<T>                 *scratch,
      const tsl::robin_set<unsigned> *const delete_set_ptr) {
    if (pool.size() == 0)
      return;

    // Truncate pool at maxc and initialize scratch spaces
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(result.size() == 0);
    if (pool.size() > maxc)
      pool.resize(maxc);
    std::vector<float> &occlude_factor = scratch->occlude_factor();
    // occlude_list can be called with the same scratch more than once by
    // search_for_point_and_add_link through inter_insert.
    occlude_factor.clear();
    // Initialize occlude_factor to pool.size() many 0.0f values for correctness
    occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);


    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
      // used for MIPS, where we store a value of eps in cur_alpha to
      // denote pruned out entries which we can skip in later rounds.
      float eps = cur_alpha + 0.01f;

      for (auto iter = pool.begin();
           result.size() < degree && iter != pool.end(); ++iter) {
        if (occlude_factor[iter - pool.begin()] > cur_alpha) {
          continue;
        }
        // Set the entry to float::max so that is not considered again
        occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
        // Add the entry to the result if its not been deleted, and doesn't add a self loop
        if (delete_set_ptr == nullptr ||
            delete_set_ptr->find(iter->id) == delete_set_ptr->end()) {
          if (iter->id != location) {
            result.push_back(iter->id);
          }
        }

        // Update occlude factor for points from iter+1 to pool.end()
        for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
          auto t = iter2 - pool.begin();
          if (occlude_factor[t] > alpha)
            continue;
          float djk =
              _distance->compare(_data + _aligned_dim * (size_t) iter2->id,
                                 _data + _aligned_dim * (size_t) iter->id,
                                 (unsigned) _aligned_dim);
          if (_dist_metric == diskann::Metric::L2 ||
              _dist_metric == diskann::Metric::COSINE) {
            occlude_factor[t] =
                (djk == 0) ? std::numeric_limits<float>::max()
                           : std::max(occlude_factor[t], iter2->distance / djk);
          } else if (_dist_metric == diskann::Metric::INNER_PRODUCT) {
            // Improvization for flipping max and min dist for MIPS
            float x = -iter2->distance;
            float y = -djk;
            if (y > cur_alpha * x) {
              occlude_factor[t] = std::max(occlude_factor[t], eps);
            }
          }
        }
      }
      cur_alpha *= 1.2;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_neighbors(const unsigned         location,
                                       std::vector<Neighbor> &pool,
                                       std::vector<unsigned> &pruned_list,
                                       InMemQueryScratch<T>  *scratch) {
    prune_neighbors(location, pool, _indexingRange, _indexingMaxC,
                    _indexingAlpha, pruned_list, scratch);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_neighbors(
      const unsigned location, std::vector<Neighbor> &pool, const _u32 range,
      const _u32 max_candidate_size, const float alpha,
      std::vector<unsigned> &pruned_list, InMemQueryScratch<T> *scratch) {
    if (pool.size() == 0) {
      throw diskann::ANNException("Pool passed to prune_neighbors is empty", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }

    _max_observed_degree = (std::max)(_max_observed_degree, range);

    // If using _pq_build, over-write the PQ distances with actual distances
    if (_pq_dist) {
      for (auto& ngh : pool)
        ngh.distance = _distance->compare(
            _data + _aligned_dim * (size_t) ngh.id,
            _data + _aligned_dim * (size_t) location, (unsigned) _aligned_dim);
    }

    // sort the pool based on distance to query and prune it with occlude_list
    std::sort(pool.begin(), pool.end());
    pruned_list.clear();
    pruned_list.reserve(range);
    occlude_list(location, pool, alpha, range, max_candidate_size, pruned_list,
                 scratch);
    assert(pruned_list.size() <= range);

    if (_saturate_graph && alpha > 1) {
      for (const auto &node : pool) {
        if (pruned_list.size() >= range)
          break;
        if ((std::find(pruned_list.begin(), pruned_list.end(), node.id) ==
             pruned_list.end()) &&
            node.id != location)
          pruned_list.push_back(node.id);
      }
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::inter_insert(unsigned               n,
                                    std::vector<unsigned> &pruned_list,
                                    const _u32             range,
                                    InMemQueryScratch<T>  *scratch) {
    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      // des.loc is the loc of the neighbors of n
      assert(des < _max_points + _num_frozen_pts);
      // des_pool contains the neighbors of the neighbors of n
      std::vector<unsigned> copy_of_neighbors;
      bool                  prune_needed = false;
      {
        LockGuard guard(_locks[des]);
        auto     &des_pool = _final_graph[des];
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < (_u64) (GRAPH_SLACK_FACTOR * range)) {
            des_pool.emplace_back(n);
            prune_needed = false;
          } else {
            copy_of_neighbors.reserve(des_pool.size() + 1);
            copy_of_neighbors = des_pool;
            copy_of_neighbors.push_back(n);
            prune_needed = true;
          }
        }
      }  // des lock is released by this point

      if (prune_needed) {
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);

        size_t reserveSize =
            (size_t) (std::ceil(1.05 * GRAPH_SLACK_FACTOR * range));
        dummy_visited.reserve(reserveSize);
        dummy_pool.reserve(reserveSize);

        for (auto cur_nbr : copy_of_neighbors) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != des) {
            float dist =
                _distance->compare(_data + _aligned_dim * (size_t) des,
                                   _data + _aligned_dim * (size_t) cur_nbr,
                                   (unsigned) _aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
            dummy_visited.insert(cur_nbr);
          }
        }
        std::vector<unsigned> new_out_neighbors;
        prune_neighbors(des, dummy_pool, new_out_neighbors, scratch);
        {
          LockGuard guard(_locks[des]);

          _final_graph[des] = new_out_neighbors;
        }
      }
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::inter_insert(unsigned               n,
                                    std::vector<unsigned> &pruned_list,
                                    InMemQueryScratch<T>  *scratch) {
    inter_insert(n, pruned_list, _indexingRange, scratch);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::link(Parameters &parameters) {
    unsigned num_threads = parameters.Get<unsigned>("num_threads");
    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    _saturate_graph = parameters.Get<bool>("saturate_graph");

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    _indexingQueueSize = parameters.Get<unsigned>("L");  // Search list size
    _indexingRange = parameters.Get<unsigned>("R");
    _indexingMaxC = parameters.Get<unsigned>("C");
    _indexingAlpha = parameters.Get<float>("alpha");

    /* visit_order is a vector that is initialized to the entire graph */
    std::vector<unsigned>          visit_order;
    std::vector<diskann::Neighbor> pool, tmp;
    tsl::robin_set<unsigned>       visited;
    visit_order.reserve(_nd + _num_frozen_pts);
    for (unsigned i = 0; i < (unsigned) _nd; i++) {
      visit_order.emplace_back(i);
    }

    if (_num_frozen_pts > 0)
      visit_order.emplace_back((unsigned) _max_points);

    // if there are frozen points, the first such one is set to be the _start
    if (_num_frozen_pts > 0)
      _start = (unsigned) _max_points;
    else
      _start = calculate_entry_point();

    for (uint64_t p = 0; p < _nd; p++) {
      _final_graph[p].reserve(
          (size_t) (std::ceil(_indexingRange * GRAPH_SLACK_FACTOR * 1.05)));
    }

    std::vector<unsigned> init_ids;
    init_ids.emplace_back(_start);

    diskann::Timer link_timer;

#pragma omp parallel for schedule(dynamic, 2048)
    for (_s64 node_ctr = 0; node_ctr < (_s64) (visit_order.size());
         node_ctr++) {
      auto node = visit_order[node_ctr];

      ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
      auto scratch = manager.scratch_space();

      std::vector<unsigned> pruned_list;
      search_for_point_and_prune(node, _indexingQueueSize, pruned_list,
                                 scratch);
      {
        LockGuard guard(_locks[node]);
        _final_graph[node].reserve(
            (_u64) (_indexingRange * GRAPH_SLACK_FACTOR * 1.05));
        _final_graph[node] = pruned_list;
        assert(_final_graph[node].size() <= _indexingRange);
      }

      inter_insert(node, pruned_list, scratch);

      if (node_ctr % 100000 == 0) {
        diskann::cout << "\r" << (100.0 * node_ctr) / (visit_order.size())
                      << "% of index build completed." << std::flush;
      }
    }

    if (_nd > 0) {
      diskann::cout << "Starting final cleanup.." << std::flush;
    }
#pragma omp parallel for schedule(dynamic, 2048)
    for (_s64 node_ctr = 0; node_ctr < (_s64) (visit_order.size());
         node_ctr++) {
      auto node = visit_order[node_ctr];
      if (_final_graph[node].size() > _indexingRange) {
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        auto scratch = manager.scratch_space();

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
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
            dummy_visited.insert(cur_nbr);
          }
        }
        prune_neighbors(node, dummy_pool, new_out_neighbors, scratch);

        _final_graph[node].clear();
        for (auto id : new_out_neighbors)
          _final_graph[node].emplace_back(id);
      }
    }
    if (_nd > 0) {
      diskann::cout << "done. Link time: "
                    << ((double) link_timer.elapsed() / (double) 1000000) << "s"
                    << std::endl;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::set_start_point(T *data) {
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    if (_nd > 0)
      throw ANNException("Can not set starting point for a non-empty index", -1,
                         __FUNCSIG__, __FILE__, __LINE__);

    memcpy(_data + _aligned_dim * _max_points, data, _aligned_dim * sizeof(T));
    _has_built = true;
    diskann::cout << "Index start point set" << std::endl;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::set_start_point_at_random(T radius) {
    std::vector<double>        real_vec;
    std::random_device         rd{};
    std::mt19937               gen{rd()};
    std::normal_distribution<> d{0.0, 1.0};
    double                     norm_sq = 0.0;
    for (size_t i = 0; i < _aligned_dim; ++i) {
      auto r = d(gen);
      real_vec.push_back(r);
      norm_sq += r * r;
    }

    double         norm = std::sqrt(norm_sq);
    std::vector<T> start_vec;
    for (auto iter : real_vec)
      start_vec.push_back(static_cast<T>(iter * radius / norm));

    set_start_point(start_vec.data());
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
        _location_to_tag.set(static_cast<unsigned>(i), tags[i]);
      }
    }

    uint32_t index_R = parameters.Get<uint32_t>("R");
    uint32_t num_threads_index = parameters.Get<uint32_t>("num_threads");
    uint32_t index_L = parameters.Get<uint32_t>("L");
    uint32_t maxc = parameters.Get<uint32_t>("C");

    if (_query_scratch.size() == 0) {
      initialize_query_scratch(5 + num_threads_index, index_L, index_L, index_R,
                               maxc, _aligned_dim);
    }

    generate_frozen_point();
    link(parameters);

    size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max = std::max(max, pool.size());
      min = std::min(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    diskann::cout << "Index built with degree: max:" << max
                  << "  avg:" << (float) total / (float) (_nd + _num_frozen_pts)
                  << "  min:" << min << "  count(deg<2):" << cnt << std::endl;

    _max_observed_degree = std::max((unsigned) max, _max_observed_degree);
    _has_built = true;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const T *data, const size_t num_points_to_load,
                             Parameters              &parameters,
                             const std::vector<TagT> &tags) {
    if (num_points_to_load == 0) {
      throw ANNException("Do not call build with 0 points", -1, __FUNCSIG__,
                         __FILE__, __LINE__);
    }
    if (_pq_dist) {
      throw ANNException(
          "ERROR: DO not use this build interface with PQ distance", -1,
          __FUNCSIG__, __FILE__, __LINE__);
    }

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    
    {
      std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
      _nd = num_points_to_load;

      memcpy((char *) _data, (char *) data, _aligned_dim * _nd * sizeof(T));

      if (_normalize_vecs) {
        for (uint64_t i = 0; i < num_points_to_load; i++) {
          normalize(_data + _aligned_dim * i, _aligned_dim);
        }
      }
    }

    build_with_data_populated(parameters, tags);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char              *filename,
                             const size_t             num_points_to_load,
                             Parameters              &parameters,
                             const std::vector<TagT> &tags) {
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    if (num_points_to_load == 0)
      throw ANNException("Do not call build with 0 points", -1, __FUNCSIG__,
                         __FILE__, __LINE__);

    if (!file_exists(filename)) {
      std::stringstream stream;
      stream << "ERROR: Data file " << filename << " does not exist."
             << std::endl;
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

      if (_pq_dist)
        aligned_free(_pq_data);
      else
        aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (num_points_to_load > file_num_points) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << num_points_to_load
             << " points and file has only " << file_num_points << " points."
             << std::endl;

      if (_pq_dist)
        aligned_free(_pq_data);
      else
        aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (file_dim != _dim) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << _dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      diskann::cerr << stream.str() << std::endl;

      if (_pq_dist)
        aligned_free(_pq_data);
      else
        aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (_pq_dist) {
      double p_val = std::min(
          1.0, ((double) MAX_PQ_TRAINING_SET_SIZE / (double) file_num_points));

      std::string suffix = _use_opq ? "_opq" : "_pq";
      suffix += std::to_string(_num_pq_chunks);
      auto pq_pivots_file = std::string(filename) + suffix + "_pivots.bin";
      auto pq_compressed_file =
          std::string(filename) + suffix + "_compressed.bin";
      generate_quantized_data<T>(std::string(filename), pq_pivots_file,
                                 pq_compressed_file, _dist_metric, p_val,
                                 _num_pq_chunks, _use_opq);

      copy_aligned_data_from_file<_u8>(pq_compressed_file.c_str(), _pq_data,
                                       file_num_points, _num_pq_chunks,
                                       _num_pq_chunks);
      _pq_table.load_pq_centroid_bin(pq_pivots_file.c_str(), _num_pq_chunks);
    }

    copy_aligned_data_from_file<T>(filename, _data, file_num_points, file_dim,
                                   _aligned_dim);
    if (_normalize_vecs) {
      for (uint64_t i = 0; i < file_num_points; i++) {
        normalize(_data + _aligned_dim * i, _aligned_dim);
      }
    }

    diskann::cout << "Using only first " << num_points_to_load
                  << " from file.. " << std::endl;

    {
      std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
      _nd = num_points_to_load;
    }
    build_with_data_populated(parameters, tags);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char  *filename,
                             const size_t num_points_to_load,
                             Parameters &parameters, const char *tag_filename) {
    std::vector<TagT> tags;

    if (_enable_tags) {
      std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
      if (tag_filename == nullptr) {
        throw ANNException("Tag filename is null, while _enable_tags is set",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
      } else {
        if (file_exists(tag_filename)) {
          diskann::cout << "Loading tags from " << tag_filename
                        << " for vamana index build" << std::endl;
          TagT  *tag_data = nullptr;
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
    build(filename, num_points_to_load, parameters, tags);
  }

  template<typename T, typename TagT>
  template<typename IdType>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T       *query,
                                                       const size_t   K,
                                                       const unsigned L,
                                                       IdType        *indices,
                                                       float *distances) {
    if (K > (uint64_t) L) {
      throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__,
                         __FILE__, __LINE__);
    }

    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto                                      scratch = manager.scratch_space();

    if (L > scratch->get_L()) {
      diskann::cout << "Attempting to expand query scratch_space. Was created "
                    << "with Lsize: " << scratch->get_L()
                    << " but search L is: " << L << std::endl;
      scratch->resize_for_new_L(L);
      diskann::cout << "Resize completed. New scratch->L is "
                    << scratch->get_L() << std::endl;
    }

    std::vector<unsigned> init_ids;
    init_ids.push_back(_start);
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    auto retval =
        iterate_to_fixed_point(query, L, init_ids, scratch, true, true);
    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();

    size_t pos = 0;
    for (int i = 0; i < best_L_nodes.size(); ++i) {
      if (best_L_nodes[i].id < _max_points) {
        // safe because Index uses uint32_t ids internally 
        // and IDType will be uint32_t or uint64_t
        indices[pos] = (IdType) best_L_nodes[i].id;
        if (distances != nullptr) {
#ifdef EXEC_ENV_OLS
          // DLVS expects negative distances
          distances[pos] = best_L_nodes[i].distance;
#else
          distances[pos] = _dist_metric == diskann::Metric::INNER_PRODUCT
                               ? -1 * best_L_nodes[i].distance
                               : best_L_nodes[i].distance;
#endif
        }
        pos++;
      }
      if (pos == K)
        break;
    }
    if (pos < K) {
      diskann::cerr << "Found fewer than K elements for query" << std::endl;
    }

    return retval;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const uint64_t K,
                                          const unsigned L, TagT *tags,
                                          float            *distances,
                                          std::vector<T *> &res_vectors) {
    if (K > (uint64_t) L) {
      throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__,
                         __FILE__, __LINE__);
    }
    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto                                      scratch = manager.scratch_space();

    if (L > scratch->get_L()) {
      diskann::cout << "Attempting to expand query scratch_space. Was created "
                    << "with Lsize: " << scratch->get_L()
                    << " but search L is: " << L << std::endl;
      scratch->resize_for_new_L(L);
      diskann::cout << "Resize completed. New scratch->L is "
                    << scratch->get_L() << std::endl;
    }

    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);

    std::vector<unsigned> init_ids(1, _start);
    iterate_to_fixed_point(query, L, init_ids, scratch, true, true);
    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    assert(best_L_nodes.size() <= L);

    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);

    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size(); ++i) {
      auto node = best_L_nodes[i];

      TagT tag;
      if (_location_to_tag.try_get(node.id, tag)) {
        tags[pos] = tag;

        if (res_vectors.size() > 0) {
          memcpy(res_vectors[pos], _data + ((size_t) node.id) * _aligned_dim,
                 _dim * sizeof(T));
        }

        if (distances != nullptr) {
#ifdef EXEC_ENV_OLS
          distances[pos] = node.distance;  // DLVS expects negative distances
#else
          distances[pos] = _dist_metric == INNER_PRODUCT ? -1 * node.distance
                                                         : node.distance;
#endif
        }
        pos++;
        // If res_vectors.size() < k, clip at the value.
        if (pos == K || pos == res_vectors.size())
          break;
      }
    }

    return pos;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::get_num_points() {
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    return _nd;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::get_max_points() {
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    return _max_points;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::generate_frozen_point() {
    if (_num_frozen_pts == 0)
      return 0;

    if (_nd == 0) {
      throw ANNException("ERROR: Can not pick a frozen point since nd=0", -1,
                         __FUNCSIG__, __FILE__, __LINE__);
    }
    size_t res = calculate_entry_point();

    if (_pq_dist) {
      // copy the PQ data corresponding to the point returned by
      // calculate_entry_point
      memcpy(_pq_data + _max_points * _num_pq_chunks,
             _pq_data + res * _num_pq_chunks,
             _num_pq_chunks * DIV_ROUND_UP(NUM_PQ_BITS, 8));
    } else {
      memcpy(_data + _max_points * _aligned_dim, _data + res * _aligned_dim,
             _aligned_dim * sizeof(T));
    }

    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::enable_delete() {
    assert(_enable_tags);

    if (!_enable_tags) {
      diskann::cerr << "Tags must be instantiated for deletions" << std::endl;
      return -2;
    }

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    if (_data_compacted) {
      for (unsigned slot = (unsigned) _nd; slot < _max_points; ++slot) {
        _empty_slots.insert(slot);
      }
    }

    return 0;
  }

  template<typename T, typename TagT>
  inline void Index<T, TagT>::process_delete(
      const tsl::robin_set<unsigned> &old_delete_set, size_t loc,
      const unsigned range, const unsigned maxc, const float alpha,
      InMemQueryScratch<T> *scratch) {
    tsl::robin_set<unsigned> &expanded_nodes_set =
        scratch->expanded_nodes_set();
    std::vector<Neighbor> &expanded_nghrs_vec = scratch->expanded_nodes_vec();

    // If this condition were not true, deadlock could result
    assert(old_delete_set.find(loc) == old_delete_set.end());

    std::vector<unsigned> adj_list;
    {
      // Acquire and release lock[loc] before acquiring locks for neighbors
      std::unique_lock<non_recursive_mutex> adj_list_lock;
      if (_conc_consolidate)
        adj_list_lock = std::unique_lock<non_recursive_mutex>(_locks[loc]);
      adj_list = _final_graph[loc];
    }

    bool modify = false;
    for (auto ngh : adj_list) {
      if (old_delete_set.find(ngh) == old_delete_set.end()) {
        expanded_nodes_set.insert(ngh);
      } else {
        modify = true;

        std::unique_lock<non_recursive_mutex> ngh_lock;
        if (_conc_consolidate)
          ngh_lock = std::unique_lock<non_recursive_mutex>(_locks[ngh]);
        for (auto j : _final_graph[ngh])
          if (j != loc && old_delete_set.find(j) == old_delete_set.end())
            expanded_nodes_set.insert(j);
      }
    }

    if (modify) {
      if (expanded_nodes_set.size() <= range) {
        std::unique_lock<non_recursive_mutex> adj_list_lock(_locks[loc]);
        _final_graph[loc].clear();
        for (auto &ngh : expanded_nodes_set)
          _final_graph[loc].push_back(ngh);
      } else {
        // Create a pool of Neighbor candidates from the expanded_nodes_set
        expanded_nghrs_vec.reserve(expanded_nodes_set.size());
        for (auto &ngh : expanded_nodes_set) {
          expanded_nghrs_vec.emplace_back(
              ngh, _distance->compare(_data + _aligned_dim * loc,
                                      _data + _aligned_dim * ngh,
                                      (unsigned) _aligned_dim));
        }
        std::sort(expanded_nghrs_vec.begin(), expanded_nghrs_vec.end());
        std::vector<unsigned> &occlude_list_output =
            scratch->occlude_list_output();
        occlude_list(loc, expanded_nghrs_vec, alpha, range, maxc,
                     occlude_list_output, scratch, &old_delete_set);
        std::unique_lock<non_recursive_mutex> adj_list_lock(_locks[loc]);
        _final_graph[loc] = occlude_list_output;
      }
    }
  }
  
   
  // Returns number of live points left after consolidation
  template<typename T, typename TagT>
  consolidation_report Index<T, TagT>::consolidate_deletes(
      const Parameters &params) {
    if (!_enable_tags)
      throw diskann::ANNException("Point tag array not instantiated", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);

    {
      std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
      std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
      std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
      if (_empty_slots.size() + _nd != _max_points) {
        std::string err = "#empty slots + nd != max points";
        diskann::cerr << err << std::endl;
        throw ANNException(err, -1, __FUNCSIG__, __FILE__, __LINE__);
      }

      if (_location_to_tag.size() + _delete_set->size() != _nd) {
        diskann::cerr << "Error: _location_to_tag.size ("
                      << _location_to_tag.size() << ")  + _delete_set->size ("
                      << _delete_set->size() << ") != _nd(" << _nd << ") ";
        return consolidation_report(diskann::consolidation_report::status_code::
                                        INCONSISTENT_COUNT_ERROR,
                                    0, 0, 0, 0, 0, 0, 0);
      }

      if (_location_to_tag.size() != _tag_to_location.size()) {
        throw diskann::ANNException(
            "_location_to_tag and _tag_to_location not of same size", -1,
            __FUNCSIG__, __FILE__, __LINE__);
      }
    }

    std::unique_lock<std::shared_timed_mutex> update_lock(_update_lock,
                                                          std::defer_lock);
    if (!_conc_consolidate)
      update_lock.lock();

    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock,
                                                 std::defer_lock);
    if (!cl.try_lock()) {
      diskann::cerr
          << "Consildate delete function failed to acquire consolidate lock"
          << std::endl;
      return consolidation_report(
          diskann::consolidation_report::status_code::LOCK_FAIL, 0, 0, 0, 0, 0,
          0, 0);
    }

    diskann::cout << "Starting consolidate_deletes... ";

    std::unique_ptr<tsl::robin_set<unsigned>> old_delete_set(
        new tsl::robin_set<unsigned>);
    {
      std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
      std::swap(_delete_set, old_delete_set);
    }

    if (old_delete_set->find(_start) != old_delete_set->end()) {
      throw diskann::ANNException("ERROR: start node has been deleted", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }

    const unsigned range = params.Get<unsigned>("R");
    const unsigned maxc = params.Get<unsigned>("C");
    const float    alpha = params.Get<float>("alpha");
    const unsigned num_threads = params.Get<unsigned>("num_threads") == 0
                                     ? omp_get_num_threads()
                                     : params.Get<unsigned>("num_threads");

    unsigned       num_calls_to_process_delete = 0;
    diskann::Timer timer;
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8192) \
    reduction(+:num_calls_to_process_delete)
    for (_s64 loc = 0; loc < (_s64) _max_points; loc++) {
      if (old_delete_set->find((_u32) loc) == old_delete_set->end() &&
          !_empty_slots.is_in_set((_u32) loc)) {
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        auto scratch = manager.scratch_space();
        process_delete(*old_delete_set, loc, range, maxc, alpha, scratch);
        num_calls_to_process_delete += 1;
      }
    }
    for (_s64 loc = _max_points; loc < (_s64) (_max_points + _num_frozen_pts);
         loc++) {
      ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
      auto scratch = manager.scratch_space();
      process_delete(*old_delete_set, loc, range, maxc, alpha, scratch);
      num_calls_to_process_delete += 1;
    }

    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    size_t ret_nd = release_locations(*old_delete_set);
    size_t max_points = _max_points;
    size_t empty_slots_size = _empty_slots.size();

    std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
    size_t delete_set_size = _delete_set->size();
    size_t old_delete_set_size = old_delete_set->size();

    if (!_conc_consolidate) {
      update_lock.unlock();
    }

    double duration = timer.elapsed() / 1000000.0;
    diskann::cout << " done in " << duration << " seconds." << std::endl;
    return consolidation_report(
        diskann::consolidation_report::status_code::SUCCESS, ret_nd, max_points,
        empty_slots_size, old_delete_set_size, delete_set_size,
        num_calls_to_process_delete, duration);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_frozen_point() {
    if (_nd < _max_points) {
      if (_num_frozen_pts == 1) {
        // set new _start to be frozen point
        _start = (_u32) _nd;
        if (!_final_graph[_max_points].empty()) {
          for (unsigned i = 0; i < _nd; i++)
            for (unsigned j = 0; j < _final_graph[i].size(); j++)
              if (_final_graph[i][j] == _max_points)
                _final_graph[i][j] = (_u32) _nd;

          _final_graph[_nd].clear();
          _final_graph[_nd].swap(_final_graph[_max_points]);

          memcpy((void *) (_data + _aligned_dim * _nd),
                 _data + (size_t) _aligned_dim * _max_points, sizeof(T) * _dim);
          memset((_data + (size_t) _aligned_dim * _max_points), 0,
                 sizeof(T) * _aligned_dim);
        }
      } else if (_num_frozen_pts > 1) {
        throw ANNException("Case not implemented.", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
      }
    }
  }

  // Should be called after acquiring _update_lock
  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data() {
    if (!_dynamic_index)
      throw ANNException("Can not compact a non-dynamic index", -1, __FUNCSIG__,
                         __FILE__, __LINE__);

    if (_data_compacted) {
      diskann::cerr
          << "Warning! Calling compact_data() when _data_compacted is true!"
          << std::endl;
      return;
    }

    if (_delete_set->size() > 0) {
      throw ANNException(
          "Can not compact data when index has non-empty _delete_set of "
          "size: " +
              std::to_string(_delete_set->size()),
          -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::Timer timer;

    std::vector<unsigned> new_location =
        std::vector<unsigned>(_max_points + _num_frozen_pts, (_u32) UINT32_MAX);

    _u32           new_counter = 0;
    std::set<_u32> empty_locations;
    for (_u32 old_location = 0; old_location < _max_points; old_location++) {
      if (_location_to_tag.contains(old_location)) {
        new_location[old_location] = new_counter;
        new_counter++;
      } else {
        empty_locations.insert(old_location);
      }
    }
    for (_u32 old_location = _max_points;
         old_location < _max_points + _num_frozen_pts; old_location++) {
      new_location[old_location] = old_location;
    }

    // If start node is removed, throw an exception
    if (_start < _max_points && !_location_to_tag.contains(_start)) {
      throw diskann::ANNException("ERROR: Start node deleted.", -1, __FUNCSIG__,
                                  __FILE__, __LINE__);
    }

    size_t num_dangling = 0;
    for (unsigned old = 0; old < _max_points + _num_frozen_pts; ++old) {
      std::vector<unsigned> new_adj_list;

      if ((new_location[old] < _max_points)  // If point continues to exist
          || (old >= _max_points && old < _max_points + _num_frozen_pts)) {
        new_adj_list.reserve(_final_graph[old].size());
        for (auto ngh_iter : _final_graph[old]) {
          if (empty_locations.find(ngh_iter) != empty_locations.end()) {
            ++num_dangling;
            diskann::cerr << "Error in compact_data(). _final_graph[" << old
                          << "] has neighbor " << ngh_iter
                          << " which is a location not associated with any tag."
                          << std::endl;

          } else {
            new_adj_list.push_back(new_location[ngh_iter]);
          }
        }
        _final_graph[old].swap(new_adj_list);

        // Move the data and adj list to the correct position
        if (new_location[old] != old) {
          assert(new_location[old] < old);
          _final_graph[new_location[old]].swap(_final_graph[old]);

          memcpy((void *) (_data + _aligned_dim * (size_t) new_location[old]),
                 (void *) (_data + _aligned_dim * (size_t) old),
                 _aligned_dim * sizeof(T));
        }
      } else {
        _final_graph[old].clear();
      }
    }
    diskann::cerr << "#dangling references after data compaction: "
                  << num_dangling << std::endl;

    _tag_to_location.clear();
    for (auto pos = _location_to_tag.find_first(); pos.is_valid();
         pos = _location_to_tag.find_next(pos)) {
      const auto tag = _location_to_tag.get(pos);
      _tag_to_location[tag] = new_location[pos._key];
    }
    _location_to_tag.clear();
    for (const auto &iter : _tag_to_location) {
      _location_to_tag.set(iter.second, iter.first);
    }

    for (_u64 old = _nd; old < _max_points; ++old) {
      _final_graph[old].clear();
    }
    _empty_slots.clear();
    for (auto i = _nd; i < _max_points; i++) {
      _empty_slots.insert((uint32_t) i);
    }

    _data_compacted = true;
    diskann::cout << "Time taken for compact_data: "
                  << timer.elapsed() / 1000000. << "s." << std::endl;
  }

  // 
  // Caller must hold unique _tag_lock and _delete_lock before calling this
  //
  template<typename T, typename TagT>
  int Index<T, TagT>::reserve_location() {
    if (_nd >= _max_points) {
      return -1;
    }
    unsigned location;
    if (_data_compacted && _empty_slots.is_empty()) {
      // This code path is encountered when enable_delete hasn't been
      // called yet, so no points have been deleted and _empty_slots
      // hasn't been filled in. In that case, just keep assigning
      // consecutive locations.
      location = (unsigned) _nd;
    } else {
      assert(_empty_slots.size() != 0);
      assert(_empty_slots.size() + _nd == _max_points);

      location = _empty_slots.pop_any();
      _delete_set->erase(location);
    }

    ++_nd;
    return location;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::release_location(int location) {
    if (_empty_slots.is_in_set(location))
      throw ANNException(
          "Trying to release location, but location already in empty slots", -1,
          __FUNCSIG__, __FILE__, __LINE__);
    _empty_slots.insert(location);

    _nd--;
    return _nd;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::release_locations(
      const tsl::robin_set<unsigned> &locations) {
    for (auto location : locations) {
      if (_empty_slots.is_in_set(location))
        throw ANNException(
            "Trying to release location, but location already in empty slots",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      _empty_slots.insert(location);

      _nd--;
    }

    if (_empty_slots.size() + _nd != _max_points)
      throw ANNException("#empty slots + nd != max points", -1, __FUNCSIG__,
                         __FILE__, __LINE__);

    return _nd;
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
    _locks = std::vector<non_recursive_mutex>(new_max_points + 1);

    reposition_point((_u32) _max_points, (_u32) new_max_points);
    _max_points = new_max_points;
    _start = (_u32) new_max_points;

    _empty_slots.reserve(_max_points);
    for (auto i = _nd; i < _max_points; i++) {
      _empty_slots.insert((uint32_t) i);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    diskann::cout << "Resizing took: "
                  << std::chrono::duration<double>(stop - start).count() << "s"
                  << std::endl;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::insert_point(const T *point, const TagT tag) {
    assert(_has_built);
    if (tag == static_cast<TagT>(0)) {
      throw diskann::ANNException(
          "Do not insert point with tag 0. That is reserved for points hidden "
          "from the user.",
          -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::shared_lock<std::shared_timed_mutex> shared_ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    // Find a vacant location in the data array to insert the new point
    auto location = reserve_location();
    if (location == -1) {
#if EXPAND_IF_FULL
      dl.unlock();
      tl.unlock();
      shared_ul.unlock();

      {
        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
        tl.lock();
        dl.lock();

        if (_nd >= _max_points) {
          auto new_max_points = (size_t) (_max_points * INDEX_GROWTH_FACTOR);
          resize(new_max_points);
        }

        dl.unlock();
        tl.unlock();
        ul.unlock();
      }

      shared_ul.lock();
      tl.lock();
      dl.lock();

      location = reserve_location();
      if (location == -1) {
        throw diskann::ANNException(
            "Cannot reserve location even after expanding graph. Terminating.",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }
#else
      return -1;
#endif
    }
    dl.unlock();

    // Insert tag and mapping to location
    if (_enable_tags) {
      if (_tag_to_location.find(tag) != _tag_to_location.end()) {
        release_location(location);
        return -1;
      }

      _tag_to_location[tag] = location;
      _location_to_tag.set(location, tag);
    }
    tl.unlock();

    // Copy the vector in to the data array
    auto offset_data = _data + (size_t) _aligned_dim * location;
    memset((void *) offset_data, 0, sizeof(T) * _aligned_dim);
    memcpy((void *) offset_data, point, sizeof(T) * _dim);

    if (_normalize_vecs) {
      normalize((float *) offset_data, _dim);
    }

    // Find and add appropriate graph edges
    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto                                      scratch = manager.scratch_space();
    std::vector<unsigned>                     pruned_list;
    search_for_point_and_prune(location, _indexingQueueSize, pruned_list,
                               scratch);
    {
      std::shared_lock<std::shared_timed_mutex> tlock(_tag_lock,
                                                      std::defer_lock);
      if (_conc_consolidate)
        tlock.lock();

      LockGuard guard(_locks[location]);
      _final_graph[location].clear();
      _final_graph[location].reserve(
          (_u64) (_indexingRange * GRAPH_SLACK_FACTOR * 1.05));

      for (auto link : pruned_list) {
        if (_conc_consolidate)
          if (!_location_to_tag.contains(link))
            continue;
        _final_graph[location].emplace_back(link);
      }
      assert(_final_graph[location].size() <= _indexingRange);

      if (_conc_consolidate)
        tlock.unlock();
    }

    inter_insert(location, pruned_list, scratch);

    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const TagT &tag) {
    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
    _data_compacted = false;

    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      diskann::cerr << "Delete tag not found " << tag << std::endl;
      return -1;
    }
    assert(_tag_to_location[tag] < _max_points);

    const auto location = _tag_to_location[tag];
    _delete_set->insert(location);
    _location_to_tag.erase(location);
    _tag_to_location.erase(tag);

    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::lazy_delete(const std::vector<TagT> &tags,
                                   std::vector<TagT>       &failed_tags) {
    if (failed_tags.size() > 0) {
      throw ANNException("failed_tags should be passed as an empty list", -1,
                         __FUNCSIG__, __FILE__, __LINE__);
    }
    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
    _data_compacted = false;

    for (auto tag : tags) {
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        failed_tags.push_back(tag);
      } else {
        const auto location = _tag_to_location[tag];
        _delete_set->insert(location);
        _location_to_tag.erase(location);
        _tag_to_location.erase(tag);
      }
    }
  }

  template<typename T, typename TagT>
  bool Index<T, TagT>::is_index_saved() {
    return _is_saved;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_active_tags(tsl::robin_set<TagT> &active_tags) {
    active_tags.clear();
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    for (auto iter : _tag_to_location) {
      active_tags.insert(iter.first);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::print_status() {
    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
    std::shared_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);

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
                  << "Data compacted: " << this->_data_compacted << std::endl;
    diskann::cout << "---------------------------------------------------------"
                     "------------"
                  << std::endl;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::count_nodes_at_bfs_levels() {
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

    boost::dynamic_bitset<> visited(_max_points + _num_frozen_pts);

    size_t MAX_BFS_LEVELS = 32;
    auto   bfs_sets = new tsl::robin_set<unsigned>[MAX_BFS_LEVELS];

    if (_dynamic_index) {
      for (unsigned i = _max_points; i < _max_points + _num_frozen_pts; ++i) {
        bfs_sets[0].insert(i);
        visited.set(i);
      }
    } else {
      bfs_sets[0].insert(_start);
      visited.set(_start);
    }

    for (size_t l = 0; l < MAX_BFS_LEVELS - 1; ++l) {
      diskann::cout << "Number of nodes at BFS level " << l << " is "
                    << bfs_sets[l].size() << std::endl;
      if (bfs_sets[l].size() == 0)
        break;
      for (auto node : bfs_sets[l]) {
        for (auto nghbr : _final_graph[node]) {
          if (!visited.test(nghbr)) {
            visited.set(nghbr);
            bfs_sets[l + 1].insert(nghbr);
          }
        }
      }
    }

    delete[] bfs_sets;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::optimize_index_layout() {  // use after build or load
    if (_dynamic_index) {
      throw diskann::ANNException(
          "Optimize_index_layout not implemented for dyanmic indices", -1,
          __FUNCSIG__, __FILE__, __LINE__);
    }

    _data_len = (_aligned_dim + 1) * sizeof(float);
    _neighbor_len = (_max_observed_degree + 1) * sizeof(unsigned);
    _node_size = _data_len + _neighbor_len;
    _opt_graph = new char[_node_size * _nd];
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

    NeighborPriorityQueue retset(L);
    std::vector<unsigned> init_ids(L);

    boost::dynamic_bitset<> flags{_nd, 0};
    unsigned                tmp_l = 0;
    unsigned               *neighbors =
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
      T    *x = (T *) (_opt_graph + _node_size * id);
      float norm_x = *x;
      x++;
      float dist =
          dist_fast->compare(x, query, norm_x, (unsigned) _aligned_dim);
      retset.insert(Neighbor(id, dist));
      flags[id] = true;
      L++;
    }

    while (retset.has_unexpanded_node()) {
      auto nbr = retset.closest_unexpanded();
      auto n = nbr.id;
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
        T    *data = (T *) (_opt_graph + _node_size * id);
        float norm = *data;
        data++;
        float dist =
            dist_fast->compare(query, data, norm, (unsigned) _aligned_dim);
        Neighbor nn(id, dist);
        retset.insert(nn);
      }
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
                                             float    *distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
  Index<uint8_t, uint64_t>::search<uint32_t>(const uint8_t *query,
                                             const size_t K, const unsigned L,
                                             uint32_t *indices,
                                             float    *distances);
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
                                             float    *distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
  Index<uint8_t, uint32_t>::search<uint32_t>(const uint8_t *query,
                                             const size_t K, const unsigned L,
                                             uint32_t *indices,
                                             float    *distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
  Index<int8_t, uint32_t>::search<uint64_t>(const int8_t *query, const size_t K,
                                            const unsigned L, uint64_t *indices,
                                            float *distances);
  template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
  Index<int8_t, uint32_t>::search<uint32_t>(const int8_t *query, const size_t K,
                                            const unsigned L, uint32_t *indices,
                                            float *distances);

}  // namespace diskann
