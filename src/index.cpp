// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <atomic>
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <thread>
#include <type_traits>
#include <omp.h>
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

#include "logger.h"
#include "exceptions.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "parameters.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "windows_customizations.h"
#include "ann_exception.h"
#include "tcmalloc/malloc_extension.h"
#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

#include "Neighbor_Tag.h"
// only L2 implemented. Need to implement inner product search

namespace diskann {
  // Initialize an index with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T, typename TagT>

  Index<T, TagT>::Index(Metric m, const size_t dim, const size_t max_points,
                        const bool dynamic_index,
                        const bool save_index_in_one_file,
                        const bool enable_tags, const bool support_eager_delete)
      : _dist_metric(m), _dim(dim), _max_points(max_points),
        _save_as_one_file(save_index_in_one_file),
        _dynamic_index(dynamic_index), _enable_tags(enable_tags),
        _support_eager_delete(support_eager_delete) {
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
      diskann::cout << "ERROR: Eager Deletes must have Dynamic Indexing "
                       "enabled. Exitting."
                    << std::endl;
      exit(-1);
    }
    // data is stored to _nd * aligned_dim matrix with necessary
    // zero-padding
    _aligned_dim = ROUND_UP(_dim, 8);

    if (dynamic_index)
      _num_frozen_pts = 1;

    alloc_aligned(((void **) &_data),
                  (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(T),
                  8 * sizeof(T));
    std::memset(_data, 0,
                (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(T));

    _ep = (unsigned) _max_points;

    _final_graph.reserve(_max_points + _num_frozen_pts);
    _final_graph.resize(_max_points + _num_frozen_pts);

    for (size_t i = 0; i < _max_points + _num_frozen_pts; i++)
      _final_graph[i].clear();

    if (_support_eager_delete) {
      _in_graph.reserve(_max_points + _num_frozen_pts);
      _in_graph.resize(_max_points + _num_frozen_pts);
    }

    diskann::cout << "Getting distance function for metric: "
                  << (m == diskann::Metric::COSINE ? "cosine" : "l2")
                  << std::endl;
    this->_distance = get_distance_function<T>(m);
    _locks = std::vector<std::mutex>(_max_points + _num_frozen_pts);

    if (_support_eager_delete)
      _locks_in = std::vector<std::mutex>(_max_points + _num_frozen_pts);

    _width = 0;
  }

  template<typename T, typename TagT>
  Index<T, TagT>::~Index() {
    delete this->_distance;
    aligned_free(_data);
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

    _delete_set.clear();
    _empty_slots.clear();
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_tags(std::string tags_file, size_t offset) {
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
      std::memset((char *) &tag_data[_ep], 0, sizeof(TagT));
    }
    tag_bytes_written =
        save_bin<TagT>(tags_file, tag_data, _nd + _num_frozen_pts, 1, offset);
    delete[] tag_data;
    return tag_bytes_written;
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_data(std::string data_file, size_t offset) {
    return save_data_in_base_dimensions(data_file, _data, _nd + _num_frozen_pts,
                                        _dim, _aligned_dim, offset);
  }

  // save the graph index on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_graph(std::string graph_file, size_t offset) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    out.seekp(offset, out.beg);
    _u64 index_size = 24;
    _u32 max_degree = 0;
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &_width, sizeof(unsigned));
    unsigned ep_u32 = _ep;
    out.write((char *) &ep_u32, sizeof(unsigned));
    out.write((char *) &_num_frozen_pts, sizeof(_u64));
    for (unsigned i = 0; i < _nd + _num_frozen_pts; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _final_graph[i].data(), GK * sizeof(unsigned));
      max_degree = _final_graph[i].size() > max_degree
                       ? (_u32) _final_graph[i].size()
                       : max_degree;
      index_size += (_u64)(sizeof(unsigned) * (GK + 1));
    }
    out.seekp(offset, out.beg);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_degree, sizeof(_u32));
    out.close();
    return index_size;  // number of bytes written
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_delete_list(const std::string &filename,
                                        _u64               file_offset) {
    if (_delete_set.size() == 0) {
      return 0;
    }
    std::unique_ptr<_u32[]> delete_list =
        std::make_unique<_u32[]>(_delete_set.size());
    _u32 i = 0;
    for (auto &del : _delete_set) {
      delete_list[i++] = del;
    }
    return save_bin<_u32>(filename, delete_list.get(), _delete_set.size(), 1,
                          file_offset);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save(const char *filename) {
    // first check if no thread is inserting
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_lock<std::shared_timed_mutex> lock(_update_lock);
    _change_lock.lock();

    // compact_data();
    compact_frozen_point();
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
      delete_file(filename);
      std::vector<size_t> cumul_bytes(5, 0);
      cumul_bytes[0] = METADATA_SIZE;
      cumul_bytes[1] =
          cumul_bytes[0] + save_graph(std::string(filename), cumul_bytes[0]);
      cumul_bytes[2] =
          cumul_bytes[1] + save_data(std::string(filename), cumul_bytes[1]);
      cumul_bytes[3] =
          cumul_bytes[2] + save_tags(std::string(filename), cumul_bytes[2]);
      cumul_bytes[4] =
          cumul_bytes[3] + save_delete_list(filename, cumul_bytes[3]);
      diskann::save_bin<_u64>(filename, cumul_bytes.data(), cumul_bytes.size(),
                              1, 0);

      diskann::cout << "Saved index as one file to " << filename << " of size "
                    << cumul_bytes[cumul_bytes.size() - 1] << "B." << std::endl;
    }

    reposition_frozen_point_to_end();

    _change_lock.unlock();
    auto stop = std::chrono::high_resolution_clock::now();
    auto timespan =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    diskann::cout << "Time taken for save: " << timespan.count() << "s."
                  << std::endl;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_tags(const std::string tag_filename,
                                   size_t            offset) {
    if (_enable_tags && !file_exists(tag_filename)) {
      diskann::cerr << "Tag file provided does not exist!" << std::endl;
      throw diskann::ANNException("Tag file provided does not exist!", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }

    if (!_enable_tags) {
      diskann::cout << "Tags not loaded as tags not enabled." << std::endl;
      return 0;
    }

    size_t file_dim, file_num_points;
    TagT * tag_data;
    load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points,
                   file_dim, offset);

    if (file_dim != 1) {
      std::stringstream stream;
      stream << "ERROR: Loading " << file_dim << " dimensions for tags,"
             << "but tag file must have 1 dimension." << std::endl;
      std::cerr << stream.str() << std::endl;
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
  size_t Index<T, TagT>::load_data(std::string filename, size_t offset) {
    if (!file_exists(filename)) {
      std::stringstream stream;
      stream << "ERROR: data file " << filename << " does not exist."
             << std::endl;
      std::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    size_t file_dim, file_num_points;
    diskann::get_bin_metadata(filename, file_num_points, file_dim, offset);

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != _dim) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << _dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      std::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (file_num_points > _max_points + _num_frozen_pts) {
      //_change_lock is already locked in load()
      std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      resize(file_num_points);
    }

    copy_aligned_data_from_file<T>(std::string(filename), _data,
                                   file_num_points, file_dim, _aligned_dim,
                                   offset);
    return file_num_points;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_delete_set(tsl::robin_set<uint32_t> &del_set) {
    del_set = _delete_set;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_delete_set(const std::string &filename,
                                         size_t             offset) {
    std::unique_ptr<_u32[]> delete_list;
    _u64                    npts, ndim;
    load_bin<_u32>(filename, delete_list, npts, ndim, offset);
    assert(ndim == 1);
    for (size_t i = 0; i < npts; i++) {
      _delete_set.insert(delete_list[i]);
    }
    return npts;
  }

  // load the index from file and update the width (max_degree), ep (navigating
  // node id), and _final_graph (adjacency list)
  template<typename T, typename TagT>
  void Index<T, TagT>::load(const char *filename) {
    _change_lock.lock();

    size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0;

    if (!_save_as_one_file) {
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

    } else {
      _u64                    nr, nc;
      std::unique_ptr<_u64[]> file_offset_data;

      std::string index_file(filename);

      diskann::load_bin<_u64>(index_file, file_offset_data, nr, nc, 0);
      // Loading data first so that we know how many points to expect.
      data_file_num_pts = load_data(index_file, file_offset_data[1]);
      graph_num_pts =
          load_graph(index_file, data_file_num_pts, file_offset_data[0]);
      if (file_offset_data[3] != file_offset_data[4]) {
        load_delete_set(index_file, file_offset_data[3]);
      }
      if (_enable_tags) {
        tags_file_num_pts = load_tags(index_file, file_offset_data[2]);
      }
    }

    if (data_file_num_pts != graph_num_pts ||
        (data_file_num_pts != tags_file_num_pts && _enable_tags)) {
      std::stringstream stream;
      stream << "ERROR: When loading index, loaded " << data_file_num_pts
             << " points from datafile, " << graph_num_pts
             << " from graph, and " << tags_file_num_pts
             << " tags, with num_frozen_pts being set to " << _num_frozen_pts
             << " in constructor." << std::endl;
      std::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    _nd = data_file_num_pts - _num_frozen_pts;
    _empty_slots.clear();
    for (_u32 i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }

    _lazy_done = _delete_set.size() != 0;

    reposition_frozen_point_to_end();
    diskann::cout << "Num frozen points:" << _num_frozen_pts << " _nd: " << _nd
                  << " _ep: " << _ep
                  << " size(_location_to_tag): " << _location_to_tag.size()
                  << " size(_tag_to_location):" << _tag_to_location.size()
                  << " Max points: " << _max_points << std::endl;

    _change_lock.unlock();
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_graph(std::string filename,
                                    size_t expected_num_points, size_t offset) {
    std::ifstream in(filename, std::ios::binary);
    in.seekg(offset, in.beg);
    size_t expected_file_size;
    _u64   file_frozen_pts;
    in.read((char *) &expected_file_size, sizeof(_u64));
    in.read((char *) &_width, sizeof(unsigned));
    in.read((char *) &_ep, sizeof(unsigned));
    in.read((char *) &file_frozen_pts, sizeof(_u64));

    if (file_frozen_pts != _num_frozen_pts) {
      std::stringstream stream;
      if (file_frozen_pts == 1)
        stream << "ERROR: When loading index, detected dynamic index, but "
                  "constructor asks for static index. Exitting."
               << std::endl;
      else
        stream << "ERROR: When loading index, detected static index, but "
                  "constructor asks for dynamic index. Exitting."
               << std::endl;
      std::cerr << stream.str() << std::endl;
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    diskann::cout << "Loading vamana index " << filename << "..." << std::flush;

    // Sanity check. In case the user gave us fewer points as max_points than
    // the number
    // of points in the dataset, resize the _final_graph to the larger size.
    if (_max_points < (expected_num_points - _num_frozen_pts)) {
      diskann::cout << "Number of points in data: " << expected_num_points
                    << " is more than max_points argument: "
                    << _final_graph.size()
                    << " Setting max points to: " << expected_num_points
                    << std::endl;
      _final_graph.resize(expected_num_points);
      _max_points = expected_num_points - _num_frozen_pts;
      // changed expected_num to expected_num - frozen_num
    }

    size_t   bytes_read = 24;
    size_t   cc = 0;
    unsigned nodes = 0;
    while (bytes_read != expected_file_size) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (k == 0) {
        diskann::cerr << "ERROR: Point found with no out-neighbors, point#"
                      << nodes << std::endl;
      }
      //      if (in.eof())
      //        break;
      cc += k;
      ++nodes;
      std::vector<unsigned> tmp(k);
      tmp.reserve(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));
      _final_graph[nodes - 1].swap(tmp);
      bytes_read += sizeof(uint32_t) * ((_u64) k + 1);
      if (nodes % 10000000 == 0)
        diskann::cout << "." << std::flush;
    }

    diskann::cout << "done. Index has " << nodes << " nodes and " << cc
                  << " out-edges, _ep is set to " << _ep << std::endl;
    return nodes;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::get_vector_by_tag(TagT &tag, T *vec) {
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      diskann::cout << "Tag " << tag << " does not exist" << std::endl;
      return -1;
    }
    unsigned location = _tag_to_location[tag];
    // memory should be allocated for vec before calling this function
    memcpy((void *) vec, (void *) (_data + (size_t)(location * _aligned_dim)),
           (size_t) _aligned_dim * sizeof(T));
    return 0;
  }

  template<typename T, typename TagT>
  const T *Index<T, TagT>::get_vector_by_tag(const TagT &tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      diskann::cout << "Tag " << tag << " does not exist in the index."
                    << std::endl;
      return nullptr;
    } else {
      unsigned location = _tag_to_location[tag];
      return _data + (size_t)(location * _aligned_dim);
    }
  }

  /**************************************************************
   *      Support for Static Index Building and Searching
   **************************************************************/

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T, typename TagT>
  unsigned Index<T, TagT>::calculate_entry_point() {
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
      float &  dist = distances[i];
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

  /* iterate_to_fixed_point():
   * node_coords : point whose neighbors to be found.
   * init_ids : ids of initial search list.
   * Lsize : size of list.
   * beam_width: beam_width when performing indexing
   * expanded_nodes_info: will contain all the node ids and distances from
   * query that are expanded
   * expanded_nodes_ids : will contain all the nodes that are expanded during
   * search.
   * best_L_nodes: ids of closest L nodes in list
   */
  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point(
      const T *node_coords, const unsigned Lsize,
      const std::vector<unsigned> &init_ids,
      std::vector<Neighbor> &      expanded_nodes_info,
      tsl::robin_set<unsigned> &   expanded_nodes_ids,
      std::vector<Neighbor> &best_L_nodes, bool ret_frozen) {
    best_L_nodes.resize(Lsize + 1);
    for (unsigned i = 0; i < Lsize + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }
    expanded_nodes_info.reserve(10 * Lsize);
    expanded_nodes_ids.reserve(10 * Lsize);

    unsigned                 l = 0;
    Neighbor                 nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    for (auto id : init_ids) {
      assert(id < _max_points + _num_frozen_pts);
      nn = Neighbor(id,
                    _distance->compare(_data + _aligned_dim * (size_t) id,
                                       node_coords, (unsigned) _aligned_dim),
                    true);
      if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
        inserted_into_pool.insert(id);
        best_L_nodes[l++] = nn;
      }
      if (l == Lsize)
        break;
    }

    /* sort best_L_nodes based on distance of each point to node_coords */
    std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
    unsigned k = 0;
    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (k < l) {
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;
        if (!(best_L_nodes[k].id == _ep && _num_frozen_pts > 0 &&
              !ret_frozen)) {
          expanded_nodes_info.emplace_back(best_L_nodes[k]);
          expanded_nodes_ids.insert(n);
        }
        std::vector<unsigned> des;
        if (_dynamic_index) {
          LockGuard guard(_locks[n]);
          for (unsigned m = 0; m < _final_graph[n].size(); m++) {
            if (_final_graph[n][m] >= _max_points + _num_frozen_pts) {
              diskann::cerr << "Wrong id found: " << _final_graph[n][m]
                            << std::endl;
              throw diskann::ANNException(
                  std::string("Wrong id found") +
                      std::to_string(_final_graph[n][m]),
                  -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            des.emplace_back(_final_graph[n][m]);
          }
        } else {
          for (unsigned m = 0; m < _final_graph[n].size(); m++) {
            if (_final_graph[n][m] >= _max_points + _num_frozen_pts) {
              diskann::cerr << "Wrong id found: " << _final_graph[n][m]
                            << std::endl;
              throw diskann::ANNException(
                  std::string("Wrong id found") +
                      std::to_string(_final_graph[n][m]),
                  -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            des.emplace_back(_final_graph[n][m]);
          }
        }

        for (unsigned m = 0; m < des.size(); ++m) {
          unsigned id = des[m];
          if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
            inserted_into_pool.insert(id);

            if ((m + 1) < des.size()) {
              auto nextn = des[m + 1];
              diskann::prefetch_vector(
                  (const char *) _data + _aligned_dim * (size_t) nextn,
                  sizeof(T) * _aligned_dim);
            }

            cmps++;
            float dist = _distance->compare(node_coords,
                                            _data + _aligned_dim * (size_t) id,
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
  void Index<T, TagT>::iterate_to_fixed_point(
      const T *node_coords, const unsigned Lindex,
      std::vector<Neighbor> &        expanded_nodes_info,
      tsl::robin_map<uint32_t, T *> &coord_map, bool return_frozen_pt) {
    std::vector<uint32_t> init_ids;
    init_ids.push_back(this->_ep);
    std::vector<Neighbor>    best_L_nodes;
    tsl::robin_set<uint32_t> expanded_nodes_ids;
    this->iterate_to_fixed_point(node_coords, Lindex, init_ids,
                                 expanded_nodes_info, expanded_nodes_ids,
                                 best_L_nodes, return_frozen_pt);
    for (Neighbor &einf : expanded_nodes_info) {
      T *coords =
          this->_data + (uint64_t) einf.id * (uint64_t) this->_aligned_dim;
      coord_map.insert(std::make_pair(einf.id, coords));
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_expanded_nodes(
      const size_t node_id, const unsigned Lindex,
      std::vector<unsigned>     init_ids,
      std::vector<Neighbor> &   expanded_nodes_info,
      tsl::robin_set<unsigned> &expanded_nodes_ids) {
    const T *             node_coords = _data + _aligned_dim * node_id;
    std::vector<Neighbor> best_L_nodes;

    if (init_ids.size() == 0)
      init_ids.emplace_back(_ep);

    iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info,
                           expanded_nodes_ids, best_L_nodes);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const float alpha, const unsigned degree,
                                    const unsigned         maxc,
                                    std::vector<Neighbor> &result) {
    auto               pool_size = (_u32) pool.size();
    std::vector<float> occlude_factor(pool_size, 0);
    occlude_list(pool, alpha, degree, maxc, result, occlude_factor);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const float alpha, const unsigned degree,
                                    const unsigned         maxc,
                                    std::vector<Neighbor> &result,
                                    std::vector<float> &   occlude_factor) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
      unsigned start = 0;

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
              _data + _aligned_dim * (size_t) pool[t].id,
              _data + _aligned_dim * (size_t) p.id, (unsigned) _aligned_dim);
          occlude_factor[t] =
              (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2f;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_neighbors(const unsigned         location,
                                       std::vector<Neighbor> &pool,
                                       const Parameters &     parameter,
                                       std::vector<unsigned> &pruned_list) {
    unsigned range = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    float    alpha = parameter.Get<float>("alpha");

    if (pool.size() == 0) {
      throw diskann::ANNException("Pool passed to prune_neighbors is empty",
                                  -1);
    }

    _width = (std::max)(_width, range);

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list(pool, alpha, range, maxc, result, occlude_factor);

    /* Add all the nodes in result into a variable called cut_graph
     * So this contains all the neighbors of id location
     */
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

  /* batch_inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::batch_inter_insert(
      unsigned n, const std::vector<unsigned> &pruned_list,
      const Parameters &parameter, std::vector<unsigned> &need_to_sync) {
    const auto range = parameter.Get<unsigned>("R");

    // assert(!src_pool.empty());

    for (auto des : pruned_list) {
      if (des == n)
        continue;
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < _max_points + _num_frozen_pts);
      if (des > _max_points)
        diskann::cout << "error. " << des << " exceeds max_pts" << std::endl;
      /* des_pool contains the neighbors of the neighbors of n */

      {
        LockGuard guard(_locks[des]);
        if (std::find(_final_graph[des].begin(), _final_graph[des].end(), n) ==
            _final_graph[des].end()) {
          _final_graph[des].push_back(n);
          if (_final_graph[des].size() > (unsigned) (range * SLACK_FACTOR))
            need_to_sync[des] = 1;
        }
      }  // des lock is released by this point
    }
  }

  /* inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::inter_insert(unsigned               n,
                                    std::vector<unsigned> &pruned_list,
                                    const Parameters &     parameter,
                                    bool                   update_in_graph) {
    const auto range = parameter.Get<unsigned>("R");
    assert(n >= 0 && n < _nd + _num_frozen_pts);

    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < _max_points + _num_frozen_pts);
      /* des_pool contains the neighbors of the neighbors of n */
      auto &                des_pool = _final_graph[des];
      std::vector<unsigned> copy_of_neighbors;
      bool                  prune_needed = false;
      {
        LockGuard guard(_locks[des]);
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < (_u64)(SLACK_FACTOR * range)) {
            des_pool.emplace_back(n);
            if (update_in_graph) {
              LockGuard guard(_locks_in[n]);
              _in_graph[n].emplace_back(des);
            }
            prune_needed = false;
          } else {
            copy_of_neighbors = des_pool;
            prune_needed = true;
          }
        }
      }  // des lock is released by this point

      if (prune_needed) {
        copy_of_neighbors.push_back(n);
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);

        size_t reserveSize = (size_t)(std::ceil(1.05 * SLACK_FACTOR * range));
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
        prune_neighbors(des, dummy_pool, parameter, new_out_neighbors);
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
          _final_graph[des].shrink_to_fit();
        }
      }
    }
  }
  /* Link():
   * The graph creation function.
   *    The graph will be updated periodically in NUM_SYNCS batches
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::link(Parameters &parameters) {
    unsigned NUM_THREADS = parameters.Get<unsigned>("num_threads");
    if (NUM_THREADS != 0)
      omp_set_num_threads(NUM_THREADS);

    uint32_t NUM_SYNCS =
        (unsigned) DIV_ROUND_UP(_nd + _num_frozen_pts, (64 * 64));
    if (NUM_SYNCS < 40)
      NUM_SYNCS = 40;
    diskann::cout << "Number of syncs: " << NUM_SYNCS << std::endl;

    _saturate_graph = parameters.Get<bool>("saturate_graph");

    if (NUM_THREADS != 0)
      omp_set_num_threads(NUM_THREADS);

    const unsigned argL = parameters.Get<unsigned>("L");  // Search list size
    const unsigned range = parameters.Get<unsigned>("R");
    const float    last_round_alpha = parameters.Get<float>("alpha");
    unsigned       L = argL;

    std::vector<unsigned> Lvec;
    Lvec.push_back(L);
    Lvec.push_back(L);
    const unsigned NUM_RNDS = 2;

    // Max degree of graph
    // Pruning parameter
    // Set alpha=1 for the first pass; use specified alpha for last pass
    parameters.Set<float>("alpha", 1);

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

    // if there are frozen points, the first such one is set to be the _ep
    if (_num_frozen_pts > 0)
      _ep = (unsigned) _max_points;
    else
      _ep = calculate_entry_point();

    if (_support_eager_delete) {
      _in_graph.reserve(_max_points + _num_frozen_pts);
      _in_graph.resize(_max_points + _num_frozen_pts);
    }

    for (uint64_t p = 0; p < _max_points + _num_frozen_pts; p++) {
      _final_graph[p].reserve((size_t)(std::ceil(range * SLACK_FACTOR * 1.05)));
    }

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // creating a initial list to begin the search process. it has _ep and
    // random other nodes
    std::set<unsigned> unique_start_points;
    unique_start_points.insert(_ep);

    std::vector<unsigned> init_ids;
    for (auto pt : unique_start_points)
      init_ids.emplace_back(pt);

    diskann::Timer link_timer;
    for (uint32_t rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
      L = Lvec[rnd_no];

      if (rnd_no == NUM_RNDS - 1) {
        if (last_round_alpha > 1)
          parameters.Set<float>("alpha", last_round_alpha);
      }

      double   sync_time = 0, total_sync_time = 0;
      double   inter_time = 0, total_inter_time = 0;
      size_t   inter_count = 0, total_inter_count = 0;
      unsigned progress_counter = 0;

      size_t round_size = DIV_ROUND_UP(_nd, NUM_SYNCS);  // size of each batch
      std::vector<unsigned> need_to_sync(_max_points + _num_frozen_pts, 0);

      std::vector<std::vector<unsigned>> pruned_list_vector(round_size);

      for (uint32_t sync_num = 0; sync_num < NUM_SYNCS; sync_num++) {
        size_t start_id = sync_num * round_size;
        size_t end_id =
            (std::min)(_nd + _num_frozen_pts, (sync_num + 1) * round_size);

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
          get_expanded_nodes(node, L, init_ids, pool, visited);
          /* check the neighbors of the query that are not part of
           * visited, check their distance to the query, and add it to
           * pool.
           */
          if (!_final_graph[node].empty())
            for (auto id : _final_graph[node]) {
              if (visited.find(id) == visited.end() && id != node) {
                float dist =
                    _distance->compare(_data + _aligned_dim * (size_t) node,
                                       _data + _aligned_dim * (size_t) id,
                                       (unsigned) _aligned_dim);
                pool.emplace_back(Neighbor(id, dist, true));
                visited.insert(id);
              }
            }
          prune_neighbors(node, pool, parameters, pruned_list);
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
          _final_graph[node].clear();
          for (auto id : pruned_list)
            _final_graph[node].emplace_back(id);
        }
        s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = start_id; node_ctr < (_s64) end_id; ++node_ctr) {
          auto                   node = visit_order[node_ctr];
          _u64                   node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          batch_inter_insert(node, pruned_list, parameters, need_to_sync);
          //          inter_insert(node, pruned_list, parameters, 0);
          pruned_list.clear();
          pruned_list.shrink_to_fit();
        }

#pragma omp parallel for schedule(dynamic, 65536)
        for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size());
             node_ctr++) {
          auto node = visit_order[node_ctr];
          if (need_to_sync[node] != 0) {
            need_to_sync[node] = 0;
            inter_count++;
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
            prune_neighbors(node, dummy_pool, parameters, new_out_neighbors);

            _final_graph[node].clear();
            for (auto id : new_out_neighbors)
              _final_graph[node].emplace_back(id);
          }
        }

        diff = std::chrono::high_resolution_clock::now() - s;
        inter_time += diff.count();

        if ((sync_num * 100) / NUM_SYNCS > progress_counter) {
          diskann::cout.precision(4);
          diskann::cout << "Completed  (round: " << rnd_no
                        << ", sync: " << sync_num << "/" << NUM_SYNCS
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
// Splittng diskann_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#ifdef USE_TCMALLOC
      MallocExtension::instance()->ReleaseFreeMemory();
#endif
      if (_nd > 0) {
        diskann::cout << "Completed Pass " << rnd_no << " of data using L=" << L
                      << " and alpha=" << parameters.Get<float>("alpha")
                      << ". Stats: ";
        diskann::cout << "search+prune_time=" << total_sync_time
                      << "s, inter_time=" << total_inter_time
                      << "s, inter_count=" << total_inter_count << std::endl;
      }
    }

    if (_nd > 0) {
      diskann::cout << "Starting final cleanup.." << std::flush;
    }
#pragma omp parallel for schedule(dynamic, 65536)
    for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size()); node_ctr++) {
      auto node = visit_order[node_ctr];
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
        prune_neighbors(node, dummy_pool, parameters, new_out_neighbors);

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
          prune_neighbors((_u32) node, dummy_pool, parameters,
                          new_out_neighbors);

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
  void Index<T, TagT>::build(const char *             filename,
                             const size_t             num_points_to_load,
                             Parameters &             parameters,
                             const std::vector<TagT> &tags) {
    if (!file_exists(filename)) {
      diskann::cerr << "Data file " << filename
                    << " does not exist!!! Exiting...." << std::endl;
      std::stringstream stream;
      stream << "Data file " << filename << " does not exist." << std::endl;
      std::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      diskann::cout << "Starting with an empty index." << std::endl;
      _nd = 0;
    } else {
      diskann::get_bin_metadata(filename, file_num_points, file_dim);
      if (file_num_points > _max_points ||
          num_points_to_load > file_num_points) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load
               << " points and file has " << file_num_points << " points, but "
               << "index can support only " << _max_points
               << " points as specified in constructor." << std::endl;
        std::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
      if (file_dim != _dim) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        std::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }

      copy_aligned_data_from_file<T>(std::string(filename), _data,
                                     file_num_points, file_dim, _aligned_dim);

      diskann::cout << "Loading only first " << num_points_to_load
                    << " from file.. " << std::endl;
      _nd = num_points_to_load;

      if (_enable_tags && tags.size() != num_points_to_load) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load
               << " points from file,"
               << "but tags vector is of size " << tags.size() << "."
               << std::endl;
        std::cerr << stream.str() << std::endl;
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
    }

    generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    if (_support_eager_delete) {
      update_in_graph();  // copying values to in_graph
    }

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
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
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char * filename,
                             const size_t num_points_to_load,
                             Parameters &parameters, const char *tag_filename) {
    if (!file_exists(filename)) {
      std::cerr << "Data file provided " << filename << " does not exist."
                << std::endl;
      std::stringstream stream;
      stream << "Data file provided " << filename << " does not exist."
             << std::endl;
      std::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      diskann::cout << "Starting with an empty index." << std::endl;
      _nd = 0;
    } else {
      diskann::get_bin_metadata(filename, file_num_points, file_dim);
      if (file_num_points > _max_points ||
          num_points_to_load > file_num_points) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load
               << " points and file has " << file_num_points << " points, but "
               << "index can support only " << _max_points
               << " points as specified in constructor." << std::endl;
        std::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
      if (file_dim != _dim) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        std::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }

      copy_aligned_data_from_file<T>(std::string(filename), _data,
                                     file_num_points, file_dim, _aligned_dim);

      diskann::cout << "Loading only first " << num_points_to_load
                    << " from file.. " << std::endl;
      _nd = num_points_to_load;
      if (_enable_tags) {
        if (tag_filename == nullptr) {
          for (unsigned i = 0; i < num_points_to_load; i++) {
            _tag_to_location[i] = i;
            _location_to_tag[i] = i;
          }
        } else {
          if (file_exists(tag_filename)) {
            diskann::cout << "Loading tags from " << tag_filename
                          << " for vamana index build" << std::endl;
            TagT * tag_data = nullptr;
            size_t npts, ndim;
            diskann::load_bin(tag_filename, tag_data, npts, ndim);
            if (npts != num_points_to_load) {
              std::stringstream sstream;
              sstream << "Loaded " << npts
                      << " tags instead of expected number: "
                      << num_points_to_load;
              diskann::cerr << sstream.str() << std::endl;
              throw diskann::ANNException(sstream.str(), -1, __FUNCSIG__,
                                          __FILE__, __LINE__);
            }
            for (size_t i = 0; i < npts; i++) {
              _tag_to_location[tag_data[i]] = (unsigned) i;
              _location_to_tag[(unsigned) i] = tag_data[i];
            }
            delete[] tag_data;
          } else {
            diskann::cerr << "Tag file " << tag_filename
                          << " does not exist. Exiting..." << std::endl;
            throw diskann::ANNException(
                std::string("Tag file") + tag_filename + " does not exist", -1,
                __FUNCSIG__, __FILE__, __LINE__);
          }
        }
      }
    }

    generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    if (_support_eager_delete) {
      update_in_graph();  // copying values to in_graph
    }

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
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
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(
      const T *query, const size_t K, const unsigned L,
      std::vector<Neighbor_Tag<TagT>> &best_K_tags) {
    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    assert(best_K_tags.size() == 0);
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor>    best, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }

    T *    aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval =
        iterate_to_fixed_point(aligned_query, L, init_ids, expanded_nodes_info,
                               expanded_nodes_ids, best, false);

    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    for (auto iter : best) {
      if (_location_to_tag.find(iter.id) != _location_to_tag.end())
        best_K_tags.emplace_back(
            Neighbor_Tag<TagT>(_location_to_tag[iter.id], iter.distance));
      if (best_K_tags.size() == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T *      query,
                                                       const size_t   K,
                                                       const unsigned L,
                                                       unsigned *     indices,
                                                       float *distances) {
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor>    best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    T *    aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval =
        iterate_to_fixed_point(aligned_query, L, init_ids, expanded_nodes_info,
                               expanded_nodes_ids, best_L_nodes);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      if (it.id < _max_points) {
        indices[pos] = it.id;
        if (distances != nullptr)
          distances[pos] = it.distance;
        pos++;
      }
      if (pos == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(
      const T *query, const uint64_t K, const unsigned L,
      std::vector<unsigned> init_ids, uint64_t *indices, float *distances) {
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor>    best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    T *    aligned_query;
    size_t allocSize = _aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, _aligned_dim * sizeof(T));
    memcpy(aligned_query, query, _dim * sizeof(T));
    auto retval = iterate_to_fixed_point(aligned_query, (unsigned) L, init_ids,
                                         expanded_nodes_info,
                                         expanded_nodes_ids, best_L_nodes);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      indices[pos] = it.id;
      distances[pos] = it.distance;
      pos++;
      if (pos == K)
        break;
    }
    aligned_free(aligned_query);
    return retval;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const uint64_t K,
                                          const unsigned L, TagT *tags,
                                          float *           distances,
                                          std::vector<T *> &res_vectors) {
    _u32 * indices = new unsigned[L];
    float *dist_interim = new float[L];
    search(query, L, L, indices, dist_interim);

    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t                                    pos = 0;
    for (int i = 0; i < (int) L; ++i)
      if (_location_to_tag.find(indices[i]) != _location_to_tag.end()) {
        tags[pos] = _location_to_tag[indices[i]];
        res_vectors[i] = _data + indices[i] * _aligned_dim;

        if (distances != nullptr)
          distances[pos] = dist_interim[i];
        pos++;
        if (pos == K)
          break;
      }
    delete[] indices;
    delete[] dist_interim;
    return pos;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const size_t K,
                                          const unsigned L, TagT *tags,
                                          float *distances) {
    _u32 * indices = new unsigned[L];
    float *dist_interim = new float[L];
    search(query, L, L, indices, dist_interim);

    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t                                    pos = 0;
    for (int i = 0; i < (int) L; ++i) {
      if (_location_to_tag.find(indices[i]) != _location_to_tag.end()) {
        tags[pos] = _location_to_tag[indices[i]];
        if (distances != nullptr)
          distances[pos] = dist_interim[i];
        pos++;
        if (pos == K)
          break;
      }
    }
    delete[] indices;
    delete[] dist_interim;
    return pos;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::get_num_points() {
    return _nd;
  }

  template<typename T, typename TagT>
  T *Index<T, TagT>::get_data() {
    if (_num_frozen_pts > 0) {
      T *    ret_data = nullptr;
      size_t allocSize = _nd * _aligned_dim * sizeof(T);
      alloc_aligned(((void **) &ret_data), allocSize, 8 * sizeof(T));
      memset(ret_data, 0, _nd * _aligned_dim * sizeof(T));
      memcpy(ret_data, _data, _nd * _aligned_dim * sizeof(T));
      return ret_data;
    }
    return _data;
  }
  template<typename T, typename TagT>
  size_t Index<T, TagT>::return_max_points() {
    return _max_points;
  }

  /*************************************************
   *      Support for Incremental Update
   *************************************************/

  // in case we add ''frozen'' auxiliary points to the dataset, these are not
  // visible to external world, we generate them here and update our dataset
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
      std::cerr << "Tags must be instantiated for deletions" << std::endl;
      return -2;
    }

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
    LockGuard guard(_change_lock);
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

    unsigned id = -1;
    {
      std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        std::cerr << "Delete tag not found" << std::endl;
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

        // constructing candidate set from out-neighbors and out-neighbors of
        // ngh and id
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
    //  diskann::cout << "Updating in_graph.....";
    for (unsigned i = 0; i < _in_graph.size(); i++)
      _in_graph[i].clear();

    for (size_t i = 0; i < _final_graph.size();
         i++)  // copying to in-neighbor graph
      for (size_t j = 0; j < _final_graph[i].size(); j++)
        _in_graph[_final_graph[i][j]].emplace_back((_u32) i);
  }

  // Do not call consolidate_deletes() if you have not locked _change_lock.
  // Returns number of live points left after consolidation
  // proxy inserts all nghrs of deleted points
  // original approach
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
    if (_eager_done) {
      diskann::cout
          << "In consolidate_deletes(), _eager_done is true. So exiting."
          << std::endl;
      return 0;
    }

    diskann::cout << "Inside Index::consolidate_deletes()" << std::endl;
    std::cout << "Empty slots size: " << _empty_slots.size() << " _nd: " << _nd
              << " max_points: " << _max_points << std::endl;
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    _u64     total_pts = _max_points + _num_frozen_pts;
    unsigned block_size = 1 << 10;
    _s64     total_blocks = DIV_ROUND_UP(total_pts, block_size);

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
    for (_s64 block = 0; block < total_blocks; ++block) {
      tsl::robin_set<unsigned> candidate_set;
      std::vector<Neighbor>    expanded_nghrs;
      std::vector<Neighbor>    result;

      for (_s64 i = block * block_size;
           i < (_s64)((block + 1) * block_size) &&
           i < (_s64)(_max_points + _num_frozen_pts);
           i++) {
        if ((_delete_set.find((_u32) i) == _delete_set.end()) &&
            (_empty_slots.find((_u32) i) == _empty_slots.end())) {
          candidate_set.clear();
          expanded_nghrs.clear();
          result.clear();

          bool modify = false;
          for (auto ngh : _final_graph[(_u32) i]) {
            if (_delete_set.find(ngh) != _delete_set.end()) {
              modify = true;

              // Add outgoing links from
              for (auto j : _final_graph[ngh])
                if (_delete_set.find(j) == _delete_set.end())
                  candidate_set.insert(j);
            } else {
              candidate_set.insert(ngh);
            }
          }
          if (modify) {
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
                  (_delete_set.find(j.id) == _delete_set.end()))
                _final_graph[(_u32) i].push_back(j.id);
            }
          }
        }
      }
    }

    if (_support_eager_delete)
      update_in_graph();

    for (auto iter : _delete_set) {
      _empty_slots.insert(iter);
    }
    _nd -= _delete_set.size();

    _data_compacted = _delete_set.size() == 0;

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for consolidate_deletes() "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     stop - start)
                     .count()
              << "s." << std::endl;

    return _nd;
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
        // set new _ep to be frozen point
        _ep = (_u32) _nd;
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

          memcpy((void *) (_data + (size_t) _aligned_dim * _nd),
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
  void Index<T, TagT>::compact_data_for_insert() {
    compact_data();

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
  void Index<T, TagT>::compact_data() {
    if (!_dynamic_index)
      return;

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
    std::cout << "Time taken for initial setup: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     stop - start)
                     .count()
              << "s." << std::endl;
    // If start node is removed, replace it.
    if (_delete_set.find(_ep) != _delete_set.end()) {
      std::cerr << "Replacing start node which has been deleted... "
                << std::flush;
      auto old_ep = _ep;
      // First active neighbor of old start node is new start node
      for (auto iter : _final_graph[_ep])
        if (_delete_set.find(iter) != _delete_set.end()) {
          _ep = iter;
          break;
        }
      if (_ep == old_ep) {
        diskann::cerr << "ERROR: Did not find a replacement for start node."
                      << std::endl;
        throw diskann::ANNException(
            "ERROR: Did not find a replacement for start node.", -1,
            __FUNCSIG__, __FILE__, __LINE__);
      } else {
        assert(_delete_set.find(_ep) == _delete_set.end());
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

            diskann::cerr << sstream.str() << std::endl;
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
    std::cout << "Time taken for moving data around: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     stop - start)
                     .count()
              << "s. Of which copy_time: " << copy_time << "s." << std::endl;

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
    for (_u32 i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }

    _lazy_done = false;
    _eager_done = false;
    _data_compacted = true;
    stop = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for tag<->index consolidation: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     stop - start)
                     .count()
              << "s." << std::endl;
    std::cout << "Time taken for compact_data(): "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     stop - fnstart)
                     .count()
              << "s." << std::endl;
  }

  // Do not call reserve_location() if you have not locked _change_lock.
  // It is not thread safe.
  template<typename T, typename TagT>
  int Index<T, TagT>::reserve_location() {
    LockGuard guard(_change_lock);
    if (_nd >= _max_points) {
      return -1;
    }
    unsigned location;
    if (_data_compacted) {
      location = (unsigned) _nd;
      _empty_slots.erase(location);
    } else {
      // no need of delete_lock here, _change_lock will ensure no other thread
      // executes this block of code
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
    reposition_point(_nd, _max_points);
    _ep = (_u32) _max_points;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::resize(uint32_t new_max_points) {
    // TODO: Check if the _change_lock and _update_lock are both locked.

    auto start = std::chrono::high_resolution_clock::now();
    assert(_empty_slots.size() ==
           0);  // should not resize if there are empty slots.
#ifndef _WINDOWS
    T *new_data;
    /*     alloc_aligned((void **) &new_data,
                      (new_max_points + 1) * _aligned_dim * sizeof(T),
                      8 * sizeof(T));
        memcpy(new_data, _data, (_max_points + 1) * _aligned_dim * sizeof(T));
        aligned_free(_data);
        _data = new_data;
        */
    realloc_aligned((void **) &_data, (void **) &new_data,
                    (_max_points + 1) * _aligned_dim * sizeof(T),
                    (new_max_points + 1) * _aligned_dim * sizeof(T),
                    8 * sizeof(T));
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

    reposition_point(_max_points, new_max_points);
    _max_points = new_max_points;
    _ep = new_max_points;

    for (_u32 i = _nd; i < _max_points; i++) {
      _empty_slots.insert(i);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    diskann::cout << "Resizing took: "
                  << std::chrono::duration<double>(stop - start).count() << "s"
                  << std::endl;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::printTagToLocation() {
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);

    std::cout << "Thread: " << std::this_thread::get_id()
              << " _tag_to_location: " << std::endl;
    for (auto tl : _tag_to_location) {
      std::cout << "(" << tl.first << "," << tl.second << "),";
    }
    std::cout << std::endl
              << "Thread: " << std::this_thread::get_id()
              << " _location_to_tag: " << std::endl;
    for (auto lt : _location_to_tag) {
      std::cout << "(" << lt.first << "," << lt.second << "),";
    }
    std::cout << std::endl;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::insert_point(const T *point, const Parameters &parameters,
                                   const TagT tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    unsigned range = parameters.Get<unsigned>("R");
    //    assert(_has_built);
    std::vector<Neighbor>    pool;
    std::vector<Neighbor>    tmp;
    tsl::robin_set<unsigned> visited;

    {
      std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
      std::shared_lock<std::shared_timed_mutex> tsl(_tag_lock);
      if (_enable_tags &&
          (_tag_to_location.find(tag) != _tag_to_location.end())) {
        // TODO! This is a repeat of lazy_delete, but we can't call
        // that function because we are taking many locks here. Hence
        // the repeated code.
        tsl.unlock();
        std::unique_lock<std::shared_timed_mutex> tul(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> tdl(_delete_lock);
        _lazy_done = true;
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
      }
    }

    auto location = reserve_location();
    if (location == -1) {
      std::cout << "Thread: " << std::this_thread::get_id()
                << " location  == -1. Waiting for unique_lock. " << std::endl
                << std::flush;
      lock.unlock();
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      std::cout << "Thread: " << std::this_thread::get_id()
                << " Obtained unique_lock. " << std::endl;
      if (_nd >= _max_points) {
        auto new_max_points = (size_t)(_max_points * INDEX_GROWTH_FACTOR);
        diskann::cerr << "Thread: " << std::this_thread::get_id()
                      << ": Increasing _max_points from " << _max_points
                      << " to " << new_max_points << " _nd is: " << _nd
                      << std::endl;
        resize(new_max_points);
      }
      growth_lock.unlock();
      lock.lock();
      location = reserve_location();
      // TODO: Consider making this a while/do_while loop so that we retry
      // instead of terminating.
      if (location == -1) {
        throw diskann::ANNException(
            "Cannot reserve location even after expanding graph. Terminating.",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }
    }

    {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);

      _tag_to_location[tag] = location;
      _location_to_tag[location] = tag;
    }

    auto offset_data = _data + (size_t) _aligned_dim * location;
    memset((void *) offset_data, 0, sizeof(T) * _aligned_dim);
    memcpy((void *) offset_data, point, sizeof(T) * _dim);

    pool.clear();
    tmp.clear();
    visited.clear();
    std::vector<unsigned> pruned_list;
    unsigned              Lindex = parameters.Get<unsigned>("L");

    std::vector<unsigned> init_ids;
    get_expanded_nodes(location, Lindex, init_ids, pool, visited);

    for (unsigned i = 0; i < pool.size(); i++)
      if (pool[i].id == (unsigned) location) {
        pool.erase(pool.begin() + i);
        visited.erase((unsigned) location);
        break;
      }

    prune_neighbors(location, pool, parameters, pruned_list);
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
    _final_graph[location].reserve((_u64)(range * SLACK_FACTOR * 1.05));

    if (pruned_list.empty()) {
      std::cout << "Thread: " << std::this_thread::get_id() << "Tag id: " << tag
                << " pruned_list.size(): " << pruned_list.size() << std::endl;
    }
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

    assert(_final_graph[location].size() <= range);
    if (_support_eager_delete)
      inter_insert(location, pruned_list, parameters, 1);
    else
      inter_insert(location, pruned_list, parameters, 0);
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const TagT &tag) {
    if ((_eager_done) && (!_data_compacted)) {
      std::cerr << "Eager delete requests were issued but data was not "
                   "compacted, cannot proceed with lazy_deletes"
                << std::endl;
      return -2;
    }
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    _lazy_done = true;

    {
      std::shared_lock<std::shared_timed_mutex> l(_tag_lock);

      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        //        diskann::cerr << "Delete tag not found" << std::endl;
        return -1;
      }
      assert(_tag_to_location[tag] < _max_points);
    }

    {
      std::unique_lock<std::shared_timed_mutex> l(_delete_lock);
      std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
      _delete_set.insert(_tag_to_location[tag]);
    }

    {
      std::unique_lock<std::shared_timed_mutex> l(_tag_lock);
      _location_to_tag.erase(_tag_to_location[tag]);
      _tag_to_location.erase(tag);
    }

    return 0;
  }

  // TODO: Check if this function needs a shared_lock on _tag_lock.
  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const tsl::robin_set<TagT> &tags,
                                  std::vector<TagT> &         failed_tags) {
    if (failed_tags.size() > 0) {
      std::cerr << "failed_tags should be passed as an empty list" << std::endl;
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
      //      assert(_tag_to_location[tag] < _max_points);
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        failed_tags.push_back(tag);
      } else {
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
      }
    }

    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::extract_data(
      T *ret_data, std::unordered_map<TagT, unsigned> &tag_to_location) {
    if (!_data_compacted) {
      std::cerr
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
  void Index<T, TagT>::get_location_to_tag(
      std::unordered_map<unsigned, TagT> &ret_loc_to_tag) {
    ret_loc_to_tag = _location_to_tag;
  }

  template<typename T, typename TagT>
  bool Index<T, TagT>::hasIndexBeenSaved() {
    return _is_saved;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_active_tags(tsl::robin_set<TagT> &active_tags) {
    active_tags.clear();
    for (auto iter : _tag_to_location) {
      active_tags.insert(iter.first);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::print_delete_set() const {
    diskann::cout << "Delete set is of size: " << _delete_set.size()
                  << std::endl;
    std::vector<uint32_t> sorted_delete_set;
    for (auto i : _delete_set) {
      sorted_delete_set.push_back(i);
    }
    std::sort(sorted_delete_set.begin(), sorted_delete_set.end());
    diskann::cout << "Sorted Delete set is of size: " << _delete_set.size()
                  << std::endl;

    // TODO: Debugging ONLY
    size_t            counter = 60000;
    std::vector<_u32> missing_ids;
    for (auto i : sorted_delete_set) {
      if (i != counter) {
        missing_ids.push_back(i);
      }
      counter++;
    }
    diskann::cout << "Missing ids in delete set for the 60k-80k case:"
                  << std::endl;
    for (auto i : missing_ids) {
      diskann::cout << i << ",";
    }
    diskann::cout << std::endl;
  }
  template<typename T, typename TagT>
  void Index<T, TagT>::are_deleted_points_in_graph() const {
    std::vector<std::pair<_u32, _u32>> start_end_pairs;
    for (size_t i = 0; i < _nd; i++) {
      for (size_t j = 0; j < _final_graph[i].size(); j++) {
        if (_delete_set.find(_final_graph[i][j]) != _delete_set.end()) {
          start_end_pairs.push_back(
              std::pair<_u32, _u32>(i, _final_graph[i][j]));
        }
      }
    }

    if (start_end_pairs.size() > 0) {
      diskann::cout << "Found " << start_end_pairs.size()
                    << " references to deleted vertices" << std::endl;
      std::sort(start_end_pairs.begin(), start_end_pairs.end(),
                [](const std::pair<_u32, _u32> &val1,
                   const std::pair<_u32, _u32> &val2) {
                  return val1.first < val2.first;
                });
      diskann::cout << "Min source id: " << start_end_pairs[0].first
                    << " Max source id: "
                    << start_end_pairs[start_end_pairs.size() - 1].first
                    << std::endl;
      std::sort(start_end_pairs.begin(), start_end_pairs.end(),
                [](const std::pair<_u32, _u32> &val1,
                   const std::pair<_u32, _u32> &val2) {
                  return val1.second < val2.second;
                });
      diskann::cout << "Min target id: " << start_end_pairs[0].second
                    << " Max target id: "
                    << start_end_pairs[start_end_pairs.size() - 1].second
                    << std::endl;
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

  /*  Internals of the library */
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
}  // namespace diskann
