#include "index.h"
#include <math_utils.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include "exceptions.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include <boost/dynamic_bitset.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include <cassert>
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "windows_customizations.h"
#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

#define INDEXING_BEAM_WIDTH 1
#define NEW_FILE_FORMAT 1
#define SATURATE_GRAPH 1

// only L2 implemented. Need to implement inner product search
namespace {
  template<typename T>
  diskann::Distance<T> *get_distance_function();

  template<>
  diskann::Distance<float> *get_distance_function() {
    return new diskann::DistanceL2();
  }

  template<>
  diskann::Distance<int8_t> *get_distance_function() {
    return new diskann::DistanceL2Int8();
  }

  template<>
  diskann::Distance<uint8_t> *get_distance_function() {
    return new diskann::DistanceL2UInt8();
  }
}  // namespace

namespace diskann {

  // Initialize an index with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T, typename TagT>
  Index<T, TagT>::Index(Metric m, const char *filename, const size_t max_points,

                        const size_t nd, const size_t num_frozen_pts,
                        const bool enable_tags, const bool store_data,
                        const bool support_eager_delete)
      : _num_frozen_pts(num_frozen_pts), _has_built(false), _width(0),
        _can_delete(false), _eager_done(true), _lazy_done(true),
        _compacted_order(true), _enable_tags(enable_tags),
        _consolidated_order(true), _support_eager_delete(support_eager_delete),
        _store_data(store_data) {
    // data is stored to _nd * aligned_dim matrix with necessary zero-padding
    std::cout << "Number of frozen points = " << _num_frozen_pts << std::endl;
    load_aligned_bin<T>(std::string(filename), _data, _nd, _dim, _aligned_dim);

    if (nd > 0) {
      if (_nd >= nd)
        _nd = nd;  // Consider the first _nd points and ignore the rest.
      else {
        std::cerr << "ERROR: Driver requests loading " << _nd << " points,"
                  << "but file has fewer (" << nd << ") points" << std::endl;
        exit(-1);
      }
    }

    _max_points = (max_points > 0) ? max_points : _nd;
    if (_max_points < _nd) {
      std::cerr << "ERROR: max_points must be >= data size; max_points: "
                << _max_points << "  n: " << _nd << std::endl;
      exit(-1);
    }

    // Allocate space for max points and frozen points,
    // and add frozen points at the end of the array
    if (_num_frozen_pts > 0) {
      _data = (T *) realloc(
          _data, (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(T));
      if (_data == NULL) {
        std::cout << "Realloc failed, killing programme" << std::endl;
        exit(-1);
      }
    }

    this->_distance = ::get_distance_function<T>();
    _locks = std::vector<std::mutex>(_max_points + _num_frozen_pts);

    _width = 0;
  }

  template<>
  Index<float>::~Index() {
    delete this->_distance;
    aligned_free(_data);
  }

  template<>
  Index<_s8>::~Index() {
    delete this->_distance;
    aligned_free(_data);
  }

  template<>
  Index<_u8>::~Index() {
    delete this->_distance;
    aligned_free(_data);
  }

  // save the graph index on a file as an adjacency list. For each point, first
  // store the number of neighbors, and then the neighbor list (each as 4 byte
  // unsigned)
  template<typename T, typename TagT>
  void Index<T, TagT>::save(const char *filename) {
    long long     total_gr_edges = 0;
    size_t        index_size = 0;
    std::ofstream out(std::string(filename), std::ios::binary | std::ios::out);

    if (_support_eager_delete)
      if (_eager_done && (!_compacted_order)) {
        if (_nd < _max_points) {
          assert(_final_graph.size() == _max_points + _num_frozen_pts);
          unsigned              active = 0;
          std::vector<unsigned> new_location = get_new_location(active);
          std::cout << "Size of new_location = " << new_location.size()
                    << std::endl;
          for (unsigned i = 0; i < new_location.size(); i++)
            if ((_delete_set.find(i) == _delete_set.end()) &&
                (new_location[i] >= _max_points + _num_frozen_pts))
              std::cout << "Wrong new_location assigned to  " << i << std::endl;
            else {
              if ((_delete_set.find(i) != _delete_set.end()) &&
                  (new_location[i] < _max_points + _num_frozen_pts))
                std::cout << "Wrong location assigned to delete point  " << i
                          << std::endl;
            }
          compact_data(new_location, active, _compacted_order);

          if (_support_eager_delete)
            update_in_graph();

        } else {
          assert(_final_graph.size() == _max_points + _num_frozen_pts);
          if (_enable_tags) {
            _change_lock.lock();
            if (_can_delete) {
              std::cerr
                  << "Disable deletes and consolidate index before saving."
                  << std::endl;
              exit(-1);
            }
          }
        }
      }
    if (_lazy_done) {
      assert(_final_graph.size() == _max_points + _num_frozen_pts);
      if (_enable_tags) {
        _change_lock.lock();
        if (_can_delete || (!_consolidated_order)) {
          std::cout << "Disable deletes and consolidate index before saving."
                    << std::endl;
          exit(-1);
        }
      }
    }
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &_width, sizeof(unsigned));
    out.write((char *) &_ep, sizeof(unsigned));
    for (unsigned i = 0; i < _nd + _num_frozen_pts; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _final_graph[i].data(), GK * sizeof(unsigned));
      total_gr_edges += GK;
    }
    index_size = out.tellp();
    out.seekp(0, std::ios::beg);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.close();

    std::cout << "Avg degree: "
              << ((float) total_gr_edges) / ((float) (_nd + _num_frozen_pts))
              << std::endl;
  }

  // load the index from file and update the width (max_degree), ep (navigating
  // node id), and _final_graph (adjacency list)
  template<typename T, typename TagT>
  void Index<T, TagT>::load(const char *filename, const bool load_tags,
                            const char *tag_filename) {
#ifdef NEW_FILE_FORMAT
    validate_file_size(filename);
#endif
    std::ifstream in(filename, std::ios::binary);
#ifdef NEW_FILE_FORMAT
    size_t expected_file_size;
    in.read((char *) &expected_file_size, sizeof(_u64));
#endif
    in.read((char *) &_width, sizeof(unsigned));
    in.read((char *) &_ep, sizeof(unsigned));
    std::cout << "Loading vamana index " << filename << "..." << std::flush;

    size_t   cc = 0;
    unsigned nodes = 0;
    while (!in.eof()) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (in.eof())
        break;
      cc += k;
      ++nodes;
      std::vector<unsigned> tmp(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));
      /* DEBUGGING CHECKS
       std::set<unsigned> unique_nbrs;
       for (auto id : tmp)
         unique_nbrs.insert(id);
       if (unique_nbrs.size() != tmp.size()) {
         std::cout << "Duplicate neighbors for point " << nodes - 1 <<
   std::endl;
         exit(-1);
       }
       if (std::find(tmp.begin(), tmp.end(), nodes - 1) != tmp.end()) {
         std::cout << "self-loop at " << nodes - 1 << std::endl;
         exit(-1);
       }
   END DEBUGGING CHECKS     */

      _final_graph.emplace_back(tmp);
      if (nodes % 10000000 == 0)
        std::cout << "." << std::flush;
    }
    if (_final_graph.size() != _nd) {
      std::cout << "ERROR. mismatch in number of points. Graph has "
                << _final_graph.size() << " points and loaded dataset has "
                << _nd << " points. " << std::endl;
      exit(-1);
    }

    std::cout << "..done. Index has " << nodes << " nodes and " << cc
              << " out-edges" << std::endl;

    // SEPARATE FUNCTION FOR LOADING TAGS? NEED TO CHECK IF SIZE MATCHES ETC?
    // USE LOAD_BIN?
    if (load_tags) {
      if (_enable_tags == false)
        std::cout << "Enabling tags." << std::endl;
      _enable_tags = true;
      std::ifstream tag_file;
      if (tag_filename == NULL)
        tag_file = std::ifstream(std::string(filename) + std::string(".tags"));
      else
        tag_file = std::ifstream(std::string(tag_filename));
      if (!tag_file.is_open()) {
        std::cerr << "Tag file not found." << std::endl;
        exit(-1);
      }
      unsigned id = 0;
      TagT     tag;
      while (tag_file >> tag) {
        _location_to_tag[id] = tag;
        _tag_to_location[tag] = id++;
      }
      tag_file.close();
      assert(id == _nd);
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
        center[j] += _data[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] /= _nd;

    // compute all to one distance
    float * distances = new float[_nd]();
#pragma omp parallel for schedule(static, 65536)
    for (_s64 i = 0; i < (_s64) _nd;
         i++) {  // GOPAL Changed from "size_t i" to "int i"
      // extract point and distance reference
      float &  dist = distances[i];
      const T *cur_vec = _data + (i * (size_t) _aligned_dim);
      dist = 0;
      float diff = 0;
      for (size_t j = 0; j < _aligned_dim; j++) {
        diff = (center[j] - cur_vec[j]) * (center[j] - cur_vec[j]);
        dist += diff;
      }
    }
    // find imin
    size_t min_idx = 0;
    float  min_dist = distances[0];
    for (size_t i = 1; i < _nd; i++) {
      if (distances[i] < min_dist) {
        min_idx = i;
        min_dist = distances[i];
      }
    }

    //    std::cout << "Medoid index = " << min_idx << std::endl;
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
      const T *node_coords, const unsigned Lsize, const unsigned beam_width,
      const std::vector<unsigned> &init_ids,
      std::vector<Neighbor> &      expanded_nodes_info,
      tsl::robin_set<unsigned> &   expanded_nodes_ids,
      std::vector<Neighbor> &      best_L_nodes) {
    best_L_nodes.resize(Lsize + 1);
    expanded_nodes_info.reserve(10 * Lsize);
    expanded_nodes_info.reserve(10 * Lsize);

    //    std::vector<Neighbor> random_visited_info;
    //    random_visited_info.reserve(500);

    unsigned l = 0;
    Neighbor nn;
    //    boost::dynamic_bitset<> inserted_into_pool(_max_points +
    //    _num_frozen_pts);
    tsl::robin_set<unsigned> inserted_into_pool;

    for (auto id : init_ids) {
      assert(id < _max_points);
      nn = Neighbor(id,
                    _distance->compare(_data + _aligned_dim * (size_t) id,
                                       node_coords, _aligned_dim),
                    true);
      if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
        inserted_into_pool.insert(id);
        //      if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
        //        inserted_into_pool.insert(id);
        best_L_nodes[l++] = nn;
      }
      if (l == Lsize)
        break;
    }

    /* sort best_L_nodes based on distance of each point to node_coords */
    std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
    unsigned              k = 0;
    uint32_t              hops = 0;
    uint32_t              cmps = 0;
    std::vector<unsigned> frontier;
    std::vector<unsigned> nbrs_to_insert;
    nbrs_to_insert.reserve(30 * Lsize);

    while (k < l) {
      unsigned nk = l;

      frontier.clear();
      nbrs_to_insert.clear();
      unsigned marker = k - 1;

      while (++marker < l && frontier.size() < (size_t) beam_width) {
        if (best_L_nodes[marker].flag) {
          frontier.emplace_back(best_L_nodes[marker].id);
          best_L_nodes[marker].flag = false;
          expanded_nodes_info.emplace_back(best_L_nodes[marker]);
          expanded_nodes_ids.insert(best_L_nodes[marker].id);
        }
      }

      if (!frontier.empty())
        hops++;
      for (auto iter = frontier.begin(); iter != frontier.end(); iter++) {
        auto n = *iter;
        if ((iter + 1) != frontier.end()) {
          auto nextn = *(iter + 1);
          diskann::prefetch_vector(
              (const char *) _final_graph[nextn].data(),
              _final_graph[nextn].size() * sizeof(unsigned));
        }

        for (unsigned m = 0; m < _final_graph[n].size(); ++m) {
          unsigned id = _final_graph[n][m];
          if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
            inserted_into_pool.insert(id);
            // Add each unique
            //            neighbor to
            // inserted to pool, if not already
            // added. we will try to insert them
            // into pool subsequently
            nbrs_to_insert.emplace_back(
                id);  // add as candidates to be inserted
          }
        }
      }

      for (uint64_t m = 0; m < nbrs_to_insert.size(); m++) {
        if (m < (nbrs_to_insert.size() - 1)) {
          // id_next = next neighbor
          unsigned id_next = nbrs_to_insert[m + 1];
          // vec_next1: data of next neighbor
          const T *vec_next1 = _data + (size_t) id_next * _aligned_dim;
          diskann::prefetch_vector((const char *) vec_next1,
                                   _aligned_dim * sizeof(T));
        }
        auto id = nbrs_to_insert[m];

        cmps++;
        // compare distance of id with node_coords
        float dist =
            _distance->compare(node_coords, _data + _aligned_dim * (size_t) id,
                               (unsigned) _aligned_dim);

        Neighbor nn(id, dist, true);
        //        random_visited_info.emplace_back(id,dist,true);

        if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
          continue;

        // if distance is smaller than largest, add to best_L_nodes, keep it
        // sorted
        unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
        if (l < Lsize)
          ++l;
        if (r < nk)
          nk = r;
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    return std::make_pair(hops, cmps);
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
    // We may optionally populate init_ids with random points
    /*      for (uint32_t i = 0; i < (Lindex - 1); i++) {
          unsigned seed = rand() % (_nd + _num_frozen_pts);
          seed = seed < _nd ? seed : seed - _nd + _max_points;
          init_ids.emplace_back(seed);
        }
        */
    iterate_to_fixed_point(node_coords, Lindex, INDEXING_BEAM_WIDTH, init_ids,
                           expanded_nodes_info, expanded_nodes_ids,
                           best_L_nodes);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const unsigned location, const float alpha,
                                    const unsigned degree, const unsigned maxc,
                                    std::vector<Neighbor> &result) {
    uint32_t           pool_size = pool.size();
    std::vector<float> occlude_factor(pool_size, 0);
    occlude_list(pool, location, alpha, degree, maxc, result, occlude_factor);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const unsigned location, const float alpha,
                                    const unsigned degree, const unsigned maxc,
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
      cur_alpha *= 1.1;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_neighbors(const unsigned location,
                                       std::vector<Neighbor> &pool,
                                       const Parameters &     parameter,
                                       std::vector<unsigned> &pruned_list) {
    unsigned range = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    float    alpha = parameter.Get<float>("alpha");

    if (pool.size() == 0)
      return;

    _width = (std::max)(_width, range);

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list(pool, location, alpha, range, maxc, result, occlude_factor);

    /* Add all the nodes in result into a variable called cut_graph
     * So this contains all the neighbors of id location
     */
    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      if (iter.id != location)
        pruned_list.emplace_back(iter.id);
    }

    if (SATURATE_GRAPH && alpha > 1) {
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
      assert(des >= 0 && des < _max_points);
      if (des > _max_points)
        std::cout << "error. " << des << " exceeds max_pts" << std::endl;
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
  void Index<T, TagT>::inter_insert(unsigned n,
                                    std::vector<unsigned> &pruned_list,
                                    const Parameters &     parameter,
                                    bool                   update_in_graph) {
    const auto range = parameter.Get<unsigned>("R");
    assert(n >= 0 && n < _nd);
    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < _max_points);
      /* des_pool contains the neighbors of the neighbors of n */
      auto &                des_pool = _final_graph[des];
      std::vector<unsigned> copy_of_neighbors;
      bool                  prune_needed = false;
      {
        LockGuard guard(_locks[des]);
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < SLACK_FACTOR * range) {
            des_pool.emplace_back(n);
            if (update_in_graph) {
              // USE APPROPRIATE LOCKS FOR IN_GRAPH
              if (std::find(_in_graph[n].begin(), _in_graph[n].end(), des) ==
                  _in_graph[n].end()) {
                _in_graph[n].emplace_back(des);
              }
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
          // DELETE IN-EDGES FROM IN_GRAPH USING APPROPRIATE LOCKS
          _final_graph[des].clear();
          for (auto new_nbr : new_out_neighbors) {
            _final_graph[des].emplace_back(new_nbr);
            if (update_in_graph) {
              _in_graph[new_nbr].emplace_back(des);
            }
          }
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

    uint32_t NUM_SYNCS = DIV_ROUND_UP(_nd + _num_frozen_pts, (128 * 128));
    if (NUM_SYNCS < 40)
      NUM_SYNCS = 40;
    std::cout << "Number of syncs: " << NUM_SYNCS << std::endl;

    const unsigned argL = parameters.Get<unsigned>("L");  // Search list size
    const unsigned range = parameters.Get<unsigned>("R");
    const float    last_round_alpha = parameters.Get<float>("alpha");
    unsigned       L = argL;

    std::vector<unsigned> Lvec;
    Lvec.push_back(2 * L / 3);
    Lvec.push_back(L);
    const unsigned NUM_RNDS = Lvec.size();

    // Max degree of graph
    // Pruning parameter
    // Set alpha=1 for the first pass; use specified alpha for last pass
    parameters.Set<float>("alpha", 1);

    /* visit_order is a vector that is initialized to the entire graph */
    std::vector<unsigned> visit_order;
    for (size_t i = 0; i < _nd; i++) {
      visit_order.emplace_back(i);
    }

    for (size_t i = 0; i < _num_frozen_pts; ++i)
      visit_order.emplace_back(_max_points + i);

    // if there are frozen points, the first such one is set to be the _ep
    if (_num_frozen_pts > 0)
      _ep = _max_points;
    else
      _ep = calculate_entry_point();

    _final_graph.reserve(_max_points + _num_frozen_pts);
    _final_graph.resize(_max_points + _num_frozen_pts);
    if (_support_eager_delete) {
      _in_graph.reserve(_max_points + _num_frozen_pts);
      _in_graph.resize(_max_points + _num_frozen_pts);
    }

    for (uint64_t p = 0; p < _max_points + _num_frozen_pts; p++) {
      _final_graph[p].reserve(range * SLACK_FACTOR * 1.05);
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

      float    sync_time = 0, total_sync_time = 0;
      float    inter_time = 0, total_inter_time = 0;
      size_t   inter_count = 0, total_inter_count = 0;
      unsigned progress_counter = 0;

      size_t round_size = DIV_ROUND_UP(_nd, NUM_SYNCS);  // size of each batch
      std::vector<unsigned> need_to_sync(_max_points + _num_frozen_pts, 0);

      std::vector<tsl::robin_set<unsigned>> sync_visited_vector(round_size);
      std::vector<std::vector<unsigned>>    pruned_list_vector(round_size);

      for (uint32_t sync_num = 0; sync_num < NUM_SYNCS; sync_num++) {
        size_t start_id = sync_num * round_size;
        size_t end_id =
            (std::min)(_nd + _num_frozen_pts, (sync_num + 1) * round_size);

        auto s = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff;

#pragma omp parallel for schedule(dynamic, 64)
        for (_u64 node_ctr = (_u64) start_id; node_ctr < (_u64) end_id;
             ++node_ctr) {
          _u64                      node = visit_order[node_ctr];
          size_t                    node_offset = node_ctr - start_id;
          tsl::robin_set<unsigned> &visited = sync_visited_vector[node_offset];
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          // get nearest neighbors of n in tmp. pool contains all the points
          // that were checked along with their distance from n. visited
          // contains all
          // the points visited, just the ids
          std::vector<Neighbor> pool;
          pool.reserve(L * 2);
          get_expanded_nodes(node, L, init_ids, pool, visited);
          /* check the neighbors of the query that are not part of visited,
           * check their distance to the query, and add it to pool.
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
          pool.clear();
          pool.shrink_to_fit();
        }
        diff = std::chrono::high_resolution_clock::now() - s;
        sync_time += diff.count();

// prune_neighbors will check pool, and remove some of the points and
// create a cut_graph, which contains neighbors for point n
#pragma omp parallel for schedule(dynamic, 64)
        for (_u64 node_ctr = (_u64) start_id; node_ctr < (_u64) end_id;
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
        for (_u64 node_ctr = start_id; node_ctr < (_u64) end_id; ++node_ctr) {
          _u64                   node = visit_order[node_ctr];
          _u64                   node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          tsl::robin_set<unsigned> &visited = sync_visited_vector[node_offset];
          batch_inter_insert(node, pruned_list, parameters, need_to_sync);
          //          inter_insert(node, pruned_list, parameters, 0);
          pruned_list.clear();
          visited.clear();
          pruned_list.shrink_to_fit();
          tsl::robin_set<unsigned>().swap(visited);
        }

#pragma omp parallel for schedule(dynamic, 64)
        for (_u64 node_ctr = 0; node_ctr < visit_order.size(); node_ctr++) {
          _u64 node = visit_order[node_ctr];
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
          std::cout.precision(4);
          std::cout << "Completed  (round: " << rnd_no << ", sync: " << sync_num
                    << "/" << NUM_SYNCS << " with L " << L << ")" << std::endl;
          total_sync_time += sync_time;
          total_inter_time += inter_time;
          total_inter_count += inter_count;
          sync_time = 0;
          inter_time = 0;
          inter_count = 0;
          progress_counter += 5;
        }
      }
      std::cout << "Completed Pass " << rnd_no << " of data using L=" << L
                << " and alpha=" << parameters.Get<float>("alpha")
                << ". Stats: ";
      std::cout << "search+prune_time=" << total_sync_time
                << "s, inter_time=" << total_inter_time
                << "s, inter_count=" << total_inter_count << std::endl;
    }

    std::cout << "Starting final cleanup.." << std::flush;
#pragma omp parallel for schedule(dynamic, 64)
    for (_u64 node_ctr = 0; node_ctr < visit_order.size(); node_ctr++) {
      size_t node = visit_order[node_ctr];
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
    std::cout << "done. Link time: "
              << ((double) link_timer.elapsed() / (double) 1000000) << "s"
              << std::endl;

    //    compute_graph_stats();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(Parameters &parameters,
                             const std::vector<TagT> &tags) {
    if (_enable_tags) {
      if (tags.size() != _nd) {
        std::cerr << "#Tags should be equal to #points" << std::endl;
        exit(-1);
      }
      for (size_t i = 0; i < tags.size(); ++i) {
        _tag_to_location[tags[i]] = i;
        _location_to_tag[i] = tags[i];
      }
    }
    std::cout << "Starting index build..." << std::endl;
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
    std::cout << "Degree: max:" << max
              << "  avg:" << (float) total / (float) _nd << "  min:" << min
              << "  count(deg<2):" << cnt << "\n"
              << "Index built." << std::endl;
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::beam_search(
      const T *query, const size_t K, const unsigned L, unsigned beam_width,
      std::vector<unsigned> start_points, unsigned *indices) {
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor>    best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    auto retval = iterate_to_fixed_point(query, L, beam_width, init_ids,
                                         expanded_nodes_info,
                                         expanded_nodes_ids, best_L_nodes);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      indices[pos] = it.id;
      pos++;
      if (pos == K)
        break;
    }
    return retval;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::beam_search(
      const T *query, const uint64_t K, const uint64_t L, unsigned beam_width,
      std::vector<unsigned> init_ids, uint64_t *indices, float *distances) {
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor>    best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    auto retval = iterate_to_fixed_point(query, (unsigned) L, beam_width,
                                         init_ids, expanded_nodes_info,
                                         expanded_nodes_ids, best_L_nodes);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      indices[pos] = it.id;
      distances[pos] = it.distance;
      pos++;
      if (pos == K)
        break;
    }
    return retval;
  }

  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::beam_search_tags(
      const T *query, const size_t K, const size_t L, TagT *tags,
      unsigned beam_width, std::vector<unsigned> start_points,
      unsigned frozen_pts, unsigned *indices_buffer) {
    const bool alloc = indices_buffer == NULL;
    auto       indices = alloc ? new unsigned[K] : indices_buffer;
    auto ret = beam_search(query, K, L, beam_width, start_points, indices);
    for (int i = 0; i < (int) K; ++i)
      tags[i] = _location_to_tag[indices[i]];
    if (alloc)
      delete[] indices;
    return ret;
  }

  /*************************************************
   *      Support for Incremental Update
   *************************************************/

  // in case we add ''frozen'' auxiliary points to the dataset, these are not
  // visible to external world, we generate them here and update our dataset
  template<typename T, typename TagT>
  int Index<T, TagT>::generate_random_frozen_points(const char *filename) {
    if (_has_built) {
      std::cout << "Index already built. Cannot add more points" << std::endl;
      return -1;
    }

    if (filename) {  // user defined frozen points
      T *frozen_pts;
      load_aligned_bin<T>(std::string(filename), frozen_pts, _num_frozen_pts,
                          _dim, _aligned_dim);
      for (unsigned i = 0; i < _num_frozen_pts; i++) {
        for (unsigned d = 0; d < _dim; d++)
          _data[(i + _max_points) * _aligned_dim + d] =
              frozen_pts[i * _dim + d];
        for (unsigned d = _dim; d < _aligned_dim; d++)
          _data[(i + _max_points) * _aligned_dim + d] = 0;
      }
    } else {  // random frozen points

      std::random_device                    device;
      std::mt19937                          generator(device());
      std::uniform_real_distribution<float> dist(0, 1);
      // Harsha: Should the distribution change with the distance metric?

      for (unsigned i = 0; i < _num_frozen_pts; ++i) {
        for (unsigned d = 0; d < _dim; d++)
          _data[(i + _max_points) * _aligned_dim + d] = dist(generator);
        for (unsigned d = _dim; d < _aligned_dim; d++)
          _data[(i + _max_points) * _aligned_dim + d] = 0;
      }
    }

    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::enable_delete() {
    LockGuard guard(_change_lock);
    assert(!_can_delete);
    assert(_enable_tags);

    if (_can_delete) {
      std::cerr << "Delete already enabled" << std::endl;
      return -1;
    }
    if (!_enable_tags) {
      std::cerr << "Tags must be instantiated for deletions" << std::endl;
      return -2;
    }

    if (_consolidated_order && _compacted_order) {
      assert(_empty_slots.size() == 0);
      for (unsigned slot = _nd; slot < _max_points; ++slot)
        _empty_slots.insert(slot);
      _consolidated_order = false;
      _compacted_order = false;
    }

    _lazy_done = false;
    _eager_done = false;
    _can_delete = true;

    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::eager_delete(const TagT tag,
                                   const Parameters &parameters) {
    if (_lazy_done && (!_consolidated_order)) {
      std::cout << "Lazy delete reuests issued but data not consolidated, "
                   "cannot proceed with eager deletes."
                << std::endl;
      return -1;
    }
    LockGuard guard(_change_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      std::cerr << "Delete tag not found" << std::endl;
      return -1;
    }

    unsigned id = _tag_to_location[tag];
    _location_to_tag.erase(_tag_to_location[tag]);
    _tag_to_location.erase(tag);
    _delete_set.insert(id);
    _empty_slots.insert(id);

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    // delete point from out-neighbors' in-neighbor list
    for (auto j : _final_graph[id])
      for (unsigned k = 0; k < _in_graph[j].size(); k++)
        if (_in_graph[j][k] == id) {
          _in_graph[j].erase(_in_graph[j].begin() + k);
          break;
        }

    tsl::robin_set<unsigned> in_nbr;
    for (unsigned i = 0; i < _in_graph[id].size(); i++)
      in_nbr.insert(_in_graph[id][i]);
    assert(_in_graph[id].size() == in_nbr.size());

    tsl::robin_set<unsigned> candidate_set;
    std::vector<Neighbor>    expanded_nghrs;
    std::vector<Neighbor>    result;
    std::vector<Neighbor>    pool, tmp;
    tsl::robin_set<unsigned> visited;

    unsigned              Lindex = parameters.Get<unsigned>("L");
    std::vector<unsigned> init_ids;

    get_expanded_nodes(id, Lindex, init_ids, pool, visited);

    for (unsigned i = 0; i < pool.size(); i++)
      if (pool[i].id == id) {
        pool.erase(pool.begin() + i);
        break;
      }

    for (auto it : in_nbr) {
      _final_graph[it].erase(
          std::remove(_final_graph[it].begin(), _final_graph[it].end(), id),
          _final_graph[it].end());
    }

    for (auto it : visited) {
      auto ngh = it;
      if (in_nbr.find(ngh) != in_nbr.end()) {
        candidate_set.clear();
        expanded_nghrs.clear();
        result.clear();

        for (auto j : _final_graph[id])
          if ((j != id) && (j != ngh) &&
              (_delete_set.find(j) == _delete_set.end()))
            candidate_set.insert(j);

        for (auto j : _final_graph[ngh])
          if ((j != id) && (j != ngh) &&
              (_delete_set.find(j) == _delete_set.end()))
            candidate_set.insert(j);

        for (auto j : candidate_set)
          expanded_nghrs.push_back(
              Neighbor(j,
                       _distance->compare(_data + _aligned_dim * (size_t) ngh,
                                          _data + _aligned_dim * (size_t) j,
                                          (unsigned) _aligned_dim),
                       true));
        std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
        occlude_list(expanded_nghrs, ngh, alpha, range, maxc, result);

        for (auto iter : _final_graph[ngh])
          for (unsigned k = 0; k < _in_graph[iter].size(); k++)
            if (_in_graph[iter][k] == ngh) {
              _in_graph[iter].erase(_in_graph[iter].begin() + k);
            }

        _final_graph[ngh].clear();

        for (auto j : result) {
          if (_delete_set.find(j.id) == _delete_set.end())
            _final_graph[ngh].push_back(j.id);
          if (std::find(_in_graph[j.id].begin(), _in_graph[j.id].end(), ngh) ==
              _in_graph[j.id].end())
            _in_graph[j.id].emplace_back(ngh);
        }
      }
    }
    _final_graph[id].clear();
    _nd--;

    _eager_done = true;
    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::update_in_graph() {
    std::cout << "Updating in_graph.....";
    for (unsigned i = 0; i < _in_graph.size(); i++)
      _in_graph[i].clear();

    for (unsigned i = 0; i < _final_graph.size();
         i++)  // copying to in-neighbor graph

      for (unsigned j = 0; j < _final_graph[i].size(); j++) {
        if (std::find(_in_graph[_final_graph[i][j]].begin(),
                      _in_graph[_final_graph[i][j]].end(),
                      i) != _in_graph[_final_graph[i][j]].end())
          std::cout << "Duplicates found" << std::endl;
        _in_graph[_final_graph[i][j]].emplace_back(i);
      }

    size_t max_in, min_in, avg_in;
    max_in = 0;
    min_in = _max_points + 1;
    avg_in = 0;
    for (unsigned i = 0; i < _in_graph.size(); i++) {
      avg_in += _in_graph[i].size();
      if (_in_graph[i].size() > max_in)
        max_in = _in_graph[i].size();
      if ((_in_graph[i].size() < min_in) && (i != _ep))
        min_in = _in_graph[i].size();
    }

    std::cout << std::endl
              << "Max in_degree = " << max_in << "; Min in_degree = " << min_in
              << "; Average in_degree = "
              << (float) (avg_in) / (float) (_nd + _num_frozen_pts)
              << std::endl;
  }

  // Do not call consolidate_deletes() if you have not locked _change_lock.
  // Returns number of live points left after consolidation
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
    if (_eager_done) {
      std::cout << "No consolidation required, eager deletes done" << std::endl;
      return 0;
    }

    assert(!_consolidated_order);
    assert(_can_delete);
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    std::vector<unsigned> new_location;
    new_location.resize(_max_points + _num_frozen_pts,
                        _max_points + _num_frozen_pts);
    unsigned active = 0;
    for (unsigned old = 0; old < _max_points + _num_frozen_pts; ++old)
      if (_empty_slots.find(old) == _empty_slots.end() &&
          _delete_set.find(old) == _delete_set.end())
        new_location[old] = active++;
    assert(active + _delete_set.size() == _max_points + _num_frozen_pts);

    tsl::robin_set<unsigned> candidate_set;
    std::vector<Neighbor>    expanded_nghrs;
    std::vector<Neighbor>    result;

    for (unsigned i = 0; i < _max_points + _num_frozen_pts; ++i) {
      if (new_location[i] < _max_points + _num_frozen_pts) {
        candidate_set.clear();
        expanded_nghrs.clear();
        result.clear();

        bool modify = false;
        for (auto ngh : _final_graph[i]) {
          if (new_location[ngh] >= _max_points + _num_frozen_pts) {
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
          for (auto j : candidate_set)
            expanded_nghrs.push_back(
                Neighbor(j,
                         _distance->compare(_data + _aligned_dim * (size_t) i,
                                            _data + _aligned_dim * (size_t) j,
                                            (unsigned) _aligned_dim),
                         true));
          std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
          occlude_list(expanded_nghrs, i, alpha, range, maxc, result);

          _final_graph[i].clear();
          for (auto j : result) {
            if (j.id != i)
              _final_graph[i].push_back(j.id);
          }
        }
      } else
        _final_graph[i].clear();
    }

    if (_support_eager_delete)
      update_in_graph();

    _nd -= _delete_set.size();
    compact_data(new_location, active, _consolidated_order);
    return _nd;
  }

  template<typename T, typename TagT>
  std::vector<unsigned> Index<T, TagT>::get_new_location(unsigned &active) {
    std::vector<unsigned> new_location;
    new_location.resize(_max_points + _num_frozen_pts,
                        _max_points + _num_frozen_pts);

    for (unsigned old = 0; old < _max_points + _num_frozen_pts; ++old)
      if (_empty_slots.find(old) == _empty_slots.end() &&
          _delete_set.find(old) == _delete_set.end())
        new_location[old] = active++;
    assert(active + _delete_set.size() == _max_points + _num_frozen_pts);

    return new_location;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data(std::vector<unsigned> new_location,
                                    unsigned active, bool &mode) {
    // If start node is removed, replace it.
    assert(!mode);
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
        std::cerr << "ERROR: Did not find a replacement for start node."
                  << std::endl;
        exit(-1);
      } else {
        assert(_delete_set.find(_ep) == _delete_set.end());
        std::cout << "New start node is " << _ep << std::endl;
      }
    }

    std::cout << "Re-numbering nodes and edges and consolidating data... "
              << std::flush;
    std::cout << "active = " << active << std::endl;
    for (unsigned old = 0; old < _max_points + _num_frozen_pts; ++old) {
      if (new_location[old] <
          _max_points + _num_frozen_pts) {  // If point continues to exist

        // Renumber nodes to compact the order
        for (size_t i = 0; i < _final_graph[old].size(); ++i) {
          assert(new_location[_final_graph[old][i]] <= _final_graph[old][i]);
          _final_graph[old][i] = new_location[_final_graph[old][i]];
        }

        if (_support_eager_delete)
          for (size_t i = 0; i < _in_graph[old].size(); ++i) {
            if (new_location[_in_graph[old][i]] <= _in_graph[old][i])
              _in_graph[old][i] = new_location[_in_graph[old][i]];
          }

        // Move the data and adj list to the correct position
        if (new_location[old] != old) {
          assert(new_location[old] < old);
          _final_graph[new_location[old]].swap(_final_graph[old]);
          if (_support_eager_delete)
            _in_graph[new_location[old]].swap(_in_graph[old]);
          memcpy((void *) (_data + _aligned_dim * (size_t) new_location[old]),
                 (void *) (_data + _aligned_dim * (size_t) old),
                 _aligned_dim * sizeof(T));
        }
      }
    }
    std::cout << "done." << std::endl;

    std::cout << "Updating mapping between tags and ids... " << std::flush;
    // Update the location pointed to by tag
    _tag_to_location.clear();
    for (auto iter : _location_to_tag)
      _tag_to_location[iter.second] = new_location[iter.first];
    _location_to_tag.clear();
    for (auto iter : _tag_to_location)
      _location_to_tag[iter.second] = iter.first;
    std::cout << "done." << std::endl;

    for (unsigned old = active; old < _max_points + _num_frozen_pts; ++old)
      _final_graph[old].clear();
    _delete_set.clear();
    _empty_slots.clear();
    mode = true;
    std::cout << "Consolidated the index" << std::endl;

    /*	  for(unsigned i = 0; i < _nd + _num_frozen_pts; i++){
          int flag = 0;
          for(unsigned j = 0; j < _final_graph[i].size(); j++)
            if(_final_graph[i][j] == i){
              std::cout << "Self loop found just after compacting inside the
       function" << std::endl;
              flag = 1;
              break;
            }
          if(flag == 1)
            break;
        } */
  }

  // Do not call reserve_location() if you have not locked _change_lock.
  // It is not thread safe.
  template<typename T, typename TagT>
  unsigned Index<T, TagT>::reserve_location() {
    assert(_nd < _max_points);

    unsigned location;
    if (_consolidated_order || _compacted_order)
      location = _nd;
    else {
      assert(_empty_slots.size() != 0);
      assert(_empty_slots.size() + _nd == _max_points);

      auto iter = _empty_slots.begin();
      location = *iter;
      _empty_slots.erase(iter);
      _delete_set.erase(iter);
    }

    ++_nd;
    return location;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::readjust_data(unsigned _num_frozen_pts) {
    if (_num_frozen_pts > 0) {
      if (_final_graph[_max_points].empty()) {
        std::cout << "Readjusting data to correctly position frozen point"
                  << std::endl;
        for (unsigned i = 0; i < _nd; i++)
          for (unsigned j = 0; j < _final_graph[i].size(); j++)
            if (_final_graph[i][j] >= _nd)
              _final_graph[i][j] = _max_points + (_final_graph[i][j] - _nd);
        for (unsigned i = 0; i < _num_frozen_pts; i++) {
          for (unsigned k = 0; k < _final_graph[_nd + i].size(); k++)
            _final_graph[_max_points + i].emplace_back(
                _final_graph[_nd + i][k]);
          _final_graph[_nd + i].clear();
        }

        if (_support_eager_delete)
          update_in_graph();

        std::cout << "Finished updating graph, updating data now" << std::endl;
        for (unsigned i = 0; i < _num_frozen_pts; i++) {
          memcpy((void *) (_data + (size_t) _aligned_dim * (_max_points + i)),
                 _data + (size_t) _aligned_dim * (_nd + i),
                 sizeof(float) * _dim);
          memset((_data + (size_t) _aligned_dim * (_nd + i)), 0,
                 sizeof(float) * _aligned_dim);
        }
        std::cout << "Readjustment done" << std::endl;
      }
    } else
      std::cout << "No frozen points. No re-adjustment required" << std::endl;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::insert_point(const T *point, const Parameters &parameters,
                                   std::vector<Neighbor> &   pool,
                                   std::vector<Neighbor> &   tmp,
                                   tsl::robin_set<unsigned> &visited,
                                   vecNgh &cut_graph, const TagT tag) {
    unsigned range = parameters.Get<unsigned>("R");
    assert(_has_built);

    LockGuard guard(_change_lock);
    if (_enable_tags &&
        (_tag_to_location.find(tag) != _tag_to_location.end())) {
      std::cerr << "Entry with the tag " << tag << " exists already"
                << std::endl;
      return -1;
    }
    if (_nd == _max_points) {
      std::cerr << "Can not insert, reached maximum(" << _max_points
                << ") points." << std::endl;
      return -2;
    }

    size_t location = reserve_location();
    _tag_to_location[tag] = location;
    _location_to_tag[location] = tag;

    auto offset_data = _data + (size_t) _aligned_dim * location;
    memset((void *) offset_data, 0, sizeof(float) * _aligned_dim);
    memcpy((void *) offset_data, point, sizeof(float) * _dim);

    pool.clear();
    tmp.clear();
    cut_graph.clear();
    visited.clear();
    std::vector<unsigned> pruned_list;
    unsigned              Lindex = parameters.Get<unsigned>("L");

    std::vector<unsigned> init_ids;
    get_expanded_nodes(location, Lindex, init_ids, pool, visited);

    for (unsigned i = 0; i < pool.size(); i++)
      if (pool[i].id == location) {
        pool.erase(pool.begin() + i);
        visited.erase(location);
        break;
      }

    prune_neighbors(location, pool, parameters, pruned_list);

    assert(_final_graph.size() == _max_points + _num_frozen_pts);

    for (unsigned i = 0; i < _final_graph[location].size(); i++)
      _in_graph[_final_graph[location][i]].erase(
          std::remove(_in_graph[_final_graph[location][i]].begin(),
                      _in_graph[_final_graph[location][i]].end(), location),
          _in_graph[_final_graph[location][i]].end());

    _final_graph[location].clear();
    _final_graph[location].reserve(range);
    assert(!pruned_list.empty());
    for (auto link : pruned_list) {
      _final_graph[location].emplace_back(link);
      if (_support_eager_delete)
        if (std::find(_in_graph[link].begin(), _in_graph[link].end(),
                      location) == _in_graph[link].end()) {
          _in_graph[link].emplace_back(location);
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
  int Index<T, TagT>::disable_delete(const Parameters &parameters,
                                     const bool consolidate) {
    LockGuard guard(_change_lock);
    if (!_can_delete) {
      std::cerr << "Delete not currently enabled" << std::endl;
      return -1;
    }
    if (!_enable_tags) {
      std::cerr << "Point tag array not instantiated" << std::endl;
      exit(-1);
    }
    if (_eager_done) {
      std::cout << "#Points after eager_delete : " << _nd + _num_frozen_pts
                << std::endl;
      if (_tag_to_location.size() != _nd) {
        std::cerr << "Tags to points array wrong sized" << std::endl;
        return -2;
      }
    } else if (_tag_to_location.size() + _delete_set.size() != _nd) {
      std::cerr << "Tags to points array wrong sized" << std::endl;
      return -2;
    }
    if (_eager_done) {
      if (_location_to_tag.size() != _nd) {
        std::cerr << "Points to tags array wrong sized" << std::endl;
        return -3;
      }
    } else if (_location_to_tag.size() + _delete_set.size() != _nd) {
      std::cerr << "Points to tags array wrong sized" << std::endl;
      return -3;
    }
    if (consolidate) {
      auto nd = consolidate_deletes(parameters);

      if (nd >= 0)
        std::cout << "#Points after consolidation: " << nd + _num_frozen_pts
                  << std::endl;
    }

    _can_delete = false;
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::delete_point(const TagT tag) {
    if ((_eager_done) && (!_compacted_order)) {
      std::cout << "Eager delete requests were issued but data was not "
                   "compacted, cannot proceed with lazy_deletes"
                << std::endl;
      return -1;
    }
    LockGuard guard(_change_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      std::cerr << "Delete tag not found" << std::endl;
      return -1;
    }
    assert(_tag_to_location[tag] < _max_points);
    _delete_set.insert(_tag_to_location[tag]);
    _location_to_tag.erase(_tag_to_location[tag]);
    _tag_to_location.erase(tag);
    return 0;
  }

  // EXPORTS
  template DISKANN_DLLEXPORT class Index<float>;
  template DISKANN_DLLEXPORT class Index<int8_t>;
  template DISKANN_DLLEXPORT class Index<uint8_t>;
}  // namespace diskann
