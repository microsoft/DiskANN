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

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include <cassert>
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "windows_customizations.h"
#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

#define MAX_ALPHA 3
#define SLACK_FACTOR 1.2

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

    if (_enable_tags) {
      std::ofstream out_tags(std::string(filename) + std::string(".tags"));
      for (unsigned i = 0; i < _nd; i++) {
        out_tags << _location_to_tag[i] << "\n";
      }
      out_tags.close();
      _change_lock.unlock();
    }

    if (_store_data) {
      std::ofstream out_data(std::string(filename) + std::string(".data"),
                             std::ios::binary);
      unsigned new_nd = _nd + _num_frozen_pts;
      out_data.write((char *) &new_nd, sizeof(_u32));
      out_data.write((char *) &_dim, sizeof(_u32));
      for (unsigned i = 0; i < _nd + _num_frozen_pts; ++i)
        out_data.write((char *) (_data + i * _aligned_dim), _dim * sizeof(T));
      out_data.close();
    }

    std::cout << "Avg degree: "
              << ((float) total_gr_edges) / ((float) (_nd + _num_frozen_pts))
              << std::endl;
  }

  // load the index from file and update the width (max_degree), ep (navigating
  // node id), and _final_graph (adjacency list)
  template<typename T, typename TagT>
  void Index<T, TagT>::load(const char *filename, const bool load_tags) {
    validate_file_size(filename);
    size_t        expected_file_size;
    std::ifstream in(filename, std::ios::binary);
    in.read((char *) &expected_file_size, sizeof(_u64));
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
      _final_graph.emplace_back(tmp);
      if (std::find(tmp.begin(), tmp.end(), nodes - 1) != tmp.end())
        std::cout << "self-loop at " << nodes - 1 << std::endl;

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

    if (load_tags) {
      if (_enable_tags == false)
        std::cout << "Enabling tags." << std::endl;
      _enable_tags = true;
      std::ifstream tag_file(std::string(filename) + std::string(".tags"));
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

  // in case we add ''frozen'' auxiliary points to the dataset, these are not
  // visible to external world, we generate them here and update our dataset
  template<typename T, typename TagT>
  int Index<T, TagT>::generate_random_frozen_points() {
    if (_has_built) {
      std::cout << "Index already built. Cannot add more points" << std::endl;
      return -1;
    }

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

    return 0;
  }

  /**************************************************************
   *      Support for Static Index Building and Searching
   **************************************************************/

  /* init_random_graph():
   * degree: degree of the random graph
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::init_random_graph(unsigned degree) {
    degree = (std::min)(degree, (unsigned) 100);
    unsigned new_max_points = _max_points + _num_frozen_pts;
    unsigned new_nd = _nd + _num_frozen_pts;
    _final_graph.resize(new_max_points);
    _final_graph.reserve(new_max_points);

    if (_support_eager_delete) {
      _in_graph.resize(new_max_points);
      _in_graph.reserve(new_max_points);
    }

    std::cout << "Generating random graph with " << new_nd << " points... "
              << std::flush;
    // PAR_BLOCK_SZ gives the number of points that can fit in a single block
    _s64 PAR_BLOCK_SZ = (1 << 16);  // = 64KB
    _s64 nblocks = DIV_ROUND_UP((_s64) _nd, PAR_BLOCK_SZ);

#pragma omp parallel for schedule(static, 1)
    for (_s64 block = 0; block < nblocks; ++block) {
      std::random_device                    rd;
      std::mt19937                          gen(rd());
      std::uniform_int_distribution<size_t> dis(0, new_nd - 1);

      /* Put random number points as neighbours to the 10% of the nodes */
      for (_u64 i = (_u64) block * PAR_BLOCK_SZ;
           i < (_u64)(block + 1) * PAR_BLOCK_SZ && i < new_nd; i++) {
        size_t             node_loc = i < _nd ? i : i - _nd + _max_points;
        std::set<unsigned> rand_set;
        while (rand_set.size() < degree && rand_set.size() < new_nd - 1) {
          unsigned cur_pt = dis(gen);
          if (cur_pt != i)
            rand_set.insert(cur_pt < _nd ? cur_pt : cur_pt - _nd + _max_points);
        }

        _final_graph[node_loc].reserve(degree);
        for (auto s : rand_set)
          _final_graph[node_loc].emplace_back(s);
        _final_graph[node_loc].shrink_to_fit();
      }
    }

    std::cout << ".. done " << std::endl;
  }

  /* iterate_to_fixed_point():
   * query : point whose neighbors to be found.
   * init_ids : ids of neighbors of navigating node.
   * retset : will contain the nearest neighbors of the query.

   * expanded_nodes_info : will contain all the node ids and distances from
   query that are
   * checked.
   * visited : will contain all the nodes that are visited during search.
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::iterate_to_fixed_point(
      const size_t node_id, const unsigned Lindex,
      std::vector<unsigned> &init_ids, std::vector<Neighbor> &retset,
      std::vector<Neighbor> &   expanded_nodes_info,
      tsl::robin_set<unsigned> &expanded_nodes_ids) {
    /* compare distance of all points in init_ids with node_coords, and put the
     * id
     * with distance
     * in retset
     */

    const T *                node_coords = _data + _aligned_dim * node_id;
    unsigned                 l = 0;
    Neighbor                 nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    for (auto id : init_ids) {
      assert(id < _max_points);
      nn = Neighbor(id,
                    _distance->compare(_data + _aligned_dim * (size_t) id,
                                       node_coords, _aligned_dim),
                    true);
      if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
        inserted_into_pool.insert(id);
        retset[l++] = nn;
      }
      if (l == Lindex)
        break;
    }

    /* sort retset based on distance of each point to node_coords */
    std::sort(retset.begin(), retset.begin() + l);
    unsigned k = 0;
    while (k < l) {
      unsigned nk = l;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;
        if (k != node_id) {
          expanded_nodes_info.emplace_back(retset[k]);
          expanded_nodes_ids.insert(n);
        }

        // prefetch _final_graph[n]
        unsigned *nbrs = _final_graph[n].data();   // nbrs: data of neighbors
        unsigned  nnbrs = _final_graph[n].size();  // nnbrs: number of neighbors
        diskann::prefetch_vector((const char *) nbrs, nnbrs * sizeof(unsigned));
        for (size_t m = 0; m < nnbrs; m++) {
          unsigned id = nbrs[m];  // id = neighbor
          if (m < (nnbrs - 1)) {
            // id_next = next neighbor
            unsigned id_next = nbrs[m + 1];
            // vec_next1: data of next neighbor
            const T *vec_next1 = _data + (size_t) id_next * _aligned_dim;
            diskann::prefetch_vector((const char *) vec_next1,
                                     _aligned_dim * sizeof(T));
          }

          if (inserted_into_pool.find(id) != inserted_into_pool.end())
            continue;

          // compare distance of id with node_coords
          float dist = _distance->compare(node_coords,
                                          _data + _aligned_dim * (size_t) id,
                                          (unsigned) _aligned_dim);
          Neighbor nn(id, dist, true);
          inserted_into_pool.insert(id);
          if (dist >= retset[l - 1].distance && (l == Lindex))
            continue;

          // if distance is smaller than largest, add to retset, keep it
          // sorted
          unsigned r = InsertIntoPool(retset.data(), l, nn);
          if (l < Lindex)
            ++l;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    assert(!expanded_nodes_info.empty());
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_neighbors(
      const size_t node, const unsigned Lindex, std::vector<Neighbor> &retset,
      std::vector<Neighbor> &   expanded_nodes_info,
      tsl::robin_set<unsigned> &expanded_nodes_ids) {
    retset.resize(Lindex + 1);
    std::vector<unsigned> init_ids;
    init_ids.reserve(Lindex);
    init_ids.emplace_back(_ep);
    for (uint32_t i = 0; i < (Lindex - 1); i++) {
      unsigned seed = rand() % (_nd + _num_frozen_pts);
      seed = seed < _nd ? seed : seed - _nd + _max_points;
      init_ids.emplace_back(seed);
    }
    iterate_to_fixed_point(node, Lindex, init_ids, retset, expanded_nodes_info,
                           expanded_nodes_ids);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::reachable_bfs(
      const unsigned                         start_node,
      std::vector<tsl::robin_set<unsigned>> &bfs_order, bool *visited) {
    auto &                    nsg = _final_graph;
    tsl::robin_set<unsigned> *cur_level = new tsl::robin_set<unsigned>();
    tsl::robin_set<unsigned> *prev_level = new tsl::robin_set<unsigned>();
    prev_level->insert(start_node);
    visited[start_node] = true;
    unsigned level = 0;
    unsigned nsg_size = nsg.size();
    while (true) {
      // clear state
      cur_level->clear();

      size_t max_deg = 0;
      size_t min_deg = 0xffffffffffL;
      size_t sum_deg = 0;

      // select candidates
      for (auto id : *prev_level) {
        max_deg = (std::max)(max_deg, nsg[id].size());
        min_deg = (std::min)(min_deg, nsg[id].size());
        sum_deg += nsg[id].size();

        for (const auto &nbr : nsg[id]) {
          if (nbr >= nsg_size) {
            std::cerr << "invalid" << std::endl;
          }
          if (!visited[nbr]) {
            cur_level->insert(nbr);
            visited[nbr] = true;
          }
        }
      }

      if (cur_level->empty()) {
        break;
      }

      std::cerr << "Level #" << level << " : " << cur_level->size() << " nodes"
                << "\tDegree max: " << max_deg
                << "  avg: " << (float) sum_deg / (float) prev_level->size()
                << "  min: " << min_deg << std::endl;

      // create a new set
      tsl::robin_set<unsigned> add(cur_level->size());
      add.insert(cur_level->begin(), cur_level->end());
      bfs_order.emplace_back(add);

      // swap cur_level and prev_level, increment level
      prev_level->clear();
      std::swap(prev_level, cur_level);
      level++;
    }

    // cleanup
    delete cur_level;
    delete prev_level;
  }

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

    unsigned start = 0;
    /* put the first node in start. This will be nearest neighbor to q */

    //    result.emplace_back(pool[start]);

    while (result.size() < degree && (start) < pool.size() && start < maxc) {
      auto &p = pool[start];
      if (occlude_factor[start] > alpha) {
        start++;
        continue;
      }
      occlude_factor[start] = std::numeric_limits<float>::max();
      result.push_back(p);
      for (unsigned t = start + 1; t < pool.size() && t < maxc; t++) {
        if (occlude_factor[t] >= MAX_ALPHA)
          continue;
        float djk = _distance->compare(
            _data + _aligned_dim * (size_t) pool[t].id,
            _data + _aligned_dim * (size_t) p.id, (unsigned) _aligned_dim);
        occlude_factor[t] =
            (std::max)(occlude_factor[t], pool[t].distance / djk);
      }
      start++;
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

    float cur_alpha = 1;
    while (cur_alpha <= alpha && !pool.empty() && result.size() < range) {
      occlude_list(pool, location, cur_alpha, range, maxc, result,
                   occlude_factor);
      cur_alpha *= 1.25;
    }

    /* Add all the nodes in result into a variable called cut_graph
     * So this contains all the neighbors of id location
     */
    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      if (iter.id != location)
        pruned_list.emplace_back(iter.id);
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
          _final_graph[des].shrink_to_fit();
          _final_graph[des].reserve(range);
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
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::link(Parameters &parameters) {
    //    The graph will be updated periodically in NUM_SYNCS batches
    size_t   true_num_pts = _nd + _num_frozen_pts;
    uint32_t NUM_SYNCS = DIV_ROUND_UP(true_num_pts, (128 * 192));
    if (NUM_SYNCS < 40)
      NUM_SYNCS = 40;
    std::cout << "Number of syncs: " << NUM_SYNCS << std::endl;

    const unsigned NUM_RNDS = parameters.Get<unsigned>(
        "num_rnds");  // num. of passes of overall algorithm
    const unsigned L = parameters.Get<unsigned>("L");  // Search list size

    // Max degree of graph
    const unsigned range = parameters.Get<unsigned>("R");
    // Pruning parameter
    const float last_round_alpha = parameters.Get<float>("alpha");
    // Set alpha=1 for the first pass; use specified alpha for last pass
    parameters.Set<float>("alpha", 1);

    /* rand_perm is a vector that is initialized to the entire graph */
    std::vector<unsigned> rand_perm;
    for (size_t i = 0; i < _nd; i++) {
      rand_perm.emplace_back(i);
    }

    for (size_t i = 0; i < _num_frozen_pts; ++i)
      rand_perm.emplace_back(_max_points + i);

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    init_random_graph(range);

    if (_num_frozen_pts > 0)
      _ep = _max_points;
    else
      _ep = calculate_entry_point();

    for (uint32_t rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
      float  sync_time = 0, total_sync_time = 0;
      float  inter_time = 0, total_inter_time = 0;
      size_t inter_count = 0, total_inter_count = 0;
      // Shuffle the dataset
      std::random_shuffle(rand_perm.begin(), rand_perm.end());
      unsigned progress_counter = 0;

      size_t round_size = DIV_ROUND_UP(_nd, NUM_SYNCS);  // size of each batch
      std::vector<unsigned> need_to_sync(_max_points + _num_frozen_pts, 0);

      std::vector<std::vector<Neighbor>>    sync_pool_vector(round_size);
      std::vector<tsl::robin_set<unsigned>> sync_visited_vector(round_size);
      std::vector<std::vector<unsigned>>    pruned_list_vector(round_size);

      for (uint32_t sync_num = 0; sync_num < NUM_SYNCS; sync_num++) {
        if (rnd_no == NUM_RNDS - 1) {
          if (last_round_alpha > 1)
            parameters.Set<float>("alpha", last_round_alpha);
        }
        size_t start_id = sync_num * round_size;
        size_t end_id = (std::min)(true_num_pts, (sync_num + 1) * round_size);

        auto s = std::chrono::high_resolution_clock::now();
#pragma omp  parallel for schedule(dynamic, 64)
        for (_u64 node_ctr = (_u64) start_id; node_ctr < (_u64) end_id;
             ++node_ctr) {
          _u64                      node = rand_perm[node_ctr];
          size_t                    node_offset = node_ctr - start_id;
          std::vector<Neighbor> &   pool = sync_pool_vector[node_offset];
          std::vector<Neighbor>     tmp;
          tsl::robin_set<unsigned> &visited = sync_visited_vector[node_offset];
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          // get nearest neighbors of n in tmp. pool contains all the points
          // that were checked along with their distance from n. visited
          // contains all
          // the points visited, just the ids
          get_neighbors(node, L, tmp, pool, visited);
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
        std::chrono::duration<double> diff =
            std::chrono::high_resolution_clock::now() - s;
        sync_time += diff.count();

// prune_neighbors will check pool, and remove some of the points and
// create a cut_graph, which contains neighbors for point n
#pragma omp parallel for schedule(dynamic, 64)
        for (_u64 node_ctr = (_u64) start_id; node_ctr < (_u64) end_id;
             ++node_ctr) {
          _u64                   node = rand_perm[node_ctr];
          size_t                 node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          _final_graph[node].clear();
          _final_graph[node].shrink_to_fit();
          //						_final_graph[node].reserve(range);
          for (auto id : pruned_list)
            _final_graph[node].emplace_back(id);
        }
        s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 64)
        for (_u64 node_ctr = start_id; node_ctr < (_u64) end_id; ++node_ctr) {
          _u64                   node = rand_perm[node_ctr];
          _u64                   node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          tsl::robin_set<unsigned> &visited = sync_visited_vector[node_offset];
          batch_inter_insert(node, pruned_list, parameters, need_to_sync);
          //  inter_insert(node, pruned_list, parameters);
          pruned_list.clear();
          pruned_list.shrink_to_fit();
          visited.clear();
          tsl::robin_set<unsigned>().swap(visited);
        }

#pragma omp parallel for schedule(dynamic, 64)
        for (_u64 node_ctr = 0; node_ctr < rand_perm.size(); node_ctr++) {
          _u64 node = rand_perm[node_ctr];
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
            _final_graph[node].shrink_to_fit();
            _final_graph[node].reserve(range);
            for (auto id : new_out_neighbors)
              _final_graph[node].emplace_back(id);
          }
        }

        diff = std::chrono::high_resolution_clock::now() - s;
        inter_time += diff.count();

        if ((sync_num * 100) / NUM_SYNCS > progress_counter) {
          std::cout.precision(4);
          std::cout << "Completed  (round: " << rnd_no << ", sync: " << sync_num
                    << "/" << NUM_SYNCS << ")" << std::endl;
          total_sync_time += sync_time;
          total_inter_time += inter_time;
          total_inter_count += inter_count;
          sync_time = 0;
          inter_time = 0;
          inter_count = 0;
          progress_counter += 5;
        }
      }
      std::cout << "Completed Pass " << rnd_no
                << " of data using L=" << parameters.Get<unsigned>("L")
                << " and alpha=" << parameters.Get<float>("alpha")
                << ". Stats: ";
      std::cout << "sync_time=" << total_sync_time
                << "s, inter_time=" << total_inter_time
                << "s, inter_count=" << total_inter_count << std::endl;
    }

    std::cout << "Starting final cleanup.." << std::flush;
#pragma omp parallel for schedule(dynamic, 64)
    for (_u64 node_ctr = 0; node_ctr < rand_perm.size(); node_ctr++) {
      size_t node = rand_perm[node_ctr];
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
        _final_graph[node].shrink_to_fit();
        //						_final_graph[node].reserve(range);
        for (auto id : new_out_neighbors)
          _final_graph[node].emplace_back(id);
      }
    }
    std::cout << "done." << std::endl;
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
  std::pair<int, int> Index<T, TagT>::beam_search(
      const T *query, const size_t K, const unsigned L, unsigned *indices,
      int beam_width, std::vector<unsigned> start_points) {
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * L);

    if (start_points.size() == 0) {
      start_points.emplace_back(_ep);
    }

    /* ep_neighbors contains all the neighbors of navigating node, and
     * their distance from the query node
     */
    std::vector<Neighbor> ep_neighbors;
    for (auto cur_pt : start_points)
      for (auto id : _final_graph[cur_pt]) {
        if (id >= _nd) {
          std::cout << "ERROR" << id << "    Cur_pt " << cur_pt << std::endl;
          exit(-1);
        }
        // std::cout << "cmp: query <-> " << id << "\n";
        ep_neighbors.emplace_back(
            Neighbor(id,
                     _distance->compare(_data + _aligned_dim * (size_t) id,
                                        query, _aligned_dim),
                     true));
      }

    /* sort the ep_neighbors based on the distance from query node */
    std::sort(ep_neighbors.begin(), ep_neighbors.end());
    for (auto iter : ep_neighbors) {
      if (init_ids.size() >= L)
        break;
      init_ids.emplace_back(iter.id);  // Add the neighbors to init_ids
      visited.insert(iter.id);         // Add the neighbors to visited list
    }

    /* Add random nodes to fill in the L_SEARCH number of nodes
     * in visited list as well as in the init_ids list
     */
    while (init_ids.size() < L) {
      unsigned id = (rand()) % _nd;
      if (visited.find(id) == visited.end())
        visited.insert(id);
      else
        continue;
      init_ids.emplace_back(id);
    }
    std::vector<Neighbor> retset(L + 1);

    /* Find out the distances of all the neighbors of navigating node
     * with the query and add it to retset. Actually not needed for all the
     * neighbors. Only needed for the random ones added later
     */
    for (size_t i = 0; i < init_ids.size(); i++) {
      if (init_ids[i] >= _nd) {
        std::cout << init_ids[i] << std::endl;
        exit(-1);
      }
      retset[i] =
          Neighbor(init_ids[i],
                   _distance->compare(_data + _aligned_dim * init_ids[i], query,
                                      (unsigned) _aligned_dim),
                   true);
    }

    /* Sort the retset based on distance of nodes from query */
    std::sort(retset.begin(), retset.begin() + L);

    std::vector<unsigned> frontier;
    std::vector<unsigned> unique_nbrs;
    unique_nbrs.reserve(10 * L);

    int hops = 0;
    int cmps = 0;
    int k = 0;
    int deleted = 0;

    /* Maximum L rounds take place to get nearest neighbor.  */
    while (k < (int) (L + deleted)) {
      int nk = L;

      frontier.clear();
      unique_nbrs.clear();
      unsigned marker = k - 1;
      while (++marker < L && frontier.size() < (size_t) beam_width) {
        if (retset[marker].flag) {
          frontier.emplace_back(retset[marker].id);
          retset[marker].flag = false;
        }
      }

      if (!frontier.empty())
        hops++;
      for (auto n : frontier) {
        /* check neighbors of each node of frontier */
        for (unsigned m = 0; m < _final_graph[n].size(); ++m) {
          unsigned id = _final_graph[n][m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);  // Add each unique neighbor to visited
          }
          unique_nbrs.emplace_back(id);  // add each neighbor to unique_nbrs
        }
      }
      auto last_iter = std::unique(unique_nbrs.begin(), unique_nbrs.end());
      for (auto iter = unique_nbrs.begin(); iter != last_iter; iter++) {
        if (iter < (last_iter - 1)) {
          unsigned id_next = *(iter + 1);
          const T *vec1 = _data + _aligned_dim * id_next;
          diskann::prefetch_vector((const char *) vec1,
                                   _aligned_dim * sizeof(T));
        }

        cmps++;
        unsigned id = *iter;
        /* compare distance of each neighbor with that of query. If the
         * distance is less than largest distance in retset, add to retset and
         * set flag to true
         */
        // Harsha: Why do we require id < _nd
        if ((id >= _nd) && (id < _max_points)) {
          std::cout << id << std::endl;
          exit(-1);
        }
        float dist = _distance->compare(_data + _aligned_dim * id, query,
                                        (unsigned) _aligned_dim);
        if (dist >= retset[L - 1].distance)
          continue;
        Neighbor nn(id, dist, true);

        // Return position in sorted list where nn inserted.
        int r = InsertIntoPool(retset.data(), L, nn);

        if (_delete_set.size() != 0)
          if (_delete_set.find(id) != _delete_set.end())
            deleted++;
        if (r < nk)
          nk = r;  // nk logs the best position in the retset that was updated
                   // due to neighbors of n.
      }
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }
    assert(retset.size() >= L + deleted);
    for (size_t i = 0; i < K;) {
      int  deleted = 0;
      auto id = retset[i + deleted].id;
      if (_delete_set.size() > 0 && _delete_set.find(id) != _delete_set.end())
        deleted++;
      else if (id < _max_points)  // Remove frozen points
        indices[i++] = id;
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename TagT>
  std::pair<int, int> Index<T, TagT>::beam_search_tags(
      const T *query, const size_t K, const size_t L, TagT *tags,
      int beam_width, std::vector<unsigned> start_points,
      unsigned *indices_buffer) {
    const bool alloc = indices_buffer == NULL;
    auto       indices = alloc ? new unsigned[K] : indices_buffer;
    auto ret = beam_search(query, K, L, indices, beam_width, start_points);
    for (int i = 0; i < (int) K; ++i)
      tags[i] = _location_to_tag[indices[i]];
    if (alloc)
      delete[] indices;
    return ret;
  }

  /*************************************************
   *      Support for Incremental Update
   *************************************************/

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

    unsigned Lindex = parameters.Get<unsigned>("L");
    get_neighbors(id, Lindex, tmp, pool, visited);

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
    assert(active + _empty_slots.size() + _delete_set.size() ==
           _max_points + _num_frozen_pts);

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
      }
    }

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
            else
              std::cout << "Wrong new location for  " << _in_graph[old][i]
                        << " is " << new_location[_in_graph[old][i]]
                        << std::endl;
          }

        // Move the data and adj list to the correct position
        if (new_location[old] != old) {
          assert(new_location[old] < old);
          _final_graph[new_location[old]].swap(_final_graph[old]);
          /*	  for(unsigned x = 0; x < _final_graph[new_location[old]].size();
             x++){
                if(_final_graph[new_location[old]][x] == new_location[old]){
                  std::cout << "Self loop after swapping" <<std::endl;
                    break;
                    }
              }*/
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

    get_neighbors(location, Lindex, tmp, pool, visited);

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
    inter_insert(location, pruned_list, parameters, 1);

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

#ifdef _WINDOWS
  template DISKANN_DLLEXPORT Index<uint8_t, int>::Index(
      Metric m, const char *filename, const size_t max_points,

      const size_t nd, const size_t num_frozen_pts, const bool enable_tags,
      const bool store_data, const bool support_eager_delete);
  template DISKANN_DLLEXPORT Index<int8_t, int>::Index(
      Metric m, const char *filename, const size_t max_points,

      const size_t nd, const size_t num_frozen_pts, const bool enable_tags,
      const bool store_data, const bool support_eager_delete);
  template DISKANN_DLLEXPORT Index<float, int>::Index(
      Metric m, const char *filename, const size_t max_points,

      const size_t nd, const size_t num_frozen_pts, const bool enable_tags,
      const bool store_data, const bool support_eager_delete);

  template DISKANN_DLLEXPORT Index<uint8_t, int>::~Index();
  template DISKANN_DLLEXPORT Index<int8_t, int>::~Index();
  template DISKANN_DLLEXPORT Index<float, int>::~Index();

  template DISKANN_DLLEXPORT void Index<uint8_t, int>::save(
      const char *filename);
  template DISKANN_DLLEXPORT void Index<int8_t, int>::save(
      const char *filename);
  template DISKANN_DLLEXPORT void Index<float, int>::save(const char *filename);

  template DISKANN_DLLEXPORT void Index<uint8_t, int>::load(
      const char *filename, const bool load_tags);
  template DISKANN_DLLEXPORT void Index<int8_t, int>::load(
      const char *filename, const bool load_tags);
  template DISKANN_DLLEXPORT void Index<float, int>::load(const char *filename,
                                                          const bool load_tags);

  template DISKANN_DLLEXPORT void Index<uint8_t, int>::build(
      Parameters &parameters, const std::vector<int> &tags);
  template DISKANN_DLLEXPORT void Index<int8_t, int>::build(
      Parameters &parameters, const std::vector<int> &tags);
  template DISKANN_DLLEXPORT void Index<float, int>::build(
      Parameters &parameters, const std::vector<int> &tags);

  template DISKANN_DLLEXPORT std::pair<int, int> Index<uint8_t>::beam_search(
      const uint8_t *query, const size_t K, const unsigned L, unsigned *indices,
      int beam_width, std::vector<unsigned> start_points);
  template DISKANN_DLLEXPORT std::pair<int, int> Index<int8_t>::beam_search(
      const int8_t *query, const size_t K, const unsigned L, unsigned *indices,
      int beam_width, std::vector<unsigned> start_points);
  template DISKANN_DLLEXPORT std::pair<int, int> Index<float>::beam_search(
      const float *query, const size_t K, const unsigned L, unsigned *indices,
      int beam_width, std::vector<unsigned> start_points);

  template DISKANN_DLLEXPORT int Index<int8_t, int>::delete_point(
      const int tag);
  template DISKANN_DLLEXPORT int Index<uint8_t, int>::delete_point(
      const int tag);
  template DISKANN_DLLEXPORT int Index<float, int>::delete_point(const int tag);
  template DISKANN_DLLEXPORT int Index<int8_t, size_t>::delete_point(
      const size_t tag);
  template DISKANN_DLLEXPORT int Index<uint8_t, size_t>::delete_point(
      const size_t tag);
  template DISKANN_DLLEXPORT int Index<float, size_t>::delete_point(
      const size_t tag);
  template DISKANN_DLLEXPORT int Index<int8_t, std::string>::delete_point(
      const std::string tag);
  template DISKANN_DLLEXPORT int Index<uint8_t, std::string>::delete_point(
      const std::string tag);
  template DISKANN_DLLEXPORT int Index<float, std::string>::delete_point(
      const std::string tag);

  template DISKANN_DLLEXPORT int Index<int8_t, int>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template DISKANN_DLLEXPORT int Index<uint8_t, int>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template DISKANN_DLLEXPORT int Index<float, int>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template DISKANN_DLLEXPORT int Index<int8_t, size_t>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template DISKANN_DLLEXPORT int Index<uint8_t, size_t>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template DISKANN_DLLEXPORT int Index<float, size_t>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template DISKANN_DLLEXPORT int Index<int8_t, std::string>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template DISKANN_DLLEXPORT int Index<uint8_t, std::string>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template DISKANN_DLLEXPORT int Index<float, std::string>::disable_delete(
      const Parameters &parameters, const bool consolidate);

  template DISKANN_DLLEXPORT int Index<int8_t, int>::eager_delete(
      const int tag, const Parameters &parameters);
  template DISKANN_DLLEXPORT int Index<int8_t, size_t>::eager_delete(
      const size_t tag, const Parameters &parameters);
  template DISKANN_DLLEXPORT int Index<int8_t, std::string>::eager_delete(
      const std::string tag, const Parameters &parameters);
  template DISKANN_DLLEXPORT int Index<uint8_t, int>::eager_delete(
      const int tag, const Parameters &parameters);
  template DISKANN_DLLEXPORT int Index<uint8_t, size_t>::eager_delete(
      const size_t tag, const Parameters &parameters);
  template DISKANN_DLLEXPORT int Index<uint8_t, std::string>::eager_delete(
      const std::string tag, const Parameters &parameters);
  template DISKANN_DLLEXPORT int Index<float, int>::eager_delete(
      const int tag, const Parameters &parameters);
  template DISKANN_DLLEXPORT int Index<float, size_t>::eager_delete(
      const size_t tag, const Parameters &parameters);
  template DISKANN_DLLEXPORT int Index<float, std::string>::eager_delete(
      const std::string tag, const Parameters &parameters);

  template DISKANN_DLLEXPORT int Index<int8_t, int>::enable_delete();
  template DISKANN_DLLEXPORT int Index<uint8_t, int>::enable_delete();
  template DISKANN_DLLEXPORT int Index<float, int>::enable_delete();
  template DISKANN_DLLEXPORT int Index<int8_t, size_t>::enable_delete();
  template DISKANN_DLLEXPORT int Index<uint8_t, size_t>::enable_delete();
  template DISKANN_DLLEXPORT int Index<float, size_t>::enable_delete();
  template DISKANN_DLLEXPORT int Index<int8_t, std::string>::enable_delete();
  template DISKANN_DLLEXPORT int Index<uint8_t, std::string>::enable_delete();
  template DISKANN_DLLEXPORT int Index<float, std::string>::enable_delete();

  template DISKANN_DLLEXPORT void Index<int8_t, int>::readjust_data(
      unsigned _num_frozen_pts);
  template DISKANN_DLLEXPORT void Index<uint8_t, int>::readjust_data(
      unsigned _num_frozen_pts);
  template DISKANN_DLLEXPORT void Index<float, int>::readjust_data(
      unsigned _num_frozen_pts);
  template DISKANN_DLLEXPORT void Index<int8_t, size_t>::readjust_data(
      unsigned _num_frozen_pts);
  template DISKANN_DLLEXPORT void Index<uint8_t, size_t>::readjust_data(
      unsigned _num_frozen_pts);
  template DISKANN_DLLEXPORT void Index<float, size_t>::readjust_data(
      unsigned _num_frozen_pts);
  template DISKANN_DLLEXPORT void Index<int8_t, std::string>::readjust_data(
      unsigned _num_frozen_pts);
  template DISKANN_DLLEXPORT void Index<uint8_t, std::string>::readjust_data(
      unsigned _num_frozen_pts);
  template DISKANN_DLLEXPORT void Index<float, std::string>::readjust_data(
      unsigned _num_frozen_pts);

  template DISKANN_DLLEXPORT int Index<int8_t, int>::insert_point(
      const int8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const int tag);
  template DISKANN_DLLEXPORT int Index<uint8_t, int>::insert_point(
      const uint8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const int tag);
  template DISKANN_DLLEXPORT int Index<float, int>::insert_point(
      const float *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const int tag);
  template DISKANN_DLLEXPORT int Index<int8_t, size_t>::insert_point(
      const int8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const size_t tag);
  template DISKANN_DLLEXPORT int Index<uint8_t, size_t>::insert_point(
      const uint8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const size_t tag);
  template DISKANN_DLLEXPORT int Index<float, size_t>::insert_point(
      const float *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const size_t tag);
  template DISKANN_DLLEXPORT int Index<int8_t, std::string>::insert_point(
      const int8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph,
      const std::string tag);
  template DISKANN_DLLEXPORT int Index<uint8_t, std::string>::insert_point(
      const uint8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph,
      const std::string tag);
  template DISKANN_DLLEXPORT int Index<float, std::string>::insert_point(
      const float *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph,
      const std::string tag);

#endif
}  // namespace diskann
