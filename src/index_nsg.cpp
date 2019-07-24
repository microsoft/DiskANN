#include "index_nsg.h"
#include <math_utils.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include "exceptions.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "util.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include <cassert>
#include "MemoryMapper.h"
#include "partition_and_pq.h"
#ifdef __NSG_WINDOWS__
#include <xmmintrin.h>
#endif

namespace {
  template<typename T>
  NSG::Distance<T> *get_distance_function();

  template<>
  NSG::Distance<float> *get_distance_function() {
    return new NSG::DistanceL2();
  }

  template<>
  NSG::Distance<int8_t> *get_distance_function() {
    return new NSG::DistanceL2Int8();
  }

  template<>
  NSG::Distance<uint8_t> *get_distance_function() {
    return new NSG::DistanceL2UInt8();
  }
}

namespace NSG {
#define _CONTROL_NUM 100
#define MAX_START_POINTS 100

  template<typename T, typename TagT>
  IndexNSG<T, TagT>::IndexNSG(const size_t dimension, const size_t n, Metric m,
                              const size_t max_points, const bool enable_tags,
                              const bool store_data)
      : _dim(dimension), _nd(n), _has_built(false), _width(0),
        _can_delete(false), _enable_tags(enable_tags),
        _consolidated_order(true), _store_data(store_data) {
    _max_points = (max_points > 0) ? max_points : n;
    if (_max_points < n) {
      std::cerr << "max_points must be >= n; max_points: " << _max_points
                << "  n: " << _nd << std::endl;
      exit(-1);
    }
    this->_distance = ::get_distance_function<T>();
    _locks = std::vector<std::mutex>(_max_points);
    _width = 0;
  }

  template<>
  IndexNSG<float>::~IndexNSG() {
    delete this->_distance;
  }

  template<>
  IndexNSG<_s8>::~IndexNSG() {
    delete this->_distance;
  }

  template<>
  IndexNSG<_u8>::~IndexNSG() {
    delete this->_distance;
  }

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::compute_in_degree_stats() {
    std::vector<size_t> in_degrees;

    size_t out_sum = 0;
    for (unsigned i = 0; i < _max_points; ++i) {
      if (_delete_set.find(i) == _delete_set.end() &&
          _empty_slots.find(i) == _empty_slots.end()) {
        out_sum += _final_graph[i].size();
        for (auto ngh : _final_graph[i])
          in_degrees[ngh]++;
      }
    }

    size_t max = 0, min = SIZE_MAX, sum = 0;
    for (const auto &deg : in_degrees) {
      max = (std::max)(max, deg);
      min = (std::min)(min, deg);
      sum += deg;
    }
    std::cout << "Max in-degree: " << max << "   Min in-degree: " << min
              << "   Avg. in-degree: " << (float) (sum) / (float) (_nd)
              << "   Avg. out-degree: " << (float) (out_sum) / (float) (_nd)
              << std::endl;
  }

  template<typename T, typename TagT>
  int IndexNSG<T, TagT>::enable_delete() {
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

    if (_consolidated_order) {
      assert(_empty_slots.size() == 0);
      for (unsigned slot = _nd; slot < _max_points; ++slot)
        _empty_slots.insert(slot);
      _consolidated_order = false;
    }
    _can_delete = true;
    return 0;
  }

  // Do not call consolidate_deletes() if you have not locked _change_lock.
  // Returns number of live points left after consolidation
  template<typename T, typename TagT>
  size_t IndexNSG<T, TagT>::consolidate_deletes(const Parameters &parameters) {
    assert(!_consolidated_order);
    assert(_can_delete);
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    std::vector<unsigned> new_location;
    new_location.resize(_max_points, _max_points);
    unsigned active = 0;
    for (unsigned old = 0; old < _max_points; ++old)
      if (_empty_slots.find(old) == _empty_slots.end() &&
          _delete_set.find(old) == _delete_set.end())
        new_location[old] = active++;
    assert(active + _empty_slots.size() + _delete_set.size() == _max_points);

    tsl::robin_set<unsigned> candidate_set;
    std::vector<Neighbor>    expanded_nghrs;
    std::vector<Neighbor>    result;

    for (unsigned i = 0; i < _max_points; ++i) {
      if (new_location[i] < _max_points) {
        candidate_set.clear();
        expanded_nghrs.clear();
        result.clear();

        bool modify = false;
        for (auto ngh : _final_graph[i]) {
          if (new_location[ngh] >= _max_points) {
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
            expanded_nghrs.push_back(Neighbor(
                j,
                _distance->compare(_data + _dim * (size_t) i,
                                   _data + _dim * (size_t) j, (unsigned) _dim),
                true));
          std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
          occlude_list(expanded_nghrs, i, alpha, range, maxc, result);

          _final_graph[i].clear();
          for (auto j : result)
            _final_graph[i].push_back(j.id);
        }
      }
    }

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
        std::cerr << "ERROR: Did not find a replacement for start node."
                  << std::endl;
        exit(-1);
      } else {
        assert(_delete_set.find(_ep) == _delete_set.end());
        std::cout << "New start node is " << _ep << std::endl;
      }
    }

    _nd -= _delete_set.size();

    std::cout << "Re-numbering nodes and edges and consolidating data... "
              << std::flush;
    for (unsigned old = 0; old < _max_points; ++old) {
      if (new_location[old] < _max_points) {  // If point continues to exist

        // Renumber nodes to compact the order
        for (size_t i = 0; i < _final_graph[old].size(); ++i) {
          assert(new_location[_final_graph[old][i]] <= _final_graph[old][i]);
          _final_graph[old][i] = new_location[_final_graph[old][i]];
        }

        // Move the data and adj list to the correct position
        if (new_location[old] != old) {
          assert(new_location[old] < old);
          _final_graph[new_location[old]].swap(_final_graph[old]);
          memcpy((void *) (_data + _dim * (size_t) new_location[old]),
                 (void *) (_data + _dim * (size_t) old), _dim * sizeof(T));
        }
      }
    }
    std::cout << "done." << std::endl;
    _store_data = true;

    std::cout << "Updating mapping between tags and ids... " << std::flush;
    // Update the location pointed to by tag
    _tag_to_location.clear();
    for (auto iter : _location_to_tag)
      _tag_to_location[iter.second] = new_location[iter.first];
    _location_to_tag.clear();
    for (auto iter : _tag_to_location)
      _location_to_tag[iter.second] = iter.first;
    std::cout << "done." << std::endl;

    for (unsigned old = active; old < _max_points; ++old)
      _final_graph[old].clear();

    _empty_slots.clear();
    _delete_set.clear();
    _consolidated_order = true;
    std::cout << "Consolidated the index" << std::endl;

    return _nd;
  }

  template<typename T, typename TagT>
  int IndexNSG<T, TagT>::disable_delete(const Parameters &parameters,
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
    if (_tag_to_location.size() + _delete_set.size() != _nd) {
      std::cerr << "Tags to points array wrong sized" << std::endl;
      return -2;
    }
    if (_location_to_tag.size() + _delete_set.size() != _nd) {
      std::cerr << "Points to tags array wrong sized" << std::endl;
      return -3;
    }
    if (consolidate) {
      auto nd = consolidate_deletes(parameters);
      std::cout << "#Points after consolidation: " << nd << std::endl;
    }

    _can_delete = false;
    return 0;
  }

  template<typename T, typename TagT>
  int IndexNSG<T, TagT>::delete_point(const TagT tag) {
    LockGuard guard(_change_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      std::cerr << "Delete tag not found" << std::endl;
      return -1;
    }
    _delete_set.insert(_tag_to_location[tag]);
    _location_to_tag.erase(_tag_to_location[tag]);
    _tag_to_location.erase(tag);
    return 0;
  }

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::save(const char *filename) {
    long long     total_gr_edges = 0;
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    assert(_final_graph.size() == _max_points);
    if (_enable_tags) {
      _change_lock.lock();
      if (_can_delete || !_consolidated_order) {
        std::cerr << "Disable deletes and consolidate index before saving."
                  << std::endl;
        exit(-1);
      }
    }

    out.write((char *) &_width, sizeof(unsigned));
    out.write((char *) &_ep, sizeof(unsigned));
    for (unsigned i = 0; i < _nd; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _final_graph[i].data(), GK * sizeof(unsigned));
      total_gr_edges += GK;
    }
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
      out_data.write((char *) &_nd, sizeof(_s32));
      out_data.write((char *) &_dim, sizeof(_s32));
      out_data.write((char *) _data, _nd * _dim * sizeof(T));
      out_data.close();
    }

    std::cout << "Avg degree: " << ((float) total_gr_edges) / ((float) _nd)
              << std::endl;
  }

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::load(const char *filename, const bool load_tags) {
    std::ifstream in(filename, std::ios::binary);
    in.read((char *) &_width, sizeof(unsigned));
    in.read((char *) &_ep, sizeof(unsigned));
    std::cout << "NSG -- width: " << _width << ", _ep: " << _ep << "\n";

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

      if (nodes % 5000000 == 0)
        std::cout << "Loaded " << nodes << " nodes, and " << cc << " neighbors"
                  << std::endl;
    }
    std::cout << "Loaded " << nodes << " nodes, and " << cc << " neighbors"
              << std::endl;
  }

  /* init_random_graph():
   * num_points: Number of points in the dataset
   * k: max degree of the graph
   * mapping: initial vector of a sample of the points in the dataset
   */
  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::init_random_graph(size_t num_points, unsigned k,
                                            std::vector<size_t> mapping) {
    k = (std::min)(k, (unsigned) 32);
    _final_graph.resize(_max_points);
    _final_graph.reserve(_max_points);
    if (!mapping.empty())
      num_points = mapping.size();
    else {
      mapping.resize(num_points);
      std::iota(std::begin(mapping), std::end(mapping), 0);
    }

    std::cout << "Generating random graph with " << num_points << " points... "
              << std::flush;
    // PAR_BLOCK_SZ gives the number of points that can fit in a single block
    size_t PAR_BLOCK_SZ = (1 << 16);  // = 64KB
    size_t nblocks = DIV_ROUND_UP(num_points, PAR_BLOCK_SZ);

#pragma omp parallel for schedule(static, 1)
    for (int64_t block = 0; block < nblocks;
         ++block) {  // GOPAL Changed from "size_t block" to "int block"
      std::random_device                    rd;
      std::mt19937                          gen(rd());
      std::uniform_int_distribution<size_t> dis(0, num_points - 1);

      /* Put random number points as neighbours to the 10% of the nodes */
      for (size_t i = block * PAR_BLOCK_SZ;
           i < (block + 1) * PAR_BLOCK_SZ && i < num_points; i++) {
        std::set<unsigned> ra_ndset;
        while (ra_ndset.size() < k && ra_ndset.size() < num_points)
          ra_ndset.insert(dis(gen));

        _final_graph[mapping[i]].reserve(k);
        for (auto s : ra_ndset)
          _final_graph[mapping[i]].emplace_back(mapping[s]);
        _final_graph[mapping[i]].shrink_to_fit();
      }
    }
    _ep = calculate_entry_point();
    std::cout << "done. Entry point set to " << _ep << "." << std::endl;
  }

  /* iterate_to_fixed_point():
   * query : point whose neighbors to be found.
   * init_ids : ids of neighbors of navigating node.
   * retset : will contain the nearest neighbors of the query.
   * fullset : will contain all the node ids and distances from query that are
   * checked.
   * visited : will contain all the nodes that are visited during search.
   */
  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::iterate_to_fixed_point(
      const T *query, const Parameters &parameter,
      std::vector<unsigned> &init_ids, std::vector<Neighbor> &retset,
      std::vector<Neighbor> &fullset, tsl::robin_set<unsigned> &visited) {
    const unsigned L = parameter.Get<unsigned>("L");

    /* put random L new ids into visited list and init_ids list */
    while (init_ids.size() < L && init_ids.size() < _nd - 1) {
      unsigned id = (rand() * rand() * rand()) % _nd;
      if (visited.find(id) != visited.end())
        continue;
      else
        visited.insert(id);
      init_ids.emplace_back(id);
    }

    /* compare distance of all points in init_ids with query, and put the id
     * with distance
     * in retset
     */
    unsigned l = 0;
    for (auto id : init_ids) {
      assert(id < _max_points);
      retset[l++] = Neighbor(
          id, _distance->compare(_data + _dim * (size_t) id, query, _dim),
          true);
      fullset.emplace_back(retset[l - 1]);
    }

    /* sort retset based on distance of each point to query */
    std::sort(retset.begin(), retset.begin() + l);
    int k = 0;
    while (k < (int) l) {
      int nk = l;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        // prefetch _final_graph[n]
        unsigned *nbrs = _final_graph[n].data();   // nbrs: data of neighbors
        unsigned  nnbrs = _final_graph[n].size();  // nnbrs: number of neighbors
        NSG::prefetch_vector((const char *) nbrs, nnbrs * sizeof(unsigned));
        for (size_t m = 0; m < nnbrs; m++) {
          unsigned id = nbrs[m];  // id = neighbor
          if (m < (nnbrs - 1)) {
            // id_next = next neighbor
            unsigned id_next = nbrs[m + 1];
            // vec_next1: data of next neighbor
            const T *vec_next1 = _data + (size_t) id_next * _dim;
            NSG::prefetch_vector((const char *) vec_next1, _dim * sizeof(T));
          }

          if (visited.find(id) == visited.end())
            visited.insert(id);  // if id is not in visited, add it to visited
          else
            continue;

          // compare distance of id with query
          float dist = _distance->compare(query, _data + _dim * (size_t) id,
                                          (unsigned) _dim);
          Neighbor nn(id, dist, true);
          fullset.emplace_back(nn);
          if (dist >= retset[l - 1].distance)
            continue;

          // if distance is smaller than largest, add to retset, keep it
          // sorted
          int r = InsertIntoPool(retset.data(), l, nn);

          if (l + 1 < retset.size())
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
    assert(!fullset.empty());
  }

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::get_neighbors(const T *query,
                                        const Parameters &     parameter,
                                        std::vector<Neighbor> &retset,
                                        std::vector<Neighbor> &fullset) {
    const unsigned           L = parameter.Get<unsigned>("L");
    tsl::robin_set<unsigned> visited(10 * L);
    get_neighbors(query, parameter, retset, fullset, visited);
  }

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::get_neighbors(const T *query,
                                        const Parameters &        parameter,
                                        std::vector<Neighbor> &   retset,
                                        std::vector<Neighbor> &   fullset,
                                        tsl::robin_set<unsigned> &visited) {
    const unsigned L = parameter.Get<unsigned>("L");

    retset.resize(L + 1);
    std::vector<unsigned> init_ids;
    init_ids.reserve(L);

    std::vector<Neighbor> ep_neighbors;
    for (auto id : _final_graph[_ep]) {
      ep_neighbors.emplace_back(Neighbor(
          id, _distance->compare(_data + _dim * (size_t) id, query, _dim),
          true));
    }

    std::sort(ep_neighbors.begin(), ep_neighbors.end());
    for (auto iter : ep_neighbors) {
      if (init_ids.size() >= L)
        break;
      init_ids.emplace_back(iter.id);
      visited.insert(iter.id);
    }

    /* Before calling this function: ep_neighbors contains the list of
     * all the neighbors of navigating node along with distance from query,
     * init_ids contains all the ids of the neighbors of _ep, visited also
     * contains all the ids of neighbors of _ep
     */
    iterate_to_fixed_point(query, parameter, init_ids, retset, fullset,
                           visited);
  }

  template<typename T, typename TagT>
  int IndexNSG<T, TagT>::insert_point(const T *point,
                                      const Parameters &        parameters,
                                      std::vector<Neighbor> &   pool,
                                      std::vector<Neighbor> &   tmp,
                                      tsl::robin_set<unsigned> &visited,
                                      vecNgh &cut_graph, const TagT tag) {
    unsigned range = parameters.Get<unsigned>("R");
    assert(_has_built);
    if (_enable_tags)
      assert(tag != NULL_TAG);

    LockGuard guard(_change_lock);

    if (_enable_tags &&
        (_tag_to_location.find(tag) != _tag_to_location.end())) {
      std::cerr << "Entry with the tag " << tag << " exists already"
                << std::endl;
      return -1;
    }
    assert(_nd <= _max_points);
    if (_nd == _max_points) {
      std::cerr << "Can not insert, reached maximum(" << _max_points
                << ") points." << std::endl;
      return -2;
    }

    size_t location = reserve_location();
    _tag_to_location[tag] = location;
    _location_to_tag[location] = tag;

    auto offset_data = _data + (size_t) _dim * location;
    memcpy((void *) offset_data, point, sizeof(T) * _dim);
    _store_data = true;

    pool.clear();
    tmp.clear();
    cut_graph.clear();
    visited.clear();

    get_neighbors(offset_data, parameters, tmp, pool, visited);
    sync_prune(_data + (size_t) _dim * location, location, pool, parameters,
               visited, cut_graph);

    assert(_final_graph.size() == _max_points);
    _final_graph[location].clear();
    _final_graph[location].reserve(range);
    assert(!cut_graph.empty());
    for (auto link : cut_graph)
      _final_graph[location].emplace_back(link.id);
    assert(_final_graph[location].size() <= range);
    inter_insert(location, cut_graph, parameters);

    return 0;
  }

  // Do not call reserve_location() if you have not locked _change_lock.
  // It is not thread safe.
  template<typename T, typename TagT>
  unsigned IndexNSG<T, TagT>::reserve_location() {
    assert(_nd < _max_points);

    unsigned location;
    if (_consolidated_order)
      location = _nd;
    else {
      assert(_empty_slots.size() != 0);
      assert(_empty_slots.size() + _nd == _max_points);

      auto iter = _empty_slots.begin();
      location = *iter;
      _empty_slots.erase(iter);
    }

    ++_nd;
    return location;
  }

  /* reachable_bfs():
   * This function fills in the order to do bfs in bfs_order
   */
  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::reachable_bfs(
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

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::populate_start_points_ep(
      std::vector<unsigned> &start_points) {
    assert(start_points.size() == 0);
    start_points.push_back(_ep);
  }

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::populate_start_points_bfs(
      std::vector<unsigned> &start_points) {
    // populate a visited array
    // WARNING: DO NOT MAKE THIS A VECTOR
    bool *visited = new bool[_nd]();
    std::fill(visited, visited + _nd, false);
    std::map<unsigned, std::vector<tsl::robin_set<unsigned>>> bfs_orders;
    unsigned start_node = _ep;
    bool     complete = false;
    bfs_orders.insert(
        std::make_pair(start_node, std::vector<tsl::robin_set<unsigned>>()));
    auto &bfs_order = bfs_orders[start_node];
    reachable_bfs(start_node, bfs_order, visited);

    start_node = 0;
    std::map<unsigned, std::vector<tsl::robin_set<unsigned>>> other_bfs_orders;
    while (!complete) {
      other_bfs_orders.insert(
          std::make_pair(start_node, std::vector<tsl::robin_set<unsigned>>()));
      auto &other_bfs_order = bfs_orders[start_node];
      reachable_bfs(start_node, other_bfs_order, visited);

      complete = true;
      for (unsigned idx = start_node; idx < _nd; idx++) {
        if (!visited[idx]) {
          complete = false;
          start_node = idx;
          break;
        }
      }
    }
    start_points.emplace_back(_ep);
    // process each component, add one node from each level if more than one
    // level, else ignore
    for (auto &k_v : bfs_orders) {
      if (k_v.second.size() > 1) {
        std::cout << "Using start points in component with nav-node: "
                  << k_v.first << std::endl;
        // add nodes from each level to `start_points`
        for (auto &lvl : k_v.second) {
          for (size_t i = 0; i < 10 && i < lvl.size() &&
                             start_points.size() < MAX_START_POINTS;
               ++i) {
            auto   iter = lvl.begin();
            size_t ra_ndoffset = rand() * rand() * rand() % lvl.size();
            for (size_t j = 0; j < ra_ndoffset; ++j)
              iter++;

            if (std::find(start_points.begin(), start_points.end(), *iter) ==
                start_points.end())
              start_points.emplace_back(*iter);
          }

          if (start_points.size() == MAX_START_POINTS)
            break;
        }
      }
    }
    std::cout << "Chose " << start_points.size() << " starting points"
              << std::endl;

    delete[] visited;
  }

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T, typename TagT>
  unsigned IndexNSG<T, TagT>::calculate_entry_point() {
    // allocate and init centroid
    float *center = new float[_dim]();
    for (size_t j = 0; j < _dim; j++)
      center[j] = 0;

    for (size_t i = 0; i < _nd; i++)
      for (size_t j = 0; j < _dim; j++)
        center[j] += _data[i * _dim + j];

    for (size_t j = 0; j < _dim; j++)
      center[j] /= _nd;

    // compute all to one distance
    float * distances = new float[_nd]();
#pragma omp parallel for schedule(static, 65536)
    for (int i = 0; i < _nd; i++) {  // GOPAL Changed from "size_t i" to "int i"
      // extract point and distance reference
      float &  dist = distances[i];
      const T *cur_vec = _data + (i * (size_t) _dim);
      dist = 0;
      float diff = 0;
      for (size_t j = 0; j < _dim; j++) {
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

  /* This function tries to add as many diverse edges as possible from current
   * node n to all the visited nodes obtained by running get_neighbors.
   * Assumes that pool is sorted in order of increasing distance.
   */
  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::occlude_list(const std::vector<Neighbor> &pool,
                                       const unsigned location,
                                       const float alpha, const unsigned degree,
                                       const unsigned         maxc,
                                       std::vector<Neighbor> &result) {
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    unsigned start = (pool[0].id == location) ? 1 : 0;
    /* put the first node in start. This will be nearest neighbor to q */
    result.emplace_back(pool[start]);

    while (result.size() < degree && (++start) < pool.size() && start < maxc) {
      auto &p = pool[start];
      bool  occlude = false;
      for (unsigned t = 0; t < result.size(); t++) {
        if (p.id == result[t].id) {
          occlude = true;
          break;
        }
        float djk =
            _distance->compare(_data + _dim * (size_t) result[t].id,
                               _data + _dim * (size_t) p.id, (unsigned) _dim);
        if (alpha * djk < p.distance /* dik */) {
          occlude = true;
          break;
        }
      }
      if (!occlude)
        result.emplace_back(p);
    }
  }

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::sync_prune(const T *x, const unsigned location,
                                     std::vector<Neighbor> &   pool,
                                     const Parameters &        parameter,
                                     tsl::robin_set<unsigned> &visited,
                                     vecNgh &                  cut_graph) {
    unsigned range = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    float    alpha = parameter.Get<float>("alpha");

    _width = (std::max)(_width, range);

    /* check the neighbors of the query that are not part of visited,
     * check their distance to the query, and add it to pool.
     */
    if (!_final_graph[location].empty())
      for (auto id : _final_graph[location]) {
        if (visited.find(id) != visited.end())
          continue;
        float dist =
            _distance->compare(x, _data + _dim * (size_t) id, (unsigned) _dim);
        pool.emplace_back(Neighbor(id, dist, true));
      }

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    occlude_list(pool, location, 1.0, range, maxc, result);

    /* create a new array result2, which contains the the points according
     * to the parameter alpha, which talks about how aggressively to keep
     * nodes during pruning
     */
    if (alpha > 1.0 && !pool.empty() && result.size() < range) {
      std::vector<Neighbor> result2;
      occlude_list(pool, location, 1.2, range - result.size(), maxc, result2);

      // add everything from result2 to result. This will lead to duplicates
      for (unsigned i = 0; i < result2.size(); i++) {
        result.emplace_back(result2[i]);
      }
      // convert it into a set, so that duplicates are all removed.
      std::set<Neighbor> s(result.begin(), result.end());
      result.assign(s.begin(), s.end());
    }

    /* Add all the nodes in result into a variable called cut_graph
     * So this contains all the neighbors of id location
     */
    cut_graph.clear();
    assert(result.size() <= range);
    for (auto iter : result)
      cut_graph.emplace_back(SimpleNeighbor(iter.id, iter.distance));
  }

  /* inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::inter_insert(unsigned n, vecNgh &cut_graph_n,
                                       const Parameters &parameter) {
    float      alpha = parameter.Get<float>("alpha");
    const auto range = parameter.Get<float>("R");
    const auto src_pool = cut_graph_n;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      assert(des.id >= 0 && des.id < _max_points);

      int dup = 0;
      /* des_pool contains the neighbors of the neighbors of n */
      auto &des_pool = _final_graph[des.id];

      std::vector<unsigned> graph_copy;
      {
        LockGuard guard(_locks[des.id]);
        for (auto nn : des_pool) {
          assert(nn >= 0 && nn < _max_points);

          if (n == nn) {
            dup = 1;
            break;
          }
        }
        if (dup)
          continue;

        if (des_pool.size() < range) {
          des_pool.emplace_back(n);
          continue;
        }

        assert(des_pool.size() == range);
        graph_copy = des_pool;
        graph_copy.emplace_back(n);
        // at this point, graph_copy contains n and neighbors of neighbor of n
      }  // des lock is released by this point

      assert(graph_copy.size() == 1 + range);
      {
        vecNgh temp_pool;
        for (auto node : graph_copy)
          /* temp_pool contains distance of each node in graph_copy, from
           * neighbor of n */
          temp_pool.emplace_back(
              SimpleNeighbor(node,
                             _distance->compare(_data + _dim * (size_t) node,
                                                _data + _dim * (size_t) des.id,
                                                (unsigned) _dim)));
        /* sort temp_pool according to distance from neighbor of n */
        std::sort(temp_pool.begin(), temp_pool.end());
        for (auto iter = temp_pool.begin(); iter + 1 != temp_pool.end(); ++iter)
          assert(iter->id != (iter + 1)->id);

        std::vector<SimpleNeighbor> result;
        result.emplace_back(temp_pool[0]);

        auto iter = temp_pool.begin();
        if (alpha > 1)
          iter = temp_pool.erase(iter);
        while (result.size() < range && iter != temp_pool.end()) {
          auto &p = *iter;
          bool  occlude = false;

          for (auto r : result) {
            if (p.id == r.id) {
              occlude = true;
              break;
            }
            float djk = _distance->compare(_data + _dim * (size_t) r.id,
                                           _data + _dim * (size_t) p.id,
                                           (unsigned) _dim);
            if (djk < p.distance /* dik */) {
              occlude = true;
              break;
            }
          }

          if (!occlude)
            result.emplace_back(p);

          if (!occlude && alpha > 1)
            iter = temp_pool.erase(iter);
          else
            ++iter;
        }

        if (alpha > 1 && result.size() < range && !temp_pool.empty()) {
          std::vector<SimpleNeighbor> result2;
          result2.emplace_back(temp_pool[0]);

          auto iter = temp_pool.begin();
          while (result2.size() + result.size() < range &&
                 ++iter != temp_pool.end()) {
            auto &p = *iter;
            bool  occlude = false;
            for (auto r : result) {
              if (p.id == r.id) {
                occlude = true;
                break;
              }
              float djk = _distance->compare(_data + _dim * (size_t) r.id,
                                             _data + _dim * (size_t) p.id,
                                             (unsigned) _dim);
              if (alpha * djk < p.distance /* dik */) {
                occlude = true;
                break;
              }
            }
            if (!occlude)
              result2.emplace_back(p);
          }
          for (auto r2 : result2)
            result.emplace_back(r2);
        }

        {
          LockGuard guard(_locks[des.id]);
          assert(result.size() <= range);
          des_pool.clear();
          for (auto iter : result)
            des_pool.emplace_back(iter.id);
        }
      }

      /* At the end of this, des_pool contains all the correct neighbors of
       * the neighbors of the query node
       */
      assert(des_pool.size() <= range);
      //      for (auto iter : des_pool)
      //        assert(iter <= _nd);  // Equality occurs when called from
      //        insert_point
    }
  }

  /* LinkHierarchy():
   * The graph creation function.
   */
  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::link(Parameters &parameters) {
    //    The graph will be updated periodically in NUM_SYNCS batches
    uint32_t NUM_SYNCS = DIV_ROUND_UP(_nd, (128 * 1024));
    if (_nd < (1 << 22))
      NUM_SYNCS = 4 * NUM_SYNCS;
    if (_nd < NUM_SYNCS) {
      NUM_SYNCS = _nd;
    }
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
    std::vector<unsigned> ra_ndperm;
    for (size_t i = 0; i < _nd; i++) {
      ra_ndperm.emplace_back(i);
    }

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    init_random_graph(_nd, range);

    assert(_final_graph.size() == _max_points);
    auto cut_graph_ = new vecNgh[_nd];

    for (uint32_t rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
      // Shuffle the dataset
      std::random_shuffle(ra_ndperm.begin(), ra_ndperm.end());
      unsigned progress_counter = 0;

      size_t round_size = DIV_ROUND_UP(_nd, NUM_SYNCS);  // size of each batch

      for (uint32_t sync_num = 0; sync_num < NUM_SYNCS; sync_num++) {
        if (rnd_no == NUM_RNDS - 1) {
          if (last_round_alpha > 1)
            parameters.Set<unsigned>(
                "L", (unsigned) (std::min)(
                         (int) L, (int) (L -
                                         (L - 30) * ((float) sync_num /
                                                     (float) NUM_SYNCS))));
          parameters.Set<float>("alpha", last_round_alpha);
        }
        size_t start_id = sync_num * round_size;
        size_t end_id = (std::min)(_nd, (sync_num + 1) * round_size);
        size_t round_size = end_id - start_id;

        size_t PAR_BLOCK_SZ =
            round_size > 1 << 20 ? 1 << 12 : (round_size + 256) / 256;
        size_t nblocks = DIV_ROUND_UP(round_size, PAR_BLOCK_SZ);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t block = 0; block < nblocks;
             ++block) {  // Gopal. changed from size_t to int64_t
          std::vector<Neighbor>    pool, tmp;
          tsl::robin_set<unsigned> visited;
          for (size_t n = start_id + block * PAR_BLOCK_SZ;
               n <
               start_id + (std::min)(round_size, (block + 1) * PAR_BLOCK_SZ);
               ++n) {
            pool.clear();
            tmp.clear();
            visited.clear();

            // get nearest neighbors of n in tmp. pool contains all the points
            // that were checked along with their distance from n. visited
            // contains all
            // the points visited, just the ids
            get_neighbors(_data + (size_t) _dim * n, parameters, tmp, pool,
                          visited);
            // sync_prune will check pool, and remove some of the points and
            // create a cut_graph, which contains neighbors for point n
            sync_prune(_data + (size_t) _dim * n, n, pool, parameters, visited,
                       cut_graph_[n]);
          }
        }

#pragma omp parallel for schedule(static, PAR_BLOCK_SZ)

        for (int node = start_id; node < end_id; ++node) {
          // clear all the neighbors of _final_graph[node]
          _final_graph[node].clear();
          _final_graph[node].reserve(range);
          assert(!cut_graph_[node].empty());
          for (auto link : cut_graph_[node]) {
            _final_graph[node].emplace_back(link.id);
            assert(link.id >= 0 && link.id < _nd);
          }
          assert(_final_graph[node].size() <= range);
        }

#pragma omp parallel for schedule(static, PAR_BLOCK_SZ)
        for (int n = start_id; n < end_id; ++n) {
          inter_insert(n, cut_graph_[n], parameters);
        }

        if ((sync_num * 100) / NUM_SYNCS > progress_counter) {
          std::cout << "Completed  (round: " << rnd_no << ", sync: " << sync_num
                    << "/" << NUM_SYNCS
                    << ") with L=" << parameters.Get<unsigned>("L")
                    << ",alpha=" << parameters.Get<float>("alpha") << std::endl;
          progress_counter += 5;
        }

#pragma omp parallel for schedule(static, PAR_BLOCK_SZ)
        for (int n = start_id; n < end_id;
             ++n) {  // Gopal. from unsigned n to int n for openmp
          auto node = n;
          assert(!cut_graph_[node].empty());
          cut_graph_[node].clear();
          cut_graph_[node].shrink_to_fit();
        }
      }
    }

    delete[] cut_graph_;
  }

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::set_data(const T *data) {
    _data = data;
  }

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::build(const T *data, Parameters &parameters,
                                const std::vector<TagT> &tags) {
    _data = data;

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
    link(parameters);  // Primary func for creating nsg graph

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
              << "  count(deg<2):" << cnt << "\n";
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  std::pair<int, int> IndexNSG<T, TagT>::beam_search(
      const T *query, const T *x, const size_t K, const Parameters &parameters,
      unsigned *indices, int beam_width,
      const std::vector<unsigned> &start_points) {
    const unsigned int L = parameters.Get<unsigned>("L_search");
    return beam_search(query, x, K, L, indices, beam_width, start_points);
  }

  template<typename T, typename TagT>
  std::pair<int, int> IndexNSG<T, TagT>::beam_search(
      const T *query, const T *x, const size_t K, const unsigned L,
      unsigned *indices, int beam_width,
      const std::vector<unsigned> &start_points) {
    _data = x;
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * L);

    if (start_points.size() == 0) {
      std::cerr << "Error: starting points must be populated for search"
                << std::endl;
      exit(-1);
    }

    /* ep_neighbors contains all the neighbors of navigating node, and
     * their distance from the query node
     */
    std::vector<Neighbor> ep_neighbors;
    for (auto curpt : start_points)
      for (auto id : _final_graph[curpt]) {
        // std::cout << "cmp: query <-> " << id << "\n";
        ep_neighbors.emplace_back(Neighbor(
            id, _distance->compare(_data + _dim * (size_t) id, query, _dim),
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
      unsigned id = (rand() * rand() * rand()) % _nd;
      if (visited.find(id) == visited.end())
        visited.insert(id);
      else
        continue;
      init_ids.emplace_back(id);
    }
    //    init_ids.resize(tmp_l);
    std::vector<Neighbor> retset(L + 1);

    /* Find out the distances of all the neighbors of navigating node
     * with the query and add it to retset. Actually not needed for all the
     * neighbors. Only needed for the random ones added later
     */
    for (size_t i = 0; i < init_ids.size(); i++) {
      // std::cout << "cmp: query <-> " << init_ids[i] << "\n";
      retset[i] = Neighbor(init_ids[i],
                           _distance->compare(_data + _dim * init_ids[i], query,
                                              (unsigned) _dim),
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
          const T *vec1 = _data + _dim * id_next;
          NSG::prefetch_vector((const char *) vec1, _dim * sizeof(T));
        }

        cmps++;
        unsigned id = *iter;
        /* compare distance of each neighbor with that of query. If the
         * distance is less than largest distance in retset, add to retset and
         * set flag to true
         */
        // std::cout << "cmp: query <-> " << id << "\n";
        float dist =
            _distance->compare(_data + _dim * id, query, (unsigned) _dim);
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
      else
        indices[i++] = id;
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename TagT>
  std::pair<int, int> IndexNSG<T, TagT>::beam_search_tags(
      const T *query, const T *x, const size_t K, const Parameters &parameters,
      TagT *tags, int beam_width, const std::vector<unsigned> &start_points,
      unsigned *indices_buffer) {
    const bool alloc = indices_buffer == NULL;
    auto       indices = alloc ? new unsigned[K] : indices_buffer;
    auto       ret =
        beam_search(query, x, K, parameters, indices, beam_width, start_points);
    for (int i = 0; i < (int) K; ++i)
      tags[i] = _location_to_tag[indices[i]];
    if (alloc)
      delete[] indices;
    return ret;
  }

  /*
    // INT8 specialization
    template<>
    void IndexNSG<_s8>::iterate_to_fixed_point(
        const _s8 *query, const Parameters &parameter,
        std::vector<unsigned> &init_ids, std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset, tsl::robin_set<unsigned> &visited) {
      const unsigned L = parameter.Get<unsigned>("L");

      // put random L new ids into visited list and init_ids list
      while (init_ids.size() < L) {
        unsigned id = (rand() * rand() * rand()) % _nd;
        if (visited.find(id) != visited.end())
          continue;
        else
          visited.insert(id);
        init_ids.emplace_back(id);
      }

      // nodes to compare against in iterations
      unsigned unvisited_nbrs[256] = {0}, unvisited_nnbrs = 0;
      float    unvisited_nbrs_dists[256] = {0.0f};

      // lambdas as helpers
      // filters all unvisited nodes and places them into out_nbrs. Additonally,
      // also inserts out_nbrs into visited
      auto filter_visited_a_ndmark_visited = [this, &visited](
          const unsigned *nbrs, const unsigned nnbrs, unsigned *out_nbrs,
          unsigned &out_nnbrs) {
        out_nnbrs = 0;
        for (unsigned i = 0; i < nnbrs; i++) {
          unsigned idx = nbrs[i];
          bool     is_not_visited = (visited.find(idx) == visited.end());
          if (is_not_visited) {
            const _s8 *point_coords = this->_data + this->_dim * idx;
            // also issue prefetches to bring vector into L2 --> prefetch easier
            // from L2->L1 later
            NSG::prefetch_vector_l2((const char *) point_coords,
                                    this->_dim * sizeof(_s8));
            out_nbrs[out_nnbrs++] = idx;
          }
        }
        visited.insert(out_nbrs, out_nbrs + out_nnbrs);
      };

      auto batch_compute_distances = [this](
          const _s8 *query, const unsigned *nbrs, const unsigned nnbrs,
          float *out_dists) {
        NSG::prefetch_vector(
            (const char *) (this->_data + (this->_dim * nbrs[0])),
            this->_dim * sizeof(_s8));
        for (unsigned i = 0; i < nnbrs; i++) {
          // prefetch next coordinates from L2 -> L1 (2 cycles for L2 hit +
          // 128-dim vectors)
          if (i < nnbrs - 1) {
            NSG::prefetch_vector(
                (const char *) (this->_data + (this->_dim * nbrs[i + 1])),
                this->_dim * sizeof(_s8));
          }
          const _s8 *point_coords = this->_data + (this->_dim * nbrs[i]);
          out_dists[i] =
              this->_distance->compare(query, point_coords, this->_dim);
        }
      };

      // compare distance of all points in init_ids with query, and put the id
      // with distance
      // in retset
      //
      unsigned l = 0;
      for (auto id : init_ids) {
        assert(id < _nd);
        retset[l++] =
            Neighbor(id,
                     _distance->compare(_data + _dim * (size_t) id, query,
                                        _dim),
                     true);
      }

      // sort retset based on distance of each point to query
      std::sort(retset.begin(), retset.begin() + l);
      int k = 0;
      while (k < (int) l) {
        int nk = l;
        if (!retset[k].flag)
          k++;

        retset[k].flag = false;
        unsigned n = retset[k].id;

        // prefetch _final_graph[n]
        unsigned *nbrs = _final_graph[n].data();   // nbrs: data of neighbors
        unsigned  nnbrs = _final_graph[n].size();  // nnbrs: number of neighbors

        // prefetch nnbrs
        NSG::prefetch_vector((const char *) nbrs, nnbrs * sizeof(unsigned));

        // filter nbrs into compare_ids
        filter_visited_a_ndmark_visited(nbrs, nnbrs, unvisited_nbrs,
                                        unvisited_nnbrs);

        // batch prefetch all unvisited vectors into L2
        for (_u64 i = 0; i < unvisited_nnbrs; i++) {
          const _s8 *vec = this->_data + (unvisited_nbrs[i] * this->_dim);
          NSG::prefetch_vector_l2((const char *) vec,
                                  this->_dim * sizeof(_s8));
        }

        // prefetch query
        NSG::prefetch_vector((const char *) query,
                             this->_dim * sizeof(_s8));

        // batch compute distances to unvisited nodes
        batch_compute_distances(query, unvisited_nbrs, unvisited_nnbrs,
                                unvisited_nbrs_dists);

        // prefetch distances vector (should be in L1/L2)
        NSG::prefetch_vector((const char *) unvisited_nbrs_dists,
                             unvisited_nnbrs * sizeof(float));
        NSG::prefetch_vector((const char *) unvisited_nbrs,
                             unvisited_nnbrs * sizeof(unsigned));

        NSG::prefetch_vector((const char *) retset.data(),
                             retset.size() * sizeof(Neighbor));

        for (size_t m = 0; m < unvisited_nnbrs; m++) {
          unsigned id = unvisited_nbrs[m];  // id = neighbor
          // compare distance of id with query
          float    dist = unvisited_nbrs_dists[m];
          Neighbor nn(id, dist, true);
          fullset.push_back(nn);
          if (dist >= retset[l - 1].distance)
            continue;

          // if distance is smaller than largest, add to retset, keep it sorted
          int r = InsertIntoPool(retset.data(), l, nn);

          if (l + 1 < retset.size())
            ++l;
          if (r < nk)
            nk = r;
        }
      }
      assert(!fullset.empty());
    }
    */

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::save_disk_opt_graph(const char *diskopt_path) {
    const _u64 SECTOR_LEN = 4096;
    std::cout << "Embedding node coords with its nhood" << std::endl;
    size_t npts_u64 = _nd, ndims_u64 = _dim;

    // amount to write in one shot
    _u64 write_blk_size = 256l * 1024l * 1024l;

    // create cached reader + writer
    cached_ofstream nsg_writer(diskopt_path, write_blk_size);

    // compute
    _u64 max_node_len, nnodes_per_sector;
    max_node_len =
        (((_u64) _width + 1) * sizeof(unsigned)) + (ndims_u64 * sizeof(T));
    nnodes_per_sector = SECTOR_LEN / max_node_len;

    std::cout << "medoid: " << _ep << "B\n";
    std::cout << "max_node_len: " << max_node_len << "B\n";
    std::cout << "nnodes_per_sector: " << nnodes_per_sector << "B\n";

    // SECTOR_LEN buffer for each sector
    char *    sector_buf = new char[SECTOR_LEN];
    char *    node_buf = new char[max_node_len];
    unsigned &nnbrs = *(unsigned *) (node_buf + ndims_u64 * sizeof(T));
    unsigned *nhood_buf =
        (unsigned *) (node_buf + (ndims_u64 * sizeof(T)) + sizeof(unsigned));

    // write first sector with metadata
    *(_u64 *) sector_buf = npts_u64;
    *(_u64 *) (sector_buf + sizeof(_u64)) = (_u64) _ep;
    *(_u64 *) (sector_buf + 2 * sizeof(_u64)) = max_node_len;
    *(_u64 *) (sector_buf + 3 * sizeof(_u64)) = nnodes_per_sector;
    nsg_writer.write(sector_buf, SECTOR_LEN);

    _u64 n_sectors = ROUND_UP(npts_u64, nnodes_per_sector) / nnodes_per_sector;
    std::cout << "# sectors: " << n_sectors << "\n";

    _u64 cur_node_id = 0;
    for (_u64 sector = 0; sector < n_sectors; sector++) {
      if (sector % 100000 == 0) {
        std::cout << "Sector #" << sector << "written\n";
      }

      memset(sector_buf, 0, SECTOR_LEN);

      for (_u64 sector_node_id = 0; sector_node_id < nnodes_per_sector;
           sector_node_id++) {
        // set cur node's nnbrs
        nnbrs = _final_graph[cur_node_id].size();

        // sanity checks on nnbrs
        assert(nnbrs > 0);
        assert(nnbrs <= _width);

        // set cur node's nhood
        nhood_buf = _final_graph[cur_node_id].data();

        // write coords of node first
        const T *node_coords = _data + ((_u64) ndims_u64 * cur_node_id);
        memcpy(node_buf, node_coords, ndims_u64 * sizeof(T));

        // write nnbrs
        *(unsigned *) (node_buf + ndims_u64 * sizeof(T)) = nnbrs;

        // write nhood next
        memcpy(node_buf + (ndims_u64 * sizeof(T)) + sizeof(unsigned), nhood_buf,
               nnbrs * sizeof(unsigned));

        // get offset into sector_buf
        char *sector_node_buf = sector_buf + (sector_node_id * max_node_len);

        // copy node buf into sector_node_buf
        memcpy(sector_node_buf, node_buf, max_node_len);
        cur_node_id++;
      }

      // flush sector to disk
      nsg_writer.write(sector_buf, SECTOR_LEN);
    }

    delete[] sector_buf;
    delete[] node_buf;

    std::cout << "Diskopt NSG written to " << diskopt_path << "\n";
  }

  template class IndexNSG<float>;
  template class IndexNSG<int8_t>;
  template class IndexNSG<uint8_t>;

}  // namespace NSG
