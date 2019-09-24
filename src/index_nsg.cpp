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
}  // namespace

namespace NSG {
#define _CONTROL_NUM 100
#define MAX_START_POINTS 100

  // Initialize an index with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T, typename TagT>
  IndexNSG<T, TagT>::IndexNSG(Metric m, const char *filename,
                              const size_t max_points, const size_t nd,
                              const bool enable_tags)
      : _has_built(false), _width(0), _can_delete(false),
        _enable_tags(enable_tags), _consolidated_order(true) {
    load_aligned_bin<T>(std::string(filename), _data, _nd, _dim, _aligned_dim);
    // data is stored to _nd * aligned_dim matrix with necessary zero-padding
    std::cout << "#points in file: " << _nd << ", dim: " << _dim
              << ", rounded_dim: " << _aligned_dim << std::endl;

    if (nd > 0) {
      if (_nd < nd) {
        std::cerr << "Error: Driver requests loading " << _nd
                  << " points, while file has " << nd << "points" << std::endl;
        exit(-1);
      } else
        _nd = nd;
    }

    _max_points = (max_points > 0) ? max_points : _nd;
    if (_max_points < _nd) {
      std::cerr << "ERROR: max_points must be >= data size; max_points: "
                << _max_points << "  n: " << _nd << std::endl;
      exit(-1);
    }

    this->_distance = ::get_distance_function<T>();

    _locks = std::vector<std::mutex>(_max_points);
    _width = 0;
  }

  template<>
  IndexNSG<float>::~IndexNSG() {
    delete this->_distance;
//    delete[] _data;
    aligned_free(_data);
  }

  template<>
  IndexNSG<_s8>::~IndexNSG() {
    delete this->_distance;
    aligned_free(_data);
//    delete[] _data;
  }

  template<>
  IndexNSG<_u8>::~IndexNSG() {
    delete this->_distance;
    aligned_free(_data);
//   delete[] _data;
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

    std::vector<unsigned> new_ids;
    new_ids.resize(_max_points, _max_points);
    unsigned active = 0;
    for (unsigned old = 0; old < _max_points; ++old)
      if (_empty_slots.find(old) == _empty_slots.end() &&
          _delete_set.find(old) == _delete_set.end())
        new_ids[old] = active++;
    assert(active + _empty_slots.size() + _delete_set.size() == _max_points);

    tsl::robin_set<unsigned> candidate_set;
    std::vector<Neighbor>    expanded_nghrs;
    std::vector<Neighbor>    result;

    for (unsigned i = 0; i < _max_points; ++i) {
      if (new_ids[i] < _max_points) {
        candidate_set.clear();
        expanded_nghrs.clear();
        result.clear();

        bool modify = false;
        for (auto ngh : _final_graph[i]) {
          if (new_ids[ngh] >= _max_points) {
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
          for (auto j : result)
            _final_graph[i].push_back(j.id);

          //          for (auto iter : _final_graph[i]) {
          //            assert(_delete_set.find(iter) == _delete_set.end());
          //            assert(_empty_slots.find(iter) == _empty_slots.end());
          //          }
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
      if (new_ids[old] < _max_points) {  // If point continues to exist

        // Renumber nodes to compact the order
        for (size_t i = 0; i < _final_graph[old].size(); ++i) {
          assert(new_ids[_final_graph[old][i]] < _max_points);
          _final_graph[old][i] = new_ids[_final_graph[old][i]];
        }

        // Move the data to the correct position
        memcpy((void *) (_data + _aligned_dim * (size_t) new_ids[old]),
               (void *) (_data + _aligned_dim * (size_t) old),
               _aligned_dim * sizeof(float));
      }
    }
    std::cout << "done." << std::endl;

    std::cout << "Updating mapping between tags and ids... " << std::flush;
    // Update the location pointed to by tag
    for (auto iter : _tag_to_point) {
      auto tag = iter.first;
      auto old_id = iter.second;
      auto new_id = new_ids[old_id];

      iter.second = new_id;
      _point_to_tag.erase(old_id);
      _point_to_tag[new_id] = tag;
    }
    std::cout << "done." << std::endl;

    for (unsigned old = active; old < _max_points; ++old)
      _final_graph[old].clear();

    _empty_slots.clear();
    _delete_set.clear();
    _consolidated_order = true;
    std::cout << "Consolidated the index" << std::endl;

    return _nd;
  }

  /*
  // Do not call consolidate_deletes() if you have not locked _change_lock.
  // Returns number of live points left after consolidation
  size_t IndexNSG::consolidate_deletes_old(const Parameters &parameters) {
    assert(!_consolidated_order);
    assert(_can_delete);
    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    std::vector<unsigned> new_ids;
    new_ids.resize(_max_points, _max_points);
    unsigned active = 0;
    for (unsigned old = 0; old < _max_points; ++old)
      if (_empty_slots.find(old) == _empty_slots.end() &&
          _delete_set.find(old) == _delete_set.end())
        new_ids[old] = active++;
    assert(active + _empty_slots.size() + _delete_set.size() == _max_points);

    tsl::robin_set<unsigned> candidate_set;
    std::vector<Neighbor>    expanded_nghrs;
    std::vector<Neighbor>    result;

    for (auto i : _delete_set) {
      // Remove point, and create new links from neighbors
      // for (auto ngh : _final_graph[i]) {
      for (auto ngh : in_graph_[i]) {
        candidate_set.clear();
        expanded_nghrs.clear();
        result.clear();

        // Add outgoing links from
        for (auto j : _final_graph[ngh])
          if (_delete_set.find(j) == _delete_set.end())
            candidate_set.insert(j);
        for (auto j : _final_graph[i])
          if (_delete_set.find(j) == _delete_set.end())
            candidate_set.insert(j);
        for (auto iter : candidate_set)
          assert(_empty_slots.find(iter) == _empty_slots.end());

        for (auto j : candidate_set)
          expanded_nghrs.push_back(
              Neighbor(j,
                       _distance->compare(_data + _dim * (size_t) ngh,
                                          _data + _dim * (size_t) j,
                                          (unsigned) _dim),
                       true));

        std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
        occlude_list(expanded_nghrs, ngh, alpha, range, maxc, result);

        _final_graph[ngh].clear();
        for (auto j : result)
          _final_graph[ngh].push_back(j.id);

        for (auto iter : _final_graph[ngh]) {
          assert(_delete_set.find(iter) == _delete_set.end());
          assert(_empty_slots.find(iter) == _empty_slots.end());
        }
      }
      --_nd;  // Decrement #points in index

      // If start node is removed, replace it.
      if (i == _ep) {
        std::cerr << "Replacing start node which has been deleted... "
                  << std::flush;
        // First active neighbor of old start node is new start node
        for (auto iter : _final_graph[i])
          if (_delete_set.find(iter) != _delete_set.end()) {
            _ep = iter;
            break;
          }
        if (_ep == i) {
          std::cerr << "ERROR: Did not find a replacement for start node."
                    << std::endl;
          exit(-1);
        } else {
          assert(_delete_set.find(_ep) == _delete_set.end());
          std::cout << "New start node is " << _ep << std::endl;
        }
      }
    }

    std::cout << "Replacing edges to deleted links... " << std::endl;
    unsigned deleted_links = 0, loops_to_start = 0;
    for (unsigned old = 0; old < _max_points; ++old) {
      if (new_ids[old] < _max_points) {  // Point still exists
        for (size_t pos = 0; pos < _final_graph[old].size(); ++pos) {
          auto link = _final_graph[old][pos];
          if (new_ids[link] >= _max_points) {
            ++deleted_links;
            for (auto twohop : _final_graph[link]) {
              if (new_ids[twohop] < _max_points) {
                _final_graph[old][pos] = twohop;
                break;
              }
              if (link == _final_graph[old][pos]) {
                ++loops_to_start;
                _final_graph[old][pos] = _ep;
              }
            }
          }
        }
      }
    }
    std::cout << "Replaced " << deleted_links << " links." << std::endl;
    std::cout << "Couldn't replace " << loops_to_start
              << " deleted out edge with 2-hop link. Linked them to start."
              << std::endl;

    std::cout << "Re-numbering nodes and edges and consolidating data... "
              << std::flush;
    for (unsigned old = 0; old < _max_points; ++old) {
      if (new_ids[old] < _max_points) {  // If point continues to exist

        // Renumber nodes to compact the order
        for (size_t i = 0; i < _final_graph[old].size(); ++i) {
          assert(new_ids[_final_graph[old][i]] < _max_points);
          _final_graph[old][i] = new_ids[_final_graph[old][i]];
        }

        // Move the data to the correct position
        memcpy((void *) (_data + _dim * (size_t) new_ids[old]),
               (void *) (_data + _dim * (size_t) old),
               _dim * sizeof(float));
      }
    }
    std::cout << "done." << std::endl;

    std::cout << "Updating mapping between tags and ids... " << std::flush;
    // Update the location pointed to by tag
    for (auto iter : _tag_to_point) {
      auto tag = iter.first;
      auto old_id = iter.second;
      auto new_id = new_ids[old_id];

      iter.second = new_id;
      _point_to_tag.erase(old_id);
      _point_to_tag[new_id] = tag;
    }
    std::cout << "done." << std::endl;

    for (unsigned old = active; old < _max_points; ++old)
      _final_graph[old].clear();

    _empty_slots.clear();
    _delete_set.clear();
    _consolidated_order = true;
    std::cout << "Consolidated the index" << std::endl;

    return _nd;
  }
  */

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
    if (_tag_to_point.size() + _delete_set.size() != _nd) {
      std::cerr << "Tags to points array wrong sized" << std::endl;
      return -2;
    }
    if (_point_to_tag.size() + _delete_set.size() != _nd) {
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
    if (_tag_to_point.find(tag) == _tag_to_point.end()) {
      std::cerr << "Delete tag not found" << std::endl;
      return -1;
    }
    _delete_set.insert(_tag_to_point[tag]);
    _point_to_tag.erase(_tag_to_point[tag]);
    _tag_to_point.erase(tag);
    return 0;
  }

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::save(const char *filename) {
    long long     total_gr_edges = 0;
    size_t        index_size = 0;
    std::ofstream out(std::string(filename), std::ios::binary | std::ios::out);

    assert(_final_graph.size() == _max_points);
    if (_enable_tags) {
      _change_lock.lock();
      if (_can_delete || !_consolidated_order) {
        std::cerr
            << "Disable deletes and consolidated order before saving index"
            << std::endl;
        exit(-1);
      }
    }

    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &_width, sizeof(unsigned));
    out.write((char *) &_ep, sizeof(unsigned));
    for (unsigned i = 0; i < _nd; i++) {
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
        out_tags << _point_to_tag[i] << "\n";
      }
      out_tags.close();
      _change_lock.unlock();
    }

    std::cout << "Avg degree: " << ((float) total_gr_edges) / ((float) _nd)
              << std::endl;
  }

  // load the SNG index if pre-computed
  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::load(const char *filename) {
    std::ifstream in(std::string(filename), std::ios::binary);
    in.seekg(0, in.end);
    size_t expected_file_size;
    size_t actual_file_size = in.tellg();
    in.seekg(0, in.beg);
    in.read((char *) &expected_file_size, sizeof(uint64_t));
    if (actual_file_size != expected_file_size) {
      std::cout << "Error loading Rand-NSG index. File size mismatch, expected "
                   "size (metadata): "
                << expected_file_size << std::endl;
      exit(-1);
    }
    in.read((char *) &_width, sizeof(unsigned));
    in.read((char *) &_ep, sizeof(unsigned));
    std::cout << "NSG -- width: " << _width << ", _ep: " << _ep << "\n";

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
        std::cout << "Loaded " << nodes << " nodes, and " << cc << " neighbors"
                  << std::endl;
    }
    std::cout << "Loaded Rand-NSG with " << nodes << " nodes, and " << cc
              << " neighbors" << std::endl;
    cc /= _nd;
    if (_final_graph.size() != _nd) {
      std::cout << "Error. mismatch in number of points. Graph has "
                << _final_graph.size() << " points and loaded dataset has "
                << _nd << " points. " << std::endl;
      exit(-1);
    }
  }

  /* init_random_graph():
   * num_points: Number of points in the dataset
   * k: max degree of the graph
   * mapping: initial vector of a sample of the points in the dataset
   */
  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::init_random_graph(size_t num_points, unsigned k,
                                            std::vector<size_t> mapping) {
    k = (std::min)(k, (unsigned) 100);
    _final_graph.resize(_max_points);
    _final_graph.reserve(_max_points);
    if (!mapping.empty())
      num_points = mapping.size();
    else {
      mapping.resize(num_points);
      std::iota(std::begin(mapping), std::end(mapping), 0);
    }

    std::cout << "Generating random graph with " << num_points << " points... ";
    // PAR_BLOCK_SZ gives the number of points that can fit in a single block
    _s64 PAR_BLOCK_SZ = (1 << 16);  // = 64KB
    _s64 nblocks = DIV_ROUND_UP((_s64) num_points, PAR_BLOCK_SZ);

#pragma omp parallel for schedule(static, 1)
    for (_s64 block = 0; block < nblocks;
         ++block) {  // GOPAL Changed from "size_t block" to "int block"
      std::random_device                    rd;
      std::mt19937                          gen(rd());
      std::uniform_int_distribution<size_t> dis(0, num_points - 1);

      /* Put random number points as neighbours to the 10% of the nodes */
      for (_s64 i = block * PAR_BLOCK_SZ;
           i < (block + 1) * PAR_BLOCK_SZ && i < (_s64) num_points; i++) {
        std::set<unsigned> ra_ndset;
        while (ra_ndset.size() < k) {
          unsigned cur_pt = dis(gen);
          if (cur_pt != i)
            ra_ndset.insert(cur_pt);
        }

        _final_graph[mapping[i]].reserve(k);
        for (auto s : ra_ndset)
          _final_graph[mapping[i]].emplace_back(mapping[s]);
        _final_graph[mapping[i]].shrink_to_fit();
      }
    }
    _ep = get_entry_point();
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
//    const unsigned L = parameter.Get<unsigned>("L");

    /* put random L new ids into visited list and init_ids list */
/*      while (init_ids.size() < L) {
      unsigned id = (rand() * rand() * rand()) % _nd;
      if (visited.find(id) == visited.end()) {
        visited.insert(id);
        fullset.emplace_back(
            Neighbor(id,
                     _distance->compare(_data + _aligned_dim * (size_t) id,
                                        query, _aligned_dim),
                     true));
        init_ids.emplace_back(id);
      }
    }
*/
    /* compare distance of all points in init_ids with query, and put the id
     * with distance
     * in retset
     */
    unsigned l = 0;
    Neighbor nn;
    for (auto id : init_ids) {
      assert(id < _max_points);
      nn = Neighbor(id,
                    _distance->compare(_data + _aligned_dim * (size_t) id,
                                       query, _aligned_dim),
                    true);
      fullset.emplace_back(nn);
      retset[l++] = nn;
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
            const T *vec_next1 = _data + (size_t) id_next * _aligned_dim;
            NSG::prefetch_vector((const char *) vec_next1,
                                 _aligned_dim * sizeof(T));
          }

          if (visited.find(id) == visited.end())
            visited.insert(id);  // if id is not in visited, add it to visited
          else
            continue;

          // compare distance of id with query
          float dist =
              _distance->compare(query, _data + _aligned_dim * (size_t) id,
                                 (unsigned) _aligned_dim);
          Neighbor nn(id, dist, true);
          fullset.emplace_back(nn);
          if (dist >= retset[l - 1].distance && (l == retset.size()-1))
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
      ep_neighbors.emplace_back(
          Neighbor(id,
                   _distance->compare(_data + _aligned_dim * (size_t) id, query,
                                      _aligned_dim),
                   true));
    }

    std::sort(ep_neighbors.begin(), ep_neighbors.end());
    for (auto iter : ep_neighbors) {
      if (init_ids.size() >= L)
        break;
      if (visited.find(iter.id) == visited.end()) {
        visited.insert(iter.id);
        fullset.emplace_back(iter);
        init_ids.emplace_back(iter.id);
      }
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
    // Gopal. Commenting out because we cannot assume the "null value"
    // if (_enable_tags)
    //  assert(tag != NULL_TAG);

    LockGuard guard(_change_lock);
    std::cout << "got lock" << std::endl;
    if (_enable_tags && (_tag_to_point.find(tag) != _tag_to_point.end())) {
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
    _tag_to_point[tag] = location;
    _point_to_tag[location] = tag;

    auto offset_data = _data + (size_t) _aligned_dim * location;
    memcpy((void *) offset_data, point, sizeof(float) * _aligned_dim);

    pool.clear();
    tmp.clear();
    cut_graph.clear();
    visited.clear();

    get_neighbors(offset_data, parameters, tmp, pool, visited);
    sync_prune(_data + (size_t) _aligned_dim * location, location, pool,
               parameters, visited, cut_graph);

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
  unsigned IndexNSG<T, TagT>::get_entry_point() {
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
    if (start == pool.size())
      return;
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
        float djk = _distance->compare(
            _data + _aligned_dim * (size_t) result[t].id,
            _data + _aligned_dim * (size_t) p.id, (unsigned) _aligned_dim);
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
        float dist = _distance->compare(x, _data + _aligned_dim * (size_t) id,
                                        (unsigned) _aligned_dim);
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
          temp_pool.emplace_back(SimpleNeighbor(
              node,
              _distance->compare(_data + _aligned_dim * (size_t) node,
                                 _data + _aligned_dim * (size_t) des.id,
                                 (unsigned) _aligned_dim)));
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
            float djk = _distance->compare(_data + _aligned_dim * (size_t) r.id,
                                           _data + _aligned_dim * (size_t) p.id,
                                           (unsigned) _aligned_dim);
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
              float djk =
                  _distance->compare(_data + _aligned_dim * (size_t) r.id,
                                     _data + _aligned_dim * (size_t) p.id,
                                     (unsigned) _aligned_dim);
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

  /* Link():
   * The graph creation function.
   */
  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::link(Parameters &parameters) {
    //    The graph will be updated periodically in NUM_SYNCS batches
    uint32_t NUM_SYNCS = DIV_ROUND_UP(_nd, (128 * 96));
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
//            parameters.Set<unsigned>(
//                "L", (unsigned) (std::min)(
//                         (int) L, (int) (L -
//                                         (L - 50) * ((float) sync_num /
//                                                     (float) NUM_SYNCS))));
            parameters.Set<unsigned>("L", (std::min)(L, (unsigned) 50));                                                 
          parameters.Set<float>("alpha", last_round_alpha);
        }
        size_t start_id = sync_num * round_size;
        size_t end_id = (std::min)(_nd, (sync_num + 1) * round_size);
        size_t round_size = end_id - start_id;

        size_t PAR_BLOCK_SZ =
            round_size > 1 << 20 ? 1 << 12 : (round_size + 256) / 256;
        size_t nblocks = DIV_ROUND_UP(round_size, PAR_BLOCK_SZ);

#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 block = 0; block < (_s64) nblocks;
             ++block) {  // Gopal. changed from size_t to _s64
          std::vector<Neighbor>    pool, tmp;
          tsl::robin_set<unsigned> visited;

          for (size_t n = start_id + block * PAR_BLOCK_SZ;
               n < start_id + (std::min<size_t>) (round_size,
                                                  (block + 1) * PAR_BLOCK_SZ);
               ++n) {
            pool.clear();
            tmp.clear();
            visited.clear();

            // get nearest neighbors of n in tmp. pool contains all the points
            // that were checked along with their distance from n. visited
            // contains all
            // the points visited, just the ids
            get_neighbors(_data + (size_t) _aligned_dim * n, parameters, tmp,
                          pool, visited);
            // sync_prune will check pool, and remove some of the points and
            // create a cut_graph, which contains neighbors for point n
            if (visited.find(n) != visited.end()) {
              unsigned itpos = 0;
              for (auto iter : pool) {
                if (iter.id == n) {
                  pool.erase(pool.begin() + itpos);
                  break;
                }
                itpos++;
              }
            }
            sync_prune(_data + (size_t) _aligned_dim * n, n, pool, parameters,
                       visited, cut_graph_[n]);
          }
        }

#pragma omp parallel for schedule(static, PAR_BLOCK_SZ)

        for (_s64 node = (_s64) start_id; node < (_s64) end_id; ++node) {
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
        for (_s64 n = start_id; n < (_s64) end_id; ++n) {
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
        for (_s64 n = start_id; n < (_s64) end_id;
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

  /*  template<typename T, typename TagT>
    void IndexNSG<T, TagT>::set_data(T *data) {
      _data = data;
    }
  */

  template<typename T, typename TagT>
  void IndexNSG<T, TagT>::build(Parameters &parameters,
                                const std::vector<TagT> &tags) {
    if (_enable_tags) {
      if (tags.size() != _nd) {
        std::cerr << "#Tags should be equal to #points" << std::endl;
        exit(-1);
      }
      for (size_t i = 0; i < tags.size(); ++i) {
        _tag_to_point[tags[i]] = i;
        _point_to_tag[i] = tags[i];
      }
    }
    std::cout << "Starting index build..." << std::endl;
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
              << "  count(deg<2):" << cnt << "\n"
              << "Index built." << std::endl;
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  std::pair<int, int> IndexNSG<T, TagT>::beam_search(
      const T *query, const size_t K, const size_t L, unsigned *indices,
      int beam_width, std::vector<unsigned> start_points) {
    //    _data = x;

    std::vector<unsigned> init_ids;

    tsl::robin_set<unsigned> visited(10 * L);

    // use start_points for init; ignore default init
    if (start_points.size() == 0)
      start_points.emplace_back(_ep);

    /* ep_neighbors contains all the neighbors of navigating node, and
     * their distance from the query node
     */
    std::vector<Neighbor> ep_neighbors;
    for (auto curpt : start_points)
      for (auto id : _final_graph[curpt]) {
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
          NSG::prefetch_vector((const char *) vec1, _aligned_dim * sizeof(T));
        }

        cmps++;
        unsigned id = *iter;
        /* compare distance of each neighbor with that of query. If the
         * distance is less than largest distance in retset, add to retset and
         * set flag to true
         */
        // std::cout << "cmp: query <-> " << id << "\n";
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
      else
        indices[i++] = id;
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename TagT>
  std::pair<int, int> IndexNSG<T, TagT>::beam_search_tags(
      const T *query, const size_t K, const size_t L, TagT *tags,
      int beam_width, std::vector<unsigned> start_points,
      unsigned *indices_buffer) {
    const bool alloc = indices_buffer == NULL;
    auto       indices = alloc ? new unsigned[K] : indices_buffer;
    auto ret = beam_search(query, K, L, indices, beam_width, start_points);
    for (int i = 0; i < (int) K; ++i)
      tags[i] = _point_to_tag[indices[i]];
    if (alloc)
      delete[] indices;
    return ret;
  }

  // EXPORTS
  template NSGDLLEXPORT class IndexNSG<float>;
  template NSGDLLEXPORT class IndexNSG<int8_t>;
  template NSGDLLEXPORT class IndexNSG<uint8_t>;

#ifdef _WINDOWS
  template NSGDLLEXPORT IndexNSG<uint8_t, int>::IndexNSG(
      Metric m, const char *filename, const size_t max_points, const size_t nd,
      const bool enable_tags);
  template NSGDLLEXPORT IndexNSG<int8_t, int>::IndexNSG(Metric m,
                                                        const char * filename,
                                                        const size_t max_points,
                                                        const size_t nd,
                                                        const bool enable_tags);
  template NSGDLLEXPORT IndexNSG<float, int>::IndexNSG(Metric m,
                                                       const char * filename,
                                                       const size_t max_points,
                                                       const size_t nd,
                                                       const bool enable_tags);

  template NSGDLLEXPORT IndexNSG<uint8_t, int>::~IndexNSG();
  template NSGDLLEXPORT IndexNSG<int8_t, int>::~IndexNSG();
  template NSGDLLEXPORT IndexNSG<float, int>::~IndexNSG();

  template NSGDLLEXPORT void IndexNSG<uint8_t, int>::save(const char *filename);
  template NSGDLLEXPORT void IndexNSG<int8_t, int>::save(const char *filename);
  template NSGDLLEXPORT void IndexNSG<float, int>::save(const char *filename);

  template NSGDLLEXPORT void IndexNSG<uint8_t, int>::load(const char *filename);
  template NSGDLLEXPORT void IndexNSG<int8_t, int>::load(const char *filename);
  template NSGDLLEXPORT void IndexNSG<float, int>::load(const char *filename);

  template NSGDLLEXPORT void IndexNSG<uint8_t, int>::build(
      Parameters &parameters, const std::vector<int> &tags);
  template NSGDLLEXPORT void IndexNSG<int8_t, int>::build(
      Parameters &parameters, const std::vector<int> &tags);
  template NSGDLLEXPORT void IndexNSG<float, int>::build(
      Parameters &parameters, const std::vector<int> &tags);

  template NSGDLLEXPORT std::pair<int, int> IndexNSG<uint8_t>::beam_search(
      const uint8_t *query, const size_t K, const size_t L, unsigned *indices,
      int beam_width, std::vector<unsigned> start_points);
  template NSGDLLEXPORT std::pair<int, int> IndexNSG<int8_t>::beam_search(
      const int8_t *query, const size_t K, const size_t L, unsigned *indices,
      int beam_width, std::vector<unsigned> start_points);
  template NSGDLLEXPORT std::pair<int, int> IndexNSG<float>::beam_search(
      const float *query, const size_t K, const size_t L, unsigned *indices,
      int beam_width, std::vector<unsigned> start_points);

  template NSGDLLEXPORT int IndexNSG<int8_t, int>::delete_point(const int tag);
  template NSGDLLEXPORT int IndexNSG<uint8_t, int>::delete_point(const int tag);
  template NSGDLLEXPORT int IndexNSG<float, int>::delete_point(const int tag);
  template NSGDLLEXPORT int IndexNSG<int8_t, size_t>::delete_point(
      const size_t tag);
  template NSGDLLEXPORT int IndexNSG<uint8_t, size_t>::delete_point(
      const size_t tag);
  template NSGDLLEXPORT int IndexNSG<float, size_t>::delete_point(
      const size_t tag);
  template NSGDLLEXPORT int IndexNSG<int8_t, std::string>::delete_point(
      const std::string tag);
  template NSGDLLEXPORT int IndexNSG<uint8_t, std::string>::delete_point(
      const std::string tag);
  template NSGDLLEXPORT int IndexNSG<float, std::string>::delete_point(
      const std::string tag);

  template NSGDLLEXPORT int IndexNSG<int8_t, int>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template NSGDLLEXPORT int IndexNSG<uint8_t, int>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template NSGDLLEXPORT int IndexNSG<float, int>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template NSGDLLEXPORT int IndexNSG<int8_t, size_t>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template NSGDLLEXPORT int IndexNSG<uint8_t, size_t>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template NSGDLLEXPORT int IndexNSG<float, size_t>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template NSGDLLEXPORT int IndexNSG<int8_t, std::string>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template NSGDLLEXPORT int IndexNSG<uint8_t, std::string>::disable_delete(
      const Parameters &parameters, const bool consolidate);
  template NSGDLLEXPORT int IndexNSG<float, std::string>::disable_delete(
      const Parameters &parameters, const bool consolidate);

  template NSGDLLEXPORT int IndexNSG<int8_t, int>::enable_delete();
  template NSGDLLEXPORT int IndexNSG<uint8_t, int>::enable_delete();
  template NSGDLLEXPORT int IndexNSG<float, int>::enable_delete();
  template NSGDLLEXPORT int IndexNSG<int8_t, size_t>::enable_delete();
  template NSGDLLEXPORT int IndexNSG<uint8_t, size_t>::enable_delete();
  template NSGDLLEXPORT int IndexNSG<float, size_t>::enable_delete();
  template NSGDLLEXPORT int IndexNSG<int8_t, std::string>::enable_delete();
  template NSGDLLEXPORT int IndexNSG<uint8_t, std::string>::enable_delete();
  template NSGDLLEXPORT int IndexNSG<float, std::string>::enable_delete();

  template NSGDLLEXPORT int IndexNSG<int8_t, int>::insert_point(
      const int8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const int tag);
  template NSGDLLEXPORT int IndexNSG<uint8_t, int>::insert_point(
      const uint8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const int tag);
  template NSGDLLEXPORT int IndexNSG<float, int>::insert_point(
      const float *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const int tag);
  template NSGDLLEXPORT int IndexNSG<int8_t, size_t>::insert_point(
      const int8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const size_t tag);
  template NSGDLLEXPORT int IndexNSG<uint8_t, size_t>::insert_point(
      const uint8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const size_t tag);
  template NSGDLLEXPORT int IndexNSG<float, size_t>::insert_point(
      const float *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph, const size_t tag);
  template NSGDLLEXPORT int IndexNSG<int8_t, std::string>::insert_point(
      const int8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph,
      const std::string tag);
  template NSGDLLEXPORT int IndexNSG<uint8_t, std::string>::insert_point(
      const uint8_t *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph,
      const std::string tag);
  template NSGDLLEXPORT int IndexNSG<float, std::string>::insert_point(
      const float *point, const Parameters &parameters,
      std::vector<Neighbor> &pool, std::vector<Neighbor> &tmp,
      tsl::robin_set<unsigned> &visited, vecNgh &cut_graph,
      const std::string tag);

#endif
}  // namespace NSG
