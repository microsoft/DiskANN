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

#include <partitionAndPQ.h>
#include <cassert>
#include "MemoryMapper.h"
#ifdef __NSG_WINDOWS__
#include <xmmintrin.h>
#endif

namespace NSG {
#define _CONTROL_NUM 100
#define MAX_START_POINTS 100

  template<typename T>
  IndexNSG<T>::IndexNSG(const size_t dimension, const size_t n, Metric m,
                        Index<T> *initializer, const size_t max_points)
      : Index<T>(dimension, n, m, max_points), initializer_{initializer} {
    locks = std::vector<std::mutex>(max_points_);
    width = 0;
  }

  template<>
  IndexNSG<float>::~IndexNSG() {
  }

  template<>
  IndexNSG<_s8>::~IndexNSG() {
  }

  template<>
  IndexNSG<_u8>::~IndexNSG() {
  }

  template<typename T>
  void IndexNSG<T>::Save(const char *filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    assert(final_graph_.size() == max_points_);

    long long total_gr_edges = 0;
    out.write((char *) &width, sizeof(unsigned));
    out.write((char *) &ep_, sizeof(unsigned));
    for (unsigned i = 0; i < nd_; i++) {
      unsigned GK = (unsigned) final_graph_[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) final_graph_[i].data(), GK * sizeof(unsigned));
      total_gr_edges += GK;
    }
    out.close();

    std::cout << "Avg degree: " << ((float) total_gr_edges) / ((float) nd_)
              << std::endl;
  }

  template<typename T>
  void IndexNSG<T>::Load(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    in.read((char *) &width, sizeof(unsigned));
    in.read((char *) &ep_, sizeof(unsigned));
    std::cout << "NSG -- width: " << width << ", ep_: " << ep_ << "\n";
    // width=100;
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
      final_graph_.emplace_back(tmp);

      if (nodes % 5000000 == 0)
        std::cout << "Loaded " << nodes << " nodes, and " << cc << " neighbors"
                  << std::endl;
    }
    cc /= nd_;
  }

  /* init_rnd_nn_graph():
   * num_points: Number of points in the dataset
   * k: max degree of the graph
   * mapping: initial vector of 10% of the points in the dataset
   */

  template<typename T>
  void IndexNSG<T>::Init_rnd_nn_graph(size_t num_points, unsigned k,
                                      std::vector<size_t> mapping) {
    k = (std::min)(k, (unsigned) 32);
    final_graph_.resize(max_points_);
    final_graph_.reserve(max_points_);
    if (!mapping.empty())
      num_points =
          mapping.size();  // num_points now = 10% of the points in dataset
    else {
      mapping.resize(num_points);
      std::iota(std::begin(mapping), std::end(mapping), 0);
    }

    std::cout << "Generating random graph.." << std::flush;
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
        std::set<unsigned> rand_set;
        while (rand_set.size() < k)
          rand_set.insert(dis(gen));

        final_graph_[mapping[i]].reserve(k);
        for (auto s : rand_set)
          final_graph_[mapping[i]].emplace_back(mapping[s]);
        final_graph_[mapping[i]].shrink_to_fit();
      }
    }
    ep_ = get_entry_point();
    std::cout << "done. Entry point set to " << ep_ << "." << std::endl;
  }

  /* iterate_to_fixed_point():
   * query : point whose neighbors to be found.
   * init_ids : ids of neighbors of navigating node.
   * retset : will contain the nearest neighbors of the query.
   * fullset : will contain all the node ids and distances from query that are
   * checked.
   * visited : will contain all the nodes that are visited during search.
   */
  template<typename T>
  void IndexNSG<T>::iterate_to_fixed_point(const T *                 query,
                                           const Parameters &        parameter,
                                           std::vector<unsigned> &   init_ids,
                                           std::vector<Neighbor> &   retset,
                                           std::vector<Neighbor> &   fullset,
                                           tsl::robin_set<unsigned> &visited) {
    const unsigned L = parameter.Get<unsigned>("L");

    /* put random L new ids into visited list and init_ids list */
    while (init_ids.size() < L) {
      unsigned id = (rand() * rand() * rand()) % nd_;
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
      assert(id < nd_);
      retset[l++] =
          Neighbor(id,
                   distance_->compare(data_ + dimension_ * (size_t) id, query,
                                      dimension_),
                   true);
    }

    /* sort retset based on distance of each point to query */
    std::sort(retset.begin(), retset.begin() + l);
    int k = 0;
    while (k < (int) l) {
      int nk = l;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        // prefetch final_graph_[n]
        unsigned *nbrs = final_graph_[n].data();   // nbrs: data of neighbors
        unsigned  nnbrs = final_graph_[n].size();  // nnbrs: number of neighbors
        NSG::prefetch_vector((const char *) nbrs, nnbrs * sizeof(unsigned));
        for (size_t m = 0; m < nnbrs; m++) {
          unsigned id = nbrs[m];  // id = neighbor
          if (m < (nnbrs - 1)) {
            unsigned id_next = nbrs[m + 1];  // id_next = next neighbor
            const T *vec_next1 =
                data_ +
                (size_t) id_next *
                    dimension_;  // vec_next1: data of next neighbor
            NSG::prefetch_vector((const char *) vec_next1,
                                 dimension_ * sizeof(T));
          }

          if (visited.find(id) == visited.end())
            visited.insert(id);  // if id is not in visited, add it to visited
          else
            continue;

          // compare distance of id with query
          float dist = distance_->compare(
              query, data_ + dimension_ * (size_t) id, (unsigned) dimension_);
          Neighbor nn(id, dist, true);
          fullset.emplace_back(nn);
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
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    assert(!fullset.empty());
  }

  template<typename T>
  void IndexNSG<T>::get_neighbors(const T *query, const Parameters &parameter,
                                  std::vector<Neighbor> &retset,
                                  std::vector<Neighbor> &fullset) {
    const unsigned           L = parameter.Get<unsigned>("L");
    tsl::robin_set<unsigned> visited(10 * L);
    get_neighbors(query, parameter, retset, fullset, visited);
  }

  template<typename T>
  void IndexNSG<T>::get_neighbors(const T *query, const Parameters &parameter,
                                  std::vector<Neighbor> &   retset,
                                  std::vector<Neighbor> &   fullset,
                                  tsl::robin_set<unsigned> &visited) {
    const unsigned L = parameter.Get<unsigned>("L");

    retset.resize(L + 1);
    std::vector<unsigned> init_ids;
    init_ids.reserve(L);

    std::vector<Neighbor> ep_neighbors;
    for (auto id : final_graph_[ep_]) {
      ep_neighbors.emplace_back(
          Neighbor(id,
                   distance_->compare(data_ + dimension_ * (size_t) id, query,
                                      dimension_),
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
     * init_ids contains all the ids of the neighbors of ep_, visited also
     * contains all the ids of neighbors of ep_
     */
    iterate_to_fixed_point(query, parameter, init_ids, retset, fullset,
                           visited);
  }

  template<typename T>
  int IndexNSG<T>::insert_point(const T *point, const Parameters &parameters,
                                std::vector<Neighbor> &   pool,
                                std::vector<Neighbor> &   tmp,
                                tsl::robin_set<unsigned> &visited,
                                vecNgh &                  cut_graph) {
    unsigned range = parameters.Get<unsigned>("R");
    if (!has_built) {
      std::cerr << "Can not insert with out building base index first."
                << std::endl;
      return -3;
    }

    LockGuard guard(incr_insert_lock);
    if (nd_ == max_points_) {
      std::cerr << "Can not insert more than " << max_points_ << " points."
                << std::endl;
      return -1;
    } else if (nd_ > max_points_) {
      std::cerr << "#points " << nd_
                << " greater than max_points_: " << max_points_ << std::endl;
      return -2;
    }

    auto offset_data = data_ + (size_t) dimension_ * nd_;
    memcpy((void *) offset_data, point, sizeof(float) * dimension_);

    // std::vector<Neighbor>    pool, tmp;
    // tsl::robin_set<unsigned> visited;
    // vecNgh                   cut_graph;

    pool.clear();
    tmp.clear();
    cut_graph.clear();
    visited.clear();

    get_neighbors(offset_data, parameters, tmp, pool, visited);
    sync_prune(nd_, pool, parameters, visited, cut_graph);

    assert(final_graph_.size() == max_points_);
    final_graph_[nd_].clear();
    final_graph_[nd_].reserve(range);
    assert(!cut_graph.empty());
    for (auto link : cut_graph) {
      final_graph_[nd_].emplace_back(link.id);
      assert(link.id >= 0 && link.id < nd_);
    }
    assert(final_graph_[nd_].size() <= range);
    InterInsertHierarchy(nd_, cut_graph, parameters);

    ++nd_;
    return 0;
  }

  /* reachable_bfs():
   * This function fills in the order to do bfs in bfs_order
   */
  template<typename T>
  void IndexNSG<T>::reachable_bfs(
      const unsigned                         start_node,
      std::vector<tsl::robin_set<unsigned>> &bfs_order, bool *visited) {
    auto &                    nsg = final_graph_;
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

  template<typename T>
  void IndexNSG<T>::populate_start_points_bfs(
      std::vector<unsigned> &start_points) {
    // populate a visited array
    // WARNING: DO NOT MAKE THIS A VECTOR
    bool *visited = new bool[nd_]();
    std::fill(visited, visited + nd_, false);
    std::map<unsigned, std::vector<tsl::robin_set<unsigned>>> bfs_orders;
    unsigned start_node = ep_;
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
      for (unsigned idx = start_node; idx < nd_; idx++) {
        if (!visited[idx]) {
          complete = false;
          start_node = idx;
          break;
        }
      }
    }
    start_points.emplace_back(ep_);
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
            size_t rand_offset = rand() * rand() * rand() % lvl.size();
            for (size_t j = 0; j < rand_offset; ++j)
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
  template<typename T>
  unsigned IndexNSG<T>::get_entry_point() {
    // allocate and init centroid
    float *center = new float[dimension_]();
    for (size_t j = 0; j < dimension_; j++)
      center[j] = 0;

    for (size_t i = 0; i < nd_; i++)
      for (size_t j = 0; j < dimension_; j++)
        center[j] += data_[i * dimension_ + j];

    for (size_t j = 0; j < dimension_; j++)
      center[j] /= nd_;

    // compute all to one distance
    float * distances = new float[nd_]();
#pragma omp parallel for schedule(static, 65536)
    for (int i = 0; i < nd_; i++) {  // GOPAL Changed from "size_t i" to "int i"
      // extract point and distance reference
      float &  dist = distances[i];
      const T *cur_vec = data_ + (i * (size_t) dimension_);
      dist = 0;
      float diff = 0;
      for (size_t j = 0; j < dimension_; j++) {
        diff = (center[j] - cur_vec[j]) * (center[j] - cur_vec[j]);
        dist += diff;
      }
    }
    // find imin
    size_t min_idx = 0;
    float  min_dist = distances[0];
    for (size_t i = 1; i < nd_; i++) {
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
   * node n to all the visited nodes obtained by running get_neighbors */
  template<typename T>
  void IndexNSG<T>::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                               const Parameters &        parameter,
                               tsl::robin_set<unsigned> &visited,
                               vecNgh &                  cut_graph_q) {
    unsigned range = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    float    alpha = parameter.Get<float>("alpha");

    width = (std::max)(width, range);

    /* check the neighbors of the query that are not part of visited,
     * check their distance to the query, and add it to pool.
     */
    if (!final_graph_[q].empty())
      for (auto id : final_graph_[q]) {
        if (visited.find(id) != visited.end())
          continue;
        float dist = distance_->compare(data_ + dimension_ * (size_t) q,
                                        data_ + dimension_ * (size_t) id,
                                        (unsigned) dimension_);
        pool.emplace_back(Neighbor(id, dist, true));
      }

    std::vector<Neighbor> result;
    /* sort the pool based on distance to query */
    std::sort(pool.begin(), pool.end());
    unsigned start = (pool[0].id == q) ? 1 : 0;
    /* put the first node in start. This will be nearest neighbor to q */
    result.emplace_back(pool[start]);

    while (result.size() < range && (++start) < pool.size() && start < maxc) {
      auto &p = pool[start];
      bool  occlude = false;
      for (unsigned t = 0; t < result.size(); t++) {
        if (p.id == result[t].id) {
          occlude = true;
          break;
        }
        /* check the distance of p from all nodes in result. If distance is less
         * than
         * distance of p from the query, then don't add it to result, otherwise
         * add
         */
        float djk = distance_->compare(
            data_ + dimension_ * (size_t) result[t].id,
            data_ + dimension_ * (size_t) p.id, (unsigned) dimension_);
        if (djk < p.distance /* dik */) {
          occlude = true;
          break;
        }
      }
      if (!occlude) {
        result.emplace_back(p);
      }
    }

    /* create a new array result2, which contains the the points according
     * to the parameter alpha, which talks about how aggressively to keep nodes
     * during pruning
     */
    if (alpha > 1.0 && !pool.empty() && result.size() < range) {
      std::vector<Neighbor> result2;
      unsigned              start2 = 0;
      if (pool[start2].id == q)
        start2++;
      result2.emplace_back(pool[start2]);
      while (result2.size() < range - result.size() &&
             (++start2) < pool.size() && start2 < maxc) {
        auto &p = pool[start2];
        bool  occlude = false;
        for (unsigned t = 0; t < result2.size(); t++) {
          if (p.id ==
              result2[t]
                  .id) {  // GOPAL. Changed from result[t].id to result2[t].id
            occlude = true;
            break;
          }
          float djk = distance_->compare(
              data_ + dimension_ * (size_t) result2[t].id,
              data_ + dimension_ * (size_t) p.id, (unsigned) dimension_);
          if (alpha * djk < p.distance /* dik */) {
            occlude = true;
            break;
          }
        }
        if (!occlude)
          result2.emplace_back(p);
      }
      /* add everything from result2 to result. This will lead to duplicates
       */
      for (unsigned i = 0; i < result2.size(); i++) {
        result.emplace_back(result2[i]);
      }
      /* convert it into a set, so that duplicates are all removed.
       */
      std::set<Neighbor> s(result.begin(), result.end());
      result.assign(s.begin(), s.end());
    }

    /* Add all the nodes in result into a variable called cut_graph_q
     * So this contains all the neighbors of q
     */
    cut_graph_q.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      assert(iter.id < nd_);
      cut_graph_q.emplace_back(SimpleNeighbor(iter.id, iter.distance));
    }
  }

  /* InterInsertHierarchy():
   * This function tries to add reverse links from all the visited nodes to the
   * current node n.
   */
  template<typename T>
  void IndexNSG<T>::InterInsertHierarchy(unsigned n, vecNgh &cut_graph_n,
                                         const Parameters &parameter) {
    float      alpha = parameter.Get<float>("alpha");
    const auto range = parameter.Get<float>("R");
    const auto src_pool = cut_graph_n;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      assert(des.id >= 0 && des.id < nd_);

      int dup = 0;
      /* des_pool contains the neighbors of the neighbors of n */
      auto &des_pool = final_graph_[des.id];

      std::vector<unsigned> graph_copy;
      {
        LockGuard guard(locks[des.id]);
        for (auto nn : des_pool) {
          assert(nn >= 0 && nn < nd_);
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
        /* at this point, graph_copy contains the neighbors of neighbor of n,
         * and also contains n
         */
      }  // des lock is released by this point

      assert(graph_copy.size() == 1 + range);
      {
        vecNgh temp_pool;
        for (auto node : graph_copy)
          /* temp_pool contains distance of each node in graph_copy, from
           * neighbor of n */
          temp_pool.emplace_back(SimpleNeighbor(
              node,
              distance_->compare(data_ + dimension_ * (size_t) node,
                                 data_ + dimension_ * (size_t) des.id,
                                 (unsigned) dimension_)));
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
            float djk = distance_->compare(data_ + dimension_ * (size_t) r.id,
                                           data_ + dimension_ * (size_t) p.id,
                                           (unsigned) dimension_);
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
              float djk = distance_->compare(data_ + dimension_ * (size_t) r.id,
                                             data_ + dimension_ * (size_t) p.id,
                                             (unsigned) dimension_);
              if (alpha * djk < p.distance /* dik */) {
                occlude = true;
                break;
              }
            }
            if (!occlude)
              result2.emplace_back(p);
          }
          for (auto r2 : result2) {
            for (auto r : result)
              assert(r.id != r2.id);
            result.emplace_back(r2);
          }
        }

        {
          LockGuard guard(locks[des.id]);
          assert(result.size() <= range);
          des_pool.clear();
          for (auto iter : result)
            des_pool.emplace_back(iter.id);
        }
      }

      /* At the end of this, des_pool contains all the correct neighbors of the
       * neighbors of the query node
       */
      assert(des_pool.size() <= range);
      for (auto iter : des_pool)
        assert(iter <= nd_);  // Equality occurs when called from insert_point
    }
  }

  /* LinkHierarchy():
   * The graph creation function.
   */
  template<typename T>
  void IndexNSG<T>::LinkHierarchy(Parameters &parameters) {
    //    The graph will be updated periodically in NUM_SYNCS batches
    // const uint32_t NUM_SYNCS = nd_ > 1 << 20 ? (nd_ / (128 * 1024 * 5))
    //                                         : 20 * (nd_ / (128 * 1024 * 5));
    uint32_t NUM_SYNCS = DIV_ROUND_UP(nd_, (128 * 1024));
    if (nd_ < (1 << 22))
      NUM_SYNCS = 4 * NUM_SYNCS;
    std::cout << "Number of syncs: " << NUM_SYNCS << std::endl;
    const uint32_t NUM_RNDS = parameters.Get<unsigned>(
        "num_rnds");  // num. of passes of overall algorithm
    const unsigned L = parameters.Get<unsigned>("L");  // Search list size
    const unsigned range =
        parameters.Get<unsigned>("R");                 // Max degree of graph
    const unsigned C = parameters.Get<unsigned>("C");  // Candidate list size
    float          last_round_alpha =
        parameters.Get<float>("alpha");  // Pruning parameter

    parameters.Set<unsigned>("L", L);
    parameters.Set<unsigned>("C", C);
    parameters.Set<float>("alpha", 1);  // alpha is hardcoded to 1 for the first
                                        // pass, for last pass alone we will use
                                        // the specified value

    /* rand_perm is a vector that is initialized to the entire graph */
    std::vector<unsigned> rand_perm;
    for (size_t i = 0; i < nd_; i++) {
      rand_perm.emplace_back(i);
    }

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    Init_rnd_nn_graph(nd_, range);

    assert(final_graph_.size() == max_points_);
    // std::vector<std::mutex> locks(max_points_);
    auto cut_graph_ = new vecNgh[nd_];

    for (uint32_t rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
      // Shuffle the dataset
      std::random_shuffle(rand_perm.begin(), rand_perm.end());
      unsigned progress_counter = 0;

      size_t round_size = DIV_ROUND_UP(nd_, NUM_SYNCS);  // size of each batch

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
        size_t end_id = (std::min)(nd_, (sync_num + 1) * round_size);
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

            /* get nearest neighbors of n in tmp. pool contains all the points
             * that were
             * checked along with their distance from n. visited contains all
             * the points
             * visited, just the ids
             */
            get_neighbors(data_ + (size_t) dimension_ * n, parameters, tmp,
                          pool, visited);
            /* sync_prune will check the pool[] list, and remove some of the
             * points and
             * create a cut_graph_ array, which contains final neighbors for
             * point n
             */
            sync_prune(n, pool, parameters, visited, cut_graph_[n]);
          }
        }

#pragma omp parallel for schedule(static, PAR_BLOCK_SZ)

        for (int node = start_id; node < end_id; ++node) {
          // clear all the neighbors of final_graph_[node]
          final_graph_[node].clear();
          final_graph_[node].reserve(range);
          assert(!cut_graph_[node].empty());
          for (auto link : cut_graph_[node]) {
            final_graph_[node].emplace_back(link.id);
            assert(link.id >= 0 && link.id < nd_);
          }
          assert(final_graph_[node].size() <= range);
        }

#pragma omp parallel for schedule(static, PAR_BLOCK_SZ)
        for (int n = start_id; n < end_id; ++n) {
          InterInsertHierarchy(n, cut_graph_[n], parameters);
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

  template<typename T>
  void IndexNSG<T>::SetData(const T* data) {
  data_ = data;
  }

  template<typename T>
  void IndexNSG<T>::BuildRandomHierarchical(const T *   data,
                                            Parameters &parameters) {
    unsigned range = parameters.Get<unsigned>("R");
    data_ = data;

    LinkHierarchy(parameters);  // Primary func for creating nsg graph

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < nd_; i++) {
      auto &pool = final_graph_[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    std::cout << "Degree: max:" << max
              << "  avg:" << (float) total / (float) nd_ << "  min:" << min
              << "  count(deg<2):" << cnt << "\n";

    width = (std::max)((unsigned) max, width);
    has_built = true;
  }

  template<typename T>
  std::pair<int, int> IndexNSG<T>::BeamSearch(
      const T *query, const T *x, const size_t K, const Parameters &parameters,
      unsigned *indices, int beam_width, std::vector<unsigned> &start_points) {
    const unsigned int L = parameters.Get<unsigned>("L_search");
    return BeamSearch(query, x, K, L, indices, beam_width, start_points);
  }

  template<typename T>
  std::pair<int, int> IndexNSG<T>::BeamSearch(
      const T *query, const T *x, const size_t K, const unsigned L,
      unsigned *indices, int beam_width, std::vector<unsigned> &start_points) {
    data_ = x;

    std::vector<unsigned> init_ids;
    // boost::dynamic_bitset<> flags{nd_, 0};
    tsl::robin_set<unsigned> visited(10 * L);
    //    unsigned                 tmp_l = 0;
    // ignore default init; use start_points for init
    if (start_points.size() == 0)
      start_points.emplace_back(ep_);

    /* ep_neighbors contains all the neighbors of navigating node, and
     * their distance from the query node
     */
    std::vector<Neighbor> ep_neighbors;
    for (auto curpt : start_points)
      for (auto id : final_graph_[curpt]) {
        // std::cout << "cmp: query <-> " << id << "\n";
        ep_neighbors.emplace_back(
            Neighbor(id,
                     distance_->compare(data_ + dimension_ * (size_t) id, query,
                                        dimension_),
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
      unsigned id = (rand() * rand() * rand()) % nd_;
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
                           distance_->compare(data_ + dimension_ * init_ids[i],
                                              query, (unsigned) dimension_),
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

    /* Maximum L rounds take place to get nearest neighbor.  */
    while (k < (int) L) {
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
        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
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
          const T *vec1 = data_ + dimension_ * id_next;
          NSG::prefetch_vector((const char *) vec1, dimension_ * sizeof(T));
        }

        cmps++;
        unsigned id = *iter;
        /* compare distance of each neighbor with that of query. If the distance
         * is less than
         * largest distance in retset, add to retset and set flag to true
         */
        // std::cout << "cmp: query <-> " << id << "\n";
        float dist = distance_->compare(data_ + dimension_ * id, query,
                                        (unsigned) dimension_);
        if (dist >= retset[L - 1].distance)
          continue;
        Neighbor nn(id, dist, true);

        // Return position in sorted list where nn inserted.
        int r = InsertIntoPool(retset.data(), L, nn);
        if (r < nk)
          nk = r;  // nk logs the best position in the retset that was updated
                   // due to neighbors of n.
      }
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }
    for (size_t i = 0; i < K; i++) {
      indices[i] = retset[i].id;
    }
    return std::make_pair(hops, cmps);
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
        unsigned id = (rand() * rand() * rand()) % nd_;
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
      auto filter_visited_and_mark_visited = [this, &visited](
          const unsigned *nbrs, const unsigned nnbrs, unsigned *out_nbrs,
          unsigned &out_nnbrs) {
        out_nnbrs = 0;
        for (unsigned i = 0; i < nnbrs; i++) {
          unsigned idx = nbrs[i];
          bool     is_not_visited = (visited.find(idx) == visited.end());
          if (is_not_visited) {
            const _s8 *point_coords = this->data_ + this->dimension_ * idx;
            // also issue prefetches to bring vector into L2 --> prefetch easier
            // from L2->L1 later
            NSG::prefetch_vector_l2((const char *) point_coords,
                                    this->dimension_ * sizeof(_s8));
            out_nbrs[out_nnbrs++] = idx;
          }
        }
        visited.insert(out_nbrs, out_nbrs + out_nnbrs);
      };

      auto batch_compute_distances = [this](
          const _s8 *query, const unsigned *nbrs, const unsigned nnbrs,
          float *out_dists) {
        NSG::prefetch_vector(
            (const char *) (this->data_ + (this->dimension_ * nbrs[0])),
            this->dimension_ * sizeof(_s8));
        for (unsigned i = 0; i < nnbrs; i++) {
          // prefetch next coordinates from L2 -> L1 (2 cycles for L2 hit +
          // 128-dim vectors)
          if (i < nnbrs - 1) {
            NSG::prefetch_vector(
                (const char *) (this->data_ + (this->dimension_ * nbrs[i + 1])),
                this->dimension_ * sizeof(_s8));
          }
          const _s8 *point_coords = this->data_ + (this->dimension_ * nbrs[i]);
          out_dists[i] =
              this->distance_->compare(query, point_coords, this->dimension_);
        }
      };

      // compare distance of all points in init_ids with query, and put the id
      // with distance
      // in retset
      //
      unsigned l = 0;
      for (auto id : init_ids) {
        assert(id < nd_);
        retset[l++] =
            Neighbor(id,
                     distance_->compare(data_ + dimension_ * (size_t) id, query,
                                        dimension_),
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

        // prefetch final_graph_[n]
        unsigned *nbrs = final_graph_[n].data();   // nbrs: data of neighbors
        unsigned  nnbrs = final_graph_[n].size();  // nnbrs: number of neighbors

        // prefetch nnbrs
        NSG::prefetch_vector((const char *) nbrs, nnbrs * sizeof(unsigned));

        // filter nbrs into compare_ids
        filter_visited_and_mark_visited(nbrs, nnbrs, unvisited_nbrs,
                                        unvisited_nnbrs);

        // batch prefetch all unvisited vectors into L2
        for (_u64 i = 0; i < unvisited_nnbrs; i++) {
          const _s8 *vec = this->data_ + (unvisited_nbrs[i] * this->dimension_);
          NSG::prefetch_vector_l2((const char *) vec,
                                  this->dimension_ * sizeof(_s8));
        }

        // prefetch query
        NSG::prefetch_vector((const char *) query,
                             this->dimension_ * sizeof(_s8));

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

  template<typename T>
  void IndexNSG<T>::save_disk_opt_graph(const char *diskopt_path) {
    const _u64 SECTOR_LEN = 4096;
    std::cout << "Embedding node coords with its nhood" << std::endl;
    size_t npts_u64 = nd_, ndims_u64 = dimension_;

    // amount to write in one shot
    _u64 write_blk_size = 256l * 1024l * 1024l;

    // create cached reader + writer
    cached_ofstream nsg_writer(diskopt_path, write_blk_size);

    // compute
    _u64 max_node_len, nnodes_per_sector;
    max_node_len =
        (((_u64) width + 1) * sizeof(unsigned)) + (ndims_u64 * sizeof(T));
    nnodes_per_sector = SECTOR_LEN / max_node_len;

    std::cout << "medoid: " << ep_ << "B\n";
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
    *(_u64 *) (sector_buf + sizeof(_u64)) = (_u64) ep_;
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
        nnbrs = final_graph_[cur_node_id].size();

        // sanity checks on nnbrs
        assert(nnbrs > 0);
        assert(nnbrs <= width);

        // set cur node's nhood
        nhood_buf = final_graph_[cur_node_id].data();

        // write coords of node first
        const T *node_coords = data_ + ((_u64) ndims_u64 * cur_node_id);
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
