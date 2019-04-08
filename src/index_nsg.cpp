#include "efanna2e/index_nsg.h"

#include <omp.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iterator>
#include <map>
#include <set>
#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"
#include "tsl/robin_set.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <cassert>

namespace NSG {
#define _CONTROL_NUM 100
#define MAX_START_POINTS 100

  IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                     Index *initializer)
      : Index(dimension, n, m), initializer_{initializer} {
    is_inner = new bool[nd_];
    for (size_t i = 0; i < nd_; i++)
      is_inner[i] = false;
  }

  IndexNSG::~IndexNSG() {
    delete[] is_inner;
  }

  unsigned IndexNSG::get_start_node() const {
    return ep_;
  }

  void IndexNSG::set_start_node(const unsigned s) {
    ep_ = s;
  }

  void IndexNSG::Save(const char *filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    assert(final_graph_.size() == nd_);

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

  void IndexNSG::SaveSmallIndex(const char *           filename,
                                std::vector<unsigned> &picked) {
    if (picked.size() != this->nd_)
      std::cerr << "In small index save: picked.size() != this->nd_"
                << std::endl;
    if (picked.size() != final_graph_.size())
      std::cerr << "In small index save: picked.size() != final_graph_.size()"
                << std::endl;

    std::ofstream out(std::string(filename) + ".small",
                      std::ios::binary | std::ios::out);

    out.write((char *) &(this->nd_), sizeof(size_t));
    out.write((char *) &(this->ep_), sizeof(unsigned));

    unsigned picked_size = picked.size();
    out.write((char *) &picked_size, sizeof(unsigned));
    out.write((char *) picked.data(), picked_size * sizeof(unsigned));

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
  }

  void IndexNSG::LoadSmallIndex(const char *           filename,
                                std::vector<unsigned> &picked) {
    std::ifstream in(std::string(filename) + ".small", std::ios::binary);
    if (in.fail()) {
      std::cerr << "Small Index file: " << std::string(filename)
                << ".small not found." << std::endl;
      exit(-1);
    }

    in.read((char *) &(this->nd_), sizeof(size_t));
    in.read((char *) &(this->ep_), sizeof(unsigned));

    unsigned picked_size;
    in.read((char *) &picked_size, sizeof(unsigned));
    assert(picked.size() == 0);
    picked.resize(picked_size);
    in.read((char *) picked.data(), picked_size * sizeof(unsigned));

    in.read((char *) &width, sizeof(unsigned));
    in.read((char *) &ep_, sizeof(unsigned));
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
      final_graph_.push_back(tmp);

      if (nodes % 5000000 == 0)
        std::cout << "Loaded " << nodes << " nodes, and " << cc << " neighbors"
                  << std::endl;
    }
    cc /= nd_;
    // std::cout<<cc<<std::endl;
  }

  void IndexNSG::Load(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    in.read((char *) &width, sizeof(unsigned));
    in.read((char *) &ep_, sizeof(unsigned));
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
      final_graph_.push_back(tmp);

      if (nodes % 5000000 == 0)
        std::cout << "Loaded " << nodes << " nodes, and " << cc << " neighbors"
                  << std::endl;
    }
    cc /= nd_;
    // std::cout<<cc<<std::endl;
  }

  /*void IndexNSG::Load_nn_graph(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    unsigned      k;
    in.read((char *) &k, sizeof(unsigned));
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t             fsize = (size_t) ss;
    size_t             num = (unsigned) (fsize / (k + 1) / 4);
    in.seekg(0, std::ios::beg);

    final_graph_.resize(num);
    final_graph_.reserve(num);
    unsigned kk = (k + 3) / 4 * 4;
    for (size_t i = 0; i < num; i++) {
      in.seekg(4, std::ios::cur);
      final_graph_[i].resize(k);
      final_graph_[i].reserve(kk);
      in.read((char *) final_graph_[i].data(), k * sizeof(unsigned));
    }
    in.close();
    ep_ = 0;
    std::cout << "Loaded EFANNA graph. Set ep_ to 0" << std::endl;
  }
*/

  void IndexNSG::Load_nn_graph(const char *filename) {
    int fd = open(filename, O_RDONLY);
    if (fd <= 0) {
      std::cerr << "Data file " << filename << " not found. Program will exit."
                << std::endl;
      exit(-1);
    }
    struct stat sb;
    if (fstat(fd, &sb) != 0) {
      std::cerr << "File load error. CHECK!" << std::endl;
      exit(-1);
    }

    off_t fileSize = sb.st_size;
    std::cout << fileSize << std::endl;
    char *buf = (char *) mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);

    unsigned k;
    std::memcpy(&k, buf, sizeof(unsigned));
    size_t num = (fileSize) / ((k + 1) * 4);

    std::cout << "k is and num is " << k << " " << num << std::endl;
    final_graph_.resize(num);
    final_graph_.reserve(num);

    unsigned kk = (k + 3) / 4 * 4;
#pragma omp  parallel for schedule(static, 65536)
    for (size_t i = 0; i < num; i++) {
      final_graph_[i].resize(k);
      final_graph_[i].reserve(kk);
      char *reader = buf + (i * (k + 1) * sizeof(unsigned));
      std::memcpy(final_graph_[i].data(), reader + sizeof(unsigned),
                  k * sizeof(unsigned));
    }
    if (munmap(buf, fileSize) != 0)
      std::cerr << "ERROR unmapping. CHECK!" << std::endl;
    close(fd);
    ep_ = 0;
    std::cout << "Loaded EFANNA graph. Set ep_ to 0" << std::endl;
  }

  void IndexNSG::Init_rnd_nn_graph(size_t num_points, unsigned k,
                                   std::vector<size_t> mapping) {
    size_t num = num_points;
    final_graph_.resize(num);
    final_graph_.reserve(num);
    if (!mapping.empty())
      num_points = mapping.size();
    std::cout << "k is and num is " << k << " " << num_points << std::endl;

    size_t PAR_BLOCK_SZ = (1 << 16);
    size_t nblocks = num_points / PAR_BLOCK_SZ;
    if (num_points % PAR_BLOCK_SZ != 0)
      nblocks++;

#pragma omp parallel for schedule(static, 1)
    for (size_t block = 0; block < nblocks; ++block) {
      std::random_device                 rd;
      size_t                             x = rd();
      std::mt19937                       gen(x);
      std::uniform_int_distribution<int> dis(0, num_points - 1);

      for (size_t i = block * PAR_BLOCK_SZ;
           i < (block + 1) * PAR_BLOCK_SZ && i < num_points; i++) {
        final_graph_[mapping[i]].reserve(k);
        for (unsigned j = 0; j < k; j++)
          final_graph_[mapping[i]].push_back(mapping[dis(gen)]);
        final_graph_[mapping[i]].shrink_to_fit();

        if (i % 1000000 == 0)
          std::cout << "Generated random neighbors for point " << i
                    << std::endl;
      }
    }
    //    ep_ = 0;
    std::cout << "Loaded Random graph. Set ep_ to 0" << std::endl;
  }

  void IndexNSG::iterate_to_fixed_point(const float *             query,
                                        const Parameters &        parameter,
                                        std::vector<unsigned> &   init_ids,
                                        std::vector<Neighbor> &   retset,
                                        std::vector<Neighbor> &   fullset,
                                        tsl::robin_set<unsigned> &visited) {
    const unsigned L = parameter.Get<unsigned>("L");

    while (init_ids.size() < L) {
      unsigned id = (rand() * rand() * rand()) % nd_;
      if (visited.find(id) != visited.end())
        continue;
      else
        visited.insert(id);
      init_ids.push_back(id);
    }

    unsigned l = 0;
    for (auto id : init_ids) {
      assert(id < nd_);
      retset[l++] =
          Neighbor(id,
                   distance_->compare(data_ + dimension_ * (size_t) id, query,
                                      dimension_),
                   true);
    }

    std::sort(retset.begin(), retset.begin() + l);
    int k = 0;
    while (k < (int) l) {
      int nk = l;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        for (auto id : final_graph_[n]) {
          if (visited.find(id) == visited.end())
            visited.insert(id);
          else
            continue;

          float dist = distance_->compare(
              query, data_ + dimension_ * (size_t) id, (unsigned) dimension_);
          Neighbor nn(id, dist, true);
          fullset.push_back(nn);
          if (dist >= retset[l - 1].distance)
            continue;
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

  void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                               std::vector<Neighbor> &retset,
                               std::vector<Neighbor> &fullset) {
    const unsigned           L = parameter.Get<unsigned>("L");
    tsl::robin_set<unsigned> visited(10 * L);
    get_neighbors(query, parameter, retset, fullset, visited);
  }

  void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                               std::vector<Neighbor> &   retset,
                               std::vector<Neighbor> &   fullset,
                               tsl::robin_set<unsigned> &visited) {
    const unsigned L = parameter.Get<unsigned>("L");

    retset.resize(L + 1);
    std::vector<unsigned> init_ids;
    init_ids.reserve(L);
    // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

    for (auto iter : final_graph_[ep_]) {
      if (init_ids.size() >= L)
        break;
      init_ids.push_back(iter);
      visited.insert(iter);
      //      fullset.insert(iter);
    }
    iterate_to_fixed_point(query, parameter, init_ids, retset, fullset,
                           visited);
  }

  void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                               std::vector<Neighbor> &      retset,
                               std::vector<Neighbor> &      fullset,
                               tsl::robin_set<unsigned> &   visited,
                               const std::vector<unsigned> &start_points) {
    const unsigned L = parameter.Get<unsigned>("L");

    retset.resize(L + 1);
    std::vector<unsigned> init_ids;
    init_ids.reserve(L);
    // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

    if (start_points.size() == 0) {
      std::cerr << "Start point set can not be empty for get_neighbor."
                << std::endl;
      exit(-1);
    }
    for (auto iter : start_points) {
      if (init_ids.size() >= L)
        break;
      init_ids.push_back(iter);
      visited.insert(iter);
    }
    for (auto s : start_points) {
      if (init_ids.size() == L)
        break;
      for (auto iter : final_graph_[s]) {
        if (init_ids.size() == L)
          break;
        init_ids.push_back(iter);
        visited.insert(iter);
      }
    }
    iterate_to_fixed_point(query, parameter, init_ids, retset, fullset,
                           visited);
  }

  void IndexNSG::reachable_bfs(const unsigned start_node,
                               std::vector<tsl::robin_set<unsigned>> &bfs_order,
                               bool *                                 visited) {
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
        max_deg = std::max(max_deg, nsg[id].size());
        min_deg = std::min(min_deg, nsg[id].size());
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
      bfs_order.push_back(add);

      // swap cur_level and prev_level, increment level
      prev_level->clear();
      std::swap(prev_level, cur_level);
      level++;
    }

    // cleanup
    delete cur_level;
    delete prev_level;
  }

  void IndexNSG::populate_start_points_bfs(
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
    start_points.push_back(ep_);
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
              start_points.push_back(*iter);
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

  void IndexNSG::init_graph(const Parameters &parameters) {
    float *center = new float[dimension_];
    for (unsigned j = 0; j < dimension_; j++)
      center[j] = 0;
    for (unsigned i = 0; i < nd_; i++) {
      for (unsigned d = 0; d < dimension_; d++) {
        center[d] += data_[i * dimension_ + d];
      }
    }
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] /= nd_;
    }

    std::vector<Neighbor> tmp, pool;
    ep_ = rand() % nd_;  // random initialize navigating point
    get_neighbors(center, parameters, tmp, pool);
    ep_ = tmp[0].id;
  }

  void IndexNSG::init_graph_outside(const float *data) {
    data_ = data;
    float *center = new float[dimension_]();
    for (size_t j = 0; j < dimension_; j++)
      center[j] = 0;
    for (size_t i = 0; i < nd_; i++)
      for (size_t j = 0; j < dimension_; j++)
        center[j] += data_[i * dimension_ + j];
    for (size_t j = 0; j < dimension_; j++)
      center[j] /= nd_;

    // compute all to one distance
    auto    distances = new float[nd_]();
#pragma omp parallel for schedule(static, 65536)
    for (size_t i = 0; i < nd_; i++) {
      // extract point and distance reference
      float &      dist = distances[i];
      const float *cur_vec = data_ + (i * (size_t) dimension_);
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
    ep_ = min_idx;
    delete[] distances;
    std::cout << "Medoid index = " << min_idx << std::endl;
  }

  void IndexNSG::init_graph_bf(const Parameters &parameters) {
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
    for (size_t i = 0; i < nd_; i++) {
      // extract point and distance reference
      float &      dist = distances[i];
      const float *cur_vec = data_ + (i * (size_t) dimension_);
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
    ep_ = min_idx;

    std::cout << "Medoid index = " << min_idx << std::endl;
    delete[] distances;
    delete[] center;
  }

  void IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                            const Parameters &        parameter,
                            tsl::robin_set<unsigned> &visited,
                            vecNgh *                  cut_graph_) {
    unsigned range = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    float    alpha = parameter.Get<float>("alpha");

    if (is_inner[q])
      range = 4 * range;
    width = range;

    if (!final_graph_[q].empty())
      for (auto id : final_graph_[q]) {
        if (visited.find(id) != visited.end())
          continue;
        float dist = distance_->compare(data_ + dimension_ * (size_t) q,
                                        data_ + dimension_ * (size_t) id,
                                        (unsigned) dimension_);
        pool.push_back(Neighbor(id, dist, true));
      }

    //    std::cout << "here5" << std::flush;
    std::vector<Neighbor> result;
    std::sort(pool.begin(), pool.end());
    unsigned start = 0;
    if (pool[start].id == q)
      start++;
    result.push_back(pool[start]);

    //    std::cout << "here4" << std::flush;
    while (result.size() < range && (++start) < pool.size() && start < maxc) {
      auto &p = pool[start];
      bool  occlude = false;
      for (unsigned t = 0; t < result.size(); t++) {
        if (p.id == result[t].id) {
          occlude = true;
          break;
        }
        float djk = distance_->compare(
            data_ + dimension_ * (size_t) result[t].id,
            data_ + dimension_ * (size_t) p.id, (unsigned) dimension_);
        if (djk < p.distance /* dik */) {
          occlude = true;
          break;
        }
      }
      if (!occlude)
        result.push_back(p);
    }

    //    std::cout << "here3" << std::flush;
    if (alpha > 1.0) {
      if (result.size() < range) {
        std::vector<Neighbor> result2;
        unsigned              start2 = 0;
        if (pool[start2].id == q)
          start2++;
        result2.push_back(pool[start2]);
        while (result2.size() < range - result.size() &&
               (++start2) < pool.size() && start2 < maxc) {
          auto &p = pool[start2];
          bool  occlude = false;
          for (unsigned t = 0; t < result2.size(); t++) {
            if (p.id == result2[t].id) {
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
            result2.push_back(p);
        }
        for (unsigned i = 0; i < result2.size(); i++) {
          result.push_back(result2[i]);
        }
        std::set<Neighbor> s(result.begin(), result.end());
        result.assign(s.begin(), s.end());
      }
    }

    //    std::cout << "here2" << std::flush;
    cut_graph_[q].clear();
    for (auto iter : result)
      cut_graph_[q].push_back(SimpleNeighbor(iter.id, iter.distance));

    //    std::cout << "here1" << std::flush;
    for (auto iter : cut_graph_[q])
      assert(iter.id < nd_);
  }

  void IndexNSG::InterInsertHierarchy(unsigned                 n,
                                      std::vector<std::mutex> &locks,
                                      const Parameters &       parameter) {
    float       alpha = parameter.Get<float>("alpha");
    auto        range = parameter.Get<float>("R");
    const auto  base_range = parameter.Get<float>("R");
    const auto &src_pool = final_graph_[n];

    for (auto des : src_pool) {
      if (is_inner[des])
        range = 4 * base_range;
      auto &                des_pool = final_graph_[des];
      std::vector<unsigned> graph_copy;
      int                   dup = 0;
      {
        LockGuard guard(locks[des]);
        for (auto nn : des_pool) {
          if (n == nn) {
            dup = 1;
            break;
          }
          graph_copy.push_back(nn);
        }
      }
      if (dup)
        continue;
      else
        graph_copy.push_back(n);

      if (graph_copy.size() > range) {
        vecNgh temp_pool;
        for (auto node : graph_copy)
          temp_pool.push_back(SimpleNeighbor(
              node,
              distance_->compare(data_ + dimension_ * (size_t) node,
                                 data_ + dimension_ * (size_t) des,
                                 (unsigned) dimension_)));

        std::vector<SimpleNeighbor> result;
        unsigned                    start = 0;
        std::sort(temp_pool.begin(), temp_pool.end());
        result.push_back(temp_pool[start]);
        while (result.size() < range && (++start) < temp_pool.size()) {
          auto &p = temp_pool[start];
          bool  occlude = false;
          for (unsigned t = 0; t < result.size(); t++) {
            if (p.id == result[t].id) {
              occlude = true;
              break;
            }
            float djk = distance_->compare(
                data_ + dimension_ * (size_t) result[t].id,
                data_ + dimension_ * (size_t) p.id, (unsigned) dimension_);
            if (djk < p.distance /* dik */) {
              occlude = true;
              break;
            }
          }
          if (!occlude)
            result.push_back(p);
        }

        if (result.size() < range) {
          std::vector<SimpleNeighbor> result2;
          unsigned                    start2 = 0;
          //		if (pool[start2].id == q)
          //			start2++;
          result2.push_back(temp_pool[start2]);
          while (result2.size() < range - result.size() &&
                 (++start2) < temp_pool.size()) {
            auto &p = temp_pool[start2];
            bool  occlude = false;
            for (unsigned t = 0; t < result2.size(); t++) {
              if (p.id == result2[t].id) {
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
              result2.push_back(p);
          }
          for (unsigned i = 0; i < result2.size(); i++) {
            result.push_back(result2[i]);
          }
          std::set<SimpleNeighbor> s(result.begin(), result.end());
          result.assign(s.begin(), s.end());
          //		std::sort(result.begin(), result.end());
          //		result.erase(unique(result.begin(), result.end()),
          // result.end());
        }
        {
          LockGuard guard(locks[des]);
          des_pool.clear();
          for (auto iter : result)
            des_pool.push_back(iter.id);
        }
      } else {
        LockGuard guard(locks[des]);
        des_pool.push_back(n);
      }
      for (auto iter : des_pool)
        assert(iter < nd_);
    }
  }

  void IndexNSG::LinkHierarchy(Parameters &parameters) {
    const float    p_val = parameters.Get<float>("p_val");
    const uint32_t NUM_SYNCS = parameters.Get<unsigned>("num_syncs");
    const uint32_t NUM_RNDS = parameters.Get<unsigned>("num_rnds");
    const uint32_t NUM_HIER = parameters.Get<unsigned>("num_hier");
    const unsigned L = parameters.Get<unsigned>("L");
    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned C = parameters.Get<unsigned>("C");
    const unsigned innerL = parameters.Get<unsigned>("innerL");
    const unsigned innerC = parameters.Get<unsigned>("innerC");
    parameters.Set<unsigned>("L", innerL);
    parameters.Set<unsigned>("C", innerC);

    float last_alpha = parameters.Get<float>("alpha");
    parameters.Set<float>("alpha", 1);

    auto size_hierarchy = new size_t[NUM_HIER];
    auto hierarchy_vertices = new std::vector<size_t>[NUM_HIER];

    hierarchy_vertices[NUM_HIER - 1].resize(nd_);
    std::iota(std::begin(hierarchy_vertices[NUM_HIER - 1]),
              std::end(hierarchy_vertices[NUM_HIER - 1]), 0);
    size_hierarchy[NUM_HIER - 1] = hierarchy_vertices[NUM_HIER - 1].size();

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    std::cout << "Entry point is " << ep_ << std::endl;
    for (int h = NUM_HIER - 2; h >= 0; h--) {
      hierarchy_vertices[h].push_back(ep_);
      for (size_t i = 0; i < hierarchy_vertices[h + 1].size(); i++) {
        float candidate = dis(gen);
        if (candidate < p_val && hierarchy_vertices[h + 1][i] != ep_)
          hierarchy_vertices[h].push_back(hierarchy_vertices[h + 1][i]);
        is_inner[hierarchy_vertices[h + 1][i]] = true;
      }
      size_hierarchy[h] = hierarchy_vertices[h].size();
      std::cout << "Generated random hierarchy level " << h << " of size "
                << size_hierarchy[h] << std::endl;
    }

    bool is_nsg = parameters.Get<bool>("is_nsg");
    bool is_rnd_nn = parameters.Get<bool>("is_rnd_nn");
    if ((!is_nsg) && is_rnd_nn) {
      std::cout << "Initializing random inner graph over " << size_hierarchy[0]
                << " vertices " << std::endl;
      Init_rnd_nn_graph(nd_, range, hierarchy_vertices[0]);
    }

    assert(final_graph_.size() == nd_);
    std::vector<std::mutex> locks(nd_);
    auto                    cut_graph_ = new vecNgh[nd_];

    for (int h = 0; h < NUM_HIER; h++) {
      std::cout << "Processing level " << h << " with " << size_hierarchy[h]
                << " vertices " << std::endl;
      if (h == NUM_HIER - 1) {
        parameters.Set<unsigned>("L", L);
        parameters.Set<unsigned>("C", C);
      }

      for (int rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
        if (rnd_no == NUM_RNDS - 1)
          parameters.Set<float>("alpha", last_alpha);
        //      std::cout << "L R and C " << L << " " << range << " " << C <<
        //      std::endl;
        size_t round_size = DIV_ROUND_UP(size_hierarchy[h], NUM_SYNCS);

        for (uint32_t rnd_no = 0; rnd_no < NUM_SYNCS; rnd_no++) {
          std::string graph_file = "/mnt/rakri/nsg-vis/random2dcube" +
                                   std::to_string(rnd_no) + ".adj";

          std::ofstream out(graph_file.c_str(), std::ofstream::binary);

          for (uint32_t i = 0; i < nd_; ++i) {
            for (uint32_t j = 0; j < final_graph_[i].size(); ++j) {
              uint32_t dest = final_graph_[i][j];
              out.write((char *) &i, sizeof(uint32_t));
              out.write((char *) &dest, sizeof(uint32_t));
            }
          }
          out.close();

          size_t start_id = rnd_no * round_size;
          size_t end_id =
              std::min(size_hierarchy[h], (rnd_no + 1) * round_size);
          size_t round_size = end_id - start_id;
          std::cout << "Round start: " << start_id << "  end: " << end_id
                    << std::endl;

          size_t PAR_BLOCK_SZ =
              round_size > 1 << 20 ? 1 << 12 : (round_size + 64) / 64;
          size_t nblocks = DIV_ROUND_UP(round_size, PAR_BLOCK_SZ);

#pragma omp parallel for schedule(static, 1)
          for (size_t block = 0; block < nblocks; ++block) {
            std::vector<Neighbor>    pool, tmp;
            tsl::robin_set<unsigned> visited;

            for (size_t n = start_id + block * PAR_BLOCK_SZ;
                 n <
                 start_id + std::min(round_size, (block + 1) * PAR_BLOCK_SZ);
                 ++n) {
              pool.clear();
              tmp.clear();
              visited.clear();
              //            if (h == NUM_HIER - 1)
              //              std::cout << "getting nbrs for " <<
              //              hierarchy_vertices[h][n]
              //                        << std::endl;
              get_neighbors(
                  data_ + (size_t) dimension_ * hierarchy_vertices[h][n],
                  parameters, tmp, pool, visited);
              //          if (h == NUM_HIER - 1)
              //           std::cout << "id: Pool Size, Retset Size, Visited
              //           Size:"
              //                    << hierarchy_vertices[h][n] << ": " <<
              //                    pool.size()
              //                   << " " << tmp.size() << " " << visited.size()
              //                  << std::endl;
              sync_prune(hierarchy_vertices[h][n], pool, parameters, visited,
                         cut_graph_);
            }
          }
          std::cout << "sync_prune completed for (level: " << h
                    << ", round: " << rnd_no << ")" << std::endl;

#pragma omp parallel for schedule(static, PAR_BLOCK_SZ)
          for (unsigned n = start_id; n < end_id; ++n) {
            auto node = hierarchy_vertices[h][n];
            final_graph_[node].clear();
            final_graph_[node].reserve(range);
            assert(!cut_graph_[node].empty());
            for (auto link : cut_graph_[node])
              final_graph_[node].push_back(link.id);
            cut_graph_[node].clear();
            cut_graph_[node].shrink_to_fit();
            assert(final_graph_[node].size() <= range);
          }

#pragma omp parallel for schedule(static, PAR_BLOCK_SZ)
          for (unsigned n = start_id; n < end_id; ++n) {
            InterInsertHierarchy(hierarchy_vertices[h][n], locks, parameters);
          }
          std::cout << "InterInsert completed for (level: " << h
                    << ", round: " << rnd_no << ")" << std::endl;
        }
      }
    }

    delete[] cut_graph_;
    delete[] size_hierarchy;
    delete[] hierarchy_vertices;
  }

  void IndexNSG::Build(size_t n, const float *data,
                       const Parameters &parameters) {
  }

  void IndexNSG::BuildFromER(size_t n, size_t nr, const float *data,
                             const Parameters &parameters) {
  }

  void IndexNSG::BuildFromSmall(size_t n, const float *data,
                                const Parameters &           parameters,
                                IndexNSG &                   small_index,
                                const std::vector<unsigned> &picked_pts) {
  }

  void IndexNSG::BuildRandomHierarchical(size_t n, const float *data,
                                         Parameters &parameters) {
    unsigned range = parameters.Get<unsigned>("R");
    bool     is_nsg = parameters.Get<bool>("is_nsg");
    bool     is_rnd_nn = parameters.Get<bool>("is_rnd_nn");
    data_ = data;

    if (is_nsg) {
      std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
      Load(nn_graph_path.c_str());
    } else {
      if (!is_rnd_nn) {
        std::string nn_graph_path =
            parameters.Get<std::string>("nn_graph_path");
        Load_nn_graph(nn_graph_path.c_str());
      }
      init_graph_bf(parameters);
    }

    LinkHierarchy(parameters);

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < nd_; i++) {
      auto &pool = final_graph_[i];
      max = std::max(max, pool.size());
      min = std::min(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    std::cout << "Degree: max:" << max
              << "  avg:" << (float) total / (float) nd_ << "  min:" << min
              << "  count(deg<2):" << cnt << "\n";

    width = std::max((unsigned) max, width);
    has_built = true;
  }

  std::pair<int, int> IndexNSG::BeamSearch(
      const float *query, const float *x, const size_t K,
      const Parameters &parameters, unsigned *indices, int beam_width,
      const std::vector<unsigned> &start_points) {
    const unsigned L = parameters.Get<unsigned>("L_search");
    data_ = x;

    std::vector<unsigned> init_ids(L);
    // boost::dynamic_bitset<> flags{nd_, 0};
    tsl::robin_set<unsigned> visited(10 * L);
    unsigned                 tmp_l = 0;
    // ignore default init; use start_points for init
    if (start_points.size() == 0)
      for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
        init_ids[tmp_l] = final_graph_[ep_][tmp_l];
        visited.insert(init_ids[tmp_l]);
      }
    else
      for (; tmp_l < L && tmp_l < start_points.size(); tmp_l++) {
        init_ids[tmp_l] = start_points[tmp_l];
        visited.insert(init_ids[tmp_l]);
      }

    while (tmp_l < L) {
      unsigned id = (rand() * rand() * rand()) % nd_;
      if (visited.find(id) == visited.end())
        visited.insert(id);
      else
        continue;
      init_ids[tmp_l++] = id;
    }

    std::vector<Neighbor> retset(L + 1);
    size_t                i = 0;
    for (auto id : init_ids)
      retset[i++] = Neighbor(id,
                             distance_->compare(data_ + dimension_ * id, query,
                                                (unsigned) dimension_),
                             true);

    int                   hops = 0;
    int                   cmps = 0;
    std::vector<unsigned> frontier;
    std::vector<unsigned> unique_nbrs;
    unique_nbrs.reserve(10 * L);
    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;

    while (k < (int) L) {
      int nk = L;

      frontier.clear();
      unique_nbrs.clear();
      unsigned marker = k - 1;
      while (++marker < L && frontier.size() < (size_t) beam_width) {
        if (retset[marker].flag) {
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
      }

      if (!frontier.empty())
        hops++;
      for (auto n : frontier) {
        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
          }
          unique_nbrs.push_back(id);
        }
      }
      auto last_iter = std::unique(unique_nbrs.begin(), unique_nbrs.end());
      for (auto iter = unique_nbrs.begin(); iter != last_iter; iter++) {
        cmps++;
        unsigned id = *iter;
        float    dist = distance_->compare(query, data_ + dimension_ * id,
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

  std::pair<int, int> IndexNSG::Search(const float *query, const float *x,
                                       const size_t      K,
                                       const Parameters &parameters,
                                       unsigned *        indices) {
    const unsigned L = parameters.Get<unsigned>("L_search");
    data_ = x;
    std::vector<Neighbor>    retset(L + 1);
    std::vector<unsigned>    init_ids(L);
    tsl::robin_set<unsigned> visited(10 * L);
    // std::mt19937 rng(rand());
    // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

    unsigned tmp_l = 0;
    for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
      init_ids[tmp_l] = final_graph_[ep_][tmp_l];
      visited.insert(init_ids[tmp_l]);
    }

    while (tmp_l < L) {
      unsigned id = rand() % nd_;
      if (visited.find(id) != visited.end())
        continue;
      else
        visited.insert(id);
      init_ids[tmp_l] = id;
      tmp_l++;
    }

    for (unsigned i = 0; i < init_ids.size(); i++) {
      unsigned id = init_ids[i];
      float    dist = distance_->compare(data_ + dimension_ * id, query,
                                      (unsigned) dimension_);
      retset[i] = Neighbor(id, dist, true);
      // flags[id] = true;
    }

    int hops = 0;
    int cmps = 0;
    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int) L) {
      int nk = L;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        hops++;
        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
          if (visited.find(id) != visited.end())
            continue;
          else
            visited.insert(id);
          cmps++;
          float dist = distance_->compare(query, data_ + dimension_ * id,
                                          (unsigned) dimension_);
          if (dist >= retset[L - 1].distance)
            continue;
          Neighbor nn(id, dist, true);
          int      r = InsertIntoPool(retset.data(), L, nn);

          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    for (size_t i = 0; i < K; i++)
      indices[i] = retset[i].id;
    return std::make_pair(hops, cmps);
  }

  unsigned long long IndexNSG::SearchWithOptGraph(const float *query, size_t K,
                                                  const Parameters &parameters,
                                                  unsigned *        indices) {
    unsigned        L = parameters.Get<unsigned>("L_search");
    DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    // std::mt19937 rng(rand());
    // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

    unsigned long long dist_comp = 0;

    tsl::robin_set<unsigned> visited;
    unsigned                 tmp_l = 0;
    unsigned *               neighbors =
        (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
    unsigned MaxM_ep = *neighbors;
    neighbors++;

    for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
      init_ids[tmp_l] = neighbors[tmp_l];
      visited.insert(init_ids[tmp_l]);
    }

    while (tmp_l < L) {
      unsigned id = rand() % nd_;
      if (visited.find(id) != visited.end())
        continue;
      else
        visited.insert(id);
      init_ids[tmp_l] = id;
      tmp_l++;
    }

    for (unsigned i = 0; i < init_ids.size(); i++) {
      unsigned id = init_ids[i];
      if (id >= nd_)
        continue;
      _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
    }
    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
      unsigned id = init_ids[i];
      if (id >= nd_)
        continue;
      float *x = (float *) (opt_graph_ + node_size * id);
      float  norm_x = *x;
      x++;
      float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
      dist_comp++;
      retset[i] = Neighbor(id, dist, true);
      visited.insert(id);
      L++;
    }
    // std::cout<<L<<std::endl;

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int) L) {
      int nk = L;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
        unsigned *neighbors =
            (unsigned *) (opt_graph_ + node_size * n + data_len);
        unsigned MaxM = *neighbors;
        neighbors++;
        for (unsigned m = 0; m < MaxM; ++m)
          _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
        for (unsigned m = 0; m < MaxM; ++m) {
          unsigned id = neighbors[m];
          if (visited.find(id) != visited.end())
            continue;
          else
            visited.insert(id);
          float *data = (float *) (opt_graph_ + node_size * id);
          float  norm = *data;
          data++;
          float dist =
              dist_fast->compare(query, data, norm, (unsigned) dimension_);
          dist_comp++;
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
    return dist_comp;
  }

  void IndexNSG::OptimizeGraph(float *data) {  // use after build or load

    data_ = data;
    data_len = (dimension_ + 1) * sizeof(float);
    neighbor_len = (width + 1) * sizeof(unsigned);
    node_size = data_len + neighbor_len;
    opt_graph_ = (char *) malloc(node_size * nd_);
    DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;
    for (unsigned i = 0; i < nd_; i++) {
      char *cur_node_offset = opt_graph_ + i * node_size;
      float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
      std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
      std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                  data_len - sizeof(float));

      cur_node_offset += data_len;
      unsigned k = final_graph_[i].size();
      std::memcpy(cur_node_offset, &k, sizeof(unsigned));
      std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                  k * sizeof(unsigned));
      std::vector<unsigned>().swap(final_graph_[i]);
    }
    free(data);
    data_ = nullptr;
    CompactGraph().swap(final_graph_);
  }

  void IndexNSG::DFS(tsl::robin_set<unsigned> &visited, unsigned root,
                     unsigned &cnt) {
    unsigned             tmp = root;
    std::stack<unsigned> s;
    s.push(root);
    if (visited.find(root) == visited.end())
      cnt++;
    visited.insert(root);
    while (!s.empty()) {
      unsigned next = nd_ + 1;
      for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
        if (visited.find(final_graph_[tmp][i]) == visited.end()) {
          next = final_graph_[tmp][i];
          break;
        }
      }
      if (next == (nd_ + 1)) {
        s.pop();
        if (s.empty())
          break;
        tmp = s.top();
        continue;
      }
      tmp = next;
      visited.insert(tmp);
      s.push(tmp);
      cnt++;
    }
  }

  void IndexNSG::findroot(tsl::robin_set<unsigned> &visited, unsigned &root,
                          const Parameters &parameter) {
    unsigned id = nd_;
    for (unsigned i = 0; i < nd_; i++) {
      if (visited.find(i) == visited.end()) {
        id = i;
        break;
      }
    }

    if (id == nd_)
      return;  // No Unlinked Node

    std::vector<Neighbor> tmp, pool;
    get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
    std::sort(pool.begin(), pool.end());

    unsigned found = 0;
    for (unsigned i = 0; i < pool.size(); i++) {
      if (visited.find(pool[i].id) != visited.end()) {
        // std::cout << pool[i].id << '\n';
        root = pool[i].id;
        found = 1;
        break;
      }
    }
    if (found == 0) {
      while (true) {
        unsigned rid = rand() % nd_;
        if (visited.find(rid) != visited.end()) {
          root = rid;
          break;
        }
      }
    }
    final_graph_[root].push_back(id);
  }

  void IndexNSG::tree_grow(const Parameters &parameter) {
    unsigned                 root = ep_;
    tsl::robin_set<unsigned> visited;
    unsigned                 unlinked_cnt = 0;
    while (unlinked_cnt < nd_) {
      DFS(visited, root, unlinked_cnt);
      std::cout << "Unlinked count: " << unlinked_cnt << std::endl;
      if (unlinked_cnt >= nd_)
        break;
      findroot(visited, root, parameter);
      std::cout << "new root"
                << ":" << root << '\n';
    }
    for (size_t i = 0; i < nd_; ++i) {
      if (final_graph_[i].size() > width) {
        width = final_graph_[i].size();
      }
    }
  }
}
