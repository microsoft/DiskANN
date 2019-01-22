#include "efanna2e/index_nsg.h"

#include <omp.h>
#include <chrono>
#include <cmath>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"

#include "tsl/robin_set.h"

namespace efanna2e {
#define _CONTROL_NUM 100

  IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                     Index *initializer)
      : Index(dimension, n, m), initializer_{initializer} {
  }

  IndexNSG::~IndexNSG() {
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

  void IndexNSG::Load(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    in.read((char *) &width, sizeof(unsigned));
    in.read((char *) &ep_, sizeof(unsigned));
    // width=100;
    unsigned cc = 0;
    while (!in.eof()) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (in.eof())
        break;
      cc += k;
      std::vector<unsigned> tmp(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));
      final_graph_.push_back(tmp);
    }
    cc /= nd_;
    // std::cout<<cc<<std::endl;
  }
  void IndexNSG::Load_nn_graph(const char *filename) {
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
  }

  void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                               std::vector<Neighbor> &retset,
                               std::vector<Neighbor> &fullset) {
    unsigned L = parameter.Get<unsigned>("L");

    retset.resize(L + 1);
    std::vector<unsigned> init_ids(L);
    // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

    tsl::robin_set<unsigned> visited(10 * L);
    L = 0;
    for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size();
         i++) {
      init_ids[i] = final_graph_[ep_][i];
      visited.insert(init_ids[i]);
      L++;
    }
    while (L < init_ids.size()) {
      unsigned id = rand() % nd_;
      if(visited.find(id) != visited.end())
        continue;
      else
        visited.insert(id);
      init_ids[L] = id;
      L++;
    }

    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
      unsigned id = init_ids[i];
      if (id >= nd_)
        continue;
      float dist = distance_->compare(data_ + dimension_ * (size_t) id, query,
                                      (unsigned) dimension_);
      retset[i] = Neighbor(id, dist, true);
      L++;
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int) L) {
      int nk = L;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
          if(visited.find(id) != visited.end())
            continue;
          else
            visited.insert(id);

          float dist = distance_->compare(query, data_ + dimension_ * (size_t) id, (unsigned) dimension_);
          Neighbor nn(id, dist, true);
          fullset.push_back(nn);
          if (dist >= retset[L - 1].distance)
            continue;
          int r = InsertIntoPool(retset.data(), L, nn);

          if (L + 1 < retset.size())
            ++L;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
  }

  void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                               tsl::robin_set<unsigned> &visited,
                               std::vector<Neighbor> &  retset,
                               std::vector<Neighbor> &  fullset) {
    unsigned L = parameter.Get<unsigned>("L");

    retset.resize(L + 1);
    std::vector<unsigned> init_ids(L);
    // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

    L = 0;
    for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size();
         i++) {
      init_ids[i] = final_graph_[ep_][i];
      visited.insert(init_ids[i]);
      L++;
    }
    while (L < init_ids.size()) {
      unsigned id = rand() % nd_;
      if(visited.find(id) != visited.end())
        continue;
      else
        visited.insert(id);
      init_ids[L] = id;
      L++;
    }

    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
      unsigned id = init_ids[i];
      if (id >= nd_)
        continue;
      float dist = distance_->compare(data_ + dimension_ * (size_t) id, query,
                                      (unsigned) dimension_);
      retset[i] = Neighbor(id, dist, true);
      fullset.push_back(retset[i]);
      // flags[id] = 1;
      L++;
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int) L) {
      int nk = L;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
          if(visited.find(id) != visited.end())
            continue;
          else
            visited.insert(id);

          float dist = distance_->compare(
              query, data_ + dimension_ * (size_t) id, (unsigned) dimension_);
          Neighbor nn(id, dist, true);
          fullset.push_back(nn);
          if (dist >= retset[L - 1].distance)
            continue;
          int r = InsertIntoPool(retset.data(), L, nn);

          if (L + 1 < retset.size())
            ++L;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
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

  void IndexNSG::init_graph_bf(const Parameters &parameters) {
    // allocate and init centroid
    float *center = new float[dimension_]();
    for (unsigned j = 0; j < dimension_; j++)
      center[j] = 0;
    for (unsigned i = 0; i < nd_; i++) {
      for (unsigned j = 0; j < dimension_; j++) {
        center[j] += data_[i * dimension_ + j];
      }
    }
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] /= nd_;
    }

    // compute all to one distance
    float * distances = new float[nd_]();
#pragma omp parallel for schedule(static, 65536)
    for (unsigned i = 0; i < nd_; i++) {
      // extract point and distance reference
      float &      dist = distances[i];
      const float *cur_vec = data_ + (i * dimension_);
      dist = 0;
      float diff = 0;
      for (unsigned j = 0; j < dimension_; j++) {
        diff = (center[j] - cur_vec[j]) * (center[j] - cur_vec[j]);
        dist += diff;
      }
    }
    // find imin
    unsigned min_idx = 0;
    float    min_dist = distances[0];
    for (unsigned i = 1; i < nd_; i++) {
      if (distances[i] < min_dist) {
        min_idx = i;
        min_dist = distances[i];
      }
    }
    ep_ = min_idx;
    std::cout << "Medoid index = " << min_idx << std::endl;
  }

  void IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                            const Parameters &       parameter,
                            tsl::robin_set<unsigned> &visited,
                            SimpleNeighbor *         cut_graph_) {
    unsigned range = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    width = range;
    unsigned start = 0;

    for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
      unsigned id = final_graph_[q][nn];
      if(visited.find(id) != visited.end())
        continue;
      float dist = distance_->compare(data_ + dimension_ * (size_t) q,
                                      data_ + dimension_ * (size_t) id,
                                      (unsigned) dimension_);
      pool.push_back(Neighbor(id, dist, true));
    }

    std::sort(pool.begin(), pool.end());
    std::vector<Neighbor> result;
    if (pool[start].id == q)
      start++;
    result.push_back(pool[start]);

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

    SimpleNeighbor *des_pool = cut_graph_ + (size_t) q * (size_t) range;
    for (size_t t = 0; t < result.size(); t++) {
      des_pool[t].id = result[t].id;
      des_pool[t].distance = result[t].distance;
    }
    if (result.size() < range) {
      des_pool[result.size()].distance = -1;
    }
    // for (unsigned t = 0; t < result.size(); t++) {
    // add_cnn(q, result[t], range, cut_graph_);
    // add_cnn(result[t].id, Neighbor(q, result[t].distance, true), range,
    // cut_graph_);
    //}
  }

  void IndexNSG::InterInsert(unsigned n, unsigned range,
                             std::vector<std::mutex> &locks,
                             SimpleNeighbor *         cut_graph_) {
    SimpleNeighbor *src_pool = cut_graph_ + (size_t) n * (size_t) range;
    for (size_t i = 0; i < range; i++) {
      if (src_pool[i].distance == -1)
        break;

      SimpleNeighbor  sn(n, src_pool[i].distance);
      size_t          des = src_pool[i].id;
      SimpleNeighbor *des_pool = cut_graph_ + des * (size_t) range;

      std::vector<SimpleNeighbor> temp_pool;
      int                         dup = 0;
      {
        LockGuard guard(locks[des]);
        for (size_t j = 0; j < range; j++) {
          if (des_pool[j].distance == -1)
            break;
          if (n == des_pool[j].id) {
            dup = 1;
            break;
          }
          temp_pool.push_back(des_pool[j]);
        }
      }
      if (dup)
        continue;

      temp_pool.push_back(sn);
      if (temp_pool.size() > range) {
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
        {
          LockGuard guard(locks[des]);
          for (unsigned t = 0; t < result.size(); t++) {
            des_pool[t] = result[t];
          }
        }
      } else {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < range; t++) {
          if (des_pool[t].distance == -1) {
            des_pool[t] = sn;
            if (t + 1 < range)
              des_pool[t + 1].distance = -1;
            break;
          }
        }
      }
    }
  }

  void IndexNSG::Link(const Parameters &parameters,
                      SimpleNeighbor *  cut_graph_) {
    std::cout << " graph link" << std::endl;
    unsigned                progress = 0;
    unsigned                percent = 100;
    unsigned                step_size = nd_ / percent;
    std::mutex              progress_lock;
    unsigned                range = parameters.Get<unsigned>("R");
    std::vector<std::mutex> locks(nd_);

#pragma omp parallel
    {
      unsigned                cnt = 0;
      std::vector<Neighbor>   pool, tmp;
      tsl::robin_set<unsigned> visited;
#pragma omp for schedule(dynamic, 100)
      for (unsigned n = 0; n < nd_; ++n) {
        pool.clear();
        tmp.clear();
        visited.clear();
        get_neighbors(data_ + dimension_ * n, parameters, visited, tmp, pool);
        sync_prune(n, pool, parameters, visited, cut_graph_);
        cnt++;
        if (cnt % step_size == 0) {
          LockGuard g(progress_lock);
          std::cout << progress++ << "/" << percent << " completed"
                    << std::endl;
        }
      }

#pragma omp for schedule(dynamic, 100)
      for (unsigned n = 0; n < nd_; ++n) {
        InterInsert(n, range, locks, cut_graph_);
      }
    }
  }

  void IndexNSG::Build(size_t n, const float *data,
                       const Parameters &parameters) {
    std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
    unsigned    range = parameters.Get<unsigned>("R");
    Load_nn_graph(nn_graph_path.c_str());
    data_ = data;
    // init_graph(parameters);
    init_graph_bf(parameters);
    SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t) range];
    std::cout << "memory allocated\n";
    Link(parameters, cut_graph_);
    final_graph_.resize(nd_);

    unsigned max = 0, min = 1e6, avg = 0, cnt = 0;
    for (size_t i = 0; i < nd_; i++) {
      SimpleNeighbor *pool = cut_graph_ + i * (size_t) range;
      unsigned        pool_size = 0;
      for (unsigned j = 0; j < range; j++) {
        if (pool[j].distance == -1)
          break;
        pool_size = j;
      }
      pool_size++;

      max = max < pool_size ? pool_size : max;
      min = min > pool_size ? pool_size : min;
      avg += pool_size;
      if (pool_size < 2)
        cnt++;

      final_graph_[i].resize(pool_size);
      for (unsigned j = 0; j < pool_size; j++) {
        final_graph_[i][j] = pool[j].id;
      }
    }

    tree_grow(parameters);

    avg /= 1.0 * nd_;
    std::cout << "Degree: max:" << max << "  avg:" << avg << "  min:" << min
              << "  count:" << cnt << "\n";

    max = 0;
    for (unsigned i = 0; i < nd_; i++) {
      max = max < final_graph_[i].size() ? final_graph_[i].size() : max;
    }
    if (max > width)
      width = max;
    has_built = true;
  }

  std::pair<int, int> IndexNSG::BeamSearch(const float *query, const float *x,
                                           size_t            K,
                                           const Parameters &parameters,
                                           unsigned *indices, int beam_width) {
    const unsigned L = parameters.Get<unsigned>("L_search");
    data_ = x;
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    // boost::dynamic_bitset<> flags{nd_, 0};
    // std::mt19937 rng(rand());
    // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);
    tsl::robin_set<unsigned> visited(10 * L);
    unsigned                 tmp_l = 0;
    for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
      init_ids[tmp_l] = final_graph_[ep_][tmp_l];
      visited.insert(init_ids[tmp_l]);
    }

    while (tmp_l < L) {
      unsigned id = rand() % nd_;
      if (visited.find(id) != visited.end()) {
        continue;
      } else {
        visited.insert(id);
      }
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

    int                   hops = 0;
    int                   cmps = 0;
    std::vector<unsigned> frontier;
    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;

    while (k < (int) L) {
      int nk = L;

      frontier.clear();
      unsigned marker = k - 1;
      while (++marker < (int) L && frontier.size() < beam_width) {
        if (retset[marker].flag) {
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
      }

      if (!frontier.empty())
        hops++;
      for (auto n : frontier) {
        // if (retset[k].flag) {
        // retset[k].flag = false;
        // unsigned n = retset[k].id;
        for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
          unsigned id = final_graph_[n][m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
          }
          cmps++;
          float dist = distance_->compare(query, data_ + dimension_ * id,
                                          (unsigned) dimension_);
          if (dist >= retset[L - 1].distance)
            continue;
          Neighbor nn(id, dist, true);

          int r = InsertIntoPool(
              retset.data(), L,
              nn);  // Return position in sorted list where nn inserted.
          if (r < nk)
            nk = r;  // nk logs the best position in the retset that was updated
                     // due to neighbors of n.
        }
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
                                       size_t K, const Parameters &parameters,
                                       unsigned *indices) {
    const unsigned L = parameters.Get<unsigned>("L_search");
    data_ = x;
    std::vector<Neighbor>   retset(L + 1);
    std::vector<unsigned>   init_ids(L);
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
      if(visited.find(id) != visited.end())
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
          if(visited.find(id) != visited.end())
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

  unsigned long long int IndexNSG::SearchWithOptGraph(
      const float *query, size_t K, const Parameters &parameters,
      unsigned *indices) {
    unsigned        L = parameters.Get<unsigned>("L_search");
    DistanceFastL2 *dist_fast = (DistanceFastL2 *) distance_;

    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    // std::mt19937 rng(rand());
    // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

    unsigned long long dist_comp = 0;

    tsl::robin_set<unsigned> visited;
    unsigned                tmp_l = 0;
    unsigned *              neighbors =
        (unsigned *) (opt_graph_ + node_size * ep_ + data_len);
    unsigned MaxM_ep = *neighbors;
    neighbors++;

    for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
      init_ids[tmp_l] = neighbors[tmp_l];
      visited.insert(init_ids[tmp_l]);
    }

    while (tmp_l < L) {
      unsigned id = rand() % nd_;
      if(visited.find(id) != visited.end())
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
          if(visited.find(id) != visited.end())
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
    if(visited.find(root) == visited.end())
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
      // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
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
    unsigned                root = ep_;
    tsl::robin_set<unsigned> visited;
    unsigned                unlinked_cnt = 0;
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
