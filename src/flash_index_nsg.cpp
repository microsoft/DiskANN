#include "efanna2e/flash_index_nsg.h"
#include <malloc.h>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <iterator>
#include <thread>
#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"

#include "tsl/robin_set.h"

namespace efanna2e {
#define _CONTROL_NUM 100

  // WARNING:: using 16KB as
  FlashIndexNSG::FlashIndexNSG(const size_t dimension, const size_t n, Metric m,
                               Index *initializer)
      : Index(dimension, n, m), initializer_{initializer} {
    this->aligned_dim = ROUND_UP(dimension, 8);
  }

  FlashIndexNSG::~FlashIndexNSG() {
    for (auto &lvl : nsg_cache) {
      for (auto &k_v : lvl) {
        lvl[k_v.first].cleanup();
      }
    }
    ep_nhood.cleanup();
  }

  void FlashIndexNSG::Save(const char *filename) {
    std::cerr << "FlashIndexNSG::Save not implemented" << std::endl;
  }

  void FlashIndexNSG::Load(const char *filename) {
    std::cerr << "FlashIndexNSG::Load not implemented" << std::endl;
  }

  SimpleNhood *FlashIndexNSG::cache_check(unsigned id) {
    for (auto &lvl : nsg_cache) {
      auto ret = lvl.find(id);
      if (ret != lvl.end()) {
        return &(lvl[id]);
      }
    }

    // not found in cache
    return nullptr;
  }

  void FlashIndexNSG::cache_bfs_levels(unsigned nlevels) {
    if (!nsg_cache.empty()) {
      std::cerr << "Cache is not empty" << std::endl;
      return;
    }

    assert(nlevels > 1);

    // create `nlevels` levels
    nsg_cache.resize(nlevels);

    // first cache nhoods of each node in ep_nhood
    std::vector<AlignedRead> read_reqs(ep_nhood.nnbrs);
    for (unsigned idx = 0; idx < ep_nhood.nnbrs; idx++) {
      unsigned node_id = ep_nhood.nbrs[idx];
      nsg_cache[0].insert(std::make_pair(node_id, SimpleNhood()));
      SimpleNhood &nhood = nsg_cache[0][node_id];
      nhood.init(node_sizes[node_id], this->dimension_);
      read_reqs[idx] =
          AlignedRead(node_offsets[node_id], node_sizes[node_id], nhood.buf);
    }
    graph_reader.read(read_reqs);
    for (auto &k_v : nsg_cache[0]) {
      SimpleNhood &nhood = nsg_cache[0][k_v.first];
      nhood.construct(this->scale_factor);
      assert(k_v.second.nnbrs > 0);
    }
    std::cerr << "Cached level-0; #nodes = " << read_reqs.size() << std::endl;

    // cache subsequent levels
    tsl::robin_set<unsigned> next_level_ids;
    for (unsigned cur_level = 1; cur_level < nlevels; cur_level++) {
      // clear old read reqs
      read_reqs.clear();
      next_level_ids.clear();

      // collect all unique IDs in next level
      for (auto &k_v : nsg_cache[cur_level - 1]) {
        for (unsigned idx = 0; idx < k_v.second.nnbrs; idx++) {
          next_level_ids.insert(k_v.second.nbrs[idx]);
        }
      }

      // allocate mem and create read-reqs
      read_reqs.resize(next_level_ids.size());
      unsigned idx = 0;
      for (const auto &id : next_level_ids) {
        nsg_cache[cur_level].insert(std::make_pair(id, SimpleNhood()));
        SimpleNhood &nhood = nsg_cache[cur_level][id];
        nhood.init(node_sizes[id], this->dimension_);
        read_reqs[idx] =
            AlignedRead(node_offsets[id], node_sizes[id], nhood.buf);
        idx++;
      }

      // execute read-reqs and verify
      graph_reader.read(read_reqs);
      for (auto &k_v : nsg_cache[cur_level]) {
        SimpleNhood &nhood = nsg_cache[cur_level][k_v.first];
        nhood.construct(this->scale_factor);
        assert(k_v.second.nnbrs > 0);
      }
      std::cerr << "Cached level-" << cur_level
                << "; #nodes = " << next_level_ids.size() << std::endl;
    }
  }

  void FlashIndexNSG::smart_cache_bfs_levels(unsigned nlevels) {
    if (!nbrs_cache.empty()) {
      std::cerr << "Cache is not empty" << std::endl;
      return;
    }

    assert(nlevels > 1);

    tsl::robin_set<unsigned> *cur_level, *prev_level;
    cur_level = new tsl::robin_set<unsigned>();
    prev_level = new tsl::robin_set<unsigned>();
    unsigned aligned_dim = ROUND_UP(this->dimension_, 8);

    // cache ep_nhood
    for (unsigned idx = 0; idx < ep_nhood.nnbrs; idx++) {
      unsigned nbr_id = ep_nhood.nbrs[idx];
      cur_level->insert(nbr_id);
      if (coords_cache.find(nbr_id) == coords_cache.end()) {
        coords_cache.insert(std::make_pair(nbr_id, nullptr));
        float *&nbr_coords = coords_cache[nbr_id];
        alloc_aligned((void **) &nbr_coords, aligned_dim * sizeof(float), 32);
        memcpy(nbr_coords, ep_nhood.aligned_fp32_coords + aligned_dim * idx,
               aligned_dim * sizeof(float));
      }
    }

    // insert ep_'s adjacency list into nbrs_cache
    nbrs_cache.insert(std::make_pair(ep_, nullptr));
    unsigned *&ep_nbrs = nbrs_cache[ep_];
    ep_nbrs = new unsigned[ep_nhood.nnbrs];
    memcpy(ep_nbrs, ep_nhood.nbrs, ep_nhood.nnbrs * sizeof(unsigned));

    for (unsigned lvl = 0; lvl < nlevels; lvl++) {
      // swap prev_level and cur_level
      std::swap(prev_level, cur_level);
      // clear cur_level
      cur_level->clear();

      // read in all pre_level nhoods
      std::vector<AlignedRead> read_reqs;
      std::vector<SimpleNhood> prev_nhoods;
      std::vector<unsigned>    prev_nhood_ids;

      // vector reserve for better perf
      read_reqs.reserve(prev_level->size());
      prev_nhoods.reserve(prev_level->size());
      prev_nhood_ids.reserve(prev_level->size());

      for (const unsigned &id : *prev_level) {
        // skip node if already read into
        if (nbrs_cache.find(id) != nbrs_cache.end()) {
          continue;
        }

        prev_nhoods.push_back(SimpleNhood());
        prev_nhoods.back().init(node_sizes[id], this->dimension_);
        read_reqs.emplace_back(node_offsets[id], node_sizes[id],
                               prev_nhoods.back().buf);
        prev_nhood_ids
      }
      std::cout << "Level : " << lvl << ", # nodes: " << read_reqs.size()
                << std::endl;
      for (SimpleNhood &nhood : prev_nhoods) {
        nhood.construct(this->scale_factor);
        assert(nhood.nnbrs > 0);
        for (unsigned nbr_idx = 0; nbr_idx < nhood.nnbrs; nbr_idx++) {
          unsigned nbr = nhood.nbrs[nbr_idx];
          float *nbr_coords = nhood.aligned_fp32_coords + aligned_dim * nbr_idx;
          // process only if not processed before
          if (coords_cache.find(nbr) == coords_cache.end()) {
            coords_cache.insert(std::make_pair(nbr, nullptr));
            float *&coords = coords_cache[nbr];
            alloc_aligned((void **) &coords, aligned_dim * sizeof(float), 32);
            memcpy(coords, nbr_coords, aligned_dim * sizeof(float));
            cur_level->insert(nbr);
          }
        }
      }
    }
    // first cache nhoods of each node in ep_nhood
    std::vector<AlignedRead> read_reqs(ep_nhood.nnbrs);
    for (unsigned idx = 0; idx < ep_nhood.nnbrs; idx++) {
      unsigned node_id = ep_nhood.nbrs[idx];
      nsg_cache[0].insert(std::make_pair(node_id, SimpleNhood()));
      SimpleNhood &nhood = nsg_cache[0][node_id];
      nhood.init(node_sizes[node_id], this->dimension_);
      read_reqs[idx] =
          AlignedRead(node_offsets[node_id], node_sizes[node_id], nhood.buf);
    }
    graph_reader.read(read_reqs);
    for (auto &k_v : nsg_cache[0]) {
      SimpleNhood &nhood = nsg_cache[0][k_v.first];
      nhood.construct(this->scale_factor);
      assert(k_v.second.nnbrs > 0);
    }
    std::cerr << "Cached level-0; #nodes = " << read_reqs.size() << std::endl;

    // cache subsequent levels
    tsl::robin_set<unsigned> next_level_ids;
    for (unsigned cur_level = 1; cur_level < nlevels; cur_level++) {
      // clear old read reqs
      read_reqs.clear();
      next_level_ids.clear();

      // collect all unique IDs in next level
      for (auto &k_v : nsg_cache[cur_level - 1]) {
        for (unsigned idx = 0; idx < k_v.second.nnbrs; idx++) {
          next_level_ids.insert(k_v.second.nbrs[idx]);
        }
      }

      // allocate mem and create read-reqs
      read_reqs.resize(next_level_ids.size());
      unsigned idx = 0;
      for (const auto &id : next_level_ids) {
        nsg_cache[cur_level].insert(std::make_pair(id, SimpleNhood()));
        SimpleNhood &nhood = nsg_cache[cur_level][id];
        nhood.init(node_sizes[id], this->dimension_);
        read_reqs[idx] =
            AlignedRead(node_offsets[id], node_sizes[id], nhood.buf);
        idx++;
      }

      // execute read-reqs and verify
      graph_reader.read(read_reqs);
      for (auto &k_v : nsg_cache[cur_level]) {
        SimpleNhood &nhood = nsg_cache[cur_level][k_v.first];
        nhood.construct(this->scale_factor);
        assert(k_v.second.nnbrs > 0);
      }
      std::cerr << "Cached level-" << cur_level
                << "; #nodes = " << next_level_ids.size() << std::endl;
    }
  }

  void FlashIndexNSG::load_embedded_index(const std::string &index_filename,
                                          const std::string &node_size_fname) {
    this->graph_reader.open(index_filename);

    // create a request
    std::vector<AlignedRead> first_sector(1);
    first_sector[0].offset = 0;
    first_sector[0].len = 512;
    efanna2e::alloc_aligned(&first_sector[0].buf, 512, 512);

    std::cout << "FlashIndexNSG::load_embedded_index --- tid: "
              << std::this_thread::get_id() << std::endl;
    // read first sector
    this->graph_reader.read(first_sector);
    this->width = *((unsigned *) first_sector[0].buf);
    this->ep_ = *((unsigned *) first_sector[0].buf + 1);
    this->scale_factor = *(float *) ((unsigned *) first_sector[0].buf + 2);
    std::cout << "NSG: width=" << width << std::endl;
    std::cout << "NSG: ep_=" << ep_ << std::endl;
    std::cout << "NSG: scale_factor=" << scale_factor << std::endl;

    // cleanup
    free(first_sector[0].buf);

    // read node sizes and deduce node offsets
    std::ifstream sizes_reader(node_size_fname,
                               std::ios::binary | std::ios::in);
    this->node_sizes.resize(this->nd_);
    sizes_reader.read((char *) this->node_sizes.data(),
                      this->nd_ * sizeof(size_t));
    this->node_offsets.push_back(512);
    for (unsigned i = 1; i < this->nd_; i++) {
      this->node_offsets.push_back(this->node_offsets[i - 1] +
                                   this->node_sizes[i - 1]);
    }
    std::cout << "Total file size (deduced from " << node_size_fname
              << " ) = " << this->node_offsets.back() << "B" << std::endl;

    // read in medoid
    ep_nhood.init(this->node_sizes[this->ep_], this->dimension_);
    std::vector<AlignedRead> ep_req;
    ep_req.emplace_back(this->node_offsets[this->ep_],
                        this->node_sizes[this->ep_], this->ep_nhood.buf);
    this->graph_reader.read(ep_req);
    this->ep_nhood.construct(scale_factor);
    std::cout << "NSG: medoid out-degree=" << this->ep_nhood.nnbrs << std::endl;
  }

  std::pair<int, int> FlashIndexNSG::BeamSearch(const float *query,
                                                const float *x, size_t K,
                                                const Parameters &parameters,
                                                unsigned *        indices,
                                                int               beam_width) {
    const unsigned L = parameters.Get<unsigned>("L_search");
    data_ = x;
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    // boost::dynamic_bitset<> flags{nd_, 0};
    // std::mt19937 rng(rand());
    // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);
    tsl::robin_set<unsigned> visited(10 * L);
    unsigned                 tmp_l = 0;
    for (; tmp_l < L && tmp_l < this->ep_nhood.nnbrs; tmp_l++) {
      init_ids[tmp_l] = this->ep_nhood.nbrs[tmp_l];
      visited.insert(init_ids[tmp_l]);
    }

    // cannot generate random init points
    unsigned dummy_inits = (L - tmp_l);
    while (dummy_inits > 0) {
      // using std::numeric_limits<unsigned>::max as dummy init points, with inf
      // distance
      unsigned id = std::numeric_limits<unsigned>::max() - dummy_inits;
      init_ids[tmp_l + dummy_inits] = id;
      dummy_inits--;
    }

    for (unsigned i = 0; i < tmp_l; i++) {
      unsigned id = init_ids[i];
      float    dist = distance_->compare(
          this->ep_nhood.aligned_fp32_coords + this->aligned_dim * i, query,
          (unsigned) dimension_);
      retset[i] = Neighbor(id, dist, true);
      // flags[id] = true;
    }
    for (unsigned i = tmp_l; i < L; i++) {
      unsigned id = init_ids[i];
      // set distance to +inf
      float dist = std::numeric_limits<float>::max();
      retset[i] = Neighbor(id, dist, true);
    }
    std::sort(retset.begin(), retset.begin() + L);

    int hops = 0;
    int cmps = 0;
    int k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    std::vector<std::pair<unsigned, float *>> id_vec_list;
    std::vector<SimpleNhood> frontier_nhoods;
    std::vector<AlignedRead> frontier_read_reqs;

    while (k < (int) L) {
      int nk = L;

      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      id_vec_list.clear();
      unsigned marker = k - 1;
      while (++marker < (unsigned) L && frontier.size() < beam_width) {
        // if dummy init, don't process
        if (retset[marker].distance == std::numeric_limits<float>::max()) {
          continue;
        }
        if (retset[marker].flag) {
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
      }

      if (!frontier.empty()) {
        hops++;
        frontier_nhoods.resize(frontier.size());
        frontier_read_reqs.resize(frontier.size());
        // alloc nhoods, read and construct fp32 variant
        // std::cout << "k = " << k << '\n';
        for (unsigned i = 0; i < frontier.size(); i++) {
          unsigned id = frontier[i];
          frontier_nhoods[i].init(this->node_sizes[id], this->dimension_);
          frontier_read_reqs[i] =
              AlignedRead(this->node_offsets[id], this->node_sizes[id],
                          frontier_nhoods[i].buf);
          // std::cout << "using buf = " << &(frontier_nhoods[i].buf) <<
          // std::endl;
        }
        graph_reader.read(frontier_read_reqs);
        for (auto &nhood : frontier_nhoods) {
          nhood.construct(this->scale_factor);
        }
      }
      // process each frontier nhood - extract id and coords of unvisited nodes
      for (auto &frontier_nhood : frontier_nhoods) {
        // if (retset[k].flag) {
        // retset[k].flag = false;
        // unsigned n = retset[k].id;
        for (unsigned m = 0; m < frontier_nhood.nnbrs; ++m) {
          unsigned id = frontier_nhood.nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
          }
          id_vec_list.push_back(std::make_pair(
              id, frontier_nhood.aligned_fp32_coords + this->aligned_dim * m));
        }
      }
      auto last_iter = std::unique(id_vec_list.begin(), id_vec_list.end());
      for (auto iter = id_vec_list.begin(); iter != last_iter; iter++) {
        cmps++;
        unsigned id = iter->first;
        float *  id_vec = iter->second;
        float    dist = distance_->compare(query, id_vec, this->aligned_dim);
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

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      // clenup nhood objects
      for (auto &nhood : frontier_nhoods) {
        nhood.cleanup();
      }
    }
    for (size_t i = 0; i < K; i++) {
      indices[i] = retset[i].id;
    }
    return std::make_pair(hops, cmps);
  }

  std::pair<int, int> FlashIndexNSG::CachedBeamSearch(
      const float *query, const float *x, size_t K,
      const Parameters &parameters, unsigned *indices, int beam_width) {
    const unsigned L = parameters.Get<unsigned>("L_search");
    data_ = x;
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    // boost::dynamic_bitset<> flags{nd_, 0};
    // std::mt19937 rng(rand());
    // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);
    tsl::robin_set<unsigned> visited(10 * L);
    unsigned                 tmp_l = 0;
    for (; tmp_l < L && tmp_l < this->ep_nhood.nnbrs; tmp_l++) {
      init_ids[tmp_l] = this->ep_nhood.nbrs[tmp_l];
      visited.insert(init_ids[tmp_l]);
    }

    // cannot generate random init points
    unsigned dummy_inits = (L - tmp_l);
    while (dummy_inits > 0) {
      // using std::numeric_limits<unsigned>::max as dummy init points, with inf
      // distance
      unsigned id = std::numeric_limits<unsigned>::max() - dummy_inits;
      init_ids[tmp_l + dummy_inits] = id;
      dummy_inits--;
    }

    for (unsigned i = 0; i < tmp_l; i++) {
      unsigned id = init_ids[i];
      float    dist = distance_->compare(
          this->ep_nhood.aligned_fp32_coords + this->aligned_dim * i, query,
          (unsigned) dimension_);
      retset[i] = Neighbor(id, dist, true);
      // flags[id] = true;
    }
    for (unsigned i = tmp_l; i < L; i++) {
      unsigned id = init_ids[i];
      // set distance to +inf
      float dist = std::numeric_limits<float>::max();
      retset[i] = Neighbor(id, dist, true);
    }
    std::sort(retset.begin(), retset.begin() + L);

    int hops = 0;
    int cmps = 0;
    int k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    std::vector<std::pair<unsigned, float *>> id_vec_list;
    std::vector<SimpleNhood> frontier_nhoods;
    std::vector<AlignedRead> frontier_read_reqs;

    while (k < (int) L) {
      int nk = L;

      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      id_vec_list.clear();
      unsigned marker = k - 1;
      while (++marker < (unsigned) L && frontier.size() < beam_width) {
        // if dummy init, don't process
        if (retset[marker].distance == std::numeric_limits<float>::max()) {
          continue;
        }
        if (retset[marker].flag) {
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
      }

      if (!frontier.empty()) {
        hops++;
        frontier_nhoods.resize(frontier.size());
        frontier_read_reqs.clear();
        // alloc nhoods, read and construct fp32 variant
        // std::cout << "k = " << k << '\n';
        for (unsigned i = 0; i < frontier.size(); i++) {
          unsigned     id = frontier[i];
          SimpleNhood *check = cache_check(id);
          if (check == nullptr) {
            frontier_nhoods[i].init(this->node_sizes[id], this->dimension_);
            frontier_read_reqs.push_back(AlignedRead(this->node_offsets[id],
                                                     this->node_sizes[id],
                                                     frontier_nhoods[i].buf));
          } else {
            // set frontier_nhoods[i].buf to nullptr to mark in-mem
            frontier_nhoods[i].aligned_fp32_coords = check->aligned_fp32_coords;
            frontier_nhoods[i].nnbrs = check->nnbrs;
            frontier_nhoods[i].nbrs = check->nbrs;
          }
          // std::cout << "using buf = " << &(frontier_nhoods[i].buf) <<
          // std::endl;
        }
        for (auto &req : frontier_read_reqs)
          assert(malloc_usable_size(req.buf) > req.len);
        graph_reader.read(frontier_read_reqs);
        for (auto &nhood : frontier_nhoods) {
          // construct fp32 only if newly created SimpleNhood
          if (nhood.buf != nullptr) {
            nhood.construct(this->scale_factor);
          }
        }
      }
      // process each frontier nhood - extract id and coords of unvisited nodes
      for (auto &frontier_nhood : frontier_nhoods) {
        for (unsigned m = 0; m < frontier_nhood.nnbrs; ++m) {
          unsigned id = frontier_nhood.nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
          }
          id_vec_list.push_back(std::make_pair(
              id, frontier_nhood.aligned_fp32_coords + this->aligned_dim * m));
        }
      }
      auto last_iter = std::unique(id_vec_list.begin(), id_vec_list.end());
      for (auto iter = id_vec_list.begin(); iter != last_iter; iter++) {
        cmps++;
        unsigned id = iter->first;
        float *  id_vec = iter->second;
        float    dist = distance_->compare(query, id_vec, this->aligned_dim);
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

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
      // cleanup all temporary nhoods
      for (auto &nhood : frontier_nhoods) {
        if (nhood.buf != nullptr) {
          nhood.cleanup();
        }
      }
    }
    for (size_t i = 0; i < K; i++) {
      indices[i] = retset[i].id;
    }
    return std::make_pair(hops, cmps);
  }

  std::pair<int, int> FlashIndexNSG::Search(const float *query, const float *x,
                                            size_t            K,
                                            const Parameters &parameters,
                                            unsigned *        indices) {
  }
}
