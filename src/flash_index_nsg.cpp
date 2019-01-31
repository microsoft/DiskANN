#include "efanna2e/flash_index_nsg.h"

#include <omp.h>
#include <chrono>
#include <cmath>
#include <iterator>
#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"

#include "tsl/robin_set.h"

#define NSG_MEDOID_NHOOD_INIT_SIZE 16384

namespace efanna2e {
#define _CONTROL_NUM 100

  // WARNING:: using 16KB as
  FlashIndexNSG::FlashIndexNSG(const size_t dimension, const size_t n, Metric m,
                               Index *initializer)
      : Index(dimension, n, m), initializer_{initializer},
        ep_nhood(NSG_MEDOID_NHOOD_INIT_SIZE, dimension) {
    this->aligned_dim = ROUND_UP(dimension, 8);
  }

  FlashIndexNSG::~FlashIndexNSG() {
  }

  void FlashIndexNSG::Save(const char *filename) {
    std::cerr << "FlashIndexNSG::Save not implemented" << std::endl;
  }

  void FlashIndexNSG::Load(const char *filename) {
    std::cerr << "FlashIndexNSG::Load not implemented" << std::endl;
  }

  void FlashIndexNSG::load_embedded_index(const std::string &index_filename,
                                          const std::string &node_size_fname) {
    this->graph_reader.open(index_filename);

    // create a request
    std::vector<AlignedRead> first_sector(1);
    first_sector[0].offset = 0;
    first_sector[0].len = 512;
    efanna2e::alloc_aligned(&first_sector[0].buf, 512, 512);

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
    for (int i = 1; i < this->nd_; i++) {
      this->node_offsets.push_back(this->node_offsets[i - 1] +
                                   this->node_sizes[i - 1]);
    }
    std::cout << "Total file size (deduced from " << node_size_fname
              << " ) = " << this->node_offsets.back() << "B" << std::endl;

    // read in medoid
    if (this->node_sizes[this->ep_] > NSG_MEDOID_NHOOD_INIT_SIZE) {
      // if medoid nhood size is greater than preset size, re-init medoid nhood
      free(this->ep_nhood.buf);
      alloc_aligned(&(this->ep_nhood.buf), this->node_sizes[this->ep_], 512);
    }
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
          this->ep_nhood.aligned_fp32_coords + this->aligned_dim * id, query,
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
      while (++marker < (int) L && frontier.size() < beam_width) {
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
        // alloc nhoods, read and construct fp32 variant
        for (auto id : frontier) {
          frontier_nhoods.emplace_back(this->node_sizes[id], this->dimension_);
          frontier_read_reqs.emplace_back(this->node_offsets[id],
                                          this->node_sizes[id],
                                          frontier_nhoods.back().buf);
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
          /*
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
                     */
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
