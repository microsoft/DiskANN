#include "efanna2e/composite_flash_index_nsg.h"
#include <malloc.h>
#include "efanna2e/index.h"

#include <omp.h>
#include <chrono>
#include <cmath>
#include <iterator>
#include <thread>
#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"
#include "efanna2e/timer.h"
#include "efanna2e/util.h"

#include "tsl/robin_set.h"

#define SECTOR_LEN 4096

namespace NSG {
  template<typename T, typename NhoodType>
  CompositeFlashNSG<T, NhoodType>::~CompositeFlashNSG() {
    medoid_nhood.cleanup();
    if (coords_cache != nullptr) {
      for (uint64_t i = 0; i < n_base; i++) {
        if (coords_cache[i] != nullptr) {
          free(coords_cache[i]);
        }
        if (!nbrs_cache[i].empty()) {
          nbrs_cache[i].clear();
        }
      }
    }
    if (nbrs_cache != nullptr) {
      for (uint64_t i = 0; i < n_base; i++) {
        if (!nbrs_cache[i].empty()) {
          nbrs_cache[i].clear();
        }
      }
    }

    delete[] coords_cache;
    delete[] nbrs_cache;

    if (node_offsets != nullptr) {
      delete[] node_offsets;
    }
    if (node_sizes != nullptr) {
      delete[] node_sizes;
    }
  }

  template<typename T, typename NhoodType>
  void CompositeFlashNSG<T, NhoodType>::cache_bfs_levels(_u64 nlevels) {
    assert(nlevels > 1);

    nbrs_cache = new std::vector<_u32>[n_base];
    coords_cache = new float *[n_base]();

    tsl::robin_set<unsigned> *cur_level, *prev_level;
    cur_level = new tsl::robin_set<unsigned>();
    prev_level = new tsl::robin_set<unsigned>();

    // add medoid nbrs to nbrs_cache
    nbrs_cache[medoid].resize(medoid_nhood.nnbrs);
    memcpy(nbrs_cache[medoid].data(), medoid_nhood.nbrs,
           medoid_nhood.nnbrs * sizeof(unsigned));

    // level 2
    for (_u64 idx = 0; idx < medoid_nhood.nnbrs; idx++) {
      unsigned nbr_id = medoid_nhood.nbrs[idx];
      cur_level->insert(nbr_id);
      if (coords_cache[nbr_id] == nullptr) {
        float *&nbr_coords = coords_cache[nbr_id];
        alloc_aligned((void **) &nbr_coords, aligned_dim * sizeof(float), 32);
        memcpy(nbr_coords, medoid_nhood.aligned_fp32_coords + aligned_dim * idx,
               aligned_dim * sizeof(float));
      }
    }

    for (_u64 lvl = 1; lvl < nlevels; lvl++) {
      // swap prev_level and cur_level
      std::swap(prev_level, cur_level);
      // clear cur_level
      cur_level->clear();

      // read in all pre_level nhoods
      std::vector<AlignedRead> read_reqs;
      std::vector<unsigned>    read_reqs_ids;
      std::vector<NhoodType>   prev_nhoods;

      for (const unsigned &id : *prev_level) {
        // skip node if already read into
        if (!nbrs_cache[id].empty()) {
          continue;
        }

        prev_nhoods.push_back(NhoodType());
        NhoodType &nhood = prev_nhoods.back();
        nhood.init(node_sizes[id], data_dim);
        read_reqs.emplace_back(node_offsets[id], node_sizes[id], nhood.buf);
        read_reqs_ids.push_back(id);
      }
      std::cout << "Level : " << lvl << ", # nodes: " << read_reqs_ids.size()
                << std::endl;

      // issue read requests
      reader.read(read_reqs);

      for (_u64 idx = 0; idx < read_reqs.size(); idx++) {
        NhoodType &nhood = prev_nhoods[idx];
        unsigned   node_id = read_reqs_ids[idx];
        nhood.construct(scale_factor);
        assert(nhood.nnbrs > 0);

        // add `node_id`'s nbrs to nbrs_cache
        if (nbrs_cache[node_id].empty()) {
          nbrs_cache[node_id].resize(nhood.nnbrs);
          memcpy(nbrs_cache[node_id].data(), nhood.nbrs,
                 nhood.nnbrs * sizeof(unsigned));
        }

        // add coords of nbrs to coords_cache
        for (_u64 nbr_idx = 0; nbr_idx < nhood.nnbrs; nbr_idx++) {
          unsigned nbr = nhood.nbrs[nbr_idx];
          // process only if not processed before
          if (coords_cache[nbr] == nullptr) {
            float *nbr_coords =
                nhood.aligned_fp32_coords + (aligned_dim * nbr_idx);
            float *&coords = coords_cache[nbr];
            alloc_aligned((void **) &coords, aligned_dim * sizeof(float), 32);
            memcpy(coords, nbr_coords, aligned_dim * sizeof(float));
            cur_level->insert(nbr);
          }
        }
        nhood.cleanup();
      }
    }

    delete cur_level;
    delete prev_level;
  }

  template<typename T, typename NhoodType>
  void CompositeFlashNSG<T, NhoodType>::load(const char *filename) {
    std::ifstream meta_reader(filename);

    // read # base_pts, # dims
    meta_reader.read((char *) &n_base, sizeof(_u64));
    meta_reader.read((char *) &data_dim, sizeof(_u64));
    aligned_dim = ROUND_UP(data_dim, 8);
    std::cout << "Index File: " << filename << std::endl;
    std::cout << "# base points: " << n_base << ", # data dims: " << data_dim
              << ", aligned # dims: " << aligned_dim << std::endl;

    // read medoid
    meta_reader.read((char *) &medoid, sizeof(_u64));
    std::cout << "Medoid: " << medoid << std::endl;

    // read scale factor
    meta_reader.read((char *) &scale_factor, sizeof(float));
    std::cout << "Scale Factor: " << scale_factor << std::endl;

    // read node_sizes, compute node_offsets
    node_sizes = new _u64[n_base];
    meta_reader.read((char *) node_sizes, n_base * sizeof(_u64));
    node_offsets = new _u64[n_base];
    _u64 first_offset =
        (3 * sizeof(_u64)) + sizeof(float) + (n_base * sizeof(_u64));
    first_offset = ROUND_UP(first_offset, SECTOR_LEN);
    std::cout << "First offset: " << first_offset << std::endl;
    node_offsets[0] = first_offset;
    for (_u64 i = 0; i < n_base - 1; i++) {
      node_offsets[i + 1] = node_offsets[i] + node_sizes[i];
    }
    meta_reader.close();

    // open file handle
    reader.open(filename);

    // read in medoid nhood
    medoid_nhood.init(node_sizes[medoid], data_dim);
    std::vector<AlignedRead> medoid_read;
    medoid_read.emplace_back(node_offsets[medoid], node_sizes[medoid],
                             medoid_nhood.buf);
    reader.read(medoid_read);
    medoid_nhood.construct(scale_factor);
    std::cout << "Medoid out-degree: " << medoid_nhood.nnbrs << std::endl;
  }

  template<typename T, typename NhoodType>
  std::pair<int, int> CompositeFlashNSG<T, NhoodType>::beam_search(
      const float *query, const _u64 k_search, const _u64 l_search,
      _u32 *indices, const _u64 beam_width, QueryStats *stats) {
    std::vector<Neighbor>    retset(l_search + 1);
    std::vector<unsigned>    init_ids(l_search);
    tsl::robin_set<unsigned> visited(10 * l_search);

    unsigned tmp_l = 0;
    // add each neighbor of medoid
    for (; tmp_l < l_search && tmp_l < medoid_nhood.nnbrs; tmp_l++) {
      unsigned id = medoid_nhood.nbrs[tmp_l];
      init_ids[tmp_l] = id;
      visited.insert(id);
      float dist = distance_cmp.compare(
          medoid_nhood.aligned_fp32_coords + aligned_dim * tmp_l, query,
          aligned_dim);
      retset[tmp_l] = Neighbor(id, dist, true);
    }

    // create dummy ids
    for (; tmp_l < l_search; tmp_l++) {
      unsigned id = std::numeric_limits<unsigned>::max() - tmp_l;
      init_ids[tmp_l] = id;
      float dist = std::numeric_limits<float>::max();
      retset[tmp_l] = Neighbor(id, dist, false);
    }

    std::sort(retset.begin(), retset.begin() + l_search);

    _u64 hops = 0;
    _u64 cmps = 0;
    _u64 k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    std::vector<std::pair<unsigned, float *>> id_vec_list;
    std::vector<NhoodType>   frontier_nhoods;
    std::vector<AlignedRead> frontier_read_reqs;

    while (k < l_search) {
      _u64 nk = l_search;

      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      id_vec_list.clear();
      _u64 marker = k - 1;
      while (++marker < l_search && frontier.size() < beam_width) {
        if (retset[marker].flag) {
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
      }

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        hops++;
        frontier_nhoods.resize(frontier.size());
        frontier_read_reqs.resize(frontier.size());
        for (_u64 i = 0; i < frontier.size(); i++) {
          unsigned id = frontier[i];
          frontier_nhoods[i].init(node_sizes[id], data_dim);
          frontier_read_reqs[i] = AlignedRead(node_offsets[id], node_sizes[id],
                                              frontier_nhoods[i].buf);
        }
        reader.read(frontier_read_reqs);
        for (auto &nhood : frontier_nhoods) {
          nhood.construct(this->scale_factor);
        }

        // process each frontier nhood - extract id and coords of unvisited
        // nodes
        for (auto &frontier_nhood : frontier_nhoods) {
          // if (retset[k].flag) {
          // retset[k].flag = false;
          // unsigned n = retset[k].id;
          for (_u64 m = 0; m < frontier_nhood.nnbrs; ++m) {
            unsigned id = frontier_nhood.nbrs[m];
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
            }
            id_vec_list.push_back(std::make_pair(
                id, frontier_nhood.aligned_fp32_coords + aligned_dim * m));
          }
        }
      }

      // process each unvisited node
      for (auto iter = id_vec_list.begin(); iter != id_vec_list.end(); iter++) {
        cmps++;
        unsigned id = iter->first;
        float *  id_vec = iter->second;
        float    dist = distance_cmp.compare(query, id_vec, this->aligned_dim);
        if (dist >= retset[l_search - 1].distance)
          continue;
        Neighbor nn(id, dist, true);

        _u64 r = InsertIntoPool(
            retset.data(), l_search,
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
    for (_u64 i = 0; i < k_search; i++) {
      indices[i] = retset[i].id;
    }
    return std::make_pair(hops, cmps);
  }

  template<typename T, typename NhoodType>
  std::pair<int, int> CompositeFlashNSG<T, NhoodType>::cached_beam_search(
      const float *query, const _u64 k_search, const _u64 l_search,
      _u32 *indices, const _u64 beam_width, QueryStats *stats) {
    if (nbrs_cache == nullptr || coords_cache == nullptr) {
      std::cerr
          << "Run CompositeFlashNSG<T, NhoodType>::cache_bfs_levels before "
             "calling cached_beam_search()"
          << std::endl;
      return std::make_pair(0, 0);
    }

    Timer timer, io_timer;

    std::vector<Neighbor>    retset(l_search + 1);
    std::vector<unsigned>    init_ids(l_search);
    tsl::robin_set<unsigned> visited(10 * l_search);

    unsigned tmp_l = 0;
    // add each neighbor of medoid
    for (; tmp_l < l_search && tmp_l < medoid_nhood.nnbrs; tmp_l++) {
      unsigned id = medoid_nhood.nbrs[tmp_l];
      init_ids[tmp_l] = id;
      visited.insert(id);
      float dist = distance_cmp.compare(
          medoid_nhood.aligned_fp32_coords + aligned_dim * tmp_l, query,
          aligned_dim);
      retset[tmp_l] = Neighbor(id, dist, true);
    }

    // create dummy ids
    for (; tmp_l < l_search; tmp_l++) {
      unsigned id = std::numeric_limits<unsigned>::max() - l_search;
      init_ids[tmp_l] = id;
      float dist = std::numeric_limits<float>::max();
      retset[tmp_l] = Neighbor(id, dist, false);
    }

    std::sort(retset.begin(), retset.begin() + l_search);

    _u64 hops = 0;
    _u64 cmps = 0;
    _u64 k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    tsl::robin_map<unsigned, float *> id_vec_map;
    std::vector<NhoodType>   frontier_nhoods;
    std::vector<AlignedRead> frontier_read_reqs;
    unsigned                 nbrs_cache_hits = 0;
    unsigned                 coords_cache_hits = 0;

    while (k < l_search) {
      _u64 nk = l_search;

      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      id_vec_map.clear();

      // populate beam
      _u64 marker = k - 1;
      while (++marker < l_search && frontier.size() < beam_width) {
        if (retset[marker].flag) {
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
      }

      if (!frontier.empty()) {
        hops++;

        for (const unsigned &id : frontier) {
          if (id >= n_base) {
            std::cout << "Found id: " << id << std::endl;
          }
          // if nbrs of `id` was not cached, make a request
          if (nbrs_cache[id].empty()) {
            frontier_nhoods.push_back(NhoodType());
            NhoodType &nhood = frontier_nhoods.back();
            nhood.init(node_sizes[id], data_dim);
            frontier_read_reqs.push_back(
                AlignedRead(node_offsets[id], node_sizes[id], nhood.buf));
            // std::cout << "Reading nhood id: " << id << std::endl;
            if (stats != nullptr) {
              stats->read_size += node_sizes[id];
              stats->n_ios++;
              if (node_sizes[id] == 4096) {
                stats->n_4k++;
              } else if (node_sizes[id] == 8192) {
                stats->n_8k++;
              } else if (node_sizes[id] == 12288) {
                stats->n_12k++;
              }
            }
          } else {
            // if nbrs of `id` are cached, use it
            nbrs_cache_hits++;
            // add cached coods from id list
            for (const unsigned &nbr_id : nbrs_cache[id]) {
              assert(coords_cache[nbr_id] != nullptr);
              // ignore if visited
              if (visited.find(nbr_id) != visited.end()) {
                continue;
              } else {
                visited.insert(nbr_id);
              }
              coords_cache_hits++;
              // if nhood of `id` is cached, its nbrs coords are also cached
              id_vec_map.insert(std::make_pair(nbr_id, coords_cache[nbr_id]));
            }
          }
        }

        // prepare and execute reads
        for (auto &req : frontier_read_reqs) {
          assert(malloc_usable_size(req.buf) >= req.len);
        }
        io_timer.reset();
        reader.read(frontier_read_reqs);
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }

        for (auto &nhood : frontier_nhoods) {
          nhood.construct(this->scale_factor);
          assert(nhood.nnbrs > 0);
          for (unsigned nbr_idx = 0; nbr_idx < nhood.nnbrs; nbr_idx++) {
            unsigned id = nhood.nbrs[nbr_idx];

            // skip if already visited
            if (visited.find(id) != visited.end()) {
              continue;
            }
            visited.insert(id);
            // add if not visited
            float *id_coords =
                nhood.aligned_fp32_coords + nbr_idx * aligned_dim;
            assert(id_vec_map.find(id) == id_vec_map.end());
            id_vec_map.insert(std::make_pair(id, id_coords));
          }
        }
      }

      for (auto &k_v : id_vec_map) {
        cmps++;
        unsigned id = k_v.first;
        float *  id_vec = k_v.second;
        float    dist = distance_cmp.compare(query, id_vec, aligned_dim);
        if (dist >= retset[l_search - 1].distance)
          continue;
        Neighbor nn(id, dist, true);

        _u64 r = InsertIntoPool(
            retset.data(), l_search,
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
        nhood.cleanup();
      }
    }
    for (_u64 i = 0; i < k_search; i++) {
      indices[i] = retset[i].id;
    }
    // std::cout << "nbrs_cache_hits = " << nbrs_cache_hits
    //          << ", coords_cache_hits = " << coords_cache_hits << std::endl;
    if (stats != nullptr) {
      stats->total_us = timer.elapsed();
    }
    return std::make_pair(hops, cmps);
  }

  template class CompositeFlashNSG<int8_t, CompositeDiskNhood<int8_t>>;
}
