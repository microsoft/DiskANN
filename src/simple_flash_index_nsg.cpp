#include "efanna2e/simple_flash_index_nsg.h"
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

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

// sector # on disk where node_id is present
#define NHOOD_SECTOR_NO(node_id) (node_id / nnodes_per_sector)

// sector read offset on disk where node_id is present
#define NHOOD_SECTOR_START(node_id) \
  ((NHOOD_SECTOR_NO(node_id) + 1) * SECTOR_LEN)

// offset into sector where node_id's nhood starts
#define NHOOD_SECTOR_OFFSET(sector_buf, node_id) \
  (unsigned *) (sector_buf + ((node_id % nnodes_per_sector) * max_node_len))

namespace {
  inline void int8_to_float(int8_t *int8_coords, float *float_coords,
                            _u64 dim) {
    for (_u64 idx = 0; idx < dim; idx++) {
      float_coords[idx] = (float) int8_coords[idx];
    }
  }
}  // namespace

namespace NSG {
  SimpleFlashNSG::SimpleFlashNSG(Distance *dist_cmp) : dist_cmp(dist_cmp) {
    medoid_nhood.second = nullptr;
  }

  SimpleFlashNSG::~SimpleFlashNSG() {
    if (data != nullptr) {
      delete[] data;
    }
    if (medoid_nhood.second != nullptr) {
      delete[] medoid_nhood.second;
    }
    reader.close();
  }

  void SimpleFlashNSG::cache_bfs_levels(_u64 nlevels) {
    std::cerr << "SimpleFlashNSG::cache_bfs_levels() not implemented"
              << std::endl;
    /*
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
    */
  }

  void SimpleFlashNSG::load(const char *data_bin, const char *nsg_file) {
    unsigned npts_u32, ndims_u32;
    NSG::load_bin<int8_t>(data_bin, data, npts_u32, ndims_u32);
    n_base = (_u64) npts_u32;
    data_dim = (_u64) ndims_u32;
    aligned_dim = ROUND_UP(data_dim, 8);

    // read nsg metadata
    std::ifstream nsg_meta(nsg_file, std::ios::binary);
    _u64          nnodes;
    READ_U64(nsg_meta, nnodes);
    assert(nnodes == n_base);
    READ_U64(nsg_meta, medoid);
    READ_U64(nsg_meta, max_node_len);
    READ_U64(nsg_meta, nnodes_per_sector);
    max_degree = (max_node_len / sizeof(unsigned)) - 1;

    std::cout << "Index File: " << nsg_file << std::endl;
    std::cout << "Medoid: " << medoid << std::endl;
    std::cout << "# nodes per sector: " << nnodes_per_sector << std::endl;
    std::cout << "max node len: " << max_node_len << std::endl;
    std::cout << "max node degree: " << max_degree << std::endl;
    nsg_meta.close();

    // open AlignedFileReader handle to nsg_file
    std::string nsg_fname(nsg_file);
    reader.open(nsg_fname);

    // read medoid nhood
    char *                   medoid_buf = new char[SECTOR_LEN];
    _u64                     medoid_sector_no = NHOOD_SECTOR_NO(medoid);
    std::vector<AlignedRead> medoid_read(1);
    medoid_read[0].len = SECTOR_LEN;
    medoid_read[0].buf = medoid_buf;
    medoid_read[0].offset = NHOOD_SECTOR_START(medoid);
    reader.read(medoid_read);

    unsigned *medoid_nhood_buf = NHOOD_SECTOR_OFFSET(medoid_buf, medoid);
    medoid_nhood.first = *(unsigned *) (medoid_nhood_buf);
    std::cout << "Medoid degree: " << medoid_nhood.first << std::endl;
    medoid_nhood.second = new unsigned[medoid_nhood.first];
    memcpy(medoid_nhood.second, (medoid_nhood_buf + 1 + sizeof(unsigned)),
           medoid_nhood.first * sizeof(unsigned));

    delete[] medoid_buf;
    std::cout << "Medoid nbrs: " << std::endl;
    for (_u64 i = 0; i < medoid_nhood.first; i++) {
      std::cout << medoid_nhood.second[i] << " ";
    }
    std::cout << std::endl;
  }

  std::pair<int, int> SimpleFlashNSG::beam_search(
      const float *query, const _u64 k_search, const _u64 l_search,
      _u32 *indices, const _u64 beam_width, QueryStats *stats) {
    // scratch space to compute distances between FP32 Query and INT8 data
    float *scratch = nullptr;
    alloc_aligned((void **) &scratch, aligned_dim, 32);
    memset(scratch, 0, aligned_dim);

    std::vector<Neighbor> retset(l_search + 1);
    std::vector<_u64>     init_ids(l_search);
    tsl::robin_set<_u64>  visited(10 * l_search);

    _u64 tmp_l = 0;
    // add each neighbor of medoid
    for (; tmp_l < l_search && tmp_l < medoid_nhood.first; tmp_l++) {
      _u64 id = medoid_nhood.second[tmp_l];
      init_ids[tmp_l] = id;
      visited.insert(id);
      int8_to_float(data + id * data_dim, scratch, data_dim);
      float dist = dist_cmp->compare(scratch, query, aligned_dim);
      retset[tmp_l] = Neighbor(id, dist, true);
      if (stats != nullptr) {
        stats->n_cmps++;
      }
    }

    // TODO:: create dummy ids
    for (; tmp_l < l_search; tmp_l++) {
      _u64 id = std::numeric_limits<unsigned>::max() - tmp_l;
      init_ids[tmp_l] = id;
      float dist = std::numeric_limits<float>::max();
      retset[tmp_l] = Neighbor(id, dist, false);
    }

    std::sort(retset.begin(), retset.begin() + l_search);

    _u64 hops = 0;
    _u64 cmps = 0;
    _u64 k = 0;

    // cleared every iteration
    std::vector<_u64> frontier;
    std::vector<std::pair<_u64, char *>> frontier_nhoods;
    std::vector<AlignedRead> frontier_read_reqs;

    while (k < l_search) {
      _u64 nk = l_search;

      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
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
          frontier_nhoods[i].first = id;
          frontier_nhoods[i].second = new char[SECTOR_LEN];

          frontier_read_reqs[i] = AlignedRead(
              NHOOD_SECTOR_START(id), SECTOR_LEN, frontier_nhoods[i].second);
        }
        reader.read(frontier_read_reqs);
        // process each frontier nhood - compute distances to unvisited nodes
        for (auto &frontier_nhood : frontier_nhoods) {
          // if (retset[k].flag) {
          // retset[k].flag = false;
          // unsigned n = retset[k].id;
          unsigned *node_buf =
              NHOOD_SECTOR_OFFSET(frontier_nhood.second, frontier_nhood.first);
          _u64      nnbrs = (_u64)(*node_buf);
          unsigned *node_nbrs = (node_buf + 1);
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned id = node_nbrs[m];
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
              cmps++;
              int8_to_float(data + id * data_dim, scratch, data_dim);
              float dist = dist_cmp->compare(scratch, query, aligned_dim);
              if (stats != nullptr) {
                stats->n_cmps++;
              }
              if (dist >= retset[l_search - 1].distance)
                continue;
              Neighbor nn(id, dist, true);
              _u64     r = InsertIntoPool(
                  retset.data(), l_search,
                  nn);  // Return position in sorted list where nn inserted.
              if (r < nk)
                nk = r;  // nk logs the best position in the retset that was
                         // updated
                         // due to neighbors of n.
            }
          }
        }
        // cleanup for the round
        for (auto &nhood : frontier_nhoods) {
          delete[] nhood.second;
        }
      }
      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }
    for (_u64 i = 0; i < k_search; i++) {
      indices[i] = retset[i].id;
    }

    return std::make_pair(hops, cmps);
  }

  std::pair<int, int> SimpleFlashNSG::cached_beam_search(
      const float *query, const _u64 k_search, const _u64 l_search,
      _u32 *indices, const _u64 beam_width, QueryStats *stats) {
    /*
if (nbrs_cache == nullptr || coords_cache == nullptr) {
  std::cerr << "Run SimpleFlashNSG::cache_bfs_levels before "
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
  float dist = dist_cmp->compare(
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
    float    dist = dist_cmp->compare(query, id_vec, aligned_dim);
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
*/
  }
}
