#include "efanna2e/ivfpq_flash_index_nsg.h"
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
#define NODE_SECTOR_NO(node_id) ((node_id / nnodes_per_sector) + 1)

// obtains region of sector containing node
#define OFFSET_TO_NODE(sector_buf, node_id) \
  ((char *) sector_buf + (node_id % nnodes_per_sector) * max_node_len)

// offset into sector where node_id's nhood starts
#define NODE_SECTOR_OFFSET(sector_buf, node_id) \
  ((char *) sector_buf + ((node_id % nnodes_per_sector) * max_node_len))

// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + data_dim * sizeof(_s8))

// returns region of `node_buf` containing [COORD(_s8)]
#define OFFSET_TO_NODE_COORDS(node_buf) (_s8 *) (node_buf)

namespace {
  inline void int8_to_float(int8_t *int8_coords, float *float_coords,
                            _u64 dim) {
    for (_u64 idx = 0; idx < dim; idx++) {
      float_coords[idx] = (float) int8_coords[idx];
    }
  }
}  // namespace

namespace NSG {
  IVFPQFlashNSG::IVFPQFlashNSG(Distance *dist_cmp, IVFPQTable *ivfpq_table)
      : dist_cmp(dist_cmp), ivfpq_table(ivfpq_table) {
    medoid_nhood.second = nullptr;
  }

  IVFPQFlashNSG::~IVFPQFlashNSG() {
    // delete backing bufs for nhood and coord cache
    delete[] nhood_cache_buf;
    delete[] coord_cache_buf;
    delete[] medoid_nhood.second;

    reader.close();
  }

  void IVFPQFlashNSG::cache_bfs_levels(_u64 nlevels) {
    assert(nlevels > 1);

    tsl::robin_set<unsigned> *cur_level, *prev_level;
    cur_level = new tsl::robin_set<unsigned>();
    prev_level = new tsl::robin_set<unsigned>();

    // add medoid nhood to cur_level
    for (_u64 idx = 0; idx < medoid_nhood.first; idx++) {
      unsigned nbr_id = medoid_nhood.second[idx];
      cur_level->insert(nbr_id);
    }

    for (_u64 lvl = 1; lvl < nlevels; lvl++) {
      // swap prev_level and cur_level
      std::swap(prev_level, cur_level);
      // clear cur_level
      cur_level->clear();

      // read in all pre_level nhoods
      std::vector<AlignedRead> read_reqs;
      std::vector<std::pair<_u64, char *>> nhoods;

      for (const unsigned &id : *prev_level) {
        // skip node if already read into
        if (nhood_cache.find(id) != nhood_cache.end()) {
          continue;
        }
        char *buf = nullptr;
        alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
        nhoods.push_back(std::make_pair(id, buf));
        AlignedRead read;
        read.len = SECTOR_LEN;
        read.buf = buf;
        read.offset = NODE_SECTOR_NO(id) * SECTOR_LEN;
        read_reqs.push_back(read);
      }

      // issue read requests
      reader.read(read_reqs);

      // process each nhood buf
      // TODO:: cache all nhoods in each sector instead of just one
      for (auto &nhood : nhoods) {
        // insert node coord into coord_cache
        char *node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
        _s8 * node_coords = OFFSET_TO_NODE_COORDS(node_buf);
        _s8 * cached_coords = new _s8[data_dim];
        memcpy(cached_coords, node_coords, data_dim * sizeof(_s8));
        coord_cache.insert(std::make_pair(nhood.first, cached_coords));

        // insert node nhood into nhood_cache
        unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);
        _u64      nnbrs = (_u64) *node_nhood;
        assert(nnbrs < 200);
        unsigned *nbrs = node_nhood + 1;
        // std::cerr << "CACHE: nnbrs = " << nnbrs << "\n";
        std::pair<_u64, unsigned *> cnhood;
        cnhood.first = nnbrs;
        cnhood.second = new unsigned[nnbrs];
        memcpy(cnhood.second, nbrs, nnbrs * sizeof(unsigned));
        nhood_cache.insert(std::make_pair(nhood.first, cnhood));

        // explore next level
        for (_u64 j = 0; j < nnbrs; j++) {
          cur_level->insert(nbrs[j]);
        }
        free(nhood.second);
      }
      std::cout << "Level: " << lvl << ", #nodes: " << nhoods.size()
                << std::endl;
    }

    delete cur_level;
    delete prev_level;

    // verify non-null
    for (auto &k_v : nhood_cache) {
      unsigned *nbrs = k_v.second.second;
      _u64      nnbrs = k_v.second.first;
      assert(malloc_usable_size(nbrs) >= nnbrs * sizeof(unsigned));
    }

    std::cerr << "Consolidating nhood_cache: # cached nhoods = "
              << nhood_cache.size() << "\n";
    // consolidate nhood_cache down to single buf
    _u64 nhood_cache_buf_len = 0;
    for (auto &k_v : nhood_cache) {
      nhood_cache_buf_len += k_v.second.first;
    }
    nhood_cache_buf = new unsigned[nhood_cache_buf_len];
    memset(nhood_cache_buf, 0, nhood_cache_buf_len);
    _u64 cur_off = 0;
    for (auto &k_v : nhood_cache) {
      std::pair<_u64, unsigned *> &val = nhood_cache[k_v.first];
      unsigned *&ptr = val.second;
      _u64       nnbrs = val.first;
      memcpy(nhood_cache_buf + cur_off, ptr, nnbrs * sizeof(unsigned));
      delete[] ptr;
      ptr = nhood_cache_buf + cur_off;
      cur_off += nnbrs;
    }

    std::cerr << "Consolidating coord_cache: # cached coords = "
              << coord_cache.size() << "\n";
    // consolidate coord_cache down to single buf
    _u64 coord_cache_buf_len = coord_cache.size() * data_dim;
    coord_cache_buf = new _s8[coord_cache_buf_len];
    memset(coord_cache_buf, 0, coord_cache_buf_len / sizeof(int));
    cur_off = 0;
    for (auto &k_v : coord_cache) {
      _s8 *&val = coord_cache[k_v.first];
      memcpy(coord_cache_buf + cur_off, val, data_dim * sizeof(_s8));
      delete[] val;
      val = coord_cache_buf + cur_off;
      cur_off += data_dim;
    }
  }

  void IVFPQFlashNSG::load(const char *nsg_file, const _u64 npts,
                           const _u64 data_dim) {
    this->n_base = npts;
    this->data_dim = data_dim;
    aligned_dim = ROUND_UP(data_dim, 8);
    std::cout << "Dataset: npts: " << n_base << ", ndims: " << data_dim
              << ", aligned_dim: " << aligned_dim << std::endl;

    // read nsg metadata
    std::ifstream nsg_meta(nsg_file, std::ios::binary);
    _u64          nnodes;
    READ_U64(nsg_meta, nnodes);
    std::cout << "nnodes: " << nnodes << std::endl;
    assert(nnodes == n_base);
    READ_U64(nsg_meta, medoid);
    READ_U64(nsg_meta, max_node_len);
    READ_U64(nsg_meta, nnodes_per_sector);
    max_degree =
        ((max_node_len - data_dim * sizeof(_s8)) / sizeof(unsigned)) - 1;

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
    char *medoid_buf = nullptr;
    alloc_aligned((void **) &medoid_buf, SECTOR_LEN, SECTOR_LEN);
    _u64                     medoid_sector_no = NODE_SECTOR_NO(medoid);
    std::vector<AlignedRead> medoid_read(1);
    medoid_read[0].len = SECTOR_LEN;
    medoid_read[0].buf = medoid_buf;
    medoid_read[0].offset = NODE_SECTOR_NO(medoid) * SECTOR_LEN;
    std::cout << "Medoid offset: " << NODE_SECTOR_NO(medoid) * SECTOR_LEN
              << "\n";
    reader.read(medoid_read);

    // all data about medoid
    char *medoid_node_buf = OFFSET_TO_NODE(medoid_buf, medoid);

    // add medoid coords to `coord_cache`
    _s8 *medoid_coords = new _s8[data_dim];
    _s8 *medoid_disk_coords = OFFSET_TO_NODE_COORDS(medoid_node_buf);
    memcpy(medoid_coords, medoid_disk_coords, data_dim * sizeof(_s8));
    coord_cache.clear();
    coord_cache.insert(std::make_pair(medoid, medoid_coords));

    // add medoid nhood to nhood_cache
    unsigned *medoid_nhood_buf = OFFSET_TO_NODE_NHOOD(medoid_node_buf);
    medoid_nhood.first = *(unsigned *) (medoid_nhood_buf);
    assert(medoid_nhood.first < 200);
    std::cout << "Medoid degree: " << medoid_nhood.first << std::endl;
    medoid_nhood.second = new unsigned[medoid_nhood.first];
    memcpy(medoid_nhood.second, (medoid_nhood_buf + 1),
           medoid_nhood.first * sizeof(unsigned));
    free(medoid_buf);

    // make a copy and insert into nhood_cache
    unsigned *medoid_nhood_copy = new unsigned[medoid_nhood.first];
    memcpy(medoid_nhood_copy, medoid_nhood.second,
           medoid_nhood.first * sizeof(unsigned));
    nhood_cache.insert(std::make_pair(
        medoid, std::make_pair(medoid_nhood.first, medoid_nhood_copy)));

    // print medoid nbrs
    std::cout << "Medoid nbrs: " << std::endl;
    for (_u64 i = 0; i < medoid_nhood.first; i++) {
      std::cout << medoid_nhood.second[i] << " ";
    }
    std::cout << std::endl;
  }

  // IGNORED -- not ported to new disk format for re-ranking
  std::pair<int, int> IVFPQFlashNSG::beam_search(
      const float *query, const _u64 k_search, const _u64 l_search,
      _u32 *indices, const _u64 beam_width, QueryStats *stats) {
    Timer query_timer, io_timer;
    // scratch space to compute distances between FP32 Query and INT8 data
    float *scratch = nullptr;
    alloc_aligned((void **) &scratch, aligned_dim * sizeof(float), 32);
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
      // inflate coords from PQ _u8 to FP32
      ivfpq_table->convert(id, scratch);
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
          alloc_aligned((void **) &frontier_nhoods[i].second, SECTOR_LEN,
                        SECTOR_LEN);

          frontier_read_reqs[i] =
              AlignedRead(NODE_SECTOR_NO(id) * SECTOR_LEN, SECTOR_LEN,
                          frontier_nhoods[i].second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
        }
        io_timer.reset();
        reader.read(frontier_read_reqs);
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }
        // process each frontier nhood - compute distances to unvisited nodes
        for (auto &frontier_nhood : frontier_nhoods) {
          // if (retset[k].flag) {
          // retset[k].flag = false;
          // unsigned n = retset[k].id;
          char *node_disk_buf =
              OFFSET_TO_NODE(frontier_nhood.second, frontier_nhood.first);
          unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
          _u64      nnbrs = (_u64)(*node_buf);
          assert(nnbrs < 200);
          unsigned *node_nbrs = (node_buf + 1);
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned id = node_nbrs[m];
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
              cmps++;
              ivfpq_table->convert(id, scratch);
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
          free(nhood.second);
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
    free(scratch);
    if (stats != nullptr) {
      stats->total_us = query_timer.elapsed();
    }
    return std::make_pair(hops, cmps);
  }

  std::pair<int, int> IVFPQFlashNSG::cached_beam_search(
      const float *query, const _u64 k_search, const _u64 l_search,
      _u32 *indices, const _u64 beam_width, QueryStats *stats,
      QueryScratch *query_scratch) {
    // reset query scratch context
    query_scratch->coord_idx = 0;
    query_scratch->sector_idx = 0;

    // scratch space to compute distances between FP32 Query and INT8 data
    float *scratch = query_scratch->aligned_scratch;

    // pointers to buffers for data
    _s8 * data_buf = query_scratch->coord_scratch;
    _u64 &data_buf_idx = query_scratch->coord_idx;

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    Timer                 query_timer, io_timer;
    std::vector<Neighbor> retset(l_search + 1);
    std::vector<_u64>     init_ids(l_search);
    tsl::robin_set<_u64>  visited(10 * l_search);

    tsl::robin_map<_u64, _s8 *> fp_coords;

    _u64 tmp_l = 0;
    // add each neighbor of medoid
    for (; tmp_l < l_search && tmp_l < medoid_nhood.first; tmp_l++) {
      _u64 id = medoid_nhood.second[tmp_l];
      init_ids[tmp_l] = id;
      visited.insert(id);
      ivfpq_table->convert(id, scratch);
      float dist = dist_cmp->compare(scratch, query, aligned_dim);
      // std::cout << "cmp: " << id << ", dist: " << dist << std::endl;
      // std::cerr << "dist: " << dist << std::endl;
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
    std::vector<std::pair<_u64, std::pair<_u64, unsigned *>>> cached_nhoods;

    while (k < l_search) {
      _u64 nk = l_search;

      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;

      // find new beam
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
        for (_u64 i = 0; i < frontier.size(); i++) {
          unsigned id = frontier[i];
          auto     iter = nhood_cache.find(id);
          if (iter != nhood_cache.end()) {
            cached_nhoods.push_back(std::make_pair(id, iter->second));
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
          } else {
            std::pair<_u64, char *> fnhood;
            fnhood.first = id;
            fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
            sector_scratch_idx++;
            frontier_nhoods.push_back(fnhood);
            frontier_read_reqs.emplace_back(NODE_SECTOR_NO(id) * SECTOR_LEN,
                                            SECTOR_LEN, fnhood.second);
            if (stats != nullptr) {
              stats->n_4k++;
              stats->n_ios++;
            }
          }
        }

        io_timer.reset();
        reader.read(frontier_read_reqs);
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }

        // process each frontier nhood - compute distances to unvisited nodes
        for (auto &frontier_nhood : frontier_nhoods) {
          // if (retset[k].flag) {
          //   retset[k].flag = false;
          //   unsigned n = retset[k].id;
          // }
          char *node_disk_buf =
              OFFSET_TO_NODE(frontier_nhood.second, frontier_nhood.first);
          unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
          _u64      nnbrs = (_u64)(*node_buf);
          assert(nnbrs < 200);
          _s8 *node_fp_coords = OFFSET_TO_NODE_COORDS(node_disk_buf);
          assert(data_buf_idx < MAX_N_CMPS);
          _s8 *node_fp_coords_copy = data_buf + (data_buf_idx * data_dim);
          data_buf_idx++;
          memcpy(node_fp_coords_copy, node_fp_coords, data_dim * sizeof(_s8));

          fp_coords.insert(
              std::make_pair(frontier_nhood.first, node_fp_coords_copy));

          unsigned *node_nbrs = (node_buf + 1);
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned id = node_nbrs[m];
            assert(id < 130000000);
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
              cmps++;
              ivfpq_table->convert(id, scratch);
              float dist = dist_cmp->compare(scratch, query, aligned_dim);
              // std::cout << "cmp: " << id << ", dist: " << dist << std::endl;
              // std::cerr << "dist: " << dist << std::endl;
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

        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods) {
          _u64      nnbrs = cached_nhood.second.first;
          unsigned *node_nbrs = cached_nhood.second.second;
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned id = node_nbrs[m];
            assert(id < 130000000);
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
              cmps++;
              ivfpq_table->convert(id, scratch);
              float dist = dist_cmp->compare(scratch, query, aligned_dim);
              // std::cout << "cmp: " << id << ", dist: " << dist << std::endl;
              // std::cerr << "dist: " << dist << std::endl;
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
      }
      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }

    // RE-RANKING STEP
    for (_u64 i = 0; i < l_search; i++) {
      _u64 idx = retset[i].id;
      _s8 *node_coords;
      auto global_cache_iter = coord_cache.find(idx);
      if (global_cache_iter == coord_cache.end()) {
        auto local_cache_iter = fp_coords.find(idx);
        assert(local_cache_iter != fp_coords.end());
        node_coords = local_cache_iter->second;
      } else {
        node_coords = global_cache_iter->second;
      }
      int8_to_float(node_coords, scratch, data_dim);
      retset[i].distance = dist_cmp->compare(query, scratch, aligned_dim);
      if (stats != nullptr) {
        stats->n_cmps++;
      }
    }

    // re-sort by distance
    std::sort(retset.begin(), retset.begin() + l_search,
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // copy k_search values
    for (_u64 i = 0; i < k_search; i++) {
      indices[i] = retset[i].id;
    }

    if (stats != nullptr) {
      stats->total_us = query_timer.elapsed();
    }

    return std::make_pair(hops, cmps);
  }
}
