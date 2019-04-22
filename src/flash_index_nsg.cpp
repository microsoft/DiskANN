#include "efanna2e/flash_index_nsg.h"
#include <malloc.h>
#include "efanna2e/index.h"
#include "linux_aligned_file_reader.h"

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
  FlashNSG::FlashNSG(Distance *dist_cmp) : dist_cmp(dist_cmp) {
    medoid_nhood.second = nullptr;
  }

  FlashNSG::~FlashNSG() {
    if (data != nullptr) {
      delete[] data;
    }
    if (medoid_nhood.second != nullptr) {
      delete[] medoid_nhood.second;
    }
    for (auto &k_v : nhood_cache) {
      delete[] k_v.second.second;
    }
    reader->close();
    delete reader;
  }

  void FlashNSG::cache_bfs_levels(_u64 nlevels) {
    assert(nlevels > 1);

    tsl::robin_set<unsigned> *cur_level, *prev_level;
    cur_level = new tsl::robin_set<unsigned>();
    prev_level = new tsl::robin_set<unsigned>();

    std::pair<_u64, unsigned *> med_nhood;
    med_nhood.first = medoid_nhood.first;
    med_nhood.second = new unsigned[med_nhood.first];
    memcpy(med_nhood.second, medoid_nhood.second,
           med_nhood.first * sizeof(unsigned));
    nhood_cache.insert(std::make_pair(medoid, med_nhood));

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
        read.offset = NHOOD_SECTOR_START(id);
        read_reqs.push_back(read);
      }

      // issue read requests
      reader->read(read_reqs);

      // process each nhood buf
      // TODO:: cache all nhoods in each sector instead of just one
      for (auto &nhood : nhoods) {
        unsigned *node_buf = NHOOD_SECTOR_OFFSET(nhood.second, nhood.first);
        _u64      nnbrs = (_u64) *node_buf;
        unsigned *nbrs = node_buf + 1;

        // insert into cache
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
  }

  void FlashNSG::load(const char *data_bin, const char *nsg_file) {
    unsigned npts_u32, ndims_u32;
    NSG::load_bin<int8_t>(data_bin, data, npts_u32, ndims_u32);
    n_base = (_u64) npts_u32;
    data_dim = (_u64) ndims_u32;
    aligned_dim = ROUND_UP(data_dim, 8);

    // read nsg metadata
    std::ifstream nsg_meta(nsg_file, std::ios::binary);
    _u64          nnodes;
    READ_U64(nsg_meta, nnodes);
    std::cout << "nnodes: " << nnodes << std::endl;
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
    reader = new LinuxAlignedFileReader();
    reader->open(nsg_fname);

    // read medoid nhood
    char *medoid_buf = nullptr;
    alloc_aligned((void **) &medoid_buf, SECTOR_LEN, SECTOR_LEN);
    _u64                     medoid_sector_no = NHOOD_SECTOR_NO(medoid);
    std::vector<AlignedRead> medoid_read(1);
    medoid_read[0].len = SECTOR_LEN;
    medoid_read[0].buf = medoid_buf;
    medoid_read[0].offset = NHOOD_SECTOR_START(medoid);
    reader->read(medoid_read);

    unsigned *medoid_nhood_buf = NHOOD_SECTOR_OFFSET(medoid_buf, medoid);
    medoid_nhood.first = *(unsigned *) (medoid_nhood_buf);
    assert(medoid_nhood.first < 200);
    std::cout << "Medoid degree: " << medoid_nhood.first << std::endl;
    medoid_nhood.second = new unsigned[medoid_nhood.first];
    memcpy(medoid_nhood.second, (medoid_nhood_buf + 1 + sizeof(unsigned)),
           medoid_nhood.first * sizeof(unsigned));

    free(medoid_buf);
    std::cout << "Medoid nbrs: " << std::endl;
    for (_u64 i = 0; i < medoid_nhood.first; i++) {
      std::cout << medoid_nhood.second[i] << " ";
    }
    std::cout << std::endl;
  }

  std::pair<int, int> FlashNSG::beam_search(const float *query,
                                            const _u64 k_search,
                                            const _u64 l_search, _u32 *indices,
                                            const _u64  beam_width,
                                            QueryStats *stats) {
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
          alloc_aligned((void **) &frontier_nhoods[i].second, SECTOR_LEN,
                        SECTOR_LEN);

          frontier_read_reqs[i] = AlignedRead(
              NHOOD_SECTOR_START(id), SECTOR_LEN, frontier_nhoods[i].second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
        }
        io_timer.reset();
        reader->read(frontier_read_reqs);
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }
        // process each frontier nhood - compute distances to unvisited nodes
        for (auto &frontier_nhood : frontier_nhoods) {
          // if (retset[k].flag) {
          // retset[k].flag = false;
          // unsigned n = retset[k].id;
          unsigned *node_buf =
              NHOOD_SECTOR_OFFSET(frontier_nhood.second, frontier_nhood.first);
          _u64 nnbrs = (_u64)(*node_buf);
          assert(nnbrs < 200);
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

  std::pair<int, int> FlashNSG::cached_beam_search(
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
      int8_to_float(data + id * data_dim, scratch, data_dim);
      float dist = dist_cmp->compare(scratch, query, aligned_dim);
      // std::cout << "cmp: " << id << ", dist: " << dist << std::endl;
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
            alloc_aligned((void **) &fnhood.second, SECTOR_LEN, SECTOR_LEN);
            frontier_nhoods.push_back(fnhood);
            frontier_read_reqs.emplace_back(NHOOD_SECTOR_START(id), SECTOR_LEN,
                                            fnhood.second);
            if (stats != nullptr) {
              stats->n_4k++;
              stats->n_ios++;
            }
          }
        }
        io_timer.reset();
        reader->read(frontier_read_reqs);
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }
        // process each frontier nhood - compute distances to unvisited nodes
        for (auto &frontier_nhood : frontier_nhoods) {
          // if (retset[k].flag) {
          // retset[k].flag = false;
          // unsigned n = retset[k].id;
          unsigned *node_buf =
              NHOOD_SECTOR_OFFSET(frontier_nhood.second, frontier_nhood.first);
          _u64 nnbrs = (_u64)(*node_buf);
          assert(nnbrs < 200);
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
              // std::cout << "cmp: " << id << ", dist: " << dist << std::endl;
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

        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods) {
          _u64      nnbrs = cached_nhood.second.first;
          unsigned *node_nbrs = cached_nhood.second.second;
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned id = node_nbrs[m];
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
              cmps++;
              int8_to_float(data + id * data_dim, scratch, data_dim);
              float dist = dist_cmp->compare(scratch, query, aligned_dim);
              // std::cout << "cmp: " << id << ", dist: " << dist << std::endl;
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
    for (_u64 i = 0; i < k_search; i++) {
      indices[i] = retset[i].id;
    }
    free(scratch);
    if (stats != nullptr) {
      stats->total_us = query_timer.elapsed();
    }
    return std::make_pair(hops, cmps);
  }
}
