// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>

#include "boost_dynamic_bitset_fwd.h"
//#include "boost/dynamic_bitset.hpp"
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include "tsl/sparse_map.h"

#include "neighbor.h"
#include "concurrent_queue.h"
#include "pq.h"
#include "aligned_file_reader.h"

#define MAX_GRAPH_DEGREE 512
#define MAX_N_CMPS 16384
#define SECTOR_LEN (_u64) 4096
#define MAX_N_SECTOR_READS 128

namespace diskann {
  //
  // Scratch space for in-memory index based search
  //
  template<typename T>
  class InMemQueryScratch {
   public:
    uint32_t search_l;
    uint32_t indexing_l;
    uint32_t r;

    ~InMemQueryScratch();
    InMemQueryScratch(uint32_t search_l, uint32_t indexing_l, uint32_t r,
                      size_t dim);
    void resize_for_query(uint32_t new_search_l);
    void clear();

    std::vector<Neighbor> &pool() {
      return _pool;
    }
    std::vector<unsigned> &des() {
      return _des;
    }
    tsl::robin_set<unsigned> &visited() {
      return _visited;
    }
    std::vector<Neighbor> &best_l_nodes() {
      return _best_l_nodes;
    }
    tsl::robin_set<unsigned> &inserted_into_pool_rs() {
      return _inserted_into_pool_rs;
    }
    boost::dynamic_bitset<> &inserted_into_pool_bs() {
      return *_inserted_into_pool_bs;
    }

    T *aligned_query() {
      return this->_aligned_query;
    }
    uint32_t *indices() {
      return this->_indices;
    }
    float *interim_dists() {
      return this->_interim_dists;
    }

   private:
    std::vector<Neighbor>    _pool;
    tsl::robin_set<unsigned> _visited;
    std::vector<unsigned>    _des;
    std::vector<Neighbor>    _best_l_nodes;
    tsl::robin_set<unsigned> _inserted_into_pool_rs;
    boost::dynamic_bitset<> *_inserted_into_pool_bs;

    T        *_aligned_query = nullptr;
    uint32_t *_indices = nullptr;
    float    *_interim_dists = nullptr;
  };

  //
  // Scratch space for SSD index based search
  //

  template<typename T>
  class SSDQueryScratch {
   public:
    T   *coord_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_CMPS * data_dim]
    _u64 coord_idx = 0;            // index of next [data_dim] scratch to use

    char *sector_scratch =
        nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

    T *aligned_query_T = nullptr;

    PQScratch<T> *_pq_scratch;

    tsl::robin_set<_u64>  visited;
    std::vector<Neighbor> retset;
    std::vector<Neighbor> full_retset;

    SSDQueryScratch(size_t aligned_dim, size_t visited_reserve);
    ~SSDQueryScratch();

    void reset();
  };

  template<typename T>
  class SSDThreadData {
   public:
    SSDQueryScratch<T> scratch;
    IOContext          ctx;

    SSDThreadData(size_t aligned_dim, size_t visited_reserve);
    void clear();
  };

  //
  // Class to avoid the hassle of pushing and popping the query scratch.
  //
  template<typename T>
  class ScratchStoreManager {
   public:
    ScratchStoreManager(ConcurrentQueue<T *> &query_scratch)
        : _scratch_pool(query_scratch) {
      _scratch = query_scratch.pop();
      while (_scratch == nullptr) {
        query_scratch.wait_for_push_notify();
        _scratch = query_scratch.pop();
      }
    }
    T *scratch_space() {
      return _scratch;
    }

    ~ScratchStoreManager() {
      _scratch->clear();
      _scratch_pool.push(_scratch);
      _scratch_pool.push_notify_all();
    }

    void destroy() {
      while (!_scratch_pool.empty()) {
        auto scratch = _scratch_pool.pop();
        while (scratch == nullptr) {
          _scratch_pool.wait_for_push_notify();
          scratch = _scratch_pool.pop();
        }
        delete scratch;
      }
    }

   private:
    T                    *_scratch;
    ConcurrentQueue<T *> &_scratch_pool;
    ScratchStoreManager(const ScratchStoreManager<T> &);
    ScratchStoreManager &operator=(const ScratchStoreManager<T> &);
  };
}  // namespace diskann
