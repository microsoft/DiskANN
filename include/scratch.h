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


// In-mem index related limits
#define GRAPH_SLACK_FACTOR 1.3

// SSD Index related limits
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
    ~InMemQueryScratch();
    InMemQueryScratch(uint32_t search_l, uint32_t indexing_l, uint32_t r,
                      uint32_t maxc, size_t dim, bool init_pq_scratch = false);
    void resize_for_new_L(uint32_t new_search_l);
    void clear();

    inline uint32_t get_L() {
      return _L;
    }
    inline uint32_t get_R() {
      return _R;
    }
    inline uint32_t get_maxc() {
      return _maxc;
    }
    inline T *aligned_query() {
      return _aligned_query;
    }
    inline PQScratch<T> *pq_scratch() {
      return _pq_scratch;
    }
    inline std::vector<Neighbor> &pool() {
      return _pool;
    }
    inline NeighborPriorityQueue &best_l_nodes() {
      return _best_l_nodes;
    }
    inline std::vector<float> &occlude_factor() {
      return _occlude_factor;
    }
    inline tsl::robin_set<unsigned> &inserted_into_pool_rs() {
      return _inserted_into_pool_rs;
    }
    inline boost::dynamic_bitset<> &inserted_into_pool_bs() {
      return *_inserted_into_pool_bs;
    }
    inline std::vector<unsigned> &id_scratch() {
      return _id_scratch;
    }
    inline std::vector<float> &dist_scratch() {
      return _dist_scratch;
    }
    inline tsl::robin_set<unsigned> &expanded_nodes_set() {
      return _expanded_nodes_set;
    }
    inline std::vector<Neighbor> &expanded_nodes_vec() {
      return _expanded_nghrs_vec;
    }
    inline std::vector<unsigned> &occlude_list_output() {
      return _occlude_list_output;
    }

   private:
    uint32_t _L;
    uint32_t _R;
    uint32_t _maxc;

    T *_aligned_query = nullptr;

    PQScratch<T> *_pq_scratch = nullptr;

    // _pool stores all neighbors explored from best_L_nodes.
    // Usually around L+R, but could be higher.
    // Initialized to 3L+R for some slack, expands as needed.
    std::vector<Neighbor> _pool;

    // _best_l_nodes is reserved for storing best L entries
    // Underlying storage is L+1 to support inserts
    NeighborPriorityQueue _best_l_nodes;

    // _occlude_factor.size() >= pool.size() in occlude_list function
    // _pool is clipped to maxc in occlude_list before affecting _occlude_factor
    // _occlude_factor is initialized to maxc size
    std::vector<float> _occlude_factor;

    // Capacity initialized to 20L
    tsl::robin_set<unsigned> _inserted_into_pool_rs;

    // Use a pointer here to allow for forward declaration of dynamic_bitset
    // in public headers to avoid making boost a dependency for clients
    // of DiskANN.
    boost::dynamic_bitset<> *_inserted_into_pool_bs;

    // _id_scratch.size() must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
    std::vector<unsigned> _id_scratch;

    // _dist_scratch must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
    // _dist_scratch should be at least the size of id_scratch
    std::vector<float> _dist_scratch;

    //  Buffers used in process delete, capacity increases as needed
    tsl::robin_set<unsigned> _expanded_nodes_set;
    std::vector<Neighbor>    _expanded_nghrs_vec;
    std::vector<unsigned>    _occlude_list_output;
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
    NeighborPriorityQueue retset;
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
