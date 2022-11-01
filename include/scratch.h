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

namespace diskann {
  template<typename T>
  struct InMemQueryScratch {
    std::vector<Neighbor>    *_pool = nullptr;
    tsl::robin_set<unsigned> *_visited = nullptr;
    std::vector<unsigned>    *_des = nullptr;
    std::vector<Neighbor>    *_best_l_nodes = nullptr;
    tsl::robin_set<unsigned> *_inserted_into_pool_rs = nullptr;
    boost::dynamic_bitset<>  *_inserted_into_pool_bs = nullptr;

    T        *aligned_query = nullptr;
    uint32_t *indices = nullptr;
    float    *interim_dists = nullptr;

    uint32_t search_l;
    uint32_t indexing_l;
    uint32_t r;

    InMemQueryScratch();
    void setup(uint32_t search_l, uint32_t indexing_l, uint32_t r, size_t dim);
    void clear();
    void resize_for_query(uint32_t new_search_l);
    void destroy();

    std::vector<Neighbor> &pool() {
      return *_pool;
    }
    std::vector<unsigned> &des() {
      return *_des;
    }
    tsl::robin_set<unsigned> &visited() {
      return *_visited;
    }
    std::vector<Neighbor> &best_l_nodes() {
      return *_best_l_nodes;
    }
    tsl::robin_set<unsigned> &inserted_into_pool_rs() {
      return *_inserted_into_pool_rs;
    }
    boost::dynamic_bitset<> &inserted_into_pool_bs() {
      return *_inserted_into_pool_bs;
    }
  };

  // Class to avoid the hassle of pushing and popping the query scratch.
  template<typename T>
  class ScratchStoreManager {
   public:
    diskann::InMemQueryScratch<T>          _scratch;
    ConcurrentQueue<InMemQueryScratch<T>> &_query_scratch;
    ScratchStoreManager(ConcurrentQueue<InMemQueryScratch<T>> &query_scratch)
        : _query_scratch(query_scratch) {
      _scratch = query_scratch.pop();
      while (_scratch.indices == nullptr) {
        query_scratch.wait_for_push_notify();
        _scratch = query_scratch.pop();
      }
    }
    InMemQueryScratch<T> scratch_space() {
      return _scratch;
    }

    ~ScratchStoreManager() {
      _scratch.clear();
      _query_scratch.push(_scratch);
      _query_scratch.push_notify_all();
    }

   private:
    ScratchStoreManager(const ScratchStoreManager<T> &);
    ScratchStoreManager<T> &operator=(const ScratchStoreManager<T> &);
  };
}  // namespace diskann
