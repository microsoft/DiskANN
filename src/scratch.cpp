// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <vector>
#include <boost/dynamic_bitset.hpp>

#include "scratch.h"

namespace diskann {
  //
  // Functions to manage scratch space for in-memory index based search
  //
  template<typename T>
  InMemQueryScratch<T>::InMemQueryScratch(uint32_t search_l,
                                          uint32_t indexing_l, uint32_t r,
                                          size_t dim)
      : search_l(search_l), indexing_l(indexing_l), r(r) {
    if (search_l == 0 || indexing_l == 0 || r == 0 || dim == 0) {
      std::stringstream ss;
      ss << "In InMemQueryScratch, one of search_l = " << search_l
         << ", indexing_l = " << indexing_l << ", dim = " << dim
         << " or r = " << r << " is zero." << std::endl;
      throw diskann::ANNException(ss.str(), -1);
    }

    _indices = new uint32_t[search_l];     // only used by search
    _interim_dists = new float[search_l];  // only used by search
    memset(_indices, 0, sizeof(uint32_t) * search_l);
    memset(_interim_dists, 0, sizeof(float) * search_l);

    auto   aligned_dim = ROUND_UP(dim, 8);
    size_t allocSize = aligned_dim * sizeof(T);
    alloc_aligned(((void **) &_aligned_query), allocSize, 8 * sizeof(T));
    memset(_aligned_query, 0, aligned_dim * sizeof(T));

    auto l_to_use = std::max(search_l, indexing_l);

    _des.reserve(2 * r);
    _pool.reserve(l_to_use * 10);
    _visited.reserve(l_to_use * 2);
    _best_l_nodes.resize(l_to_use + 1);
    _inserted_into_pool_rs.reserve(l_to_use * 20);
    _inserted_into_pool_bs = new boost::dynamic_bitset<>();
  }

  template<typename T>
  void InMemQueryScratch<T>::clear() {
    memset(_indices, 0, sizeof(uint32_t) * search_l);
    memset(_interim_dists, 0, sizeof(float) * search_l);
    _pool.clear();
    _visited.clear();
    _des.clear();
    _inserted_into_pool_rs.clear();
    _inserted_into_pool_bs->reset();
  }

  template<typename T>
  void InMemQueryScratch<T>::resize_for_query(uint32_t new_search_l) {
    if (new_search_l > std::max(search_l, indexing_l)) {
      if (_indices != nullptr) {
        delete[] _indices;
      }
      _indices = new uint32_t[new_search_l];

      if (_interim_dists != nullptr) {
        delete[] _interim_dists;
      }
      _interim_dists = new float[new_search_l];
      search_l = new_search_l;
    }
  }

  template<typename T>
  InMemQueryScratch<T>::~InMemQueryScratch() {
    delete _inserted_into_pool_bs;

    delete[] _indices;
    delete[] _interim_dists;

    if (_aligned_query != nullptr) {
      aligned_free(_aligned_query);
    }
  }

  //
  // Functions to manage scratch space for SSD based search
  //
  template<typename T>
  void SSDQueryScratch<T>::reset() {
    coord_idx = 0;
    sector_idx = 0;
    visited.clear();
    retset.clear();
    full_retset.clear();
  }

  template<typename T>
  SSDQueryScratch<T>::SSDQueryScratch(size_t aligned_dim,
                                      size_t visited_reserve) {
    _u64 coord_alloc_size = ROUND_UP(MAX_N_CMPS * aligned_dim, 256);

    diskann::alloc_aligned((void **) &coord_scratch, coord_alloc_size, 256);
    diskann::alloc_aligned((void **) &sector_scratch,
                           (_u64) MAX_N_SECTOR_READS * (_u64) SECTOR_LEN,
                           SECTOR_LEN);
    diskann::alloc_aligned((void **) &aligned_query_T, aligned_dim * sizeof(T),
                           8 * sizeof(T));

    _pq_scratch = new PQScratch<T>(MAX_GRAPH_DEGREE, aligned_dim);

    memset(coord_scratch, 0, MAX_N_CMPS * aligned_dim);
    memset(aligned_query_T, 0, aligned_dim * sizeof(T));

    visited.reserve(visited_reserve);
    full_retset.reserve(visited_reserve);
  }

  template<typename T>
  SSDQueryScratch<T>::~SSDQueryScratch() {
    diskann::aligned_free((void *) coord_scratch);
    diskann::aligned_free((void *) sector_scratch);

    delete[] _pq_scratch;
  }

  template<typename T>
  SSDThreadData<T>::SSDThreadData(size_t aligned_dim, size_t visited_reserve)
      : scratch(aligned_dim, visited_reserve) {
  }

  template<typename T>
  void SSDThreadData<T>::clear() {
    scratch.reset();
  }

  template DISKANN_DLLEXPORT class InMemQueryScratch<int8_t>;
  template DISKANN_DLLEXPORT class InMemQueryScratch<uint8_t>;
  template DISKANN_DLLEXPORT class InMemQueryScratch<float>;

  template DISKANN_DLLEXPORT class SSDQueryScratch<_u8>;
  template DISKANN_DLLEXPORT class SSDQueryScratch<_s8>;
  template DISKANN_DLLEXPORT class SSDQueryScratch<float>;

  template DISKANN_DLLEXPORT class SSDThreadData<_u8>;
  template DISKANN_DLLEXPORT class SSDThreadData<_s8>;
  template DISKANN_DLLEXPORT class SSDThreadData<float>;
}  // namespace diskann