// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <vector>
#include <boost/dynamic_bitset.hpp>

#include "scratch.h"

namespace diskann {

  // QueryScratch functions
  template<typename T>
  InMemQueryScratch<T>::InMemQueryScratch() {
    search_l = indexing_l = r = 0;
    // pointers are initialized in the header itself.
  }
  template<typename T>
  void InMemQueryScratch<T>::setup(uint32_t search_l, uint32_t indexing_l,
                                   uint32_t r, size_t dim) {
    if (search_l == 0 || indexing_l == 0 || r == 0 || dim == 0) {
      std::stringstream ss;
      ss << "In InMemQueryScratch, one of search_l = " << search_l
         << ", indexing_l = " << indexing_l << ", dim = " << dim
         << " or r = " << r << " is zero." << std::endl;
      throw diskann::ANNException(ss.str(), -1);
    }
    indices = new uint32_t[search_l];     // only used by search
    interim_dists = new float[search_l];  // only used by search
    memset(indices, 0, sizeof(uint32_t) * search_l);
    memset(interim_dists, 0, sizeof(float) * search_l);
    this->search_l = search_l;
    this->indexing_l = indexing_l;
    this->r = r;

    auto   aligned_dim = ROUND_UP(dim, 8);
    size_t allocSize = aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset(aligned_query, 0, aligned_dim * sizeof(T));

    auto l_to_use = std::max(search_l, indexing_l);

    _des = new std::vector<unsigned>();
    _des->reserve(2 * r);
    _pool = new std::vector<Neighbor>();
    _pool->reserve(l_to_use * 10);
    _visited = new tsl::robin_set<unsigned>();
    _visited->reserve(l_to_use * 2);
    _best_l_nodes = new std::vector<Neighbor>();
    _best_l_nodes->resize(l_to_use + 1);
    _inserted_into_pool_rs = new tsl::robin_set<unsigned>();
    _inserted_into_pool_rs->reserve(l_to_use * 20);
    _inserted_into_pool_bs = new boost::dynamic_bitset<>();

    //_pq_scratch.setup();
  }

  template<typename T>
  void InMemQueryScratch<T>::clear() {
    memset(indices, 0, sizeof(uint32_t) * search_l);
    memset(interim_dists, 0, sizeof(float) * search_l);
    _pool->clear();
    _visited->clear();
    _des->clear();
    _inserted_into_pool_rs->clear();
    _inserted_into_pool_bs->reset();
  }

  template<typename T>
  void InMemQueryScratch<T>::resize_for_query(uint32_t new_search_l) {
    if (search_l < new_search_l) {
      if (indices != nullptr) {
        delete[] indices;
      }
      indices = new uint32_t[new_search_l];

      if (interim_dists != nullptr) {
        delete[] interim_dists;
      }
      interim_dists = new float[new_search_l];
      search_l = new_search_l;
    }
  }

  template<typename T>
  void InMemQueryScratch<T>::destroy() {
    if (indices != nullptr) {
      delete[] indices;
      indices = nullptr;
    }
    if (interim_dists != nullptr) {
      delete[] interim_dists;
      interim_dists = nullptr;
    }
    if (_pool != nullptr) {
      delete _pool;
      _pool = nullptr;
    }
    if (_visited != nullptr) {
      delete _visited;
      _visited = nullptr;
    }
    if (_des != nullptr) {
      delete _des;
      _des = nullptr;
    }
    if (_best_l_nodes != nullptr) {
      delete _best_l_nodes;
      _best_l_nodes = nullptr;
    }
    if (aligned_query != nullptr) {
      aligned_free(aligned_query);
      aligned_query = nullptr;
    }

    if (_inserted_into_pool_rs != nullptr) {
      delete _inserted_into_pool_rs;
      _inserted_into_pool_rs = nullptr;
    }
    if (_inserted_into_pool_bs != nullptr) {
      delete _inserted_into_pool_bs;
      _inserted_into_pool_bs = nullptr;
    }

    search_l = indexing_l = r = 0;
  }

  template DISKANN_DLLEXPORT InMemQueryScratch<int8_t>::InMemQueryScratch();
  template DISKANN_DLLEXPORT InMemQueryScratch<uint8_t>::InMemQueryScratch();
  template DISKANN_DLLEXPORT InMemQueryScratch<float>::InMemQueryScratch();

  template DISKANN_DLLEXPORT void InMemQueryScratch<int8_t>::setup(
      uint32_t search_l, uint32_t indexing_l, uint32_t r, size_t dim);
  template DISKANN_DLLEXPORT void InMemQueryScratch<uint8_t>::setup(
      uint32_t search_l, uint32_t indexing_l, uint32_t r, size_t dim);
  template DISKANN_DLLEXPORT void InMemQueryScratch<float>::setup(
      uint32_t search_l, uint32_t indexing_l, uint32_t r, size_t dim);

  template DISKANN_DLLEXPORT void InMemQueryScratch<int8_t>::clear();
  template DISKANN_DLLEXPORT void InMemQueryScratch<uint8_t>::clear();
  template DISKANN_DLLEXPORT void InMemQueryScratch<float>::clear();

  template DISKANN_DLLEXPORT void InMemQueryScratch<int8_t>::resize_for_query(
      uint32_t new_search_l);
  template DISKANN_DLLEXPORT void InMemQueryScratch<uint8_t>::resize_for_query(
      uint32_t new_search_l);
  template DISKANN_DLLEXPORT void InMemQueryScratch<float>::resize_for_query(
      uint32_t new_search_l);

  template DISKANN_DLLEXPORT void InMemQueryScratch<int8_t>::destroy();
  template DISKANN_DLLEXPORT void InMemQueryScratch<uint8_t>::destroy();
  template DISKANN_DLLEXPORT void InMemQueryScratch<float>::destroy();

}  // namespace diskann