// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "utils.h"

#define NUM_PQ_CENTROIDS 256

namespace diskann {
  class FixedChunkPQTable {
    float* tables = nullptr;  // pq_tables = float array of size [256 * ndims]
    _u64   ndims = 0;         // ndims = true dimension of vectors
    _u64   n_chunks = 0;
    _u32*  chunk_offsets = nullptr;
    _u32*  rearrangement = nullptr;
    float* centroid = nullptr;
    float* tables_T = nullptr;  // same as pq_tables, but col-major
   public:
    FixedChunkPQTable() {
    }

    virtual ~FixedChunkPQTable() {
#ifndef EXEC_ENV_OLS
      if (tables != nullptr)
        delete[] tables;
      if (tables_T != nullptr)
        delete[] tables_T;
      if (rearrangement != nullptr)
        delete[] rearrangement;
      if (chunk_offsets != nullptr)
        delete[] chunk_offsets;
      if (centroid != nullptr)
        delete[] centroid;
#endif
    }

#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles& files,
                              const char* pq_table_file, size_t num_chunks){
#else
    void load_pq_centroid_bin(const char* pq_table_file, size_t num_chunks) {
#endif

        _u64 nr, nc;
#ifdef EXEC_ENV_OLS
    _u64* file_offset_data;  // since load_bin only sets the pointer, no need
                             // to delete.
    diskann::load_bin<_u64>(files, pq_table_file, file_offset_data, nr, nc);
#else
      std::unique_ptr<_u64[]> file_offset_data;
      diskann::load_bin<_u64>(pq_table_file, file_offset_data, nr, nc);
#endif

    if (nr != 5) {
      diskann::cout << "Error reading pq_pivots file " << pq_table_file
                    << ". Offsets dont contain correct metadata, # offsets = "
                    << nr << ", but expecting " << 5;
      throw diskann::ANNException(
          "Error reading pq_pivots file at offsets data.", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }

    diskann::cout << "Offsets: " << file_offset_data[0] << " "
                  << file_offset_data[1] << " " << file_offset_data[2] << " "
                  << file_offset_data[3] << " " << file_offset_data[4]
                  << std::endl;

#ifdef EXEC_ENV_OLS
    diskann::load_bin<float>(files, pq_table_file, tables, nr, nc,
                             file_offset_data[0]);
#else
      diskann::load_bin<float>(pq_table_file, tables, nr, nc,
                               file_offset_data[0]);
#endif

    if ((nr != NUM_PQ_CENTROIDS)) {
      diskann::cout << "Error reading pq_pivots file " << pq_table_file
                    << ". file_num_centers  = " << nr << " but expecting "
                    << NUM_PQ_CENTROIDS << " centers";
      throw diskann::ANNException(
          "Error reading pq_pivots file at pivots data.", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }

    this->ndims = nc;

#ifdef EXEC_ENV_OLS
    diskann::load_bin<float>(files, pq_table_file, centroid, nr, nc,
                             file_offset_data[1]);
#else
      diskann::load_bin<float>(pq_table_file, centroid, nr, nc,
                               file_offset_data[1]);
#endif

    if ((nr != this->ndims) || (nc != 1)) {
      diskann::cerr << "Error reading centroids from pq_pivots file "
                    << pq_table_file << ". file_dim  = " << nr
                    << ", file_cols = " << nc << " but expecting "
                    << this->ndims << " entries in 1 dimension.";
      throw diskann::ANNException(
          "Error reading pq_pivots file at centroid data.", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }

#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint32_t>(files, pq_table_file, rearrangement, nr, nc,
                                file_offset_data[2]);
#else
      diskann::load_bin<uint32_t>(pq_table_file, rearrangement, nr, nc,
                                  file_offset_data[2]);
#endif
    if ((nr != this->ndims) || (nc != 1)) {
      diskann::cerr << "Error reading re-arrangement data pq_pivots file "
                    << pq_table_file << ". file_dim  = " << nr
                    << ", file_cols = " << nc << " but expecting "
                    << this->ndims << " entries in 1 dimension.";
      throw diskann::ANNException(
          "Error reading pq_pivots file at re-arrangement data.", -1,
          __FUNCSIG__, __FILE__, __LINE__);
    }

#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint32_t>(files, pq_table_file, chunk_offsets, nr, nc,
                                file_offset_data[3]);
#else
      diskann::load_bin<uint32_t>(pq_table_file, chunk_offsets, nr, nc,
                                  file_offset_data[3]);
#endif

    if (nc != 1 || (nr != num_chunks + 1 && num_chunks != 0)) {
      diskann::cerr << "Error loading chunk offsets file. numc: " << nc
                    << " (should be 1). numr: " << nr << " (should be "
                    << num_chunks + 1 << " or 0 if we need to infer)"
                    << std::endl;
      throw diskann::ANNException("Error loading chunk offsets file", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }

    this->n_chunks = nr - 1;
    diskann::cout << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS
                  << ", #dims: " << this->ndims
                  << ", #chunks: " << this->n_chunks << std::endl;

    // alloc and compute transpose
    tables_T = new float[256 * this->ndims];
    for (_u64 i = 0; i < 256; i++) {
      for (_u64 j = 0; j < this->ndims; j++) {
        tables_T[j * 256 + i] = tables[i * this->ndims + j];
      }
    }
  }

  _u32
  get_num_chunks() {
    return static_cast<_u32>(n_chunks);
  }
  void populate_chunk_distances(const float* query_vec, float* dist_vec) {
    memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
    // chunk wise distance computation
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float* chunk_dists = dist_vec + (256 * chunk);
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         permuted_dim_in_query = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        for (_u64 idx = 0; idx < 256; idx++) {
          double diff =
              centers_dim_vec[idx] - (query_vec[permuted_dim_in_query] -
                                      centroid[permuted_dim_in_query]);
          chunk_dists[idx] += (float) (diff * diff);
        }
      }
    }
  }

  float l2_distance(const float* query_vec, _u8* base_vec) {
    float res = 0;
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         permuted_dim_in_query = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        float        diff = centers_dim_vec[base_vec[chunk]] -
                     (query_vec[permuted_dim_in_query] -
                      centroid[permuted_dim_in_query]);
        res += diff * diff;
      }
    }
    return res;
  }

  float inner_product(const float* query_vec, _u8* base_vec) {
    float res = 0;
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         permuted_dim_in_query = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        float        diff =
            centers_dim_vec[base_vec[chunk]] *
            query_vec[permuted_dim_in_query];  // assumes centroid is 0 to
                                               // prevent translation errors
        res += diff;
      }
    }
    return -res;  // returns negative value to simulate distances (max -> min
                  // conversion)
  }

  void inflate_vector(_u8* base_vec, float* out_vec) {
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         original_dim = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        out_vec[original_dim] =
            centers_dim_vec[base_vec[chunk]] + centroid[original_dim];
      }
    }
  }

  void populate_chunk_inner_products(const float* query_vec, float* dist_vec) {
    memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
    // chunk wise distance computation
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float* chunk_dists = dist_vec + (256 * chunk);
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         permuted_dim_in_query = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        for (_u64 idx = 0; idx < 256; idx++) {
          double prod =
              centers_dim_vec[idx] *
              query_vec[permuted_dim_in_query];  // assumes that we are not
                                                 // shifting the vectors to mean
                                                 // zero, i.e., centroid array
                                                 // should be all zeros
          chunk_dists[idx] -=
              (float) prod;  // returning negative to keep the search code clean
                             // (max inner product vs min distance)
        }
      }
    }
  }
};  // namespace diskann
}  // namespace diskann
