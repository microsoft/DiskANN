// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.


#include "pq.h"

namespace diskann {
  FixedChunkPQTable::FixedChunkPQTable() {
  }

  FixedChunkPQTable::~FixedChunkPQTable() {
#ifndef EXEC_ENV_OLS
    if (tables != nullptr)
      delete[] tables;
    if (tables_tr != nullptr)
      delete[] tables_tr;
    if (chunk_offsets != nullptr)
      delete[] chunk_offsets;
    if (centroid != nullptr)
      delete[] centroid;
    if (rotmat_tr != nullptr)
      delete[] rotmat_tr;
#endif
  }

#ifdef EXEC_ENV_OLS
  void FixedChunkPQTable::load_pq_centroid_bin(MemoryMappedFiles& files,
                                               const char*        pq_table_file,
                                               size_t             num_chunks) {
#else
  void FixedChunkPQTable::load_pq_centroid_bin(const char* pq_table_file,
                                               size_t      num_chunks) {
#endif

    _u64        nr, nc;
    std::string rotmat_file =
        std::string(pq_table_file) + "_rotation_matrix.bin";

#ifdef EXEC_ENV_OLS
    _u64* file_offset_data;  // since load_bin only sets the pointer, no need
                             // to delete.
    diskann::load_bin<_u64>(files, pq_table_file, file_offset_data, nr, nc);
#else
    std::unique_ptr<_u64[]> file_offset_data;
    diskann::load_bin<_u64>(pq_table_file, file_offset_data, nr, nc);
#endif

    if (nr != 4) {
      diskann::cout << "Error reading pq_pivots file " << pq_table_file
                    << ". Offsets dont contain correct metadata, # offsets = "
                    << nr << ", but expecting " << 4;
      throw diskann::ANNException(
          "Error reading pq_pivots file at offsets data.", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }

    diskann::cout << "Offsets: " << file_offset_data[0] << " "
                  << file_offset_data[1] << " " << file_offset_data[2] << " "
                  << file_offset_data[3] << std::endl;

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
    diskann::load_bin<uint32_t>(files, pq_table_file, chunk_offsets, nr, nc,
                                file_offset_data[2]);
#else
    diskann::load_bin<uint32_t>(pq_table_file, chunk_offsets, nr, nc,
                                file_offset_data[2]);
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

    if (file_exists(rotmat_file)) {
#ifdef EXEC_ENV_OLS
      diskann::load_bin<float>(files, rotmat_file, (float*&) rotmat_tr, nr, nc);
#else
      diskann::load_bin<float>(rotmat_file, rotmat_tr, nr, nc);
#endif
      if (nr != this->ndims || nc != this->ndims) {
        diskann::cerr << "Error loading rotation matrix file" << std::endl;
        throw diskann::ANNException("Error loading rotation matrix file", -1,
                                    __FUNCSIG__, __FILE__, __LINE__);
      }
      use_rotation = true;
    }

    // alloc and compute transpose
    tables_tr = new float[256 * this->ndims];
    for (_u64 i = 0; i < 256; i++) {
      for (_u64 j = 0; j < this->ndims; j++) {
        tables_tr[j * 256 + i] = tables[i * this->ndims + j];
      }
    }
  }

  _u32 FixedChunkPQTable::get_num_chunks() {
    return static_cast<_u32>(n_chunks);
  }

  void FixedChunkPQTable::preprocess_query(float* query_vec) {
    for (_u32 d = 0; d < ndims; d++) {
      query_vec[d] -= centroid[d];
    }
    std::vector<float> tmp(ndims, 0);
    if (use_rotation) {
      for (_u32 d = 0; d < ndims; d++) {
        for (_u32 d1 = 0; d1 < ndims; d1++) {
          tmp[d] += query_vec[d1] * rotmat_tr[d1 * ndims + d];
        }
      }
      std::memcpy(query_vec, tmp.data(), ndims * sizeof(float));
    }
  }

  // assumes pre-processed query
  void FixedChunkPQTable::populate_chunk_distances(const float* query_vec,
                                                   float*       dist_vec) {
    memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
    // chunk wise distance computation
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float* chunk_dists = dist_vec + (256 * chunk);
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        const float* centers_dim_vec = tables_tr + (256 * j);
        for (_u64 idx = 0; idx < 256; idx++) {
          double diff = centers_dim_vec[idx] - (query_vec[j]);
          chunk_dists[idx] += (float) (diff * diff);
        }
      }
    }
  }

  float FixedChunkPQTable::l2_distance(const float* query_vec, _u8* base_vec) {
    float res = 0;
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        const float* centers_dim_vec = tables_tr + (256 * j);
        float        diff = centers_dim_vec[base_vec[chunk]] - (query_vec[j]);
        res += diff * diff;
      }
    }
    return res;
  }

  float FixedChunkPQTable::inner_product(const float* query_vec,
                                         _u8*         base_vec) {
    float res = 0;
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        const float* centers_dim_vec = tables_tr + (256 * j);
        float        diff = centers_dim_vec[base_vec[chunk]] *
                     query_vec[j];  // assumes centroid is 0 to
                                    // prevent translation errors
        res += diff;
      }
    }
    return -res;  // returns negative value to simulate distances (max -> min
                  // conversion)
  }

  // assumes no rotation is involved
  void FixedChunkPQTable::inflate_vector(_u8* base_vec, float* out_vec) {
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        const float* centers_dim_vec = tables_tr + (256 * j);
        out_vec[j] = centers_dim_vec[base_vec[chunk]] + centroid[j];
      }
    }
  }

  void FixedChunkPQTable::populate_chunk_inner_products(const float* query_vec,
                                                        float*       dist_vec) {
    memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
    // chunk wise distance computation
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float* chunk_dists = dist_vec + (256 * chunk);
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        const float* centers_dim_vec = tables_tr + (256 * j);
        for (_u64 idx = 0; idx < 256; idx++) {
          double prod =
              centers_dim_vec[idx] * query_vec[j];  // assumes that we are not
                                                    // shifting the vectors to
                                                    // mean zero, i.e., centroid
                                                    // array should be all zeros
          chunk_dists[idx] -=
              (float) prod;  // returning negative to keep the search code clean
                             // (max inner product vs min distance)
        }
      }
    }
  }

  void aggregate_coords(const unsigned* ids, const _u64 n_ids,
                        const _u8* all_coords, const _u64 ndims, _u8* out) {
    for (_u64 i = 0; i < n_ids; i++) {
      memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(_u8));
    }
  }

  void pq_dist_lookup(const _u8* pq_ids, const _u64 n_pts,
                      const _u64 pq_nchunks, const float* pq_dists,
                      float* dists_out) {
    _mm_prefetch((char*) dists_out, _MM_HINT_T0);
    _mm_prefetch((char*) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char*) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char*) (pq_ids + 128), _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
      const float* chunk_dists = pq_dists + 256 * chunk;
      if (chunk < pq_nchunks - 1) {
        _mm_prefetch((char*) (chunk_dists + 256), _MM_HINT_T0);
      }
      for (_u64 idx = 0; idx < n_pts; idx++) {
        _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
        dists_out[idx] += chunk_dists[pq_centerid];
      }
    }
  }
}  // namespace diskann
