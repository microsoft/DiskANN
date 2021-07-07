// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "utils.h"

#define NUM_PQ_CENTROIDS 256
#define NUM_PQ_OFFSETS 5

namespace diskann {
  template<typename T>
  class FixedChunkPQTable {
    // data_dim = n_chunks * chunk_size;
    float* tables =
        nullptr;  // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    //    _u64   n_chunks;    // n_chunks = # of chunks ndims is split into
    //    _u64   chunk_size;  // chunk_size = chunk size of each dimension chunk
    _u64   ndims;  // ndims = chunk_size * n_chunks
    _u64   n_chunks;
    _u32*  chunk_offsets = nullptr;
    _u32*  rearrangement = nullptr;
    float* centroid = nullptr;
    float* tables_T = nullptr;  // same as pq_tables, but col-major
    float* all_to_all_dists = nullptr;

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
      if (all_to_all_dists != nullptr)
        delete[] all_to_all_dists;
#endif
    }

    _u64 get_dim() {
      return ndims;
    }

#ifdef EXEC_ENV_OLS
    void load_pq_pivots_old(MemoryMappedFiles& files,
                            const std::string& pq_pivots_path,
                            size_t             num_chunks) {
#else
    void load_pq_pivots_old(const std::string& pq_pivots_path,
                            size_t             num_chunks) {
#endif
      _u64 nr, nc;
// Load the pq pivots.
#ifdef EXEC_ENV_OLS
      diskann::load_bin<float>(files, pq_pivots_path.c_str(), tables, nr, nc);
#else
      diskann::load_bin<float>(pq_pivots_path.c_str(), tables, nr, nc);
#endif

      if ((nr != NUM_PQ_CENTROIDS)) {
        diskann::cout << "Error reading pq_pivots file " << pq_pivots_path
                      << ". file_num_centers  = " << nr << " but expecting "
                      << NUM_PQ_CENTROIDS << " centers";
        throw diskann::ANNException(
            "Error reading pq_pivots file at pivots data.", -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }

      this->ndims = nc;

// Load the PQ centroids
//_pq_pivots.bin_centroid.bin
//_chunk_offsets.bin
//_rearrangement_perm.bin
#ifdef EXEC_ENV_OLS
      diskann::load_bin<float>(files, pq_pivots_path + "_centroid.bin",
                               centroid, nr, nc);
#else
      diskann::load_bin<float>(pq_pivots_path + "_centroid.bin", centroid, nr,
                               nc);
#endif

      if ((nr != this->ndims) || (nc != 1)) {
        diskann::cerr << "Error reading centroids from pq_pivots file "
                      << pq_pivots_path << ". file_dim  = " << nr
                      << ", file_cols = " << nc << " but expecting "
                      << this->ndims << " entries in 1 dimension.";
        throw diskann::ANNException(
            "Error reading pq_pivots file at centroid data.", -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }

#ifdef EXEC_ENV_OLS
      diskann::load_bin<uint32_t>(files,
                                  pq_pivots_path + "_rearrangement_perm.bin",
                                  rearrangement, nr, nc);
#else
      diskann::load_bin<uint32_t>(pq_pivots_path + "_rearrangement_perm.bin",
                                  rearrangement, nr, nc);
#endif
      if ((nr != this->ndims) || (nc != 1)) {
        diskann::cerr << "Error reading re-arrangement data pq_pivots file "
                      << pq_pivots_path << ". file_dim  = " << nr
                      << ", file_cols = " << nc << " but expecting "
                      << this->ndims << " entries in 1 dimension.";
        throw diskann::ANNException(
            "Error reading pq_pivots file at re-arrangement data.", -1,
            __FUNCSIG__, __FILE__, __LINE__);
      }

#ifdef EXEC_ENV_OLS
      diskann::load_bin<uint32_t>(files, pq_pivots_path + "_chunk_offsets.bin",
                                  chunk_offsets, nr, nc);
#else
      diskann::load_bin<uint32_t>(pq_pivots_path + "_chunk_offsets.bin",
                                  chunk_offsets, nr, nc);
#endif

      if (nr != (uint64_t) num_chunks + 1 || nc != 1) {
        diskann::cerr
            << "Error reading pq_pivots file at chunk offsets; file has nr="
            << nr << ",nc=" << nc << ", expecting nr=" << num_chunks + 1
            << ", nc=1." << std::endl;
        throw diskann::ANNException(
            "Error reading pq_pivots file at chunk offsets.", -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }

      this->n_chunks = num_chunks;
      diskann::cout << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS
                    << ", #dims: " << this->ndims
                    << ", #chunks: " << this->n_chunks << std::endl;
    }

#ifdef EXEC_ENV_OLS
    void load_pq_pivots_new(MemoryMappedFiles& files,
                            const std::string& pq_pivots_path,
                            size_t num_chunks, size_t offset) {
#else
    void load_pq_pivots_new(const std::string& pq_pivots_path,
                            size_t num_chunks, size_t offset) {
#endif

      _u64 nr, nc;
#ifdef EXEC_ENV_OLS
      _u64* file_offset_data;  // since load_bin only sets the pointer, no need
                               // to delete.
      diskann::load_bin<_u64>(files, pq_pivots_path, file_offset_data, nr, nc,
                              offset);
#else
      std::unique_ptr<_u64[]> file_offset_data;
      diskann::load_bin<_u64>(pq_pivots_path, file_offset_data, nr, nc, offset);
#endif

      if (nr != NUM_PQ_OFFSETS) {
        diskann::cout << "Error reading pq_pivots file " << pq_pivots_path
                      << ". Offsets dont contain correct metadata, # offsets = "
                      << nr << ", but expecting " << NUM_PQ_OFFSETS;
        throw diskann::ANNException(
            "Error reading pq_pivots file at offsets data.", -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }

      diskann::cout << "Offsets: " << file_offset_data[0] << " "
                    << file_offset_data[1] << " " << file_offset_data[2] << " "
                    << file_offset_data[3] << " " << file_offset_data[4]
                    << std::endl;

#ifdef EXEC_ENV_OLS
      diskann::load_bin<float>(files, pq_pivots_path.c_str(), tables, nr, nc,
                               file_offset_data[0] + offset);
#else
      diskann::load_bin<float>(pq_pivots_path.c_str(), tables, nr, nc,
                               file_offset_data[0] + offset);
#endif

      if ((nr != NUM_PQ_CENTROIDS)) {
        diskann::cout << "Error reading pq_pivots file " << pq_pivots_path
                      << ". file_num_centers  = " << nr << " but expecting "
                      << NUM_PQ_CENTROIDS << " centers";
        throw diskann::ANNException(
            "Error reading pq_pivots file at pivots data.", -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }

      this->ndims = nc;

#ifdef EXEC_ENV_OLS
      diskann::load_bin<float>(files, pq_pivots_path.c_str(), centroid, nr, nc,
                               file_offset_data[1] + offset);
#else
      diskann::load_bin<float>(pq_pivots_path.c_str(), centroid, nr, nc,
                               file_offset_data[1] + offset);
#endif

      if ((nr != this->ndims) || (nc != 1)) {
        diskann::cerr << "Error reading centroids from pq_pivots file "
                      << pq_pivots_path << ". file_dim  = " << nr
                      << ", file_cols = " << nc << " but expecting "
                      << this->ndims << " entries in 1 dimension.";
        throw diskann::ANNException(
            "Error reading pq_pivots file at centroid data.", -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }

#ifdef EXEC_ENV_OLS
      diskann::load_bin<uint32_t>(files, pq_pivots_path.c_str(), rearrangement,
                                  nr, nc, file_offset_data[2] + offset);
#else
      diskann::load_bin<uint32_t>(pq_pivots_path.c_str(), rearrangement, nr, nc,
                                  file_offset_data[2] + offset);
#endif
      if ((nr != this->ndims) || (nc != 1)) {
        diskann::cerr << "Error reading re-arrangement data pq_pivots file "
                      << pq_pivots_path << ". file_dim  = " << nr
                      << ", file_cols = " << nc << " but expecting "
                      << this->ndims << " entries in 1 dimension.";
        throw diskann::ANNException(
            "Error reading pq_pivots file at re-arrangement data.", -1,
            __FUNCSIG__, __FILE__, __LINE__);
      }

#ifdef EXEC_ENV_OLS
      diskann::load_bin<uint32_t>(files, pq_pivots_path.c_str(), chunk_offsets,
                                  nr, nc, file_offset_data[3] + offset);
#else
      diskann::load_bin<uint32_t>(pq_pivots_path.c_str(), chunk_offsets, nr, nc,
                                  file_offset_data[3] + offset);
#endif

      if (nr != (uint64_t) num_chunks + 1 || nc != 1) {
        diskann::cerr
            << "Error reading pq_pivots file at chunk offsets; file has nr="
            << nr << ",nc=" << nc << ", expecting nr=" << num_chunks + 1
            << ", nc=1." << std::endl;
        throw diskann::ANNException(
            "Error reading pq_pivots file at chunk offsets.", -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }

      this->n_chunks = num_chunks;
      diskann::cout << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS
                    << ", #dims: " << this->ndims
                    << ", #chunks: " << this->n_chunks << std::endl;
    }

#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles& files,
                              const char* pq_table_file, size_t num_chunks,
                              size_t offset = 0){
#else
    void load_pq_centroid_bin(const char* pq_table_file, size_t num_chunks,
                              size_t offset = 0) {
#endif

        std::string pq_pivots_path(pq_table_file);
    _u64 nr, nc;

#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_file, nr, nc, offset);
#else
      get_bin_metadata(pq_table_file, nr, nc, offset);
#endif

    if (nr == NUM_PQ_OFFSETS) {
#ifdef EXEC_ENV_OLS
      load_pq_pivots_new(files, pq_table_file, num_chunks, offset);
#else
        load_pq_pivots_new(pq_table_file, num_chunks, offset);
#endif
    } else if (nr == NUM_PQ_CENTROIDS) {
#ifdef EXEC_ENV_OLS
      load_pq_pivots_old(files, pq_table_file, num_chunks);
#else
        load_pq_pivots_old(pq_table_file, num_chunks);
#endif
    }

    // alloc and compute transpose
    tables_T = new float[256 * ndims];
    for (_u64 i = 0; i < 256; i++) {
      for (_u64 j = 0; j < ndims; j++) {
        tables_T[j * 256 + i] = tables[i * ndims + j];
      }
    }

    // added this for easy PQ-PQ squared-distance calculations
    // TODO: Create only for StreamingMerger.
    all_to_all_dists = new float[256 * 256 * n_chunks];
    std::memset(all_to_all_dists, 0, 256 * 256 * n_chunks * sizeof(float));
    // should perhaps optimize later
    for (_u32 i = 0; i < 256; i++) {
      for (_u32 j = 0; j < 256; j++) {
        for (_u32 c = 0; c < n_chunks; c++) {
          for (_u64 d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
            float diff = (tables[i * ndims + d] - tables[j * ndims + d]);
            all_to_all_dists[i * 256 * n_chunks + j * n_chunks + c] +=
                diff * diff;
          }
        }
      }
    }
    diskann::cout << "Finished optimizing for PQ-PQ distance compuation "
                  << std::endl;

  }

  void
  populate_chunk_distances(const T* query_vec, float* dist_vec) {
    memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
    // chunk wise distance computation
    for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float* chunk_dists = dist_vec + (256 * chunk);
      for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
        _u64         permuted_dim_in_query = rearrangement[j];
        const float* centers_dim_vec = tables_T + (256 * j);
        for (_u64 idx = 0; idx < 256; idx++) {
          // Fixing crash in v14 machines.
          // float diff = centers_dim_vec[idx] -
          //             ((float) query_vec[permuted_dim_in_query] -
          //              centroid[permuted_dim_in_query]);
          // chunk_dists[idx] += (diff * diff);
          double diff =
              centers_dim_vec[idx] - (query_vec[permuted_dim_in_query] -
                                      centroid[permuted_dim_in_query]);
          chunk_dists[idx] += (float) (diff * diff);
        }
      }
    }
  }

  // computes PQ distance between comp_src and comp_dsts in efficient manner
  // comp_src: [nchunks]
  // comp_dsts: count * [nchunks]
  // dists: [count]
  // TODO (perf) :: re-order computation to get better locality
  void compute_distances(const _u8* comp_src, const _u8* comp_dsts,
                         float* dists, const _u32 count) {
    std::memset(dists, 0, count * sizeof(float));
    for (_u64 i = 0; i < count; i++) {
      for (_u64 c = 0; c < n_chunks; c++) {
        dists[i] +=
            all_to_all_dists[(_u64) comp_src[c] * 256 * n_chunks +
                             (_u64) comp_dsts[i * n_chunks + c] * n_chunks + c];
      }
    }
  }

  // fp_vec: [ndims]
  // out_pq_vec : [nchunks]
  void deflate_vec(const float* fp_vec, _u8* out_pq_vec) {
    // permute the vector according to PQ rearrangement, compute all distances
    // to 256 centroids and choose the closest (for each chunk)
    for (_u32 c = 0; c < n_chunks; c++) {
      float closest_dist = std::numeric_limits<float>::max();
      for (_u32 i = 0; i < 256; i++) {
        float cur_dist = 0;
        for (_u64 d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
          float diff =
              (tables[i * ndims + d] -
               ((float) fp_vec[rearrangement[d]] - centroid[rearrangement[d]]));
          cur_dist += diff * diff;
        }
        if (cur_dist < closest_dist) {
          closest_dist = cur_dist;
          out_pq_vec[c] = (_u8) i;
        }
      }
    }
  }
};  // namespace diskann
}  // namespace diskann
