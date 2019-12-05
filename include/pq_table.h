#pragma once
#include "utils.h"

namespace diskann {
  template<typename T>
  class FixedChunkPQTable {
    // data_dim = n_chunks * chunk_size;
    float* tables =
        nullptr;        // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    _u64   n_chunks;    // n_chunks = # of chunks ndims is split into
    _u64   chunk_size;  // chunk_size = chunk size of each dimension chunk
    _u64   ndims;       // ndims = chunk_size * n_chunks
    float* tables_T = nullptr;  // same as pq_tables, but col-major
   public:
    FixedChunkPQTable() {
    }

    virtual ~FixedChunkPQTable() {
      if (tables != nullptr) {
        delete[] tables;
      }
      if (tables_T != nullptr) {
        delete[] tables_T;
      }
    }

    void load_pq_centroid_bin(const char* filename, _u64 nchunks,
                              _u64 chunksize) {
      this->n_chunks = nchunks;
      this->chunk_size = chunksize;
      // bin structure: [256][ndims][ndims(float)]
      unsigned npts_u32, ndims_u32;
      size_t   npts_u64, ndims_u64;
      diskann::load_bin<float>(filename, tables, npts_u64, ndims_u64);
      npts_u32 = npts_u64;
      ndims_u32 = ndims_u64;
      std::cout << "PQ Pivots: #ctrs: " << npts_u32 << ", #dims: " << ndims_u32
                << ", #chunks: " << nchunks << ", chunk_size: " << chunksize
                << std::endl;
      this->ndims = ndims_u32;
      //      assert((_u64) ndims_u32 == n_chunks * chunk_size);
      // alloc and compute transpose
      tables_T = new float[256 * ndims_u32];
      for (_u64 i = 0; i < 256; i++) {
        for (_u64 j = 0; j < ndims_u32; j++) {
          tables_T[j * 256 + i] = tables[i * ndims_u32 + j];
        }
      }
    }

    // in_vec = _u8 * [n_chunks]
    // out_vec = float* [ndims]
    virtual void convert(const _u8* in_vec, float* out_vec) {
      // _mm_prefetch((char*) tables, 3);
      _mm_prefetch((char*) in_vec, _MM_HINT_T1);
      // prefetch full out_vec
      _u64 n_floats_per_line = 16;
      _u64 n_lines = ROUND_UP(ndims, n_floats_per_line) / n_floats_per_line;
      for (_u64 line = 0; line < n_lines; line++) {
        _mm_prefetch((char*) (out_vec + 16 * line), _MM_HINT_T1);
      }

      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        const _u8    pq_idx = *(in_vec + chunk);
        const float* vals = (tables + (ndims * pq_idx)) + (chunk * chunk_size);
        float*       chunk_out = out_vec + (chunk * chunk_size);
        // avoiding memcpy as chunk size is at most 10
        switch (chunk_size) {
          case 2:
            *chunk_out++ = *vals++;
            *chunk_out++ = *vals++;
            break;
          case 3:
            *chunk_out++ = *vals++;
            *chunk_out++ = *vals++;
            *chunk_out++ = *vals++;
            break;
          default:
            for (_u64 i = 0; i < chunk_size; i++) {
              *(chunk_out + i) = *(vals + i);
            }
        }
      }
    }

    void populate_chunk_distances(const T* query_vec, float* dist_vec) {
      memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
      // chunk wise distance computation
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float* chunk_dists = dist_vec + (256 * chunk);
        for (_u64 j = 0; j < chunk_size; j++) {
          _u64 dim_no = (chunk * chunk_size) + j;
          if (dim_no == this->ndims)
            break;
          const float* centers_dim_vec = tables_T + (256 * dim_no);
          for (_u64 idx = 0; idx < 256; idx++) {
            float diff = centers_dim_vec[idx] - (float) query_vec[dim_no];
            chunk_dists[idx] += (diff * diff);
          }
        }
      }
    }
  };
}  // namespace diskann
