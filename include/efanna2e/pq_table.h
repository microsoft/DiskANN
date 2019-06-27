#pragma once
#include "efanna2e/util.h"

namespace NSG {
  template<typename T>
  class PQTable {
   public:
    virtual void load_bin(const char* filename) = 0;
    virtual void convert(const T* in_vec, float* out_vec) = 0;
  };

  class FixedChunkPQTable : PQTable<_u8> {
    // data_dim = n_chunks * chunk_size;
    float* tables;      // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    _u64   n_chunks;    // n_chunks = # of chunks ndims is split into
    _u64   chunk_size;  // chunk_size = chunk size of each dimension chunk
    _u64   ndims;       // ndims = chunk_size * n_chunks
    float* tables_T;    // same as pq_tables, but col-major
   public:
    FixedChunkPQTable(_u64 nchunks, _u64 chunksize)
        : n_chunks(nchunks), chunk_size(chunksize) {
    }

    ~FixedChunkPQTable() {
      delete[] tables;
      delete[] tables_T;
    }

    void load_bin(const char* filename) override {
      // bin structure: [256][ndims][ndims(float)]
      unsigned npts_u32, ndims_u32;
      size_t   npts_u64, ndims_u64;
      NSG::load_bin<float>(filename, tables, npts_u64, ndims_u64);
      npts_u32 = npts_u64;
      ndims_u32 = ndims_u64;
      std::cout << "PQ Pivots: # ctrs: " << npts_u32
                << ", # dims: " << ndims_u32 << std::endl;
      ndims = n_chunks * chunk_size;
      assert((_u64) ndims_u32 == n_chunks * chunk_size);
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
    virtual void convert(const _u8* in_vec, float* out_vec) override {
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

    void populate_chunk_distances(const float* query_vec, float* dist_vec) {
      memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
      // chunk wise distance computation
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float* chunk_dists = dist_vec + (256 * chunk);
        for (_u64 j = 0; j < chunk_size; j++) {
          _u64         dim_no = (chunk * chunk_size) + j;
          const float* centers_dim_vec = tables_T + (256 * dim_no);
          for (_u64 idx = 0; idx < 256; idx++) {
            float diff = centers_dim_vec[idx] - query_vec[dim_no];
            chunk_dists[idx] += (diff * diff);
          }
        }
      }
    }
  };

  class IVFPQTable {
   public:
    // ivf_npivots X data_dim
    float* ivf_pivots = nullptr;
    // pq_npivots X data_dim
    float* pq_pivots = nullptr;
    // npts X 1
    _u16* ivf_data = nullptr;
    // npts X pq_nchunks
    _u16* pq_data = nullptr;
    _u64  npts, data_dim, ivf_npivots, pq_npivots, pq_nchunks, pq_chunk_size;
    IVFPQTable(const char* ivf_piv_filename, const char* ivf_data_ivecs,
               const char* pq_piv_filename, const char* pq_data_ivecs,
               _u64 pq_chunk_size)
        : pq_chunk_size(pq_chunk_size) {
      // load ivf pivots from file
      unsigned ivf_npivs_u32, data_ndims_u32;
      load_Tvecs<float>(ivf_piv_filename, ivf_pivots, ivf_npivs_u32,
                        data_ndims_u32);
      ivf_npivots = (_u64) ivf_npivs_u32;
      data_dim = (_u64) data_ndims_u32;
      std::cout << "IVF Pivots: # pivots = " << ivf_npivots
                << ", # dims = " << data_dim << "\n";

      // load ivf assignments in _u32 form to _u16
      _u64 ivf_data_dim = 0;
      block_load_convert_Tvecs<_u32, _u16>(ivf_data_ivecs, ivf_data, npts,
                                           ivf_data_dim);
      assert(ivf_data_dim == 1);

      // load PQ pivots from file
      _u32 pq_data_dim_u32 = 0, pq_npivots_u32;
      load_Tvecs<float>(pq_piv_filename, pq_pivots, pq_npivots_u32,
                        pq_data_dim_u32);
      pq_npivots = (_u64) pq_npivots_u32;
      std::cout << "PQ Pivots: # pivots = " << pq_npivots
                << ", # dims = " << pq_data_dim_u32 << "\n";
      assert(data_dim == (_u64) pq_data_dim_u32);

      // load PQ data in _u32 form to _u16
      _u64 pq_data_dim = 0, pq_data_npts = 0;
      block_load_convert_Tvecs<_u32, _u16>(pq_data_ivecs, pq_data, pq_data_npts,
                                           pq_data_dim);
      std::cout << "PQ Data: # pts = " << pq_data_npts
                << ", # chunks = " << pq_data_dim
                << ", chunk size = " << pq_chunk_size << "\n";
      pq_nchunks = pq_data_dim;
    }

    ~IVFPQTable() {
      free(ivf_pivots);
      free(pq_pivots);
      free(ivf_data);
      free(pq_data);
    }

    void convert(const _u64 id, float* out_vec) {
      // extract PQ data for ID
      _u16* id_pq_data = pq_data + (pq_nchunks * id);
      /*
      std::cout << "Input: " << id << " -- PQ: ";
      for(_u64 k=0;k<pq_nchunks;k++)
        std::cout << id_pq_data[k] << " ";
      std::cout << "\n";
      */
      // construct residual vector for ID
      for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
        _u16   chunk_val = id_pq_data[chunk];
        float* chunk_residuals =
            pq_pivots + (data_dim * chunk_val) + (chunk * pq_chunk_size);
        _u64 chunk_size =
            (std::min)(data_dim - (chunk * pq_chunk_size), pq_chunk_size);
        float* chunk_out_vec = out_vec + (chunk * pq_chunk_size);
        // avoid memcpy for fast memcpy using loops
        for (_u64 d = 0; d < chunk_size; d++) {
          *(chunk_out_vec + d) = *(chunk_residuals + d);
        }
      }

      // add center value
      _u16   id_ivf = ivf_data[id];
      float* ivf_center = ivf_pivots + (data_dim * id_ivf);
      for (_u64 d = 0; d < data_dim; d++) {
        *(out_vec + d) += *(ivf_center + d);
      }
    }
  };
}  // namespace NSG
