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
   public:
    FixedChunkPQTable(_u64 nchunks, _u64 chunksize)
        : n_chunks(nchunks), chunk_size(chunksize) {
    }

    ~FixedChunkPQTable() {
      delete[] tables;
    }

    void load_bin(const char* filename) override {
      // bin structure: [256][ndims][ndims(float)]
      unsigned npts, ndims;
      NSG::load_bin<float>(filename, tables, npts, ndims);
      std::cout << "PQ Pivots: # ctrs: " << npts << ", # dims: " << ndims
                << std::endl;
      assert((_u64) ndims == n_chunks * chunk_size);
    }

    // in_vec = _u8 * [n_chunks]
    // out_vec = float* [ndims]
    virtual void convert(const _u8* in_vec, float* out_vec) override {
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        const _u8    pq_idx = *(in_vec + chunk);
        const float* vals = tables + (ndims * pq_idx) + (chunk * n_chunks);
        float*       chunk_out = out_vec + (chunk * chunk_size);
        // avoiding memcpy as chunk size is at most 10
        for (_u64 i = 0; i < chunk_size; i++) {
          *(chunk_out + i) = *(vals + i);
        }
      }
    }
  };
}  // namespace NSG