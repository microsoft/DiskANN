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
    bool   use_rotation = false;
    _u32*  chunk_offsets = nullptr;
    float* centroid = nullptr;
    float* tables_tr = nullptr;  // same as pq_tables, but col-major
    float* rotmat_tr = nullptr;

   public:
    FixedChunkPQTable();

    virtual ~FixedChunkPQTable();

#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles& files,
                              const char* pq_table_file, size_t num_chunks);
#else
    void load_pq_centroid_bin(const char* pq_table_file, size_t num_chunks);
#endif

    _u32 get_num_chunks();

    void preprocess_query(float* query_vec);

    // assumes pre-processed query
    void populate_chunk_distances(const float* query_vec, float* dist_vec);

    float l2_distance(const float* query_vec, _u8* base_vec);

    float inner_product(const float* query_vec, _u8* base_vec);

    // assumes no rotation is involved
    void inflate_vector(_u8* base_vec, float* out_vec);

    void populate_chunk_inner_products(const float* query_vec, float* dist_vec);
  }; 
}  // namespace diskann
