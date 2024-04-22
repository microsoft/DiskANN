// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common_includes.h"

#ifdef EXEC_ENV_OLS
#include "content_buf.h"
#include "memory_mapped_files.h"
#endif

namespace diskann
{
    class FixedChunkPQTable
    {
    public:
        FixedChunkPQTable();
        virtual ~FixedChunkPQTable();

    #ifdef EXEC_ENV_OLS
        void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks);
    #else
        void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks);
    #endif

        void preprocess_query(float *query_vec);
        // assumes pre-processed query
        void populate_chunk_distances(const float *query_vec, float *dist_vec);
        float l2_distance(const float *query_vec, uint8_t *base_vec);
        float inner_product(const float *query_vec, uint8_t *base_vec);
        // assumes no rotation is involved
        void inflate_vector(uint8_t *base_vec, float *out_vec);
        void populate_chunk_inner_products(const float *query_vec, float *dist_vec);

        float *tables = nullptr; // pq_tables = float array of size [256 * ndims]
        uint64_t ndims = 0;      // ndims = true dimension of vectors
        uint64_t n_chunks = 0;
        bool use_rotation = false;
        uint32_t *chunk_offsets = nullptr;
        float *centroid = nullptr;
        float *tables_tr = nullptr; // same as pq_tables, but col-major
        float *rotmat_tr = nullptr;
    };
} // namespace diskann
