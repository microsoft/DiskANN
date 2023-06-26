#pragma once
#include <cstdint>
#include "pq_common.h"
#include "utils.h"

namespace diskann
{

template <typename T> struct PQScratch
{
    float *aligned_pqtable_dist_scratch = nullptr; // MUST BE AT LEAST [256 * NCHUNKS]
    float *aligned_dist_scratch = nullptr;         // MUST BE AT LEAST diskann MAX_DEGREE
    uint8_t *aligned_pq_coord_scratch = nullptr;   // AT LEAST  [N_CHUNKS * MAX_DEGREE]
    float *rotated_query = nullptr;
    float *aligned_query_float = nullptr;

    PQScratch(size_t graph_degree, size_t aligned_dim)
    {
        diskann::alloc_aligned((void **)&aligned_pq_coord_scratch,
                               (size_t)graph_degree * (size_t)MAX_PQ_CHUNKS * sizeof(uint8_t), 256);
        diskann::alloc_aligned((void **)&aligned_pqtable_dist_scratch, 256 * (size_t)MAX_PQ_CHUNKS * sizeof(float),
                               256);
        diskann::alloc_aligned((void **)&aligned_dist_scratch, (size_t)graph_degree * sizeof(float), 256);
        diskann::alloc_aligned((void **)&aligned_query_float, aligned_dim * sizeof(float), 8 * sizeof(float));
        diskann::alloc_aligned((void **)&rotated_query, aligned_dim * sizeof(float), 8 * sizeof(float));

        memset(aligned_query_float, 0, aligned_dim * sizeof(float));
        memset(rotated_query, 0, aligned_dim * sizeof(float));
    }

    void initialize(size_t dim, const T *query, const float norm = 1.0f)
    {
        for (size_t d = 0; d < dim; ++d)
        {
            if (norm != 1.0f)
                rotated_query[d] = aligned_query_float[d] = static_cast<float>(query[d]) / norm;
            else
                rotated_query[d] = aligned_query_float[d] = static_cast<float>(query[d]);
        }
    }
};

} // namespace diskann