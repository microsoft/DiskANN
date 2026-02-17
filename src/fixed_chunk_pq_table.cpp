// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "fixed_chunk_pq_table.h"
#include "pq_common.h"
#include "utils.h"

namespace diskann
{
    FixedChunkPQTable::FixedChunkPQTable()
    {
    }

    FixedChunkPQTable::~FixedChunkPQTable()
    {
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
    void FixedChunkPQTable::load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks)
    {
    #else
    void FixedChunkPQTable::load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks)
    {
    #endif

        uint64_t nr, nc;
        std::string rotmat_file = std::string(pq_table_file) + "_rotation_matrix.bin";

    #ifdef EXEC_ENV_OLS
        size_t *file_offset_data; // since load_bin only sets the pointer, no need
                                  // to delete.
        diskann::load_bin<size_t>(files, pq_table_file, file_offset_data, nr, nc);
    #else
        std::unique_ptr<size_t[]> file_offset_data;
        diskann::load_bin<size_t>(pq_table_file, file_offset_data, nr, nc);
    #endif

        bool use_old_filetype = false;

        if (nr != 4 && nr != 5)
        {
            diskann::cout << "Error reading pq_pivots file " << pq_table_file
                          << ". Offsets dont contain correct metadata, # offsets = " << nr << ", but expecting " << 4
                          << " or " << 5;
            throw diskann::ANNException("Error reading pq_pivots file at offsets data.", -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }

        if (nr == 4)
        {
            diskann::cout << "Offsets: " << file_offset_data[0] << " " << file_offset_data[1] << " " << file_offset_data[2]
                          << " " << file_offset_data[3] << std::endl;
        }
        else if (nr == 5)
        {
            use_old_filetype = true;
            diskann::cout << "Offsets: " << file_offset_data[0] << " " << file_offset_data[1] << " " << file_offset_data[2]
                          << " " << file_offset_data[3] << file_offset_data[4] << std::endl;
        }
        else
        {
            throw diskann::ANNException("Wrong number of offsets in pq_pivots", -1, __FUNCSIG__, __FILE__, __LINE__);
        }

    #ifdef EXEC_ENV_OLS
        diskann::load_bin<float>(files, pq_table_file, tables, nr, nc, file_offset_data[0]);
    #else
        diskann::load_bin<float>(pq_table_file, tables, nr, nc, file_offset_data[0]);
    #endif

        if ((nr != NUM_PQ_CENTROIDS))
        {
            diskann::cout << "Error reading pq_pivots file " << pq_table_file << ". file_num_centers  = " << nr
                          << " but expecting " << NUM_PQ_CENTROIDS << " centers";
            throw diskann::ANNException("Error reading pq_pivots file at pivots data.", -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }

        this->ndims = nc;

    #ifdef EXEC_ENV_OLS
        diskann::load_bin<float>(files, pq_table_file, centroid, nr, nc, file_offset_data[1]);
    #else
        diskann::load_bin<float>(pq_table_file, centroid, nr, nc, file_offset_data[1]);
    #endif
    
        if ((nr != this->ndims) || (nc != 1))
        {
            diskann::cerr << "Error reading centroids from pq_pivots file " << pq_table_file << ". file_dim  = " << nr
                          << ", file_cols = " << nc << " but expecting " << this->ndims << " entries in 1 dimension.";
            throw diskann::ANNException("Error reading pq_pivots file at centroid data.", -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }
    
        int chunk_offsets_index = 2;
        if (use_old_filetype)
        {
            chunk_offsets_index = 3;
        }
    #ifdef EXEC_ENV_OLS
        diskann::load_bin<uint32_t>(files, pq_table_file, chunk_offsets, nr, nc, file_offset_data[chunk_offsets_index]);
    #else
        diskann::load_bin<uint32_t>(pq_table_file, chunk_offsets, nr, nc, file_offset_data[chunk_offsets_index]);
    #endif

        if (nc != 1 || (nr != num_chunks + 1 && num_chunks != 0))
        {
            diskann::cerr << "Error loading chunk offsets file. numc: " << nc << " (should be 1). numr: " << nr
                          << " (should be " << num_chunks + 1 << " or 0 if we need to infer)" << std::endl;
            throw diskann::ANNException("Error loading chunk offsets file", -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        this->n_chunks = nr - 1;
        diskann::cout << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS << ", #dims: " << this->ndims
                      << ", #chunks: " << this->n_chunks << std::endl;

    #ifdef EXEC_ENV_OLS
        if (files.fileExists(rotmat_file))
        {
            diskann::load_bin<float>(files, rotmat_file, (float *&)rotmat_tr, nr, nc);
    #else
        if (file_exists(rotmat_file))
        {
            diskann::load_bin<float>(rotmat_file, rotmat_tr, nr, nc);
    #endif
            if (nr != this->ndims || nc != this->ndims)
            {
                diskann::cerr << "Error loading rotation matrix file" << std::endl;
                throw diskann::ANNException("Error loading rotation matrix file", -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            use_rotation = true;
        }

        // alloc and compute transpose
        tables_tr = new float[256 * this->ndims];
        for (size_t i = 0; i < 256; i++)
        {
            for (size_t j = 0; j < this->ndims; j++)
            {
                tables_tr[j * 256 + i] = tables[i * this->ndims + j];
            }
        }
    }

    void FixedChunkPQTable::preprocess_query(float *query_vec)
    {
        for (uint32_t d = 0; d < ndims; d++)
        {
            query_vec[d] -= centroid[d];
        }
        std::vector<float> tmp(ndims, 0);
        if (use_rotation)
        {
            for (uint32_t d = 0; d < ndims; d++)
            {
                for (uint32_t d1 = 0; d1 < ndims; d1++)
                {
                    tmp[d] += query_vec[d1] * rotmat_tr[d1 * ndims + d];
                }
            }
            std::memcpy(query_vec, tmp.data(), ndims * sizeof(float));
        }
    }

    // assumes pre-processed query
    void FixedChunkPQTable::populate_chunk_distances(const float *query_vec, float *dist_vec)
    {
        memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
        // chunk wise distance computation
        for (size_t chunk = 0; chunk < n_chunks; chunk++)
        {
            // sum (q-c)^2 for the dimensions associated with this chunk
            float *chunk_dists = dist_vec + (256 * chunk);
            for (size_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++)
            {
                const float *centers_dim_vec = tables_tr + (256 * j);
                for (size_t idx = 0; idx < 256; idx++)
                {
                    double diff = centers_dim_vec[idx] - (query_vec[j]);
                    chunk_dists[idx] += (float)(diff * diff);
                }
            }
        }
    }

    float FixedChunkPQTable::l2_distance(const float *query_vec, uint8_t *base_vec)
    {
        float res = 0;
        for (size_t chunk = 0; chunk < n_chunks; chunk++)
        {
            for (size_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++)
            {
                const float *centers_dim_vec = tables_tr + (256 * j);
                float diff = centers_dim_vec[base_vec[chunk]] - (query_vec[j]);
                res += diff * diff;
            }
        }
        return res;
    }

    float FixedChunkPQTable::inner_product(const float *query_vec, uint8_t *base_vec)
    {
        float res = 0;
        for (size_t chunk = 0; chunk < n_chunks; chunk++)
        {
            for (size_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++)
            {
                const float *centers_dim_vec = tables_tr + (256 * j);
                float diff = centers_dim_vec[base_vec[chunk]] * query_vec[j]; // assumes centroid is 0 to
                                                                              // prevent translation errors
                res += diff;
            }
        }
        return -res; // returns negative value to simulate distances (max -> min
                     // conversion)
    }

    // assumes no rotation is involved
    template <typename InputType, typename OutputType>
    void FixedChunkPQTable::inflate_vector(InputType *base_vec, OutputType *out_vec) const
    {
        for (size_t chunk = 0; chunk < n_chunks; chunk++)
        {
            for (size_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++)
            {
                const float *centers_dim_vec = tables_tr + (256 * j);
                out_vec[j] = static_cast<OutputType> (centers_dim_vec[static_cast<uint8_t>(base_vec[chunk])] + centroid[j]);
            }
        }
    }

    void FixedChunkPQTable::populate_chunk_inner_products(const float *query_vec, float *dist_vec)
    {
        memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
        // chunk wise distance computation
        for (size_t chunk = 0; chunk < n_chunks; chunk++)
        {
            // sum (q-c)^2 for the dimensions associated with this chunk
            float *chunk_dists = dist_vec + (256 * chunk);
            for (size_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++)
            {
                const float *centers_dim_vec = tables_tr + (256 * j);
                for (size_t idx = 0; idx < 256; idx++)
                {
                    double prod = centers_dim_vec[idx] * query_vec[j]; // assumes that we are not
                                                                       // shifting the vectors to
                                                                       // mean zero, i.e., centroid
                                                                       // array should be all zeros
                    chunk_dists[idx] -= (float)prod;                   // returning negative to keep the search code
                                                                       // clean (max inner product vs min distance)
                }
            }
        }
    }

    template void FixedChunkPQTable::inflate_vector<uint8_t, float>(uint8_t *base_vec, float *out_vec) const;
    template void FixedChunkPQTable::inflate_vector<uint8_t, uint8_t>(uint8_t *base_vec, uint8_t *out_vec) const;
    template void FixedChunkPQTable::inflate_vector<int8_t, int8_t>(int8_t *base_vec, int8_t *out_vec) const;
    template void FixedChunkPQTable::inflate_vector<float, float>(float *base_vec, float *out_vec) const;
    } // namespace diskann
