
#include "pq.h"
#include "pq_l2_distance.h"
#include "pq_scratch.h"

// block size for reading/processing large files and matrices in blocks
#define BLOCK_SIZE 5000000

namespace diskann
{

template <typename data_t>
PQL2Distance<data_t>::PQL2Distance(uint32_t num_chunks, bool use_opq) : _num_chunks(num_chunks), _is_opq(use_opq)
{
}

template <typename data_t> PQL2Distance<data_t>::~PQL2Distance()
{
}

template <typename data_t> bool PQL2Distance<data_t>::is_opq() const
{
    return this->_is_opq;
}

#ifdef EXEC_ENV_OLS
template <typename data_t>
void PQL2Distance<data_t>::load_pivot_data(MemoryMappedFiles &files, const std::string &pq_table_file)
{
    _pq_table.load_pq_centroid_bin(files, pq_table_file.c_str(), _num_chunks);
}
#else
template <typename data_t>
void PQL2Distance<data_t>::load_pivot_data(const std::string &pq_table_file)
{
    _pq_table.load_pq_centroid_bin(pq_table_file.c_str(), _num_chunks);
}
#endif

template <typename data_t> uint32_t PQL2Distance<data_t>::get_num_chunks() const
{
    return static_cast<uint32_t>(_num_chunks);
}

template <typename data_t>
const FixedChunkPQTable & PQL2Distance<data_t>::get_pq_table() const
{
    return _pq_table;
}

// REFACTOR: Instead of doing half the work in the caller and half in this
// function, we let this function
//  do all of the work, making it easier for the caller.
template <typename data_t>
void PQL2Distance<data_t>::preprocess_query(const data_t *aligned_query, uint32_t dim, PQScratch<data_t> &scratch)
{
    // Copy query vector to float and then to "rotated" query
    for (size_t d = 0; d < dim; d++)
    {
        scratch.aligned_query_float[d] = (float)aligned_query[d];
    }
    scratch.initialize(dim, aligned_query);

    for (uint32_t d = 0; d < _pq_table.ndims; d++)
    {
        scratch.rotated_query[d] -= _pq_table.centroid[d];
    }
    std::vector<float> tmp(_pq_table.ndims, 0);
    if (_is_opq)
    {
        for (uint32_t d = 0; d < _pq_table.ndims; d++)
        {
            for (uint32_t d1 = 0; d1 < _pq_table.ndims; d1++)
            {
                tmp[d] += scratch.rotated_query[d1] * _pq_table.rotmat_tr[d1 * _pq_table.ndims + d];
            }
        }
        std::memcpy(scratch.rotated_query, tmp.data(), _pq_table.ndims * sizeof(float));
    }
    this->prepopulate_chunkwise_distances(scratch.rotated_query, scratch.aligned_pqtable_dist_scratch);
}

template <typename data_t>
void PQL2Distance<data_t>::preprocessed_distance(PQScratch<data_t> &pq_scratch, const uint32_t n_ids, float *dists_out)
{
    pq_dist_lookup(pq_scratch.aligned_pq_coord_scratch, n_ids, _num_chunks, pq_scratch.aligned_pqtable_dist_scratch,
                   dists_out);
}

template <typename data_t>
void PQL2Distance<data_t>::preprocessed_distance(PQScratch<data_t> &pq_scratch, const uint32_t n_ids,
                                                 std::vector<float> &dists_out)
{
    pq_dist_lookup(pq_scratch.aligned_pq_coord_scratch, n_ids, _num_chunks, pq_scratch.aligned_pqtable_dist_scratch,
                   dists_out);
}

template <typename data_t> float PQL2Distance<data_t>::brute_force_distance(const float *query_vec, uint8_t *base_vec)
{
    float res = 0;
    for (size_t chunk = 0; chunk < _num_chunks; chunk++)
    {
        for (size_t j = _pq_table.chunk_offsets[chunk]; j < _pq_table.chunk_offsets[chunk + 1]; j++)
        {
            const float *centers_dim_vec = _pq_table.tables + (256 * j);
            float diff = centers_dim_vec[base_vec[chunk]] - (query_vec[j]);
            res += diff * diff;
        }
    }
    return res;
}

template <typename data_t>
void PQL2Distance<data_t>::prepopulate_chunkwise_distances(const float *query_vec, float *dist_vec)
{
    memset(dist_vec, 0, 256 * _num_chunks * sizeof(float));
    // chunk wise distance computation
    for (size_t chunk = 0; chunk < _num_chunks; chunk++)
    {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float *chunk_dists = dist_vec + (256 * chunk);
        for (size_t j = _pq_table.chunk_offsets[chunk]; j < _pq_table.chunk_offsets[chunk + 1]; j++)
        {
            const float *centers_dim_vec = _pq_table.tables_tr + (256 * j);
            for (size_t idx = 0; idx < 256; idx++)
            {
                double diff = centers_dim_vec[idx] - (query_vec[j]);
                chunk_dists[idx] += (float)(diff * diff);
            }
        }
    }
}

template DISKANN_DLLEXPORT class PQL2Distance<int8_t>;
template DISKANN_DLLEXPORT class PQL2Distance<uint8_t>;
template DISKANN_DLLEXPORT class PQL2Distance<float>;

} // namespace diskann