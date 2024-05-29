#include <exception>

#include "pq_common.h"
#include "pq_data_store.h"
#include "pq.h"
#include "pq_scratch.h"
#include "distance.h"

namespace diskann
{

// REFACTOR TODO: Assuming that num_pq_chunks is known already. Must verify if
// this is true.
template <typename data_t>

#ifdef EXEC_ENV_OLS
PQDataStore<data_t>::PQDataStore(size_t dim, location_t num_points, size_t num_pq_chunks,
                                 std::unique_ptr<Distance<data_t>> distance_fn,
                                 std::unique_ptr<QuantizedDistance<data_t>> pq_distance_fn,
                                 MemoryMappedFiles &files,
                                 const std::string &codebook_path)
#else
PQDataStore<data_t>::PQDataStore(size_t dim, location_t num_points, size_t num_pq_chunks,
                                 std::unique_ptr<Distance<data_t>> distance_fn,
                                 std::unique_ptr<QuantizedDistance<data_t>> pq_distance_fn,
                                 const std::string &codebook_path)
#endif
    : AbstractDataStore<data_t>(num_points, num_pq_chunks), _num_chunks(num_pq_chunks),
      _pq_pivot_file_path(codebook_path)
{
    if (num_pq_chunks > dim)
    {
        throw diskann::ANNException("ERROR: num_pq_chunks > dim", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    _distance_fn = std::move(distance_fn);
    _pq_distance_fn = std::move(pq_distance_fn);

    _aligned_dim = ROUND_UP(num_pq_chunks, _distance_fn->get_required_alignment());
    alloc_aligned(((void **)&_quantized_data), this->_capacity * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    std::memset(_quantized_data, 0, this->_capacity * _aligned_dim * sizeof(data_t));

#ifdef EXEC_ENV_OLS
    if (!codebook_path.empty())
    {
        _pq_distance_fn->load_pivot_data(files, codebook_path);
    }
#else
    if (!codebook_path.empty())
    {
        _pq_distance_fn->load_pivot_data(codebook_path.c_str());
    }
#endif
}

template <typename data_t> PQDataStore<data_t>::~PQDataStore()
{
    if (_quantized_data != nullptr)
    {
        aligned_free(_quantized_data);
        _quantized_data = nullptr;
    }
}

template <typename data_t> location_t PQDataStore<data_t>::load(const std::string &filename)
{
    return load_impl(filename);
}
template <typename data_t> size_t PQDataStore<data_t>::save(const std::string &filename, const location_t num_points)
{
    return diskann::save_bin(filename, _quantized_data, this->capacity(), _num_chunks, 0);
}

template <typename data_t> size_t PQDataStore<data_t>::get_aligned_dim() const
{
    return _aligned_dim;
}

// Populate quantized data from regular data.
template <typename data_t> void PQDataStore<data_t>::populate_data(const data_t *vectors, const location_t num_pts)
{
    memset(_quantized_data, 0, _aligned_dim * sizeof(data_t) * num_pts);
    for (location_t i = 0; i < num_pts; i++)
    {
        std::memmove(_quantized_data + i * _aligned_dim, vectors + i * this->_dim, this->_dim * sizeof(data_t));
    }
}

template <typename data_t> void PQDataStore<data_t>::populate_data(const std::string &filename, const size_t offset)
{
    if (_quantized_data != nullptr)
    {
        aligned_free(_quantized_data);
    }

    uint64_t file_num_points = 0, file_dim = 0;
    get_bin_metadata(filename, file_num_points, file_dim, offset);
    this->_capacity = (location_t)file_num_points;
    this->_dim = file_dim;

    double p_val = std::min(1.0, ((double)MAX_PQ_TRAINING_SET_SIZE / (double)file_num_points));

    auto pivots_file = _pq_pivot_file_path.empty()
                           ? get_pivot_data_filename(filename, _use_opq, static_cast<uint32_t>(_num_chunks))
                           : _pq_pivot_file_path;

    auto compressed_file = get_quantized_vectors_filename(filename, _use_opq, static_cast<uint32_t>(_num_chunks));

    generate_quantized_data<data_t>(filename, pivots_file, compressed_file, _distance_fn->get_metric(), p_val,
                                    _num_chunks,
                                    _pq_distance_fn->is_opq());

    // REFACTOR TODO: Not sure of the alignment. Just copying from index.cpp
    alloc_aligned(((void **)&_quantized_data), file_num_points * _num_chunks * sizeof(uint8_t), 1);
    copy_aligned_data_from_file<uint8_t>(compressed_file.c_str(), _quantized_data, file_num_points, _num_chunks,
                                         _num_chunks);
#ifdef EXEC_ENV_OLS
    throw ANNException("load_pq_centroid_bin should not be called when "
                       "EXEC_ENV_OLS is defined.",
                       -1, __FUNCSIG__, __FILE__, __LINE__);
#else
    _pq_distance_fn->load_pivot_data(pivots_file);
#endif
}

template <typename data_t>
void PQDataStore<data_t>::extract_data_to_bin(const std::string &filename, const location_t num_pts)
{
    diskann::save_bin(filename, _quantized_data, this->capacity(), _num_chunks, 0);
}

template <typename data_t> void PQDataStore<data_t>::get_vector(const location_t i, data_t *dest) const
{
    // REFACTOR TODO: Should we inflate the compressed vector here?
    if (i < this->capacity())
    {
        const FixedChunkPQTable &pq_table = _pq_distance_fn->get_pq_table();
        pq_table.inflate_vector<data_t, data_t>((data_t *)(_quantized_data + i * _aligned_dim), dest);
    }
    else
    {
        std::stringstream ss;
        ss << "Requested vector " << i << " but only  " << this->capacity() << " vectors are present";
        throw diskann::ANNException(ss.str(), -1);
    }
}
template <typename data_t> void PQDataStore<data_t>::set_vector(const location_t loc, const data_t *const vector)
{
    if (_pq_distance_fn == nullptr)
    {
        throw diskann::ANNException("PQ distance is not loaded, cannot set vector for PQDataStore.", -1);
    }

    const FixedChunkPQTable &pq_table = _pq_distance_fn->get_pq_table();

    if (pq_table.tables == nullptr)
    {
        throw diskann::ANNException("PQ table is not loaded for PQ distance, cannot set vector for PQDataStore.", -1);
    }

    uint64_t full_dimension = pq_table.ndims;
    uint64_t num_chunks = _num_chunks;

    std::vector<float> vector_float(full_dimension);
    diskann::convert_types<data_t, float>(vector, vector_float.data(), 1, full_dimension);
    std::vector<uint8_t> compressed_vector(num_chunks * sizeof(data_t));
    std::vector<data_t> compressed_vector_T(num_chunks);

    generate_pq_data_from_pivots_simplified(vector_float.data(), 1, pq_table.tables, 256 * full_dimension,
                                            full_dimension,
                                            num_chunks, compressed_vector);

    diskann::convert_types<uint8_t, data_t>(compressed_vector.data(), compressed_vector_T.data(), 1, num_chunks);

    size_t offset_in_data = loc * _aligned_dim;
    memset(_quantized_data + offset_in_data, 0, _aligned_dim * sizeof(data_t));
    memcpy(_quantized_data + offset_in_data, compressed_vector_T.data(), this->_dim * sizeof(data_t));
}

template <typename data_t> void PQDataStore<data_t>::prefetch_vector(const location_t loc)
{
    const uint8_t *ptr = _quantized_data + ((size_t)loc) * _num_chunks * sizeof(data_t);
    diskann::prefetch_vector((const char *)ptr, _num_chunks * sizeof(data_t));
}

template <typename data_t>
void PQDataStore<data_t>::move_vectors(const location_t old_location_start, const location_t new_location_start,
                                       const location_t num_locations)
{
    if (num_locations == 0 || old_location_start == new_location_start)
    {
        return;
    }

    // The [start, end) interval which will contain obsolete points to be
    // cleared.
    uint32_t mem_clear_loc_start = old_location_start;
    uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

    if (new_location_start < old_location_start)
    {
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_start < new_location_start + num_locations)
        {
            // Clear only after the end of the new range.
            mem_clear_loc_start = new_location_start + num_locations;
        }
    }
    else
    {
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_end_limit > new_location_start)
        {
            // Clear only up to the beginning of the new range.
            mem_clear_loc_end_limit = new_location_start;
        }
    }

    // Use memmove to handle overlapping ranges.
    copy_vectors(old_location_start, new_location_start, num_locations);
    memset(_quantized_data + _aligned_dim * mem_clear_loc_start, 0,
           sizeof(data_t) * _aligned_dim * (mem_clear_loc_end_limit - mem_clear_loc_start));
}

template <typename data_t>
void PQDataStore<data_t>::copy_vectors(const location_t from_loc, const location_t to_loc, const location_t num_points)
{
    // REFACTOR TODO: Is the number of bytes correct?
    memcpy(_quantized_data + to_loc * _num_chunks, _quantized_data + from_loc * _num_chunks, _num_chunks * num_points);
}

// REFACTOR TODO: Currently, we take aligned_query as parameter, but this
// function should also do the alignment.
template <typename data_t>
void PQDataStore<data_t>::preprocess_query(const data_t *aligned_query, AbstractScratch<data_t> *scratch) const
{
    if (scratch == nullptr)
    {
        throw diskann::ANNException("Scratch space is null", -1);
    }

    PQScratch<data_t> *pq_scratch = scratch->pq_scratch();

    if (pq_scratch == nullptr)
    {
        throw diskann::ANNException("PQScratch space has not been set in the scratch object.", -1);
    }

    _pq_distance_fn->preprocess_query(aligned_query, (location_t)this->get_dims(), *pq_scratch);
}

template <typename data_t> float PQDataStore<data_t>::get_distance(const data_t *query, const location_t loc) const
{
    throw diskann::ANNException("get_distance(const data_t *query, const location_t loc) hasn't been implemented for PQDataStore", -1);
}

template <typename data_t> float PQDataStore<data_t>::get_distance(const location_t loc1, const location_t loc2) const
{
    throw diskann::ANNException("get_distance(const location_t loc1, const location_t loc2) hasn't been implemented for PQDataStore", -1);
}

template <typename data_t>
void PQDataStore<data_t>::get_distance(const data_t *preprocessed_query, const location_t *locations,
                                       const uint32_t location_count, float *distances,
                                       AbstractScratch<data_t> *scratch_space) const
{
    if (scratch_space == nullptr)
    {
        throw diskann::ANNException("Scratch space is null", -1);
    }
    PQScratch<data_t> *pq_scratch = scratch_space->pq_scratch();
    if (pq_scratch == nullptr)
    {
        throw diskann::ANNException("PQScratch not set in scratch space.", -1);
    }
    diskann::aggregate_coords(locations, location_count, _quantized_data, this->_num_chunks,
                              pq_scratch->aligned_pq_coord_scratch);
    _pq_distance_fn->preprocessed_distance(*pq_scratch, location_count, distances);
}

template <typename data_t>
void PQDataStore<data_t>::get_distance(const data_t *preprocessed_query, const std::vector<location_t> &ids,
                                       std::vector<float> &distances, AbstractScratch<data_t> *scratch_space) const
{
    if (scratch_space == nullptr)
    {
        throw diskann::ANNException("Scratch space is null", -1);
    }
    PQScratch<data_t> *pq_scratch = scratch_space->pq_scratch();
    if (pq_scratch == nullptr)
    {
        throw diskann::ANNException("PQScratch not set in scratch space.", -1);
    }
    diskann::aggregate_coords(ids, _quantized_data, this->_num_chunks, pq_scratch->aligned_pq_coord_scratch);
    _pq_distance_fn->preprocessed_distance(*pq_scratch, (location_t)ids.size(), distances);
}

template <typename data_t> location_t PQDataStore<data_t>::calculate_medoid() const
{
    // REFACTOR TODO: Must calculate this just like we do with data store.
    size_t r = (size_t)rand() * (size_t)RAND_MAX + (size_t)rand();
    return (uint32_t)(r % (size_t)this->capacity());
}

template <typename data_t> size_t PQDataStore<data_t>::get_alignment_factor() const
{
    return _distance_fn->get_required_alignment();
}

template <typename data_t> Distance<data_t> *PQDataStore<data_t>::get_dist_fn() const
{
    return _distance_fn.get();
}

template <typename data_t> location_t PQDataStore<data_t>::load_impl(const std::string &file_prefix)
{
    if (_quantized_data != nullptr)
    {
        aligned_free(_quantized_data);
    }
    auto quantized_vectors_file =
        get_quantized_vectors_filename(file_prefix, _use_opq, static_cast<uint32_t>(_num_chunks));

    size_t num_points;
    load_aligned_bin(quantized_vectors_file, _quantized_data, num_points, _num_chunks, _num_chunks);
    this->_capacity = (location_t)num_points;

    auto pivots_file = get_pivot_data_filename(file_prefix, _use_opq, static_cast<uint32_t>(_num_chunks));
    _pq_distance_fn->load_pivot_data(pivots_file);

    return this->_capacity;
}

template <typename data_t> location_t PQDataStore<data_t>::expand(const location_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size < this->capacity())
    {
        std::stringstream ss;
        ss << "Cannot 'expand' datastore when new capacity (" << new_size << ") < existing capacity("
           << this->capacity() << ")" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
#ifndef _WINDOWS
    data_t *new_data;
    alloc_aligned((void **)&new_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    memcpy(new_data, _quantized_data, this->capacity() * _aligned_dim * sizeof(data_t));
    aligned_free(_quantized_data);
    _quantized_data = new_data;
#else
    realloc_aligned((void **)&_quantized_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
#endif
    this->_capacity = new_size;
    return this->_capacity;
}

template <typename data_t> location_t PQDataStore<data_t>::shrink(const location_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size > this->capacity())
    {
        std::stringstream ss;
        ss << "Cannot 'shrink' datastore when new capacity (" << new_size << ") > existing capacity("
           << this->capacity() << ")" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
#ifndef _WINDOWS
    data_t *new_data;
    alloc_aligned((void **)&new_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    memcpy(new_data, _quantized_data, new_size * _aligned_dim * sizeof(data_t));
    aligned_free(_quantized_data);
    _quantized_data = new_data;
#else
    realloc_aligned((void **)&_quantized_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
#endif
    this->_capacity = new_size;
    return this->_capacity;
}

#ifdef EXEC_ENV_OLS
template <typename data_t> location_t PQDataStore<data_t>::load_impl(AlignedFileReader &reader)
{
}
#endif

template DISKANN_DLLEXPORT class PQDataStore<int8_t>;
template DISKANN_DLLEXPORT class PQDataStore<float>;
template DISKANN_DLLEXPORT class PQDataStore<uint8_t>;

} // namespace diskann