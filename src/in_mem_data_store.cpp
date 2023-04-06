// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "in_mem_data_store.h"

#include "utils.h"

namespace diskann
{

template <typename data_t>
InMemDataStore<data_t>::InMemDataStore(const location_t capacity, 
                                             const size_t dim, std::shared_ptr<Distance<data_t>> distance_metric)
    : AbstractDataStore<data_t>(capacity, dim), _aligned_dim(ROUND_UP(dim, 8)), 
      _distance_metric(distance_metric)
{
    alloc_aligned(((void **)&_data), capacity * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    std::memset(_data, 0, capacity * _aligned_dim * sizeof(data_t));
}

template <typename data_t> InMemDataStore<data_t>::~InMemDataStore()
{
    if (_data != nullptr)
    {
        aligned_free(this->_data);
    }
}

template <typename data_t> location_t InMemDataStore<data_t>::load(const std::string &filename)
{
    load_data(filename);
}

template <typename data_t> void InMemDataStore<data_t>::store(const std::string &filename)
{

}

template <typename data_t> void InMemDataStore<data_t>::populate_data(const data_t *vectors, const location_t num_pts)
{
    for (auto i = 0; i < num_pts; i++)
    {
        memset(_data + i * _aligned_dim, 0, _aligned_dim * sizeof(data_t));
        std::memmove(_data + i * _aligned_dim, vectors + i * _dim, _dim * sizeof(data_t));
    }

    if (_distance_metric->normalization_required())
    {
        _distance_metric->normalize(_data, num_pts);
    }
}

template<typename data_t> void InMemDataStore<data_t>::populate_data(const std::string &filename, const size_t offset)
{
    copy_aligned_data_from_file(filename.c_str(), _data, _num_points, _dim, _aligned_dim, offset);
    if (_distance_metric->normalization_required())
    {
        _distance_metric->normalize(_data, _num_points);
    }
}

#ifdef EXEC_ENV_OLS
template <typename data_t> location_t Index<data_t>::load_data(AlignedFileReader &reader)
{
    size_t file_dim, file_num_points;

    diskann::get_bin_metadata(reader, file_num_points, file_dim);

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != _dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > _max_points + _num_frozen_pts)
    {
        resize(file_num_points - _num_frozen_pts);
    }

    return file_num_points;
}
#endif

template <typename data_t>
location_t InMemDataStore<data_t>::load_data(const std::string &filename)
{
    size_t file_dim, file_num_points;
    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: data file " << filename << " does not exist." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    diskann::get_bin_metadata(filename, file_num_points, file_dim);

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != this->_dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << this->_dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > this->capacity())
    {
        resize(file_num_points);
    }

    copy_aligned_data_from_file<data_t>(filename.c_str(), _data, file_num_points, file_dim, _aligned_dim);

    return file_num_points;
}

template <typename data_t> 
void InMemDataStore<data_t>::get_vector(const location_t i, data_t* dest) const 
{
    memcpy(dest, _data + i * _aligned_dim, this->_dim * sizeof(data_t));
}

template <typename data_t>
void InMemDataStore<data_t>::set_vector(const location_t loc, const data_t *const vector)
{
    size_t offset_in_data = loc * _aligned_dim;
    memset(_data + offset_in_data, 0, _aligned_dim * sizeof(data_t));
    memcpy(_data + offset_in_data, vector, this->_dim * sizeof(data_t));
    if (_distance_metric->normalization_required())
    {
        _distance_metric->normalize(_data + offset_in_data, _aligned_dim);
    }
}

template<typename data_t> void InMemDataStore<data_t>::prefetch_vector(const location_t loc)
{
    diskann::prefetch_vector((const char *)_data + _aligned_dim * (size_t)loc, sizeof(T) * _aligned_dim);
}

template<typename data_t> float InMemDataStore<data_t>::get_distance(const data_t *query, const location_t loc) const
{
    return _distance_metric->compare(query, _data + _aligned_dim * loc, _aligned_dim);
}

template<typename data_t>
void InMemDataStore<data_t>::get_distance(const data_t *query, const location_t *locations, const uint32_t location_count, float *distances) const
{
    for (auto i = 0; i < location_count; i++)
    {
        distances[i] = _distance_metric->compare(query, _data + locations[i] * _aligned_dim, this->_aligned_dim);
    }
}

template <typename data_t> 
float InMemDataStore<data_t>::get_distance(const location_t loc1, const location_t loc2) const
{
    return _distance_metric->compare(_data + loc1 * _aligned_dim, _data + loc2 * _aligned_dim, this->_aligned_dim);
}

template <typename data_t> void InMemDataStore<data_t>::expand(const location_t new_size)
{
#ifndef _WINDOWS
    data_t *new_data;
    alloc_aligned((void **)&new_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    memcpy(new_data, _data, (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(data_t));
    aligned_free(_data);
    _data = new_data;
#else
    realloc_aligned((void **)&_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
#endif
    _capacity = new_size;

}

template <typename data_t> void InMemDataStore<data_t>::shrink(const location_t new_size)
{
#ifndef _WINDOWS
    data_t *new_data;
    alloc_aligned((void **)&new_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    memcpy(new_data, _data, new_size * _aligned_dim * sizeof(data_t));
    aligned_free(_data);
    _data = new_data;
#else
    realloc_aligned((void **)&_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
#endif
}

template<typename data_t>
void InMemDataStore<data_t>::reposition_points(const location_t old_location_start, const location_t new_location_start, const location_t num_locations)
{
    if (num_locations == 0 || old_location_start == new_location_start)
    {
        return;
    }

    // Update pointers to the moved nodes. Note: the computation is correct even
    // when new_location_start < old_location_start given the C++ uint32_t
    // integer arithmetic rules.
    const uint32_t location_delta = new_location_start - old_location_start;

    // The [start, end) interval which will contain obsolete points to be
    // cleared.
    uint32_t mem_clear_loc_start = old_location_start;
    uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

    // Move the adjacency lists. Make sure that overlapping ranges are handled
    // correctly.
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
    copy_points(old_location_start, new_location_start, num_locations);
    memset(_data + _aligned_dim * mem_clear_loc_start, 0,
           sizeof(data_t) * _aligned_dim * (mem_clear_loc_end_limit - mem_clear_loc_start));
}

template<typename data_t>
void InMemDataStore<data_t>::copy_points(const location_t from_loc, const location_t to_loc,
                                         const location_t num_points)
{
    assert(from_loc < _capacity);
    assert(to_loc < _capacity);
    assert(num_points < _capacity);
    memmove(_data + _aligned_dim * to_loc, _data + _aligned_dim * from_loc, num_points * _aligned_dim * sizeof(data_t));
}

template <typename data_t> location_t InMemDataStore<data_t>::calculate_medoid() const
{
    // allocate and init centroid
    float *center = new float[_aligned_dim];
    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] = 0;

    for (size_t i = 0; i < _num_points; i++)
        for (size_t j = 0; j < _aligned_dim; j++)
            center[j] += (float)_data[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] /= (float)_num_points;

    // compute all to one distance
    float *distances = new float[_num_points];

//TODO: REFACTOR. Removing pragma might make this slow. Must revisit. 
// Problem is that we need to pass num_threads here, it is not clear
// if data store must be aware of threads!
//#pragma omp parallel for schedule(static, 65536) 
    for (int64_t i = 0; i < (int64_t)_num_points; i++)
    {
        // extract point and distance reference
        float &dist = distances[i];
        const data_t *cur_vec = _data + (i * (size_t)_aligned_dim);
        dist = 0;
        float diff = 0;
        for (size_t j = 0; j < _aligned_dim; j++)
        {
            diff = (center[j] - (float)cur_vec[j]) * (center[j] - (float)cur_vec[j]);
            dist += diff;
        }
    }
    // find imin
    uint32_t min_idx = 0;
    float min_dist = distances[0];
    for (uint32_t i = 1; i < _num_points; i++)
    {
        if (distances[i] < min_dist)
        {
            min_idx = i;
            min_dist = distances[i];
        }
    }

    delete[] distances;
    delete[] center;
    return min_idx;
}

} // namespace diskann