// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <vector>
#include "abstract_data_store.h"

namespace diskann
{

template <typename data_t>
AbstractDataStore<data_t>::AbstractDataStore(const location_t capacity, const size_t dim)
    : _capacity(capacity), _dim(dim)
{
}

template <typename data_t> location_t AbstractDataStore<data_t>::capacity() const
{
    return _capacity;
}

template <typename data_t> size_t AbstractDataStore<data_t>::get_dims() const
{
    return _dim;
}

template <typename data_t> location_t AbstractDataStore<data_t>::resize(const location_t new_num_points)
{
    if (new_num_points > _capacity)
    {
        return expand(new_num_points);
    }
    else if (new_num_points < _capacity)
    {
        return shrink(new_num_points);
    }
    else
    {
        return _capacity;
    }
}

template <typename data_t>
void AbstractDataStore<data_t>::preprocess_query(const data_t *query, AbstractScratch<data_t> *query_scratch) const
{
}

template <typename data_t>
void AbstractDataStore<data_t>::get_distance(const data_t *preprocessed_query, const std::vector<location_t> &ids,
                                             std::vector<float> &distances,
                                             AbstractScratch<data_t> *scratch_space) const
{
}

template DISKANN_DLLEXPORT class AbstractDataStore<float>;
template DISKANN_DLLEXPORT class AbstractDataStore<int8_t>;
template DISKANN_DLLEXPORT class AbstractDataStore<uint8_t>;
} // namespace diskann
