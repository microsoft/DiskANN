// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>

#include "abstract_data_store.h"

namespace diskann
{

template<typename data_t>
AbstractDataStore<data_t>::AbstractDataStore(const location_t capacity, const size_t dim) : _capacity(capacity), _dim(dim)
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

} // namespace diskann
