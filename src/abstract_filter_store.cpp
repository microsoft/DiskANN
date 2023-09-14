// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <vector>

#include "abstract_filter_store.h"
#include "types.h"

namespace diskann
{

template <typename label_t>
AbstractDataStore<label_t>::AbstractFilterStore(const location_t num_points)
    : _num_points(num_points)
{
}

template <typename label_t> location_t AbstractFilterStore<label_t>::get_number_points() const
{
    return _num_points;
}

template DISKANN_DLLEXPORT class AbstractFilterStore<uint16_t>;
template DISKANN_DLLEXPORT class AbstractFilterStore<uint32_t>;
} // namespace diskann
