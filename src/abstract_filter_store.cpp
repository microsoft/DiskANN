// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "abstract_filter_store.h"

namespace diskann
{

template <typename label_type>
AbstractFilterStore<label_type>::AbstractFilterStore(const size_t num_points) : _num_points(num_points)
{
}

template DISKANN_DLLEXPORT class AbstractFilterStore<uint16_t>;
template DISKANN_DLLEXPORT class AbstractFilterStore<uint32_t>;
} // namespace diskann