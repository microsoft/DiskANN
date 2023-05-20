// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "allocator.h"
#include "utils.h"

namespace diskann
{

template <class T> T *Allocator<T>::allocate(std::size_t count, const T *hint)
{
    auto ret = _allocator.allocate(count, hint);
    update_memory_usage(count, /*is_allocation*/ true);
    return ret;
}

template <class T> void Allocator<T>::deallocate(T *ptr, std::size_t count)
{
    _allocator.deallocate(ptr, count);
    update_memory_usage(count, /*is_allocation*/ false);
}

template <class T> void Allocator<T>::update_memory_usage(std::size_t count, bool is_allocation)
{
#ifdef FORCE_TO_TRACK_MEMORY_IN_ALLOCATOR
    assert(_memory_used_in_bytes != nullptr);
#endif //  FORCE_TO_TRACK_MEMORY_IN_ALLOCATOR
    if (_memory_used_in_bytes == nullptr)
    {
        return;
    }

    if (is_allocation)
    {
        *_memory_used_in_bytes += sizeof(T) * count;
    }
    else
    {
        *_memory_used_in_bytes -= sizeof(T) * count;
    }
}

template class Allocator<uint32_t>;
template class Allocator<diskann::vector<uint32_t>>;
template class Allocator<std::_Container_proxy>;

} // namespace diskann
