// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "allocator.h"
#include "utils.h"

namespace diskann
{

template <class T> T *Allocator<T>::allocate(std::size_t count)
{
    auto ret = _allocator.allocate(count);
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
#ifdef TRACK_MEM_ALLOCATED
    assert(_memory_used_in_bytes_ptr != nullptr);
#endif //  TRACK_MEM_ALLOCATED
    if (_memory_used_in_bytes_ptr == nullptr)
    {
        return;
    }

    if (is_allocation)
    {
        *_memory_used_in_bytes_ptr += sizeof(T) * count;
    }
    else
    {
        *_memory_used_in_bytes_ptr -= sizeof(T) * count;
    }
}

template class Allocator<uint8_t>;
template class Allocator<uint16_t>;
template class Allocator<uint32_t>;
template class Allocator<uint64_t>;

template class Allocator<char>;
template class Allocator<int8_t>;
template class Allocator<int16_t>;
template class Allocator<int32_t>;
template class Allocator<int64_t>;

template class Allocator<float>;
template class Allocator<double>;

template class Allocator<diskann::vector<uint8_t>>;
template class Allocator<diskann::vector<uint16_t>>;
template class Allocator<diskann::vector<uint32_t>>;
template class Allocator<diskann::vector<uint64_t>>;

template class Allocator<diskann::vector<char>>;
template class Allocator<diskann::vector<int8_t>>;
template class Allocator<diskann::vector<int16_t>>;
template class Allocator<diskann::vector<int32_t>>;
template class Allocator<diskann::vector<int64_t>>;

template class Allocator<diskann::vector<float>>;
template class Allocator<diskann::vector<double>>;

#ifdef _WINDOWS
template class Allocator<std::_Container_proxy>;
#endif // _WINDOWS

} // namespace diskann
