// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "allocator.h"
#include "utils.h"

namespace diskann
{
template <class _Ty> _CONSTEXPR20 void Allocator<_Ty>::deallocate(_Ty *const _Ptr, const size_t _Count)
{
    Base::deallocate(_Ptr, _Count);
#ifdef FORCE_TO_TRACK_MEMORY_IN_ALLOCATOR
    _STL_ASSERT(_memory_used_in_bytes != nullptr, "_memory_used_in_bytes should not be nullptr");
#endif
    if (_memory_used_in_bytes != nullptr)
    {
        *_memory_used_in_bytes -= _Count * sizeof(_Ty);
    }
}

template <class _Ty>
_NODISCARD_RAW_PTR_ALLOC _CONSTEXPR20 __declspec(Allocator) _Ty *Allocator<_Ty>::allocate(
    _CRT_GUARDOVERFLOW const size_t _Count)
{
    auto ret = Base::allocate(_Count);
#ifdef FORCE_TO_TRACK_MEMORY_IN_ALLOCATOR
    _STL_ASSERT(_memory_used_in_bytes != nullptr, "_memory_used_in_bytes should not be nullptr");
#endif
    if (_memory_used_in_bytes != nullptr)
    {
        *_memory_used_in_bytes += _Count * sizeof(_Ty);
    }
    return ret;
}

template class Allocator<uint32_t>;
template class Allocator<diskann::vector<uint32_t>>;
template class Allocator<std::_Container_proxy>;

} // namespace diskann
