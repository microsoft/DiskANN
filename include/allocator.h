// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <atomic>
#include <memory>
#include <vector>

namespace diskann
{

/// <summary>
/// Below class overrides the allocate\deallocate method of std::allocator<_Ty> to track memory usage
/// </summary>
template <class _Ty> class Allocator : public std::allocator<_Ty>
{
  public:
    using Base = std::allocator<_Ty>;

#if _HAS_DEPRECATED_ALLOCATOR_MEMBERS
    template <class _Other> struct _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS rebind
    {
        using other = Allocator<_Other>;
    };

#endif // _HAS_DEPRECATED_ALLOCATOR_MEMBERS

    constexpr Allocator(std::atomic<size_t> *memory_used) noexcept
    {
        _memory_used = memory_used;
    }

    constexpr Allocator() noexcept
    {
        _memory_used = nullptr;
    }

    constexpr Allocator(const Allocator &a) noexcept
    {
        _memory_used = a._memory_used;
    }

    template <class _Other> constexpr Allocator(const Allocator<_Other> &a) noexcept
    {
        _memory_used = a._memory_used;
    }

    _CONSTEXPR20 Allocator &operator=(const Allocator &a)
    {
        _memory_used = a._memory_used;
    }

    _CONSTEXPR20 void deallocate(_Ty *const _Ptr, const size_t _Count)
    {
        Base::deallocate(_Ptr, _Count);
        #ifdef FORCE_TO_TRACK_MEMORY_IN_ALLOCATOR
        _STL_ASSERT(_memory_used != nullptr, "_memory_used should not be nullptr");
        #endif
        if (_memory_used != nullptr)
        {
            *_memory_used -= _Count * sizeof(_Ty);
        }
    }

    _NODISCARD_RAW_PTR_ALLOC _CONSTEXPR20 __declspec(allocator) _Ty *allocate(_CRT_GUARDOVERFLOW const size_t _Count)
    {
        auto ret = Base::allocate(_Count);
#ifdef FORCE_TO_TRACK_MEMORY_IN_ALLOCATOR
        _STL_ASSERT(_memory_used != nullptr, "_memory_used should not be nullptr");
#endif
        if (_memory_used != nullptr)
        {
            *_memory_used += _Count * sizeof(_Ty);
        }
        return ret;
    }

#if _HAS_CXX23
    _NODISCARD_RAW_PTR_ALLOC constexpr allocation_result<_Ty *> allocate_at_least(
        _CRT_GUARDOVERFLOW const size_t _Count)
    {
        return {allocate(_Count), _Count};
    }
#endif // _HAS_CXX23

#if _HAS_DEPRECATED_ALLOCATOR_MEMBERS
    _CXX17_DEPRECATE_OLD_ALLOCATOR_MEMBERS _NODISCARD_RAW_PTR_ALLOC __declspec(allocator) _Ty *allocate(
        _CRT_GUARDOVERFLOW const size_t _Count, const void *)
    {
        return allocate(_Count);
    }

#endif // _HAS_DEPRECATED_ALLOCATOR_MEMBERS

    std::atomic<size_t>* _memory_used;
};

template <class T>
using vector = std::vector<T, Allocator<T>>;

//template <class T>
//using robin_set = std::vector<T, Allocator<T>>;

class MemoryManager
{
  public:
    MemoryManager() : _memory_used(0)
    {
    }

    size_t get_memory_used() const;
    void alloc_aligned(void **ptr, size_t size, size_t align);

    void realloc_aligned(void **ptr, size_t size, size_t align);

    void aligned_free(void *ptr);

    template <typename T> Allocator<T> create_allocator();


  private:
    std::atomic<size_t> _memory_used;
};

} // namespace diskann
