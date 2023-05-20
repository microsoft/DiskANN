// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <atomic>
#include <memory>
#include <vector>
#include <unordered_map>
#include "tsl/robin_set.h"
#include "windows_customizations.h"

namespace diskann
{

/// <summary>
/// The wrapper of std::allocator which tracks the memory usage
/// </summary>
template <class T> class Allocator
{
  public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    Allocator(std::atomic<size_t> *memory_used = nullptr)
    {
        _memory_used_in_bytes = memory_used;
    }

    template <class U = T> Allocator(const Allocator<U> &a)
    {
        _memory_used_in_bytes = a._memory_used_in_bytes;
    }

    Allocator &operator=(const Allocator &a)
    {
        _memory_used_in_bytes = a._memory_used_in_bytes;
    }

    DISKANN_DLLEXPORT T *allocate(std::size_t count, const T *hint = nullptr);

    DISKANN_DLLEXPORT void deallocate(T *ptr, std::size_t count);

    std::atomic<size_t> *_memory_used_in_bytes;

  private:
    void update_memory_usage(std::size_t count, bool is_allocation);

    std::allocator<T> _allocator;
};

template <class T> inline bool operator==(const Allocator<T> &a, const Allocator<T> &b)
{
    return a._memory_used_in_bytes == b._memory_used_in_bytes;
}

template <class _Ty, class _Alloc = Allocator<_Ty>> using vector = std::vector<_Ty, _Alloc>;

template <class Key, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>, class Allocator = Allocator<Key>,
          bool StoreHash = false, class GrowthPolicy = tsl::rh::power_of_two_growth_policy<2>>
using robin_set = tsl::robin_set<Key, Hash, KeyEqual, Allocator, StoreHash, GrowthPolicy>;

template <class _Kty, class _Ty, class _Hasher = std::hash<_Kty>, class _Keyeq = std::equal_to<_Kty>,
          class _Alloc = Allocator<std::pair<const _Kty, _Ty>>>
using unordered_map = std::unordered_map<_Kty, _Ty, _Hasher, _Keyeq, _Alloc>;

} // namespace diskann
